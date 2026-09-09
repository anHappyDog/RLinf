#!/usr/bin/env python3
"""Compare critic input heads on one frozen OpenPI feature cache."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn

from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.critic_batch import atomic_torch_save


class MeanTokenCritic(nn.Module):
    """Current RLinf critic: masked mean VLM feature followed by an MLP."""

    def __init__(self, feature_dim: int, state_dim: int):
        super().__init__()
        del state_dim
        self.value_head = ValueHead(
            input_dim=feature_dim,
            hidden_sizes=(1024, 512, 256),
            output_dim=1,
            activation="relu",
            bias_last=True,
        )

    def forward(
        self,
        pooled: torch.Tensor,
        state: torch.Tensor,
        prefix_out: torch.Tensor | None,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        del state, prefix_out, prefix_mask
        return self.value_head(pooled.float())


class StateFusionCritic(nn.Module):
    """Fuse normalized proprioception with the pooled VLM representation."""

    def __init__(self, feature_dim: int, state_dim: int):
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.value_head = ValueHead(
            input_dim=feature_dim + 128,
            hidden_sizes=(1024, 512, 256),
            output_dim=1,
            activation="relu",
            bias_last=True,
        )

    def forward(
        self,
        pooled: torch.Tensor,
        state: torch.Tensor,
        prefix_out: torch.Tensor | None,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        del prefix_out, prefix_mask
        fused = torch.cat(
            [self.feature_norm(pooled.float()), self.state_encoder(state.float())],
            dim=-1,
        )
        return self.value_head(fused)


class StateAttentionCritic(nn.Module):
    """Use proprioception to select useful visual-language prefix tokens."""

    def __init__(self, feature_dim: int, state_dim: int, attention_dim: int = 256):
        super().__init__()
        self.token_norm = nn.LayerNorm(feature_dim)
        self.token_projection = nn.Linear(feature_dim, attention_dim)
        self.pooled_projection = nn.Linear(feature_dim, attention_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.value_head = ValueHead(
            input_dim=attention_dim * 3,
            hidden_sizes=(512, 256),
            output_dim=1,
            activation="relu",
            bias_last=True,
        )
        self.scale = math.sqrt(attention_dim)

    def forward(
        self,
        pooled: torch.Tensor,
        state: torch.Tensor,
        prefix_out: torch.Tensor | None,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_out is None:
            raise ValueError("StateAttentionCritic requires token-level prefix_out.")
        tokens = self.token_projection(self.token_norm(prefix_out.float()))
        state_feature = self.state_encoder(state.float())
        scores = torch.einsum("bsd,bd->bs", tokens, state_feature) / self.scale
        scores = scores.masked_fill(~prefix_mask.to(torch.bool), -torch.inf)
        attended = torch.einsum("bs,bsd->bd", scores.softmax(-1), tokens)
        fused = torch.cat(
            [attended, self.pooled_projection(pooled.float()), state_feature], dim=-1
        )
        return self.value_head(fused)


_ARCHITECTURES = {
    "mean_token": MeanTokenCritic,
    "state_fusion": StateFusionCritic,
    "state_attention": StateAttentionCritic,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(_ARCHITECTURES),
        default=list(_ARCHITECTURES),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split-mode",
        choices=("episode", "none"),
        default="episode",
        help=(
            "Use complete auto-reset episodes for held-out evaluation, or fit "
            "all valid targets for a pure fixed-batch capacity test."
        ),
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("fp32", "bf16"),
        default="bf16",
        help="Head compute dtype; optimizer master parameters remain fp32.",
    )
    return parser.parse_args()


def _explained_variance(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target_variance = torch.var(target, correction=0)
    if target_variance <= 1e-12:
        return float("nan")
    residual_variance = torch.var(target - prediction, correction=0)
    return float(1.0 - residual_variance / target_variance)


def _weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    error = prediction - target
    absolute = error.abs()
    loss = torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _predict(
    model: nn.Module,
    cache: dict,
    indices: torch.Tensor,
    device: torch.device,
    compute_dtype: str,
) -> torch.Tensor:
    autocast_enabled = compute_dtype == "bf16" and device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        prediction = model(
            cache["pooled_prefix"][indices].to(device),
            cache["state"][indices].to(device),
            (
                cache["prefix_out"][indices].to(device)
                if cache["prefix_out"] is not None
                else None
            ),
            cache["prefix_mask"][indices].to(device),
        )
    return prediction.float()


def _heldout_groups(
    group_count: int,
    outcomes: torch.Tensor | None,
    complete: torch.Tensor | None,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    if outcomes is not None:
        eligible = complete if complete is not None else torch.ones_like(outcomes)
        successes = torch.where(outcomes & eligible)[0].tolist()
        failures = torch.where(~outcomes & eligible)[0].tolist()
        if len(successes) >= 2 and len(failures) >= 2:
            return [rng.choice(successes), rng.choice(failures)]
    candidates = (
        torch.where(complete)[0].tolist()
        if complete is not None and complete.any()
        else list(range(group_count))
    )
    rng.shuffle(candidates)
    return sorted(candidates[: max(1, len(candidates) // 4)])


def _evaluate(
    model: nn.Module,
    cache: dict,
    indices: torch.Tensor,
    device: torch.device,
    delta: float,
    compute_dtype: str,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        prediction = _predict(
            model,
            cache,
            indices,
            device,
            compute_dtype,
        )
        target = cache["returns"][indices].float().to(device)
        weights = (
            cache["sample_weights"][indices].float().to(device)
            if cache["sample_weights"] is not None
            else torch.ones_like(target)
        )
        loss = _weighted_huber(prediction, target, weights, delta)
    return {
        "loss": float(loss),
        "explained_variance": _explained_variance(target, prediction),
        "prediction_min": float(prediction.min()),
        "prediction_mean": float(prediction.mean()),
        "prediction_max": float(prediction.max()),
        "target_min": float(target.min()),
        "target_mean": float(target.mean()),
        "target_max": float(target.max()),
    }


def main() -> None:
    args = parse_args()
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    mask = cache["loss_mask"].to(torch.bool).reshape(-1)
    group_indices = cache.get("episode_indices", cache["trajectory_indices"])
    group_outcomes = cache.get("episode_outcomes", cache.get("trajectory_outcomes"))
    group_complete = cache.get("episode_complete")
    group_count = (
        int(group_outcomes.numel())
        if group_outcomes is not None
        else int(group_indices.max()) + 1
    )
    if args.split_mode == "episode":
        heldout = _heldout_groups(
            group_count,
            group_outcomes,
            group_complete,
            args.seed,
        )
        heldout_mask = torch.zeros_like(mask)
        for group_index in heldout:
            heldout_mask |= group_indices == group_index
        train_indices = torch.where(mask & ~heldout_mask)[0]
        test_indices = torch.where(mask & heldout_mask)[0]
        if train_indices.numel() == 0 or test_indices.numel() == 0:
            raise ValueError("Episode split produced an empty train or test set.")
    else:
        heldout = []
        train_indices = torch.where(mask)[0]
        test_indices = None
        if train_indices.numel() == 0:
            raise ValueError("Feature cache has no valid critic targets.")

    device = torch.device(args.device)
    feature_dim = int(cache["pooled_prefix"].shape[-1])
    state_dim = int(cache["state"].shape[-1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {
        "settings": vars(args)
        | {
            "split_unit": (
                "none"
                if args.split_mode == "none"
                else (
                    "auto_reset_episode" if "episode_indices" in cache else "trajectory"
                )
            ),
            "heldout_groups": heldout,
        },
        "architectures": {},
    }
    all_results["settings"]["features"] = str(args.features)
    all_results["settings"]["output_dir"] = str(args.output_dir)

    for architecture in args.architectures:
        torch.manual_seed(args.seed)
        model = _ARCHITECTURES[architecture](feature_dim, state_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        generator = torch.Generator().manual_seed(args.seed)
        history = []
        for epoch in range(args.epochs):
            model.train()
            order = train_indices[
                torch.randperm(train_indices.numel(), generator=generator)
            ]
            for start in range(0, order.numel(), args.batch_size):
                indices = order[start : start + args.batch_size]
                prediction = _predict(
                    model,
                    cache,
                    indices,
                    device,
                    args.compute_dtype,
                )
                target = cache["returns"][indices].float().to(device)
                weights = (
                    cache["sample_weights"][indices].float().to(device)
                    if cache["sample_weights"] is not None
                    else torch.ones_like(target)
                )
                loss = _weighted_huber(prediction, target, weights, args.huber_delta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                record: dict[str, object] = {
                    "epoch": epoch + 1,
                    "train": _evaluate(
                        model,
                        cache,
                        train_indices,
                        device,
                        args.huber_delta,
                        args.compute_dtype,
                    ),
                }
                if test_indices is not None:
                    record["heldout"] = _evaluate(
                        model,
                        cache,
                        test_indices,
                        device,
                        args.huber_delta,
                        args.compute_dtype,
                    )
                history.append(record)
                heldout_text = (
                    f" heldout_ev={record['heldout']['explained_variance']:.4f}"
                    if "heldout" in record
                    else ""
                )
                print(
                    f"{architecture} epoch={epoch + 1} "
                    f"train_ev={record['train']['explained_variance']:.4f} "
                    f"{heldout_text}",
                    flush=True,
                )

        all_results["architectures"][architecture] = history
        atomic_torch_save(
            {"architecture": architecture, "state_dict": model.cpu().state_dict()},
            args.output_dir / f"{architecture}.pt",
        )

    (args.output_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

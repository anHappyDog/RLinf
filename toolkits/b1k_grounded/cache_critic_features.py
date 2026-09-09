#!/usr/bin/env python3
"""Cache frozen OpenPI prefix features from an exported critic batch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.utils.critic_batch import atomic_torch_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Merged critic batch artifact.")
    parser.add_argument(
        "config",
        type=Path,
        help="Resolved RLinf config.yaml containing actor.model.",
    )
    parser.add_argument("output", type=Path, help="Output feature-cache path.")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pooled-only",
        action="store_true",
        help="Omit token-level prefix features to create a much smaller cache.",
    )
    return parser.parse_args()


def _flatten_time_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(
            f"Expected a [time, batch, ...] tensor, got {tuple(tensor.shape)}."
        )
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _observation_slice(
    forward_inputs: dict[str, torch.Tensor],
    start: int,
    stop: int,
    device: torch.device,
) -> Observation:
    flattened = {
        key: _flatten_time_batch(value)[start:stop].to(device)
        for key, value in forward_inputs.items()
    }
    images = {
        key.removeprefix("obs_image__"): value
        for key, value in flattened.items()
        if key.startswith("obs_image__")
    }
    image_masks = {
        key.removeprefix("obs_image_mask__"): value
        for key, value in flattened.items()
        if key.startswith("obs_image_mask__")
    }
    return Observation(
        images=images,
        image_masks=image_masks,
        state=flattened["obs_state"],
        tokenized_prompt=flattened["tokenized_prompt"],
        tokenized_prompt_mask=flattened["tokenized_prompt_mask"],
    )


def _flatten_optional(batch: dict[str, Any], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    return _flatten_time_batch(value) if isinstance(value, torch.Tensor) else None


def _episode_metadata(
    dones: torch.Tensor,
    terminations: torch.Tensor,
    *,
    time_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each chunk to an auto-reset episode and reduce its outcome."""
    if dones.shape[:2] != terminations.shape[:2]:
        raise ValueError("dones and terminations must share time and batch axes.")
    if time_size is not None:
        if dones.shape[0] < time_size:
            raise ValueError(
                f"Episode metadata has {dones.shape[0]} time steps, expected at "
                f"least {time_size}."
            )
        dones = dones[:time_size]
        terminations = terminations[:time_size]
    time_size, trajectory_count = dones.shape[:2]
    done_flags = dones.to(torch.bool).reshape(time_size, trajectory_count, -1).any(-1)
    termination_flags = (
        terminations.to(torch.bool).reshape(time_size, trajectory_count, -1).any(-1)
    )
    done_ids = done_flags.to(torch.int64)
    local_episode_ids = done_ids.cumsum(dim=0) - done_ids
    episode_counts = local_episode_ids.max(dim=0).values + 1
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), episode_counts[:-1].cumsum(0)]
    )
    episode_ids = local_episode_ids + offsets.unsqueeze(0)
    episode_count = int(episode_counts.sum())
    outcomes = torch.zeros(episode_count, dtype=torch.bool)
    complete = torch.zeros(episode_count, dtype=torch.bool)
    flat_ids = episode_ids.reshape(-1)
    outcomes.scatter_reduce_(
        0, flat_ids, termination_flags.reshape(-1), reduce="amax", include_self=True
    )
    complete.scatter_reduce_(
        0, flat_ids, done_flags.reshape(-1), reduce="amax", include_self=True
    )
    return flat_ids, outcomes, complete


def main() -> None:
    args = parse_args()
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be positive.")

    artifact = torch.load(args.batch, map_location="cpu", weights_only=False)
    batch = artifact["batch"]
    forward_inputs = batch["forward_inputs"]
    time_size, trajectory_count = batch["returns"].shape[:2]
    sample_count = time_size * trajectory_count

    cfg = OmegaConf.load(args.config)
    model = get_model(cfg.actor.model)
    if model is None:
        raise ValueError(f"Unsupported model type {cfg.actor.model.model_type!r}.")
    device = torch.device(args.device)
    model = model.to(device).eval()

    pooled_features = []
    prefix_features = []
    prefix_masks = []
    with torch.inference_mode():
        for start in range(0, sample_count, args.micro_batch_size):
            stop = min(start + args.micro_batch_size, sample_count)
            observation = _observation_slice(forward_inputs, start, stop, device)
            prefix_out, prefix_mask, _ = model.model.build_prefix_cache(observation)
            mask = prefix_mask.to(prefix_out.dtype).unsqueeze(-1)
            pooled = (prefix_out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled_features.append(pooled.to(torch.bfloat16).cpu())
            prefix_masks.append(prefix_mask.cpu())
            if not args.pooled_only:
                prefix_features.append(prefix_out.to(torch.bfloat16).cpu())
            print(f"Cached prefix features {stop}/{sample_count}", flush=True)

    trajectory_indices = torch.arange(trajectory_count).repeat(time_size)
    time_indices = torch.arange(time_size).repeat_interleave(trajectory_count)
    episode_indices, episode_outcomes, episode_complete = _episode_metadata(
        batch["dones"], batch["terminations"], time_size=time_size
    )
    cache = {
        "metadata": {
            "format_version": 1,
            "source_batch": str(args.batch),
            "source_global_step": artifact["metadata"]["global_step"],
            "time_size": time_size,
            "trajectory_count": trajectory_count,
            "sample_count": sample_count,
            "model_path": str(cfg.actor.model.model_path),
            "feature_dtype": "bfloat16",
            "contains_token_features": not args.pooled_only,
        },
        "pooled_prefix": torch.cat(pooled_features),
        "prefix_mask": torch.cat(prefix_masks),
        "prefix_out": torch.cat(prefix_features) if prefix_features else None,
        "state": _flatten_time_batch(forward_inputs["obs_state"]),
        "returns": _flatten_time_batch(batch["returns"]),
        "loss_mask": _flatten_time_batch(batch["loss_mask"]),
        "sample_weights": _flatten_optional(batch, "sample_weights"),
        "prev_values": (
            _flatten_time_batch(batch["prev_values"][:time_size])
            if isinstance(batch.get("prev_values"), torch.Tensor)
            else None
        ),
        "subtask_ids": _flatten_optional(batch, "subtask_ids"),
        "trajectory_indices": trajectory_indices,
        "time_indices": time_indices,
        "trajectory_outcomes": artifact.get("trajectory_outcomes"),
        "episode_indices": episode_indices,
        "episode_outcomes": episode_outcomes,
        "episode_complete": episode_complete,
    }
    atomic_torch_save(cache, args.output)
    print(
        f"Saved {sample_count} samples to {args.output} with pooled shape "
        f"{tuple(cache['pooled_prefix'].shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

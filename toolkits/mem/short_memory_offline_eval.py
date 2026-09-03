#!/usr/bin/env python3
"""Measure short-memory sensitivity with paired demonstration losses."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_data_loader import (
    create_behavior_sft_data_loader,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    load_base_safetensors,
    load_full_wrapper_weights,
    resolve_full_weights,
    resolve_model_safetensors,
)

_CONDITIONS = ("correct", "repeat_current", "shuffle_past")


def ablate_observation_history(observation: Observation, condition: str) -> Observation:
    """Return a paired history control without changing masks or timestamps."""
    if condition not in _CONDITIONS:
        raise ValueError(f"Unknown short-memory condition: {condition!r}.")
    if condition == "correct":
        return observation
    if observation.history_frame_mask is None or observation.history_states is None:
        raise ValueError("Short-memory controls require history tensors.")

    frame_mask = observation.history_frame_mask

    def _control(values: torch.Tensor) -> torch.Tensor:
        controlled = values.clone()
        if condition == "repeat_current":
            content_mask = frame_mask.reshape(
                *frame_mask.shape,
                *((1,) * (values.ndim - frame_mask.ndim)),
            )
            current = values[:, -1:].expand_as(values)
            return torch.where(content_mask, current, controlled)

        for batch_index, mask in enumerate(frame_mask):
            valid_past = torch.nonzero(mask[:-1], as_tuple=False).flatten()
            controlled[batch_index, valid_past] = values[
                batch_index, valid_past.flip(0)
            ]
        return controlled

    return dataclasses.replace(
        observation,
        images={key: _control(value) for key, value in observation.images.items()},
        history_states=_control(observation.history_states),
    )


def summarize_losses(losses: dict[str, list[float]]) -> dict:
    """Aggregate paired loss differences and the directional causal gate."""
    if not losses["correct"]:
        raise ValueError("At least one paired loss is required.")
    means = {
        condition: sum(values) / len(values) for condition, values in losses.items()
    }
    per_sample = [
        {condition: losses[condition][index] for condition in _CONDITIONS}
        for index in range(len(losses["correct"]))
    ]
    correct_wins = {
        condition: sum(
            correct < controlled
            for correct, controlled in zip(
                losses["correct"], losses[condition], strict=True
            )
        )
        for condition in _CONDITIONS[1:]
    }
    return {
        "num_samples": len(losses["correct"]),
        "mean_loss": means,
        "mean_delta_vs_correct": {
            condition: means[condition] - means["correct"]
            for condition in _CONDITIONS[1:]
        },
        "correct_win_count": correct_wins,
        "directional_gate": all(
            means["correct"] < means[condition] for condition in _CONDITIONS[1:]
        ),
        "per_sample": per_sample,
    }


def move_observation_to_device(
    observation: Observation, device: torch.device
) -> Observation:
    def _move(value):
        return value.to(device) if isinstance(value, torch.Tensor) else value

    return Observation(
        images={key: _move(value) for key, value in observation.images.items()},
        image_masks={
            key: _move(value) for key, value in observation.image_masks.items()
        },
        state=_move(observation.state),
        tokenized_prompt=_move(observation.tokenized_prompt),
        tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
        token_ar_mask=_move(observation.token_ar_mask),
        token_loss_mask=_move(observation.token_loss_mask),
        pcd_xyz=_move(observation.pcd_xyz),
        history_states=_move(observation.history_states),
        history_frame_mask=_move(observation.history_frame_mask),
        history_time_offsets=_move(observation.history_time_offsets),
    )


def load_short_memory_model(checkpoint: Path, device: torch.device):
    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        short_memory=True,
        short_memory_temporal_layers=(3, 7, 11, 15),
        short_memory_drop_history_layer=15,
        short_memory_state_dim=23,
        discrete_state_input=False,
    )
    core = config.create()
    full_weights = resolve_full_weights(checkpoint)
    if full_weights is not None:
        wrapper = OpenPiPytorchSFTActionModel(
            core,
            num_steps=5,
            action_env_dim=23,
        )
        load_full_wrapper_weights(wrapper, full_weights, expect_rlt=False)
        core = wrapper.model
        source = full_weights
    else:
        safetensors = resolve_model_safetensors(checkpoint)
        if safetensors is None:
            raise FileNotFoundError(
                f"No supported checkpoint found under {checkpoint}."
            )
        load_base_safetensors(
            core,
            safetensors,
            allow_missing_prefixes=("history_state_encoder.",),
        )
        source = safetensors
    return core.to(device=device, dtype=torch.bfloat16).eval(), source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument(
        "--asset-id",
        default="physical-intelligence/behavior",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    device = torch.device("cuda")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model, checkpoint_source = load_short_memory_model(checkpoint, device)
    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=args.dataset_root,
        assets_dir=args.assets_dir,
        asset_id=args.asset_id,
        model_path=str(checkpoint),
        config_name="pi05_behavior",
        repo_id="behavior-1k/2025-challenge-demos",
        tasks=[args.task],
        modalities=["rgb"],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=1,
        num_workers=0,
        fine_grained_level=0,
        tolerance_s=1.0e-4,
        shuffle=False,
        seed=args.seed,
        skill_labels=None,
        use_skill=False,
        prompt_source="mixed",
        primitive_prompt_probability=0.5,
        mixed_boundary_fallback_to_task=False,
        history_length=6,
        history_frame_stride=30,
        history_state_dim=23,
        discrete_state_input=False,
        enable_gap=True,
        allow_left=0,
        allow_right=0,
        dist_rank=0,
        dist_world_size=1,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    losses = {condition: [] for condition in _CONDITIONS}
    skipped = 0
    with torch.no_grad():
        for observation, actions in loader:
            if not torch.all(observation.history_frame_mask):
                skipped += 1
                continue
            observation = move_observation_to_device(observation, device)
            actions = actions.to(device)
            noise = torch.randn(
                actions.shape,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            time = torch.full((actions.shape[0],), 0.5, device=device)
            for condition in _CONDITIONS:
                controlled = ablate_observation_history(observation, condition)
                loss = model.compute_loss(
                    controlled,
                    actions,
                    train=False,
                    noise=noise,
                    time=time,
                )
                losses[condition].append(float(loss.mean()))
            if len(losses["correct"]) == args.num_samples:
                break

    metrics = summarize_losses(losses)
    metrics.update(
        checkpoint=str(checkpoint_source),
        task=args.task,
        seed=args.seed,
        skipped_incomplete_history=skipped,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

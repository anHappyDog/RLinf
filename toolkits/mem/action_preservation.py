#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure π0.5 action-output drift after high-level text-tail training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
from openpi.models.tokenizer import PaligemmaTokenizer

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    R1ManifestEntry,
    read_r1_manifest,
)
from rlinf.data.datasets.openpi_rlinf.behavior.high_level_state import (
    BehaviorStateReader,
    StateNormStats,
    read_state_norm_stats,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from toolkits.mem.r0_overfit import load_pi05_model
from toolkits.mem.r1_train import _VIDEO_KEYS, decode_video_frame


@dataclasses.dataclass(frozen=True)
class ActionCalibrationSample:
    """One original π0.5 action prompt and its matching observation."""

    manifest: R1ManifestEntry
    images: dict[str, np.ndarray]
    state: np.ndarray
    tokens: np.ndarray
    token_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--state-norm-stats", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tail-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument("--max-relative-rmse", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Compare deterministic base and tuned action samples."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Action-preservation evaluation requires a CUDA GPU.")
    if args.sample_count <= 0 or args.batch_size <= 0 or args.denoise_steps <= 0:
        raise ValueError("Sample count, batch size, and denoise steps must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = read_r1_manifest(args.calibration_manifest)
    if not entries:
        raise ValueError("Calibration manifest is empty.")
    selected = random.Random(args.seed).sample(
        entries, k=min(args.sample_count, len(entries))
    )
    selected.sort(key=lambda entry: (entry.episode_index, entry.frame_index))
    state_stats = read_state_norm_stats(args.state_norm_stats)
    samples = load_action_calibration_samples(
        args.dataset_root,
        selected,
        state_stats,
        max_token_len=args.max_token_len,
    )

    device = torch.device("cuda")
    model = load_pi05_model(args.model_path, device, args.max_token_len)
    noise_generator = torch.Generator().manual_seed(args.seed)
    noises = torch.randn(
        len(samples),
        model.action_horizon,
        model.action_dim,
        generator=noise_generator,
    )
    base_actions = sample_calibration_actions(
        model,
        samples,
        noises,
        batch_size=args.batch_size,
        denoise_steps=args.denoise_steps,
        device=device,
    )
    loaded_names = load_trainable_tail(model, args.tail_checkpoint)
    tuned_actions = sample_calibration_actions(
        model,
        samples,
        noises,
        batch_size=args.batch_size,
        denoise_steps=args.denoise_steps,
        device=device,
    )

    metrics = action_drift_metrics(base_actions, tuned_actions)
    metrics.update(
        {
            "action_preservation_pass": metrics["relative_rmse"]
            <= args.max_relative_rmse,
            "max_relative_rmse": args.max_relative_rmse,
            "sample_count": len(samples),
            "denoise_steps": args.denoise_steps,
            "loaded_parameter_tensor_count": len(loaded_names),
            "tail_checkpoint": str(args.tail_checkpoint),
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "noise": noises,
            "base_actions": base_actions,
            "tuned_actions": tuned_actions,
        },
        args.output_dir / "action_outputs.pt",
    )
    print(json.dumps(metrics, indent=2))


def load_action_calibration_samples(
    dataset_root: Path,
    entries: list[R1ManifestEntry],
    state_stats: StateNormStats,
    *,
    max_token_len: int,
) -> list[ActionCalibrationSample]:
    """Load observations with the original π0.5 ``Action:`` prompt format."""
    tokenizer = PaligemmaTokenizer(max_len=max_token_len)
    state_reader = BehaviorStateReader(dataset_root)
    samples = []
    for entry in entries:
        images = {}
        for model_key, video_key in _VIDEO_KEYS.items():
            video_path = (
                dataset_root
                / "videos"
                / f"task-{entry.task_index:04d}"
                / video_key
                / f"episode_{entry.episode_index:08d}.mp4"
            )
            images[model_key] = decode_video_frame(video_path, entry.frame_index)
        state = state_stats.normalize(
            state_reader.read(entry.episode_index, entry.frame_index)
        )
        tokens, token_mask = tokenizer.tokenize(entry.task, state)
        samples.append(
            ActionCalibrationSample(
                manifest=entry,
                images=images,
                state=state,
                tokens=tokens,
                token_mask=token_mask,
            )
        )
    return samples


@torch.no_grad()
def sample_calibration_actions(
    model: Pi0,
    samples: list[ActionCalibrationSample],
    noises: torch.Tensor,
    *,
    batch_size: int,
    denoise_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample deterministic action chunks for all calibration observations."""
    model.eval()
    outputs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        observation = collate_action_samples(batch, device)
        batch_noise = noises[start : start + len(batch)].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = model.sample_actions(
                observation, num_steps=denoise_steps, noise=batch_noise
            )
        outputs.append(actions.float().cpu())
    return torch.cat(outputs)


def collate_action_samples(
    samples: list[ActionCalibrationSample], device: torch.device
) -> Observation:
    """Build a padded 32D π0.5 action observation batch."""
    states = np.stack([sample.state for sample in samples])
    states = np.pad(states, ((0, 0), (0, 32 - states.shape[1])))
    observation = Observation.from_dict(
        {
            "image": {
                key: torch.from_numpy(
                    np.stack([sample.images[key] for sample in samples])
                )
                for key in _VIDEO_KEYS
            },
            "image_mask": {
                key: torch.ones(len(samples), dtype=torch.bool) for key in _VIDEO_KEYS
            },
            "state": torch.from_numpy(states).float(),
            "tokenized_prompt": torch.from_numpy(
                np.stack([sample.tokens for sample in samples])
            ).long(),
            "tokenized_prompt_mask": torch.from_numpy(
                np.stack([sample.token_mask for sample in samples])
            ).bool(),
        }
    )
    return Observation(
        images={key: value.to(device) for key, value in observation.images.items()},
        image_masks={
            key: value.to(device) for key, value in observation.image_masks.items()
        },
        state=observation.state.to(device),
        tokenized_prompt=observation.tokenized_prompt.to(device),
        tokenized_prompt_mask=observation.tokenized_prompt_mask.to(device),
    )


def load_trainable_tail(model: Pi0, checkpoint_path: Path) -> set[str]:
    """Copy a saved R1 tail into the corresponding model parameters."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    unexpected = set(checkpoint).difference(parameters)
    if unexpected:
        raise ValueError(f"Tail checkpoint has unexpected keys: {sorted(unexpected)}")
    with torch.no_grad():
        for name, value in checkpoint.items():
            if parameters[name].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {parameters[name].shape}, "
                    f"got {value.shape}."
                )
            parameters[name].copy_(value.to(parameters[name].device))
    return set(checkpoint)


def action_drift_metrics(
    base_actions: torch.Tensor, tuned_actions: torch.Tensor
) -> dict[str, float]:
    """Return scale-aware action drift statistics."""
    if base_actions.shape != tuned_actions.shape:
        raise ValueError("Base and tuned action tensors must have the same shape.")
    difference = tuned_actions.float() - base_actions.float()
    rmse = difference.square().mean().sqrt()
    base_rms = base_actions.float().square().mean().sqrt()
    flat_base = base_actions.float().flatten(1)
    flat_tuned = tuned_actions.float().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(flat_base, flat_tuned).mean()
    first_difference = difference[:, 0]
    first_base = base_actions[:, 0].float()
    return {
        "rmse": rmse.item(),
        "relative_rmse": (rmse / (base_rms + 1e-8)).item(),
        "mean_cosine_similarity": cosine.item(),
        "max_absolute_drift": difference.abs().max().item(),
        "first_action_relative_rmse": (
            first_difference.square().mean().sqrt()
            / (first_base.square().mean().sqrt() + 1e-8)
        ).item(),
    }


if __name__ == "__main__":
    main()

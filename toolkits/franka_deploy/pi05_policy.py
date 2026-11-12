# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Official Pi0.5 inference for canonical pickup checkpoints with joint deltas."""

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np

from .model_adapters import preprocess_image


def quantile_transform(
    values: np.ndarray, stats: dict, *, inverse: bool = False
) -> np.ndarray:
    """Apply the training quantile transform without clipping outlying values."""
    source = np.asarray(values)
    source = source.astype(np.result_type(source.dtype, np.float32), copy=True)
    if stats["transform"] != "quantile":
        raise ValueError("Pi0.5 canonical pickup requires quantile statistics.")
    lo = np.asarray(stats["q01"], dtype=source.dtype)
    hi = np.asarray(stats["q99"], dtype=source.dtype)
    if lo.shape != (8,) or hi.shape != (8,) or source.shape[-1:] != (8,):
        raise ValueError(
            "Expected eight-dimensional canonical values and quantile statistics."
        )
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise ValueError(
            "Expected finite q99 >= q01 for every normalization dimension."
        )
    center, scale = (lo + hi) / 2, (hi - lo) / 2
    scale = np.where(scale == 0, 1, scale)
    result = source * scale + center if inverse else (source - center) / scale
    indices = stats.get("passthrough_indices", [])
    result[..., indices] = source[..., indices]
    return result


class Pi05QtargetPolicy:
    """Load trained Pi0.5 weights and decode each chunk relative to its anchor qpos."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        action_dt: float,
        device: str = "cuda:0",
        denoise_steps: int = 10,
        seed: int = 0,
    ) -> None:
        import torch
        from openpi import transforms
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.training.config import ModelTransformFactory

        if not np.isfinite(action_dt) or action_dt <= 0:
            raise ValueError("action_dt must be explicit positive seconds per action.")
        if denoise_steps < 1:
            raise ValueError("denoise_steps must be positive.")
        checkpoint = Path(checkpoint).resolve()
        actor = (
            checkpoint
            if (checkpoint / "upstream_assets").is_dir()
            else checkpoint / "actor"
        )
        assets = actor / "upstream_assets"
        contract = json.loads((assets / "canonical_qtarget.json").read_text())
        recipe = contract["recipe"]
        normalization = contract["normalization"]
        if (
            recipe["name"] != "canonical_qtarget"
            or recipe["state_layout"] != "qpos_width8"
        ):
            raise ValueError("Expected a canonical_qtarget/qpos_width8 checkpoint.")
        if normalization["model_adapter"] != {"action_representation": "joint_delta"}:
            raise ValueError(
                "Expected canonical joint_delta encoding for this Pi0.5 checkpoint."
            )
        self.stats = normalization["schemas"]["molmobot"]
        for field in ("state", "action"):
            stats = self.stats[field]
            quantile_transform(np.zeros(8, dtype=np.float32), stats)
            if np.asarray(stats["count"]).shape != (8,) or np.any(
                np.asarray(stats["count"]) <= 0
            ):
                raise ValueError(
                    f"Expected all eight Molmo {field} dimensions to be valid."
                )
        with (assets / "train_config.pkl").open("rb") as stream:
            config = pickle.load(stream)
        config = dataclasses.replace(
            config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
        )
        self.horizon = config.model.action_horizon
        if (
            not config.model.pi05
            or config.model.action_dim != 32
            or self.horizon != 16
            or not config.model.discrete_state_input
            or config.model.max_token_len != 200
        ):
            raise ValueError(
                "Expected trained Pi0.5 action32/H16 with discrete state and 200 tokens."
            )
        if (
            normalization["action_horizon"] != self.horizon
            or normalization["action_dim"] != 8
        ):
            raise ValueError(
                "Expected H16 normalization of eight physical action dimensions."
            )
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        self.core = PI0Pytorch(config.model)
        source = actor / "model_state_dict" / "full_weights.pt"
        state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if any(not key.startswith("core.") for key in state):
            raise ValueError("Expected only core.* parameters in full_weights.pt.")
        self.core.load_state_dict(
            {key.removeprefix("core."): value for key, value in state.items()},
            strict=True,
        )
        del state
        # Preserve the constructor's selective BF16/FP32 layout when restoring weights.
        self.core.to(device).eval()
        self.core.gradient_checkpointing_disable()
        self.transform = transforms.compose(
            ModelTransformFactory()(config.model).inputs
        )
        self.denoise_steps = denoise_steps
        self.device = torch.device(device)
        self.metadata = {
            "adapter": "official_pi05_canonical_qtarget",
            "checkpoint_actor_dir": str(actor),
            "horizon": self.horizon,
            "model_action_dim": 32,
            "action_dt": float(action_dt),
            "denoise_steps": denoise_steps,
            "pi05": True,
            "discrete_state_input": True,
            "action_semantics": "absolute joint targets in radians; continuous closedness 0=open,1=closed",
            "state_semantics": "measured qpos7 rad plus measured gripper aperture metres; all eight valid",
            "model_action_representation": "joint_delta",
            "gripper_feedback_used": True,
            "legacy_state_width_mask0": False,
            "normalization_schema": "molmobot",
            "normalization_transform": "quantile",
            "seed": seed,
        }

    def prepare_observation(self, request: dict) -> dict:
        """Normalize real state8 before official Pi0.5 tokenization and padding."""
        qpos = np.asarray(request["qpos"], dtype=np.float32)
        width = float(request["gripper_width_m"])
        prompt = request["prompt"]
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError(
                "qpos must contain seven finite measured joint angles in radians."
            )
        if not np.isfinite(width) or width < 0:
            raise ValueError(
                "gripper_width_m must be a finite nonnegative measurement."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty original task string.")
        state = quantile_transform(np.r_[qpos, np.float32(width)], self.stats["state"])
        return self.transform(
            {
                "state": state.astype(np.float32),
                "image": {
                    "base_0_rgb": preprocess_image(request["images"]["external"]),
                    "left_wrist_0_rgb": preprocess_image(request["images"]["wrist"]),
                    "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                },
                "image_mask": {
                    "base_0_rgb": True,
                    "left_wrist_0_rgb": True,
                    "right_wrist_0_rgb": False,
                },
                "prompt": prompt,
            }
        )

    def decode_actions(self, predicted: np.ndarray, anchor_qpos: np.ndarray) -> dict:
        """Restore absolute H16 targets using the same observed qpos for every frame."""
        predicted = np.asarray(predicted)
        if predicted.shape != (self.horizon, 32):
            raise ValueError("Expected a model output with shape H16 x 32.")
        actions = quantile_transform(
            predicted[:, :8], self.stats["action"], inverse=True
        )
        actions[:, :7] += np.asarray(anchor_qpos, dtype=np.float32)[None, :]
        if not np.isfinite(actions).all():
            raise ValueError("Expected finite decoded canonical actions.")
        return {
            "qtarget": actions[:, :7],
            "gripper_closedness": np.clip(actions[:, 7], 0, 1),
            "metadata": dict(self.metadata),
        }

    def infer(self, request: dict) -> dict:
        """Predict an H16 chunk and return absolute joint targets and closedness."""
        import torch
        from openpi.models.model import Observation

        prepared = self.prepare_observation(request)
        observation = Observation.from_dict(
            {
                "state": torch.as_tensor(prepared["state"][None], device=self.device),
                "image": {
                    key: torch.as_tensor(value[None], device=self.device)
                    for key, value in prepared["image"].items()
                },
                "image_mask": {
                    key: torch.as_tensor(np.asarray([value]), device=self.device)
                    for key, value in prepared["image_mask"].items()
                },
                "tokenized_prompt": torch.as_tensor(
                    prepared["tokenized_prompt"][None], device=self.device
                ),
                "tokenized_prompt_mask": torch.as_tensor(
                    prepared["tokenized_prompt_mask"][None], device=self.device
                ),
            }
        )
        with torch.inference_mode():
            predicted = (
                self.core.sample_actions(
                    self.device, observation, num_steps=self.denoise_steps
                )[0]
                .float()
                .cpu()
                .numpy()
            )
        return self.decode_actions(
            predicted, np.asarray(request["qpos"], dtype=np.float32)
        )

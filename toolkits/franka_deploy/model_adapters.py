# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Official OpenPI inference for legacy canonical Franka pickup checkpoints."""

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preserve the MolmoBot 360x640 center crop and 224x224 letterbox."""
    from openpi_client.image_tools import resize_with_pad

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an HxWx3 RGB uint8 image.")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Expected a nonempty RGB image.")
    scale = max(360 / height, 640 / width)
    target_height, target_width = int(height * scale), int(width * scale)
    resized = Image.fromarray(image).resize(
        (target_width, target_height), resample=Image.Resampling.BILINEAR
    )
    left, top = (target_width - 640) // 2, (target_height - 360) // 2
    cropped = np.asarray(resized.crop((left, top, left + 640, top + 360)))
    return resize_with_pad(cropped, 224, 224)


def normalize(values: np.ndarray, stats: dict, *, inverse: bool = False) -> np.ndarray:
    """Apply the checkpoint's eight-dimensional mean/std statistics."""
    if stats.get("transform", "mean_std") != "mean_std":
        raise ValueError("This legacy adapter requires checkpoint mean_std statistics.")
    values = np.asarray(values, dtype=np.float32)
    mean, std = (
        np.asarray(stats["mean"], np.float32),
        np.asarray(stats["std"], np.float32),
    )
    if mean.shape != (8,) or std.shape != (8,) or values.shape[-1] != 8:
        raise ValueError("Expected canonical eight-dimensional values and statistics.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Expected finite means and nonnegative standard deviations.")
    scale = np.where(std == 0, 1, std)
    result = values * scale + mean if inverse else (values - mean) / scale
    indices = stats.get("passthrough_indices", [])
    result[..., indices] = values[..., indices]
    return result


class CanonicalQtargetPolicy:
    """Read raw core.* weights without altering the trained canonical contract.

    This adapter deliberately reproduces the legacy Molmo route whose width was
    masked during training. Measured hardware width is accepted but not used.
    """

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
        if normalization.get("model_adapter") not in (None, {}, "absolute"):
            raise ValueError(
                "This adapter supports only legacy absolute targets, not model-adapted checkpoints."
            )
        self.stats = normalization["schemas"]["molmobot"]
        if self.stats["state"]["count"][7] != 0:
            raise ValueError(
                "This adapter only supports legacy Molmo width-mask0 checkpoints."
            )
        with (assets / "train_config.pkl").open("rb") as stream:
            config = pickle.load(stream)
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, pytorch_compile_mode=None),
        )
        self.horizon = config.model.action_horizon
        if config.model.pi05 or config.model.action_dim != 8 or self.horizon != 16:
            raise ValueError("Expected the trained pi0 action8/H16 architecture.")
        if normalization["action_horizon"] != self.horizon:
            raise ValueError(
                "Normalization horizon differs from checkpoint model horizon."
            )
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        self.core = PI0Pytorch(config.model)
        source = actor / "model_state_dict" / "full_weights.pt"
        state = torch.load(source, map_location="cpu", weights_only=True)
        if any(not key.startswith("core.") for key in state):
            raise ValueError("Expected only core.* parameters in full_weights.pt.")
        self.core.load_state_dict(
            {key.removeprefix("core."): value for key, value in state.items()},
            strict=True,
        )
        del state
        # Construction established the original selective BF16/FP32 layout.
        # Recasting after restoration would round the trained FP32 norm weights.
        self.core.to(device).eval()
        self.core.gradient_checkpointing_disable()
        self.transform = transforms.compose(
            ModelTransformFactory()(config.model).inputs
        )
        self.denoise_steps = denoise_steps
        self.device = torch.device(device)
        self.metadata = {
            "adapter": "official_canonical_qtarget_legacy_molmo",
            "checkpoint_actor_dir": str(actor),
            "horizon": self.horizon,
            "action_dt": float(action_dt),
            "action_dt_source": "explicit_server_argument",
            "denoise_steps": denoise_steps,
            "action_semantics": "absolute joint targets in radians; closedness 0=open,1=closed",
            "state_semantics": "measured qpos7; legacy width masked to normalized zero",
            "gripper_feedback_used": False,
            "legacy_state_width_mask0": True,
            "normalization_schema": "molmobot",
            "seed": seed,
        }

    def prepare_observation(self, request: dict) -> dict:
        """Translate role-labelled RGB and hardware state to trained model inputs."""
        qpos = np.asarray(request["qpos"], dtype=np.float32)
        width = float(request["gripper_width_m"])
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError(
                "qpos must contain seven finite measured joint angles in radians."
            )
        if not np.isfinite(width) or width < 0:
            raise ValueError(
                "gripper_width_m must be a finite nonnegative measurement."
            )
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty original task string.")
        state = normalize(np.r_[qpos, width], self.stats["state"])
        state[7] = 0.0
        return self.transform(
            {
                "state": state,
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

    def infer(self, request: dict) -> dict:
        """Generate a full native-horizon chunk; leave physical dispatch to the client."""
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
        actions = normalize(predicted, self.stats["action"], inverse=True)
        if actions.shape != (self.horizon, 8) or not np.isfinite(actions).all():
            raise ValueError("Expected a finite H16 x action8 model output.")
        return {
            "qtarget": actions[:, :7],
            "gripper_closedness": np.clip(actions[:, 7], 0, 1),
            "metadata": dict(self.metadata),
        }


def load_policy(model_kind: str, checkpoint: str | Path, **kwargs):
    """Require a model-specific contract instead of guessing the base gripper map."""
    if model_kind != "canonical-qtarget":
        raise ValueError(
            "MolmoBot base deployment is not supported until its measured gripper mapping is confirmed."
        )
    return CanonicalQtargetPolicy(checkpoint, **kwargs)

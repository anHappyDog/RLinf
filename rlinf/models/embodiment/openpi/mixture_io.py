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

"""Observation preparation and native action decoding for pickup mixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_map

from rlinf.data.datasets.openpi.control_schema import (
    ControlAdapter,
    normalize,
    unnormalize,
)

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype in (torch.float16, torch.bfloat16):
            value = value.float()
    return np.asarray(value)


def resize_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Resize HWC RGB (uint8 or float in [0, 1]) to 224, pad, and map to [-1, 1]."""
    image = _numpy(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, found shape {image.shape}")
    pixels = torch.as_tensor(image.astype(np.float32))
    if image.dtype == np.uint8:
        pixels = pixels / 255
    height, width = pixels.shape[:2]
    scale = 224 / max(height, width)
    resized = torch.nn.functional.interpolate(
        pixels.permute(2, 0, 1)[None],
        size=(round(height * scale), round(width * scale)),
        mode="bilinear",
        align_corners=False,
    )[0]
    dh, dw = 224 - resized.shape[-2], 224 - resized.shape[-1]
    padded = torch.nn.functional.pad(
        resized, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    )
    return padded.permute(1, 2, 0).numpy() * 2 - 1


def build_prompt(schema_id: str, control_mode: str, fps: float, task: str) -> str:
    """Describe the same native control contract during training and inference."""
    return f"Schema: {schema_id}; control: {control_mode}; native fps: {fps:g}. {task}"


def build_observation(
    normalized_state: np.ndarray,
    images: dict[str, np.ndarray | torch.Tensor],
    prompt: str,
    tokenizer: Any,
) -> dict:
    """Build one model observation from padded state and model-slot HWC images."""
    tokens, token_mask = tokenizer.tokenize(prompt, normalized_state)
    return {
        "image": {
            slot: resize_image(images[slot])
            if slot in images
            else np.zeros((224, 224, 3), dtype=np.float32)
            for slot in IMAGE_KEYS
        },
        "image_mask": {slot: slot in images for slot in IMAGE_KEYS},
        "state": normalized_state,
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": token_mask,
    }


class MixturePolicyIO:
    """Use one prepared training schema for native-controller inference.

    ``manifest_dir`` accepts the prepared directory or its normalization JSON.
    Env batches contain ``states``, ``task_descriptions``, and either ``images``
    keyed by model slots or the conventional ``main_images``, ``wrist_images``,
    ``extra_view_images``. Images are HWC uint8 or float in [0, 1]. Decoding takes
    the original observation state explicitly; normalized state is never used as
    the joint reference.
    """

    def __init__(
        self,
        manifest_dir: str | Path,
        schema_id: str,
        *,
        action_dim: int = 32,
        max_token_len: int = 200,
    ) -> None:
        path = Path(manifest_dir)
        if path.is_dir():
            path = path / "normalization.json"
        contents = path.read_bytes()
        contract_path = path.parent / "contract.json"
        if contract_path.exists():
            expected = json.loads(contract_path.read_text())["files"][
                "normalization.json"
            ]
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError(
                    "Prepared normalization.json differs from its contract"
                )
        self.manifest = json.loads(contents)
        expected_dim = self.manifest.get("action_dim")
        if expected_dim is not None and int(expected_dim) != action_dim:
            raise ValueError(
                f"Prepared action_dim {expected_dim} differs from model {action_dim}"
            )
        self.schema_id = schema_id
        self.schema = self.manifest["schemas"][schema_id]
        self.adapter = ControlAdapter.from_config(self.schema.get("adapter"))
        self.action_dim, self.max_token_len = action_dim, max_token_len
        self.native_action_dim = self._feature_dim(self.schema["action"])
        state_dim = self._feature_dim(self.schema["state"])
        if max(self.native_action_dim, state_dim) > action_dim:
            raise ValueError(
                f"Schema {schema_id} exceeds model dimension {action_dim}: "
                f"action={self.native_action_dim}, state={state_dim}"
            )
        self._tokenizer = None

    @staticmethod
    def _feature_dim(stats: dict) -> int:
        return len(
            stats["q01"] if stats.get("transform") == "quantile" else stats["mean"]
        )

    def prepare_observation(
        self, raw_state: np.ndarray, images: dict, task: str
    ) -> dict:
        """Encode one native state and its configured camera views."""
        if self._tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer

            self._tokenizer = PaligemmaTokenizer(self.max_token_len)
        state = normalize(
            self.adapter.encode_state(_numpy(raw_state)), self.schema["state"]
        )
        padded = np.zeros(self.action_dim, dtype=np.float32)
        padded[: len(state)] = state
        cameras = {slot: images[slot] for slot in self.schema["cameras"]}
        prompt = build_prompt(
            self.schema_id, self.schema["control_mode"], self.schema["fps"], task
        )
        return build_observation(padded, cameras, prompt, self._tokenizer)

    def batch_observation(self, env_obs: dict) -> dict:
        """Prepare a native env-worker batch without retaining mutable state."""
        states = _numpy(env_obs["states"])
        if states.ndim != 2:
            raise ValueError(
                f"Expected batched native states [B,D], found {states.shape}"
            )
        if "images" in env_obs:
            images = env_obs["images"]
        else:
            images = {
                slot: env_obs[key]
                for slot, key in zip(
                    IMAGE_KEYS,
                    ("main_images", "wrist_images", "extra_view_images"),
                    strict=True,
                )
                if key in env_obs and env_obs[key] is not None
            }
        tasks = env_obs["task_descriptions"]
        if len(tasks) != len(states):
            raise ValueError(
                "Task descriptions and native states must have equal batch size"
            )
        samples = [
            self.prepare_observation(
                state, {slot: image[i] for slot, image in images.items()}, tasks[i]
            )
            for i, state in enumerate(states)
        ]
        return tree_map(lambda *xs: torch.as_tensor(np.stack(xs)), *samples)

    def decode(
        self, normalized_actions: np.ndarray, raw_state: np.ndarray
    ) -> np.ndarray:
        """Unnormalize valid action dimensions and restore native joint targets."""
        actions = _numpy(normalized_actions)[..., : self.native_action_dim]
        actions = unnormalize(actions, self.schema["action"])
        return self.adapter.decode_actions(actions, _numpy(raw_state)).astype(
            np.float32
        )

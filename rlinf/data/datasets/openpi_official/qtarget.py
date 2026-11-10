# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical command targets at each source's original frame rate."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.openpi.control_schema import normalize, unnormalize
from rlinf.data.datasets.openpi.mixture import MixtureSftDataset, NumericEpisodeCache

SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RECIPE = {
    "name": "canonical_qtarget",
    "state_layout": "qpos_width8",
    "network_input_mask": False,
    "closed_loop_environment_adapter": False,
    "target_hz": "source_native",
    "offsets": "consecutive source frames, k=0..H-1",
    "selection": "original command at each source frame",
    "tail": "repeat final command with loss masked beyond episode end",
    "state_mask": "zero after normalization; no model mask channel",
    "prompt": "original task only",
}


class CanonicalQtargetDataset(MixtureSftDataset):
    """Consume prepared command/state8 rows; state masks gate normalization.

    The checkpoint has no state-mask input channel. Missing measured dimensions
    are zero in normalized space; the mask remains sample metadata for auditing.
    Robot and gripper identity select source statistics, never text prefixes.
    """

    def __init__(
        self, manifest_dir: str | Path, *, model_path: str | Path, **kwargs: Any
    ) -> None:
        super().__init__(manifest_dir, action_dim=8, **kwargs)
        self.model_path = Path(model_path)
        self.assets = qtarget_assets(self.root)
        self._model_transform = None
        self.incomplete_chunk_dropped = 0

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_model_transform"] = None
        return state

    def raw_item(self, index: int) -> tuple:
        """Read consecutive source commands and measured-state validity."""
        episode = int(np.searchsorted(self.ends, index, side="right"))
        row = self.episodes[episode]
        offset = index - (int(self.ends[episode - 1]) if episode else 0)
        frame = int(row["anchor_indices"][offset])
        if self._cache is None:
            self._cache = NumericEpisodeCache(self.cache_files)
        columns = ["action", "observation.state", "observation.state_mask"]
        if row.get("action_mask_feature"):
            columns.append(row["action_mask_feature"])
        if row.get("timestamp_feature"):
            columns.append(row["timestamp_feature"])
        if row.get("supervision_mask_feature"):
            columns.append(row["supervision_mask_feature"])
        arrays = self._cache.read_columns(row, columns)
        clock = (
            arrays[row["timestamp_feature"]].reshape(-1).astype(np.float64)
            if row.get("timestamp_feature")
            else np.arange(row["frames"]) / row["fps"]
        )
        if not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
            raise ValueError(
                f"Expected strictly increasing finite timestamps: {row['catalog_id']}"
            )
        source_indices = frame + np.arange(self.action_horizon)
        valid = source_indices < row["frames"]
        selected = np.minimum(source_indices, row["frames"] - 1)
        target_times = clock[selected]
        actions = arrays["action"][selected].astype(np.float32)
        lo, hi = row["core_range"]
        valid &= (selected >= lo) & (selected < hi)
        if row.get("supervision_mask_feature"):
            valid &= (
                arrays[row["supervision_mask_feature"]]
                .reshape(-1)[selected]
                .astype(bool)
            )
        action_mask = valid[:, None] & np.isfinite(actions)
        if row.get("action_mask_feature"):
            dimension_mask = arrays[row["action_mask_feature"]]
            if dimension_mask.shape != arrays["action"].shape:
                raise ValueError(
                    "Expected action_mask to match canonical action8 shape"
                )
            action_mask &= dimension_mask[selected].astype(bool)
        state = arrays["observation.state"][frame].astype(np.float32)
        state_mask = arrays["observation.state_mask"][frame].astype(bool) & np.isfinite(
            state
        )
        if actions.shape[-1] != 8 or state.shape != (8,) or state_mask.shape != (8,):
            raise ValueError("Canonical qtarget requires action8/state8/state_mask8")
        return row, frame, state, state_mask, actions, action_mask, clock, target_times

    def numeric_item(self, index: int) -> tuple:
        row, frame, state, state_mask, actions, action_mask, _, _ = self.raw_item(index)
        stats = self.schemas[row["schema_id"]]
        state = np.where(state_mask, normalize(state, stats["state"]), 0).astype(
            np.float32
        )
        actions = np.where(action_mask, normalize(actions, stats["action"]), 0).astype(
            np.float32
        )
        return row, frame, state, actions, action_mask

    def __getitem__(self, index: int) -> dict:
        from openpi import transforms
        from openpi.training.config import ModelTransformFactory

        from rlinf.data.datasets.video import VideoFrameCache
        from rlinf.models.embodiment.openpi_official.config import load_train_config

        from .image import preprocess_molmo_image

        row, frame, state, state_mask, actions, loss_mask, clock, target_times = (
            self.raw_item(index)
        )
        stats = self.schemas[row["schema_id"]]
        state = np.where(state_mask, normalize(state, stats["state"]), 0).astype(
            np.float32
        )
        actions = np.where(loss_mask, normalize(actions, stats["action"]), 0).astype(
            np.float32
        )
        if self._video_cache is None:
            self._video_cache = VideoFrameCache(self.cache_files)
        images = {}
        video_feature = row.get("video_timestamp_feature")
        if video_feature:
            video_clock = self._cache.read_columns(row, [video_feature])[
                video_feature
            ].reshape(-1)
            video_offset = float(video_clock[frame] - video_clock[0])
        else:
            video_offset = frame / row["fps"]
        for slot, video in row["videos"].items():
            pixels = self._video_cache.read(
                video["path"],
                video["from_timestamp"] + video_offset,
                tolerance_s=0.5 / row["fps"] + 1e-5,
            )
            images[slot] = preprocess_molmo_image(pixels)
        observation = {
            "state": state,
            "image": {
                slot: images.get(slot, np.zeros((224, 224, 3), dtype=np.uint8))
                for slot in SLOTS
            },
            "image_mask": {slot: slot in images for slot in SLOTS},
            "prompt": row["task"],
        }
        if self._model_transform is None:
            config = load_train_config(str(self.model_path))
            if (
                config.model.action_dim != 8
                or config.model.action_horizon != self.action_horizon
            ):
                raise ValueError(
                    "Canonical action8/horizon differs from checkpoint architecture"
                )
            self._model_transform = transforms.compose(
                ModelTransformFactory()(config.model).inputs
            )
        return {
            "observation": self._model_transform(observation),
            "actions": actions,
            "action_loss_mask": loss_mask,
            "state_mask": state_mask,
            "target_timestamps": target_times,
            "sample_schema": row["schema_id"],
            "sample_group": row["source_group_id"],
        }


def qtarget_assets(manifest_dir: str | Path) -> dict:
    """Read source-routed train-only statistics for native-frame command chunks."""
    root = Path(manifest_dir)
    metadata = json.loads((root / "normalization.json").read_text())
    if metadata["target_recipe"] != RECIPE:
        raise ValueError("Expected canonical qtarget native-frame recipe")
    return {"version": 1, "recipe": RECIPE, "normalization": metadata}


def save_qtarget_assets(
    manifest_dir: str | Path, output_assets_dir: str | Path
) -> None:
    """Export canonical units, source routes and statistics with the model."""
    root = Path(output_assets_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "canonical_qtarget.json").write_text(
        json.dumps(qtarget_assets(manifest_dir), indent=2) + "\n"
    )


class QtargetInputTransform:
    """Normalize canonical physical state using an explicit source-schema route."""

    def __init__(self, train_config: Any, assets: dict) -> None:
        from openpi import transforms
        from openpi.training.config import ModelTransformFactory

        if (
            train_config.model.action_dim != 8
            or train_config.model.action_horizon
            != assets["normalization"]["action_horizon"]
        ):
            raise ValueError(
                "Canonical action8/horizon differs from checkpoint architecture"
            )
        self.schemas = assets["normalization"]["schemas"]
        self.model_transform = transforms.compose(
            ModelTransformFactory()(train_config.model).inputs
        )

    def __call__(self, observation: dict) -> dict:
        observation = dict(observation)
        schema = observation.pop("source_schema")
        stats = self.schemas[schema]
        mask = np.asarray(observation.pop("state_mask"), dtype=bool)
        observation["state"] = np.where(
            mask, normalize(observation["state"], stats["state"]), 0
        ).astype(np.float32)
        if "actions" in observation:
            observation["actions"] = normalize(observation["actions"], stats["action"])
        return self.model_transform(observation)


class QtargetOutputTransform:
    """Decode canonical targets; native robot command conversion is external.

    The gripper output remains closedness in [0, 1] units. This transform does
    not produce MolmoBot actuator commands or recover native gripper state.
    """

    def __init__(self, assets: dict) -> None:
        self.schemas = assets["normalization"]["schemas"]

    def __call__(self, output: dict) -> dict:
        output = dict(output)
        schema = output.pop("source_schema")
        output["actions"] = unnormalize(
            output["actions"], self.schemas[schema]["action"]
        )
        return output

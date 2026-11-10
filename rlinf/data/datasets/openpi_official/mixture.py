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

"""Native absolute-joint data with checkpoint-owned OpenPI preprocessing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.data.datasets.openpi.manifest_index import ManifestIndex
from rlinf.data.datasets.openpi.mixture import (
    MixtureSampler,
    MixtureSftDataset,
    NumericEpisodeCache,
)

_MOLMO_JOINT_CONTROLS = {
    "native_molmobot_arm7_gripper1_command_unscaled",
    "native_molmobot_joint_position_rad7_gripper_command_0_255",
    "franka_droid_absolute_joint_position_and_robotiq_actuator_command",
}
_DROID_JOINT_CONTROL = "droid_absolute_joint_position_and_normalized_closure"


class OfficialMixtureDataset(Dataset):
    """Read complete native action chunks and apply the checkpoint's transforms.

    A named mapping of manifest directories namespaces schemas as ``name/schema``.
    MolmoBot actuator commands and DROID normalized closure retain their native
    units until source-specific transforms. Prepared numeric normalization and
    relative-action adapters are not applied.
    """

    def __init__(
        self,
        manifest_dir: str | Path | Mapping[str, str | Path],
        *,
        model_path: str | Path,
        action_horizon: int,
        split: str = "train",
        source_schemas: list[str] | None = None,
        max_samples: int | None = None,
        sample_multiple: int = 1,
        cache_files: int = 4,
    ) -> None:
        self.roots = (
            {name: Path(path) for name, path in manifest_dir.items()}
            if isinstance(manifest_dir, Mapping)
            else {"": Path(manifest_dir)}
        )
        self.root = next(iter(self.roots.values()))
        self.model_path = Path(model_path)
        self.action_horizon = action_horizon
        self.split = split
        self.cache_files = cache_files
        self._cache = NumericEpisodeCache(cache_files)
        self._input_transforms = {}
        self._video_cache = None
        self.schemas = {}
        self.episodes = []
        contracts = {}
        selected = None if source_schemas is None else set(source_schemas)
        available_samples = 0
        for name, root in self.roots.items():
            contract_path = root / "contract.json"
            contract = json.loads(contract_path.read_text())
            for filename, expected in contract["files"].items():
                if (
                    hashlib.sha256((root / filename).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError(f"Prepared manifest changed: {root / filename}")
            contracts[name] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            metadata = json.loads((root / "normalization.json").read_text())
            for line in (root / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                original_schema = row["schema_id"]
                schema_id = f"{name}/{original_schema}" if name else original_schema
                if selected is not None and schema_id not in selected:
                    continue
                schema = metadata["schemas"][original_schema]
                control_mode = schema["native_control_mode"]
                if control_mode in _MOLMO_JOINT_CONTROLS:
                    state_dim = 9
                elif control_mode == _DROID_JOINT_CONTROL:
                    state_dim = 8
                else:
                    raise ValueError(
                        f"Official mixture requires absolute arm7 with MolmoBot "
                        f"actuator commands or DROID normalized closure; "
                        f"schema {schema_id} uses {control_mode}"
                    )
                self.schemas[schema_id] = schema
                action, state, supervised = self._cache.read(row)
                if action.shape[-1] != 8 or state.shape[-1] != state_dim:
                    raise ValueError(
                        f"Schema {schema_id} requires native action8/state{state_dim}; "
                        f"found action{action.shape[-1]}/state{state.shape[-1]}"
                    )
                available_samples += len(row["anchor_indices"])
                complete = np.zeros(len(action), dtype=bool)
                if len(action) >= action_horizon:
                    complete[: len(action) - action_horizon + 1] = (
                        np.convolve(
                            supervised.astype(np.int64),
                            np.ones(action_horizon, dtype=np.int64),
                            mode="valid",
                        )
                        == action_horizon
                    )
                anchors = [i for i in row["anchor_indices"] if complete[i]]
                if anchors:
                    self.episodes.append(
                        {**row, "schema_id": schema_id, "anchor_indices": anchors}
                    )
        if selected is not None and selected != set(self.schemas):
            raise ValueError(f"Unknown source schemas: {selected - set(self.schemas)}")
        if not self.episodes:
            raise ValueError(
                f"No complete supervised H{action_horizon} chunks in {split}"
            )
        ids = [row["catalog_id"] for row in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Official mixture manifests contain duplicate catalog IDs")
        self.available_samples = available_samples
        complete_samples = sum(len(row["anchor_indices"]) for row in self.episodes)
        self.incomplete_chunk_dropped = available_samples - complete_samples
        self.alignment_dropped = 0
        if max_samples is not None:
            count = min(max_samples, complete_samples)
            self.alignment_dropped = count % sample_multiple
            count -= self.alignment_dropped
            if count == 0:
                raise ValueError(
                    "Evaluation needs at least one complete batch per rank"
                )
            streams = {
                schema: iter(
                    (i, anchor)
                    for i, row in enumerate(self.episodes)
                    if row["schema_id"] == schema
                    for anchor in row["anchor_indices"]
                )
                for schema in sorted(self.schemas)
            }
            chosen = {}
            while count:
                for schema, stream in list(streams.items()):
                    item = next(stream, None)
                    if item is None:
                        del streams[schema]
                        continue
                    i, anchor = item
                    chosen.setdefault(i, []).append(anchor)
                    count -= 1
                    if count == 0:
                        break
            self.episodes = [
                {**row, "anchor_indices": chosen[i]}
                for i, row in enumerate(self.episodes)
                if i in chosen
            ]
        self.ends = np.cumsum([len(row["anchor_indices"]) for row in self.episodes])
        self.sampling_unit = "episode"
        self.requires_contract = True
        self.manifest_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "contracts": contracts,
                    "episodes": self.episodes,
                    "action_horizon": action_horizon,
                    "checkpoint_assets": {
                        str(path.relative_to(self.model_path)): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in (
                            self.model_path / "assets/train_config.pkl",
                            *sorted(
                                (self.model_path / "assets").glob("*/norm_stats.json")
                            ),
                        )
                    },
                    "profile": "official_molmobot_absolute_arm7_robotiq",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_cache"] = None
        state["_input_transforms"] = {}
        state["_video_cache"] = None
        return state

    def numeric_item(self, index: int) -> tuple[dict, int, dict]:
        """Return the current native state and a complete absolute action chunk."""
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        row = self.episodes[episode_index]
        offset = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        frame = row["anchor_indices"][offset]
        if self._cache is None:
            self._cache = NumericEpisodeCache(self.cache_files)
        action, state, _ = self._cache.read(row)
        return (
            row,
            frame,
            {
                "observation/joint_position": state[frame, :7].copy(),
                "observation/gripper_position": state[frame, 7:].copy(),
                "actions": action[frame : frame + self.action_horizon].copy(),
                "prompt": row["task"],
            },
        )

    def _build_input_transform(self, control_mode: str):
        from openpi import transforms
        from openpi.training.checkpoints import load_norm_stats

        from rlinf.models.embodiment.openpi_official.config import load_train_config

        config = load_train_config(str(self.model_path))
        if config.model.action_horizon != self.action_horizon:
            raise ValueError("Dataset horizon differs from the checkpoint horizon")
        data = config.data.create(config.assets_dirs, config.model)
        data_inputs = data.data_transforms.inputs
        if control_mode == _DROID_JOINT_CONTROL:
            from molmobot_pi0.dataset_openpi import (
                GripperActionTransform,
                GripperStateTransform,
                MlSpacesDatasetConfigFactory,
            )

            if (
                not isinstance(config.data, MlSpacesDatasetConfigFactory)
                or not config.data.joint_pos_actions
            ):
                raise ValueError(
                    "DROID closure requires a MolmoBot absolute-joint checkpoint factory"
                )
            # DROID already supplies closure; retain every other checkpoint transform.
            data_inputs = [
                transform
                for transform in data_inputs
                if not isinstance(
                    transform, (GripperStateTransform, GripperActionTransform)
                )
            ]
        stats = load_norm_stats(self.model_path / "assets", data.asset_id)
        return transforms.compose(
            [
                transforms.InjectDefaultPrompt(None),
                *data_inputs,
                transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ]
        )

    def __getitem__(self, index: int) -> dict:
        from rlinf.data.datasets.video import VideoFrameCache

        from .image import preprocess_molmo_image

        if self._video_cache is None:
            self._video_cache = VideoFrameCache(self.cache_files)

        row, frame, raw = self.numeric_item(index)
        for slot, input_key in (
            ("base_0_rgb", "observation/exterior_image_1_left"),
            ("left_wrist_0_rgb", "observation/wrist_image_left"),
        ):
            video = row["videos"][slot]
            pixels = self._video_cache.read(
                video["path"],
                video["from_timestamp"] + frame / row["fps"],
                tolerance_s=0.5 / row["fps"] + 1e-5,
            )
            raw[input_key] = preprocess_molmo_image(pixels)
        control_mode = self.schemas[row["schema_id"]]["native_control_mode"]
        if control_mode not in self._input_transforms:
            self._input_transforms[control_mode] = self._build_input_transform(
                control_mode
            )
        sample = self._input_transforms[control_mode](raw)
        actions = sample.pop("actions")
        sample["image_mask"] = {
            key: bool(value) for key, value in sample["image_mask"].items()
        }
        return {
            "observation": sample,
            "actions": actions,
            "sample_schema": row["schema_id"],
            "sample_group": row["source_group_id"],
        }


class RealizedOfficialMixtureDataset(MixtureSftDataset):
    """Bridge shared 15 Hz measured-joint targets to the original OpenPI core."""

    def __init__(
        self, manifest_dir: str | Path, *, model_path: str | Path, **kwargs: Any
    ) -> None:
        from rlinf.models.embodiment.openpi_official.trajectory import trajectory_assets

        super().__init__(manifest_dir, action_dim=8, **kwargs)
        self.model_path = Path(model_path)
        trajectory_assets(self.root)
        self._model_transform = None
        self.incomplete_chunk_dropped = 0

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_model_transform"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        from rlinf.data.datasets.video import VideoFrameCache
        from rlinf.models.embodiment.openpi_official.config import load_train_config
        from rlinf.models.embodiment.openpi_official.trajectory import (
            build_trajectory_prompt,
            trajectory_model_transform,
        )

        from .image import preprocess_molmo_image

        row, frame, state, actions, loss_mask = self.numeric_item(index)
        if row["target_recipe"]["name"] != "realized_joint_trajectory":
            raise ValueError("Expected measured-joint trajectory targets")
        if self._video_cache is None:
            self._video_cache = VideoFrameCache(self.cache_files)
        images = {}
        for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            if slot in row["videos"]:
                video = row["videos"][slot]
                pixels = self._video_cache.read(
                    video["path"],
                    video["from_timestamp"] + frame / row["fps"],
                    tolerance_s=0.5 / row["fps"] + 1e-5,
                )
                images[slot] = preprocess_molmo_image(pixels)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        prompt = build_trajectory_prompt(
            row["task"],
            row["schema_id"],
            row["robot_identity"]["robot_type"],
            row["target_recipe"]["gripper_identity"],
            bool(np.isfinite(self._trajectory_cache.read(row)["gripper_state"][frame])),
        )
        observation = {
            "state": state,
            "image": {
                slot: images.get(slot, image)
                for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "image_mask": {
                slot: slot in images
                for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "prompt": prompt,
        }
        if self._model_transform is None:
            self._model_transform = trajectory_model_transform(
                load_train_config(str(self.model_path)),
            )
        return {
            "observation": self._model_transform(observation),
            "actions": actions,
            "action_loss_mask": loss_mask,
            "sample_schema": row["schema_id"],
            "sample_group": row["source_group_id"],
        }


class ExposureMatchedSampler(MixtureSampler):
    """Insert auxiliary batches without changing the target sample stream."""

    def __init__(
        self,
        dataset: OfficialMixtureDataset,
        *,
        target_schemas: list[str],
        target_samples_per_epoch: int,
        auxiliary_samples_per_epoch: int,
        global_batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if (
            target_samples_per_epoch <= 0
            or auxiliary_samples_per_epoch < 0
            or global_batch_size % world_size
            or target_samples_per_epoch % global_batch_size
            or auxiliary_samples_per_epoch % global_batch_size
        ):
            raise ValueError("Exposure budgets must be whole positive global batches")
        super().__init__(
            dataset,
            batch_size=global_batch_size // world_size,
            world_size=world_size,
            rank=rank,
            seed=seed,
            samples_per_epoch=target_samples_per_epoch + auxiliary_samples_per_epoch,
        )
        self.target_schemas = sorted(target_schemas)
        if not self.target_schemas or not set(self.target_schemas) <= set(
            dataset.schemas
        ):
            raise ValueError("target_schemas must name existing mixture schemas")
        self.target_samples = target_samples_per_epoch
        self.auxiliary_samples = auxiliary_samples_per_epoch
        self.global_batch_size = global_batch_size
        self.target_mask = torch.cat(
            [
                torch.full(
                    (len(row["anchor_indices"]),),
                    row["schema_id"] in self.target_schemas,
                    dtype=torch.bool,
                )
                for row in dataset.episodes
            ]
        )
        if auxiliary_samples_per_epoch and self.target_mask.all():
            raise ValueError(
                "Auxiliary exposure requires at least one auxiliary schema"
            )

    def __iter__(self):
        target = torch.multinomial(
            self.weights * self.target_mask,
            self.target_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(self.seed + self.epoch),
        ).tolist()
        auxiliary = (
            torch.multinomial(
                self.weights * ~self.target_mask,
                self.auxiliary_samples,
                replacement=True,
                generator=torch.Generator().manual_seed(
                    self.seed + self.epoch + 1_000_003
                ),
            ).tolist()
            if self.auxiliary_samples
            else []
        )
        indices = []
        for start in range(0, max(len(target), len(auxiliary)), self.global_batch_size):
            indices.extend(target[start : start + self.global_batch_size])
            indices.extend(auxiliary[start : start + self.global_batch_size])
        local = indices[self.rank :: self.world_size]
        for position in range(self.position, len(local)):
            self.position = position + 1
            yield local[position]

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "target_schemas": self.target_schemas,
            "target_samples": self.target_samples,
            "auxiliary_samples": self.auxiliary_samples,
            "global_batch_size": self.global_batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        current = self.state_dict()
        for key in (
            "target_schemas",
            "target_samples",
            "auxiliary_samples",
            "global_batch_size",
        ):
            if state[key] != current[key]:
                raise ValueError(
                    "Exposure-matched resume requires the same exposure plan"
                )
        super().load_state_dict(state)


def build_official_mixture_sft_dataloader(
    cfg: Any, world_size: int, rank: int, data_paths: Any, eval_dataset: bool = False
) -> tuple[Any, dict]:
    """Build the official-input native mixture loader for RLinf SFT workers."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    from rlinf.models.embodiment.openpi_official.config import load_train_config

    if isinstance(data_paths, Sequence) and not isinstance(data_paths, str):
        if len(data_paths) != 1:
            raise ValueError(
                "Use a named manifest mapping for multiple official sources"
            )
        data_paths = data_paths[0]
    model_path = cfg.actor.model.model_path
    train_config = load_train_config(model_path)
    batch_size = int(
        cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size)
        if eval_dataset
        else cfg.actor.micro_batch_size
    )
    settings = cfg.data.official
    dataset_kwargs = {
        "model_path": model_path,
        "action_horizon": train_config.model.action_horizon,
        "split": "val" if eval_dataset else "train",
        "max_samples": int(cfg.data.eval_max_samples) if eval_dataset else None,
        "sample_multiple": world_size * batch_size if eval_dataset else 1,
        "cache_files": int(settings.cache_files),
    }
    if settings.get("recipe") == "canonical_qtarget":
        from .qtarget import CanonicalQtargetDataset

        dataset = CanonicalQtargetDataset(data_paths, **dataset_kwargs)
    elif settings.get("recipe") == "realized_trajectory":
        if not isinstance(data_paths, (str, Path)):
            raise ValueError(
                "Realized trajectory requires one shared prepared manifest"
            )
        dataset = RealizedOfficialMixtureDataset(data_paths, **dataset_kwargs)
    else:
        dataset = OfficialMixtureDataset(
            data_paths,
            source_schemas=(
                None
                if settings.source_schemas is None
                else list(settings.source_schemas)
            ),
            **dataset_kwargs,
        )
    if settings.get("recipe") == "realized_trajectory":
        from rlinf.data.datasets.openpi.realized_trajectory import (
            RealizedTrajectorySampler,
        )

        sampler = RealizedTrajectorySampler(
            dataset,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            seed=int(cfg.actor.seed),
            shuffle=not eval_dataset,
            samples_per_epoch=int(cfg.data.samples_per_epoch),
        )
    elif not eval_dataset and settings.sampling == "target_exposure":
        sampler = ExposureMatchedSampler(
            dataset,
            target_schemas=list(settings.target_schemas),
            target_samples_per_epoch=int(settings.target_samples_per_epoch),
            auxiliary_samples_per_epoch=int(settings.auxiliary_samples_per_epoch),
            global_batch_size=int(cfg.actor.global_batch_size),
            world_size=world_size,
            rank=rank,
            seed=int(cfg.actor.seed),
        )
    elif settings.sampling == "weighted" or eval_dataset:
        sampler = MixtureSampler(
            dataset,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            seed=int(cfg.actor.seed),
            shuffle=not eval_dataset,
            samples_per_epoch=int(cfg.data.samples_per_epoch),
        )
    else:
        raise ValueError(f"Unknown official mixture sampling: {settings.sampling}")
    workers = int(cfg.data.num_workers)
    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        prefetch_factor=int(cfg.data.get("prefetch_factor", 2)) if workers else None,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=workers > 0,
        pin_memory=True,
        drop_last=False,
    )
    return loader, {
        "num_samples": len(dataset),
        "available_samples": dataset.available_samples,
        "incomplete_chunk_dropped": dataset.incomplete_chunk_dropped,
        "alignment_dropped": dataset.alignment_dropped,
        "schemas": list(dataset.schemas),
        "sampling_unit": dataset.sampling_unit,
        "fps_by_schema": dataset.episodes.metadata["fps_by_schema"]
        if isinstance(dataset.episodes, ManifestIndex)
        else {
            schema: sorted(
                {row["fps"] for row in dataset.episodes if row["schema_id"] == schema}
            )
            for schema in dataset.schemas
        },
        "time_policy": (
            "source-native consecutive command frames, masked tail"
            if settings.get("recipe") == "canonical_qtarget"
            else "future measured joints at (k+1)/15 seconds"
            if isinstance(dataset, RealizedOfficialMixtureDataset)
            else "source-native frame indices; no resampling"
        ),
        "manifest_fingerprint": dataset.manifest_fingerprint,
        "action_horizon": dataset.action_horizon,
        "image_processing": "official policy image geometry; core training augmentation",
        "normalization": (
            "source-routed train-only canonical qtarget statistics"
            if settings.get("recipe") == "canonical_qtarget"
            else "shared train-only realized-trajectory statistics"
            if isinstance(dataset, RealizedOfficialMixtureDataset)
            else "checkpoint-owned official transforms"
        ),
        "sampling": (
            "evaluation"
            if eval_dataset
            else "source_episode_anchor"
            if isinstance(dataset, RealizedOfficialMixtureDataset)
            else settings.sampling
        ),
        "samples_per_rank": len(sampler),
        "global_sample_budget": len(sampler) * world_size,
        "target_sample_budget": (
            sampler.target_samples
            if isinstance(sampler, ExposureMatchedSampler)
            else None
        ),
        "auxiliary_sample_budget": (
            sampler.auxiliary_samples
            if isinstance(sampler, ExposureMatchedSampler)
            else None
        ),
    }

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

"""Native-control LeRobot mixtures with explicit supervision and source schemas."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_QUALITIES = {
    "native_source_success",
    "clean_pickup_annotation",
    "sampled_visual_confirmed",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _episode_metadata(root: Path, cameras: dict[str, str]) -> dict[int, dict]:
    import pyarrow.parquet as pq

    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    for feature in cameras.values():
        columns.extend(
            f"videos/{feature}/{key}"
            for key in ("chunk_index", "file_index", "from_timestamp")
        )
    records = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        records.extend(pq.read_table(path, columns=columns).to_pylist())
    file_starts = {}
    for row in records:
        key = (row["data/chunk_index"], row["data/file_index"])
        file_starts[key] = min(
            file_starts.get(key, row["dataset_from_index"]), row["dataset_from_index"]
        )
    for row in records:
        key = (row["data/chunk_index"], row["data/file_index"])
        row["row_offset"] = row["dataset_from_index"] - file_starts[key]
    return {int(row["episode_index"]): row for row in records}


class NumericEpisodeCache:
    """Bounded process-local cache of selected numeric Parquet columns."""

    def __init__(self, max_files: int = 4) -> None:
        self.max_files = max_files
        self._tables: OrderedDict = OrderedDict()

    def read(self, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read native action/state and the intersection of supervision masks."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        feature = row.get("supervision_mask_feature")
        columns = ["action", "observation.state"] + ([feature] if feature else [])
        key = (row["data_file"], tuple(columns))
        if key not in self._tables:
            table = pq.read_table(row["data_file"], columns=columns)
            arrays = {}
            for name in columns:
                array = table[name].combine_chunks()
                if pa.types.is_fixed_size_list(array.type) or pa.types.is_list(
                    array.type
                ):
                    arrays[name] = (
                        array.flatten()
                        .to_numpy(zero_copy_only=False)
                        .reshape(len(array), -1)
                    )
                else:
                    arrays[name] = array.to_numpy(zero_copy_only=False)
            self._tables[key] = arrays
            if len(self._tables) > self.max_files:
                self._tables.popitem(last=False)
        self._tables.move_to_end(key)
        arrays = self._tables[key]
        start, frames = row["row_offset"], row["frames"]
        action = arrays["action"][start : start + frames]
        state = arrays["observation.state"][start : start + frames]
        if len(action) != frames or len(state) != frames:
            raise ValueError(f"Episode exceeds its Parquet file: {row['catalog_id']}")
        mask = np.zeros(frames, dtype=bool)
        lo, hi = row["core_range"]
        mask[lo:hi] = True
        if feature:
            native_mask = arrays[feature][start : start + frames].reshape(frames)
            mask &= native_mask.astype(bool)
        return action, state, mask


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, values: np.ndarray) -> None:
        values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite training action/state values")
        count = len(values)
        mean = values.mean(axis=0)
        m2 = np.square(values - mean).sum(axis=0)
        if not self.count:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + np.square(delta) * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def result(self) -> dict:
        std = np.sqrt(self.m2 / self.count)
        std[std == 0] = 1
        return {"count": self.count, "mean": self.mean.tolist(), "std": std.tolist()}


def prepare_mixture(
    catalog_root: str | Path,
    output_dir: str | Path,
    sources: list[dict],
    val_fraction: float = 0.1,
    seed: int = 0,
    allow_development: bool = False,
    max_train_episodes: int | None = None,
    max_val_episodes: int | None = None,
) -> dict:
    """Prepare grouped splits, native numeric statistics, and v3 media references.

    Each source names one independent native control schema and camera mapping.
    Episode limits apply after grouping the complete eligible pool into splits.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    names = [source["name"] for source in sources]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Mixture source names must be nonempty and unique")
    catalog_root, output_dir = Path(catalog_root).resolve(), Path(output_dir)
    catalog_path = catalog_root / "catalog/episodes.jsonl"
    pool = _jsonl(catalog_path)
    cache = NumericEpisodeCache()
    provenance = {}
    metadata = {}
    infos = {}
    excluded = {"quality": 0, "development": 0, "unknown_core": 0}
    selected = []
    for source in sources:
        if source["weight"] <= 0:
            raise ValueError(f"Source weight must be positive: {source['name']}")
        cameras = source["cameras"]
        if not cameras or not set(cameras) <= set(IMAGE_KEYS):
            raise ValueError(f"Invalid OpenPI camera slots: {source['name']}")
        rows = [
            row
            for row in pool
            if row["source_dataset"] == source["source_dataset"]
            and ("domain" not in source or row["domain"] == source["domain"])
        ]
        eligible = []
        for row in rows:
            if row["quality_tier"] not in _QUALITIES:
                excluded["quality"] += 1
                continue
            development = row["split"] == "development_only"
            if development and not allow_development:
                excluded["development"] += 1
                continue
            root = catalog_root / row["partition"]
            if str(root) not in provenance:
                prov_path = catalog_root / row["provenance_file"]
                provenance[str(root)] = {
                    item["episode_index"]: item["provenance"]
                    for item in _jsonl(prov_path)
                }
            prov = provenance[str(root)][row["episode_index"]]
            if prov.get("semantic_core_status") == "UNKNOWN":
                excluded["unknown_core"] += 1
                continue
            core = row["core_range_in_export"]
            if isinstance(core, str):
                core = json.loads(core)
            if core is None or not 0 <= core[0] < core[1] <= row["frames"]:
                raise ValueError(f"Missing valid core bounds: {row['catalog_id']}")
            group = (
                f"{row['source_dataset']}:scene:{row['scene_id']}"
                if row.get("scene_id")
                else row["source_group_id"]
            )
            eligible.append(
                {
                    **row,
                    "schema_id": source["name"],
                    "root": str(root),
                    "development_only": development,
                    "split_group": group,
                    "core_range": core,
                    "cameras": cameras,
                    "supervision_mask_feature": prov.get("supervision_mask_feature"),
                }
            )
        development_groups = {
            row["split_group"] for row in eligible if row["development_only"]
        }
        groups_by_lineage = {}
        for row in eligible:
            groups_by_lineage.setdefault(row["source_group_id"], set()).add(
                row["split_group"]
            )
        if any(len(groups) != 1 for groups in groups_by_lineage.values()):
            raise ValueError("One source group maps to multiple scene groups")
        split_rows = {"train": [], "val": []}
        for row in eligible:
            val = (
                row["split_group"] not in development_groups
                and _fraction(row["split_group"], seed) < val_fraction
            )
            split_rows["val" if val else "train"].append(row)
        for split, limit_key, default_limit in [
            ("train", "max_train_episodes", max_train_episodes),
            ("val", "max_val_episodes", max_val_episodes),
        ]:
            rows = sorted(
                split_rows[split], key=lambda row: _fraction(row["catalog_id"], seed)
            )
            limit = source.get(limit_key, default_limit)
            if limit is not None:
                if limit < 0:
                    raise ValueError("Episode limits must be nonnegative")
                rows = rows[:limit]
            rows.sort(key=lambda row: (row["partition"], row["episode_index"]))
            for row in rows:
                root = Path(row["root"])
                key = (str(root), tuple(cameras.items()))
                if key not in metadata:
                    info = json.loads((root / "meta/info.json").read_text())
                    if info["codebase_version"] != "v3.0":
                        raise ValueError(f"Expected LeRobot v3.0: {root}")
                    infos[str(root)] = info
                    metadata[key] = _episode_metadata(root, cameras)
                info, episode = infos[str(root)], metadata[key][row["episode_index"]]
                if episode["length"] != row["frames"]:
                    raise ValueError(
                        f"Catalog/metadata length mismatch: {row['catalog_id']}"
                    )
                row["data_file"] = str(
                    root
                    / info["data_path"].format(
                        chunk_index=episode["data/chunk_index"],
                        file_index=episode["data/file_index"],
                    )
                )
                row["row_offset"] = episode["row_offset"]
                row["videos"] = {
                    slot: {
                        "path": str(
                            root
                            / info["video_path"].format(
                                video_key=feature,
                                chunk_index=episode[f"videos/{feature}/chunk_index"],
                                file_index=episode[f"videos/{feature}/file_index"],
                            )
                        ),
                        "from_timestamp": episode[f"videos/{feature}/from_timestamp"],
                    }
                    for slot, feature in cameras.items()
                }
                _, _, mask = cache.read(row)
                row["anchor_indices"] = np.flatnonzero(mask).tolist()
                if row["anchor_indices"]:
                    row["prepared_split"] = split
                    selected.append(row)
    schemas = {}
    for source in sources:
        train = [
            row
            for row in selected
            if row["schema_id"] == source["name"] and row["prepared_split"] == "train"
        ]
        if not train:
            raise ValueError(
                f"No eligible training episodes for schema {source['name']}"
            )
        controls = {(row["control_mode"], row["fps"]) for row in train}
        if len(controls) != 1:
            raise ValueError(
                f"One schema must have one control mode/FPS: {source['name']}"
            )
        action_stats, state_stats = _Moments(), _Moments()
        for row in sorted(train, key=lambda row: (row["data_file"], row["row_offset"])):
            action, state, mask = cache.read(row)
            action_stats.add(action[mask])
            state_stats.add(state[mask])
        control, fps = controls.pop()
        schemas[source["name"]] = {
            "weight": source["weight"],
            "action": {**action_stats.result(), "transform": "mean_std"},
            "state": {**state_stats.result(), "transform": "tanh_mean_std"},
            "control_mode": control,
            "fps": fps,
            "cameras": source["cameras"],
            "train_catalog_ids": [row["catalog_id"] for row in train],
        }
    train_groups = {
        row["source_group_id"] for row in selected if row["prepared_split"] == "train"
    }
    val_groups = {
        row["source_group_id"] for row in selected if row["prepared_split"] == "val"
    }
    if train_groups & val_groups:
        raise ValueError("Source group leakage between training and validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 1,
        "catalog_root": str(catalog_root),
        "seed": seed,
        "val_fraction": val_fraction,
        "allow_development": allow_development,
        "num_obs_steps": 1,
        "time_policy": "native indices; no resampling",
        "excluded": excluded,
        "schemas": {},
    }
    for split in ("train", "val"):
        rows = [row for row in selected if row["prepared_split"] == split]
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        summary[split] = {
            "episodes": len(rows),
            "anchors": sum(len(row["anchor_indices"]) for row in rows),
        }
    for name in schemas:
        summary["schemas"][name] = {
            split: {
                "episodes": sum(
                    row["schema_id"] == name and row["prepared_split"] == split
                    for row in selected
                ),
                "anchors": sum(
                    len(row["anchor_indices"])
                    for row in selected
                    if row["schema_id"] == name and row["prepared_split"] == split
                ),
            }
            for split in ("train", "val")
        }
    _write_json(output_dir / "normalization.json", {"version": 1, "schemas": schemas})
    _write_json(output_dir / "summary.json", summary)
    return summary


class MixtureSftDataset(Dataset):
    """Current-frame native-control samples and masked future action chunks."""

    def __init__(
        self,
        manifest_dir: str | Path,
        *,
        split: str = "train",
        action_horizon: int = 50,
        action_dim: int = 32,
        max_token_len: int = 200,
        cache_files: int = 4,
        max_samples: int | None = None,
        sample_multiple: int = 1,
    ) -> None:
        self.root = Path(manifest_dir)
        self.episodes = _jsonl(self.root / f"{split}.jsonl")
        self.available_samples = sum(
            len(row["anchor_indices"]) for row in self.episodes
        )
        self.alignment_dropped = 0
        if max_samples is not None:
            count = min(max_samples, self.available_samples)
            self.alignment_dropped = count % sample_multiple
            count -= self.alignment_dropped
            if not count:
                raise ValueError(
                    f"Evaluation needs at least {sample_multiple} anchors after the sample limit "
                    "to give every rank complete batches"
                )

            def anchors(schema):
                for i, row in enumerate(self.episodes):
                    if row["schema_id"] == schema:
                        yield from ((i, frame) for frame in row["anchor_indices"])

            streams = {
                schema: anchors(schema)
                for schema in sorted({row["schema_id"] for row in self.episodes})
            }
            chosen = {}
            while count and streams:
                for schema, stream in list(streams.items()):
                    item = next(stream, None)
                    if item is None:
                        del streams[schema]
                        continue
                    episode, frame = item
                    chosen.setdefault(episode, []).append(frame)
                    count -= 1
                    if not count:
                        break
            self.episodes = [
                {**row, "anchor_indices": chosen[i]}
                for i, row in enumerate(self.episodes)
                if i in chosen
            ]
        self.schemas = json.loads((self.root / "normalization.json").read_text())[
            "schemas"
        ]
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.max_token_len, self.cache_files = max_token_len, cache_files
        if action_horizon < 1 or action_dim < 1:
            raise ValueError("Action horizon and dimension must be positive")
        self.ends = np.cumsum([len(row["anchor_indices"]) for row in self.episodes])
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError(f"No supervised anchors in {split} manifest")
        self._cache = None
        self._tokenizer = None
        for row in self.episodes:
            schema = self.schemas[row["schema_id"]]
            if (
                max(len(schema["action"]["mean"]), len(schema["state"]["mean"]))
                > action_dim
            ):
                raise ValueError(
                    f"Native vector exceeds model dimension: {row['schema_id']}"
                )

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_cache"], state["_tokenizer"] = None, None
        return state

    def numeric_item(
        self, index: int
    ) -> tuple[dict, int, np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized native values and exact core/tail/dimension masks."""
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        row = self.episodes[episode_index]
        offset = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        frame = row["anchor_indices"][offset]
        if self._cache is None:
            self._cache = NumericEpisodeCache(self.cache_files)
        action, state, supervised = self._cache.read(row)
        schema = self.schemas[row["schema_id"]]
        indices = frame + np.arange(self.action_horizon)
        valid = indices < len(action)
        indices = np.minimum(indices, len(action) - 1)
        native_dim = action.shape[-1]
        actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        actions[:, :native_dim] = (action[indices] - schema["action"]["mean"]) / schema[
            "action"
        ]["std"]
        loss_mask = np.zeros_like(actions, dtype=bool)
        loss_mask[:, :native_dim] = (valid & supervised[indices])[:, None]
        normalized_state = np.zeros(self.action_dim, dtype=np.float32)
        normalized_state[: state.shape[-1]] = (
            state[frame] - schema["state"]["mean"]
        ) / schema["state"]["std"]
        normalized_state = np.tanh(normalized_state)
        return row, frame, normalized_state, actions, loss_mask

    def __getitem__(self, index: int) -> dict:
        from lerobot.datasets.video_utils import decode_video_frames
        from openpi.models.tokenizer import PaligemmaTokenizer

        row, frame, state, actions, loss_mask = self.numeric_item(index)
        if self._tokenizer is None:
            self._tokenizer = PaligemmaTokenizer(self.max_token_len)
        prompt = (
            f"Schema: {row['schema_id']}; control: {row['control_mode']}; "
            f"native fps: {row['fps']:g}. {row['task']}"
        )
        tokens, token_mask = self._tokenizer.tokenize(prompt, state)
        images, image_masks = {}, {}
        for slot in IMAGE_KEYS:
            image_masks[slot] = slot in row["videos"]
            if slot in row["videos"]:
                video = row["videos"][slot]
                timestamp = video["from_timestamp"] + frame / row["fps"]
                image = decode_video_frames(
                    video["path"],
                    [timestamp],
                    tolerance_s=0.5 / row["fps"] + 1e-5,
                    backend="pyav",
                )[0]
                height, width = image.shape[-2:]
                scale = 224 / max(height, width)
                resized = torch.nn.functional.interpolate(
                    image[None],
                    size=(round(height * scale), round(width * scale)),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                dh, dw = 224 - resized.shape[-2], 224 - resized.shape[-1]
                image = torch.nn.functional.pad(
                    resized, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
                )
                images[slot] = image.permute(1, 2, 0).numpy() * 2 - 1
            else:
                images[slot] = np.zeros((224, 224, 3), dtype=np.float32)
        return {
            "observation": {
                "image": images,
                "image_mask": image_masks,
                "state": state,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
            },
            "actions": actions,
            "action_loss_mask": loss_mask,
        }


class MixtureSampler(Sampler[int]):
    """Epoch-seeded weighted global draws, sharded deterministically by rank."""

    def __init__(
        self,
        dataset: MixtureSftDataset,
        *,
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        samples_per_epoch: int = 4096,
    ) -> None:
        if not 0 <= rank < world_size or batch_size < 1:
            raise ValueError("Invalid rank/world size or batch size")
        self.dataset, self.rank, self.world_size = dataset, rank, world_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self.position = 0
        self.num_samples = (
            math.ceil(samples_per_epoch / (world_size * batch_size)) * batch_size
            if shuffle
            else len(range(rank, len(dataset), world_size))
        )
        counts = {}
        for row in dataset.episodes:
            counts[row["schema_id"]] = counts.get(row["schema_id"], 0) + len(
                row["anchor_indices"]
            )
        self.weights = torch.cat(
            [
                torch.full(
                    (len(row["anchor_indices"]),),
                    dataset.schemas[row["schema_id"]]["weight"]
                    / counts[row["schema_id"]],
                    dtype=torch.float64,
                )
                for row in dataset.episodes
            ]
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.position = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        total = self.num_samples * self.world_size
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.multinomial(
                self.weights, total, replacement=True, generator=generator
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))
        local_indices = indices[self.rank : len(indices) : self.world_size]
        for position in range(self.position, len(local_indices)):
            self.position = position + 1
            yield local_indices[position]

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.set_epoch(state["epoch"])
        self.position = state["position"]


def build_mixture_sft_dataloader(
    cfg: Any, world_size: int, rank: int, data_paths: Any, eval_dataset: bool = False
) -> tuple[Any, dict]:
    """Build a checkpointable native-control mixture loader for the SFT worker."""
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader

    if not isinstance(data_paths, (str, Path)):
        if len(data_paths) != 1:
            raise ValueError(
                "Mixture data_paths must name one prepared manifest directory"
            )
        data_paths = data_paths[0]
    if OmegaConf.select(cfg, "data.num_obs_steps", default=1) != 1:
        raise ValueError("The mixture input contract currently uses num_obs_steps=1")
    batch_size = int(
        cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size)
        if eval_dataset
        else cfg.actor.micro_batch_size
    )
    dataset = MixtureSftDataset(
        data_paths,
        split="val" if eval_dataset else "train",
        action_horizon=int(
            OmegaConf.select(cfg, "actor.model.openpi.action_horizon", default=50)
        ),
        action_dim=int(cfg.actor.model.openpi.model_action_dim),
        max_token_len=int(cfg.actor.model.openpi.max_token_len),
        cache_files=int(OmegaConf.select(cfg, "data.mixture.cache_files", default=4)),
        max_samples=(
            int(OmegaConf.select(cfg, "data.eval_max_samples", default=128))
            if eval_dataset
            else None
        ),
        sample_multiple=world_size * batch_size if eval_dataset else 1,
    )
    sampler = MixtureSampler(
        dataset,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        seed=int(cfg.actor.seed),
        shuffle=not eval_dataset,
        samples_per_epoch=int(
            OmegaConf.select(cfg, "data.samples_per_epoch", default=4096)
        ),
    )
    workers = int(cfg.data.num_workers)
    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=workers > 0,
        pin_memory=True,
        drop_last=False,
    )
    return loader, {
        "num_samples": len(dataset),
        "num_obs_steps": 1,
        "manifest_dir": str(dataset.root),
        "schemas": dataset.schemas,
        "eval_available_anchors": dataset.available_samples if eval_dataset else None,
        "eval_alignment_dropped": dataset.alignment_dropped if eval_dataset else 0,
    }

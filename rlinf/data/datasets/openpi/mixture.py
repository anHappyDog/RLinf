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

import gzip
import hashlib
import heapq
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from rlinf.data.datasets.openpi.control_schema import ControlAdapter, normalize
from rlinf.data.datasets.openpi.manifest_index import (
    ManifestIndex,
    build_manifest_indexes,
)
from rlinf.data.datasets.parquet import ParquetEpisodeCache
from rlinf.data.datasets.video import VideoFrameCache
from rlinf.utils.logging import get_logger

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_QUALITIES = {
    "native_source_success",
    "clean_pickup_annotation",
    "sampled_visual_confirmed",
}


def _iter_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _jsonl(path: Path) -> list[dict]:
    return list(_iter_jsonl(path))


def _catalog_rows(root: Path) -> list[dict]:
    """Read either a legacy collection or a relocated final LeRobot export."""
    if (root / "catalog/episodes.jsonl").exists():
        return [
            {**row, "catalog_root": str(root)}
            for row in _jsonl(root / "catalog/episodes.jsonl")
        ]
    get_logger().info("Indexing final export: %s", root)
    partition = json.loads((root / "meta/partition.json").read_text())
    metadata = partition["metadata"]
    provenance = {}
    with (root / "meta/clip_provenance.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            prov = item["provenance"]
            if provenance and len(provenance) % 100000 == 0:
                get_logger().info(
                    "Indexed %d provenance episodes: %s", len(provenance), root.name
                )
            provenance[item["episode_index"]] = {
                key: prov.get(key)
                for key in (
                    "domain",
                    "core_range_in_export",
                    "semantic_core_status",
                    "supervision_mask_feature",
                )
            }
    rows = []
    for item in _iter_jsonl(root / "meta/collection_index.jsonl"):
        prov = provenance[item["episode_index"]]
        dataset = item["source_dataset"]
        # Keep split grouping at the publisher group while balancing original episodes.
        episode_group = (
            f"{dataset}:{item['source_revision']}:{item['source_episode_id']}"
        )
        rows.append(
            {
                **item,
                "catalog_id": f"{root.name}:{item['clip_id']}",
                "catalog_root": str(root),
                "partition": ".",
                "provenance_file": "meta/clip_provenance.jsonl",
                "domain": prov["domain"] or metadata.get("domain"),
                "arms": metadata["arms"],
                "robot_identity": {
                    "robot_type": partition.get("robot_type"),
                    "domain": metadata.get("domain"),
                    "source_contract": metadata.get("source_contract", {}),
                },
                "source_group_id": episode_group,
                "split_group_id": f"{dataset}:{item['source_group_id']}",
                "publisher_group_id": item["source_group_id"],
                "core_range_in_export": prov.get("core_range_in_export"),
                "export_provenance": prov,
                "accepted_visual_review": item.get("visual_review_status") == "accepted"
                and item.get("visual_pickup_confirmed") is True,
            }
        )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _episode_metadata(
    root: Path, cameras: dict[str, str], episode_ids: list[int] | None = None
) -> dict[int, dict]:
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
    file_starts = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        if episode_ids is not None:
            starts = pq.read_table(
                path,
                columns=["data/chunk_index", "data/file_index", "dataset_from_index"],
            )
            starts = starts.group_by(["data/chunk_index", "data/file_index"]).aggregate(
                [("dataset_from_index", "min")]
            )
            for row in starts.to_pylist():
                key = (row["data/chunk_index"], row["data/file_index"])
                value = row["dataset_from_index_min"]
                file_starts[key] = min(file_starts.get(key, value), value)
        records.extend(
            pq.read_table(
                path,
                columns=columns,
                filters=[("episode_index", "in", episode_ids)]
                if episode_ids is not None
                else None,
            ).to_pylist()
        )
    if episode_ids is None:
        for row in records:
            key = (row["data/chunk_index"], row["data/file_index"])
            file_starts[key] = min(
                file_starts.get(key, row["dataset_from_index"]),
                row["dataset_from_index"],
            )
    for row in records:
        key = (row["data/chunk_index"], row["data/file_index"])
        row["row_offset"] = row["dataset_from_index"] - file_starts[key]
    return {int(row["episode_index"]): row for row in records}


class NumericEpisodeCache:
    """Bounded process-local cache of selected numeric Parquet columns."""

    def __init__(self, max_files: int = 4) -> None:
        self.max_files = max_files
        self._parquet = ParquetEpisodeCache(max_files)

    def read_columns(self, row: dict, columns: list[str]) -> dict[str, np.ndarray]:
        """Read only requested columns and row groups, sliced to one episode."""
        return self._parquet.read_columns(row, columns)

    def read(self, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read native action/state and the intersection of supervision masks."""
        feature = row.get("supervision_mask_feature")
        columns = ["action", "observation.state"] + ([feature] if feature else [])
        arrays = self.read_columns(row, columns)
        frames = row["frames"]
        action, state = arrays["action"], arrays["observation.state"]
        if len(action) != frames or len(state) != frames:
            raise ValueError(f"Episode exceeds its Parquet file: {row['catalog_id']}")
        mask = np.zeros(frames, dtype=bool)
        lo, hi = row["core_range"]
        mask[lo:hi] = True
        if feature:
            mask &= arrays[feature].reshape(frames).astype(bool)
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


def action_chunk(
    action: np.ndarray,
    state: np.ndarray,
    supervised: np.ndarray,
    frame: int,
    horizon: int,
    adapter: ControlAdapter,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one anchor-relative chunk and its native time supervision mask."""
    indices = frame + np.arange(horizon)
    valid = indices < len(action)
    indices = np.minimum(indices, len(action) - 1)
    return (
        adapter.encode_actions(action[indices], state[frame]),
        valid & supervised[indices],
    )


def _schema_statistics(train, cache, source, horizon, seed, max_anchors):
    adapter = ControlAdapter.from_config(source.get("adapter"))
    groups = {}
    for row in train:
        groups.setdefault(row["source_group_id"], []).append(row)

    def candidates():
        for rows in groups.values():
            anchors = ((row, frame) for row in rows for frame in row["anchor_indices"])
            yield from heapq.nsmallest(
                32,
                anchors,
                key=lambda item: _fraction(
                    f"stats:{item[0]['catalog_id']}:{item[1]}", seed
                ),
            )

    chosen = heapq.nsmallest(
        max_anchors,
        candidates(),
        key=lambda item: _fraction(f"stats:{item[0]['catalog_id']}:{item[1]}", seed),
    )
    actions, states = [], []
    for row, frame in chosen:
        action, state, supervised = cache.read(row)
        chunk, mask = action_chunk(action, state, supervised, frame, horizon, adapter)
        actions.append(chunk[mask])
        states.append(adapter.encode_state(state[frame]))
    transform = source.get("normalization", {}).get("transform", "quantile")
    result = []
    for values, indices in (
        (np.concatenate(actions), adapter.gripper_action_indices),
        (np.stack(states), adapter.gripper_state_indices),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite transformed training values: {source['name']}"
            )
        moments = _Moments()
        moments.add(values)
        stats = {
            **moments.result(),
            "transform": transform,
            "passthrough_indices": list(indices),
        }
        if transform == "quantile":
            stats.update(
                zip(("q01", "q99"), np.quantile(values, (0.01, 0.99), axis=0).tolist())
            )
        result.append(stats)
    return (
        *result,
        {
            "policy": "up to32anchors/source_episode then hash sample; valid transformed future elements",
            "anchors": len(chosen),
            "max_anchors": max_anchors,
            "seed": seed,
        },
    )


def prepare_mixture(
    catalog_root: str | Path | list[str | Path],
    output_dir: str | Path,
    sources: list[dict],
    val_fraction: float = 0.1,
    seed: int = 0,
    allow_development: bool = False,
    max_train_episodes: int | None = None,
    max_val_episodes: int | None = None,
    action_horizon: int = 50,
    action_dim: int = 32,
    sampling_unit: str = "frame",
    statistics_max_anchors: int = 8192,
    target_recipe: str | None = None,
    source_weight_policy: str | None = None,
) -> dict:
    """Prepare grouped splits, native numeric statistics, and v3 media references.

    Each source names one independent native control schema and camera mapping.
    Episode limits apply after grouping the complete eligible pool into splits.
    """
    if target_recipe is not None:
        from rlinf.data.datasets.openpi.realized_trajectory import RECIPE

        if target_recipe != RECIPE["name"]:
            raise ValueError(f"Unknown target recipe: {target_recipe}")
        if action_horizon != RECIPE["horizon"] or action_dim < 8:
            raise ValueError(
                "realized_joint_trajectory v1 requires horizon=16 and action_dim>=8"
            )
        if any(source.get("arms") != 1 for source in sources):
            raise ValueError(
                "realized_joint_trajectory v1 supports explicit single-arm sources only"
            )
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    names = [source["name"] for source in sources]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Mixture source names must be nonempty and unique")
    if action_horizon < 1 or action_dim < 1 or statistics_max_anchors < 1:
        raise ValueError(
            "Horizon, model dimension and statistics budget must be positive"
        )
    if sampling_unit not in {"frame", "episode"}:
        raise ValueError(f"Unknown sampling unit: {sampling_unit}")
    roots = [catalog_root] if isinstance(catalog_root, (str, Path)) else catalog_root
    roots = [Path(root).resolve() for root in roots]
    output_dir = Path(output_dir)
    pool = [row for root in roots for row in _catalog_rows(root)]
    ids = [row["catalog_id"] for row in pool]
    if len(ids) != len(set(ids)):
        raise ValueError("Combined catalogs must have unique catalog IDs")
    cache = NumericEpisodeCache()
    trajectory_cache = None
    if target_recipe is not None:
        from rlinf.data.datasets.openpi.realized_trajectory import (
            RealizedTrajectoryCache,
        )

        trajectory_cache = RealizedTrajectoryCache(cache)
    provenance = {}
    metadata = {}
    infos = {}
    excluded = {"quality": 0, "development": 0, "unknown_core": 0}
    selected = []
    logger = get_logger()
    for source in sources:
        logger.info("Preparing source %s", source["name"])
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
            and ("arms" not in source or row.get("arms") == source["arms"])
            and (
                "partitions" not in source
                or Path(row["catalog_root"]).name in source["partitions"]
            )
        ]
        eligible = []
        for row in rows:
            if row["quality_tier"] not in _QUALITIES and not row.get(
                "accepted_visual_review", False
            ):
                excluded["quality"] += 1
                continue
            development = row["split"] == "development_only"
            if development and not allow_development:
                excluded["development"] += 1
                continue
            collection = Path(row["catalog_root"])
            root = collection / row["partition"]
            if "export_provenance" in row:
                prov = row["export_provenance"]
            else:
                if str(root) not in provenance:
                    prov_path = collection / row["provenance_file"]
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
                else row.get("split_group_id", row["source_group_id"])
            )
            if target_recipe is not None:
                if not row.get("scene_id"):
                    group = row["source_group_id"]
                row = {
                    **row,
                    "target_recipe": {
                        **RECIPE,
                        **source["trajectory"],
                        "gripper_identity": source["gripper_identity"],
                    },
                    "state_gripper_present": source["trajectory"].get("gripper_state")
                    is not None,
                    "gripper_identity": source["gripper_identity"],
                }
                fixed_split = source.get("fixed_partition_splits", {}).get(
                    collection.name
                )
                if fixed_split is not None:
                    row["assigned_split"] = fixed_split
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
        logger.info(
            "Source %s eligible exported episodes: %d", source["name"], len(eligible)
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
        fixed = {}
        for row in eligible:
            assigned = row.get("assigned_split")
            if assigned is not None:
                if assigned not in {"train", "val", "test"}:
                    raise ValueError(f"Unknown assigned split: {assigned}")
                group = row["split_group"]
                if group in fixed and fixed[group] != assigned:
                    raise ValueError(f"Conflicting fixed splits for group {group}")
                if group in development_groups and assigned != "train":
                    raise ValueError(f"Development group cannot enter holdout: {group}")
                fixed[group] = assigned
        split_rows = {"train": [], "val": [], "test": []}
        for row in eligible:
            val = (
                row["split_group"] not in development_groups
                and _fraction(row["split_group"], seed) < val_fraction
            )
            split_rows[fixed.get(row["split_group"], "val" if val else "train")].append(
                row
            )
        for split, limit_key, default_limit in [
            ("train", "max_train_episodes", max_train_episodes),
            ("val", "max_val_episodes", max_val_episodes),
            ("test", "max_test_episodes", None),
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
                    subset_ids = (
                        [
                            item["episode_index"]
                            for item in eligible
                            if item["root"] == row["root"]
                        ]
                        if len(eligible) < 256
                        else None
                    )
                    metadata[key] = _episode_metadata(root, cameras, subset_ids)
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
                if trajectory_cache is None:
                    _, _, mask = cache.read(row)
                else:
                    mask = trajectory_cache.valid_anchors(row)
                row["anchor_indices"] = np.flatnonzero(mask).tolist()
                if row["anchor_indices"]:
                    row["prepared_split"] = split
                    selected.append(row)
                    if len(selected) % 10000 == 0:
                        logger.info(
                            "Prepared %d exported episodes; current source=%s split=%s",
                            len(selected),
                            source["name"],
                            split,
                        )
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
        adapted = "adapter" in source or "normalization" in source
        if target_recipe is not None:
            action_stats, state_stats = {}, {}
            stats_policy = {
                "policy": "shared train-only realized targets; bounded original-episode sample"
            }
        elif adapted:
            action_stats, state_stats, stats_policy = _schema_statistics(
                train, cache, source, action_horizon, seed, statistics_max_anchors
            )
        else:
            action_moments, state_moments = _Moments(), _Moments()
            for row in sorted(
                train, key=lambda row: (row["data_file"], row["row_offset"])
            ):
                action, state, mask = cache.read(row)
                action_moments.add(action[mask])
                state_moments.add(state[mask])
            action_stats = {**action_moments.result(), "transform": "mean_std"}
            state_stats = {**state_moments.result(), "transform": "tanh_mean_std"}
            stats_policy = {"policy": "all supervised native training rows"}
        control, fps = controls.pop()
        schemas[source["name"]] = {
            "weight": source["weight"],
            "action": action_stats,
            "state": state_stats,
            "adapter": source.get("adapter"),
            "control_contract": source.get("control_contract"),
            "statistics_policy": stats_policy,
            "control_mode": "realized_joint_trajectory_v1"
            if target_recipe is not None
            else source.get("model_control_mode", control),
            "native_control_mode": control,
            "fps": fps,
            "cameras": source["cameras"],
            **(
                {
                    "target_recipe": train[0]["target_recipe"],
                    "robot_identity": train[0].get("robot_identity"),
                }
                if target_recipe is not None
                else {}
            ),
            "train_catalog_ids": [row["catalog_id"] for row in train],
        }
    if target_recipe is not None:
        from rlinf.data.datasets.openpi.realized_trajectory import MaskedMoments

        action_moments, state_moments = MaskedMoments(), MaskedMoments()
        stats_anchors = 0
        for source in sources:
            logger.info("Computing train-only realized statistics: %s", source["name"])
            grouped = {}
            for row in selected:
                if (
                    row["schema_id"] == source["name"]
                    and row["prepared_split"] == "train"
                ):
                    grouped.setdefault(row["source_group_id"], []).append(row)
            chosen_groups = heapq.nsmallest(
                statistics_max_anchors,
                grouped,
                key=lambda group: _fraction(f"realized_stats:{group}", seed),
            )
            chosen_anchors = []
            for group in chosen_groups:
                rows = grouped[group]
                total = sum(len(row["anchor_indices"]) for row in rows)
                offset = min(
                    int(_fraction(f"realized_anchor:{group}", seed) * total), total - 1
                )
                for row in rows:
                    if offset < len(row["anchor_indices"]):
                        chosen_anchors.append((row, row["anchor_indices"][offset]))
                        break
                    offset -= len(row["anchor_indices"])
            for row, frame in sorted(
                chosen_anchors,
                key=lambda item: (item[0]["data_file"], item[0]["row_offset"], item[1]),
            ):
                sample = trajectory_cache.sample(row, frame)
                action_moments.add(sample["actions"], sample["action_mask"])
                state_moments.add(sample["state"][None], sample["state_mask"][None])
                stats_anchors += 1
        if np.any(action_moments.count[:7] == 0):
            raise ValueError(
                "Realized recipe has no supervised future joint targets in training statistics"
            )
        for schema in schemas.values():
            schema["action"], schema["state"] = (
                action_moments.result(),
                state_moments.result(),
            )
            schema["statistics_policy"].update(
                {
                    "anchors": stats_anchors,
                    "max_anchors_per_source": statistics_max_anchors,
                    "seed": seed,
                }
            )

    if source_weight_policy is not None:
        if source_weight_policy != "molmobot_60_sqrt_episodes" or target_recipe is None:
            raise ValueError(
                "Source weight policy requires realized_joint_trajectory and molmobot_60_sqrt_episodes"
            )
        if "molmobot" not in schemas or len(schemas) < 2:
            raise ValueError(
                "MolmoBot 60 percent policy requires molmobot and auxiliary sources"
            )
        sizes = {
            name: len(
                {
                    row["source_group_id"]
                    for row in selected
                    if row["schema_id"] == name and row["prepared_split"] == "train"
                }
            )
            for name in schemas
        }
        auxiliary = sum(
            math.sqrt(count) for name, count in sizes.items() if name != "molmobot"
        )
        for name, schema in schemas.items():
            schema["weight"] = (
                0.6 if name == "molmobot" else 0.4 * math.sqrt(sizes[name]) / auxiliary
            )
    for key in ("source_group_id", "split_group"):
        assignments = {}
        for row in selected:
            group, split = row[key], row["prepared_split"]
            if group in assignments and assignments[group] != split:
                raise ValueError(f"Group leakage between splits: {group}")
            assignments[group] = split
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 2,
        "catalog_roots": [str(root) for root in roots],
        "seed": seed,
        "val_fraction": val_fraction,
        "allow_development": allow_development,
        "num_obs_steps": 1,
        "time_policy": "future source-clock measured joints at 15 Hz"
        if target_recipe is not None
        else "native indices; no resampling",
        "target_recipe": RECIPE if target_recipe is not None else None,
        "excluded": excluded,
        "schemas": {},
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "sampling_unit": sampling_unit,
        "source_weight_policy": source_weight_policy,
    }
    for split in ("train", "val", "test"):
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
            "weight": schemas[name]["weight"],
            **{
                split: {
                    "episodes": sum(
                        row["schema_id"] == name and row["prepared_split"] == split
                        for row in selected
                    ),
                    "original_episodes": len(
                        {
                            row["source_group_id"]
                            for row in selected
                            if row["schema_id"] == name
                            and row["prepared_split"] == split
                        }
                    ),
                    "anchors": sum(
                        len(row["anchor_indices"])
                        for row in selected
                        if row["schema_id"] == name and row["prepared_split"] == split
                    ),
                }
                for split in ("train", "val", "test")
            },
        }
    bound = target_recipe is not None or any(
        "adapter" in source or "normalization" in source for source in sources
    )
    _write_json(
        output_dir / "normalization.json",
        {
            "version": 2,
            "schemas": schemas,
            "target_recipe": RECIPE if target_recipe is not None else None,
            "sampling_unit": sampling_unit,
            "action_horizon": action_horizon if bound else None,
            "action_dim": action_dim if bound else None,
        },
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "contract.json",
        {
            "version": 2,
            "files": {
                name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                for name in (
                    "train.jsonl",
                    "val.jsonl",
                    "test.jsonl",
                    "normalization.json",
                )
            },
        },
    )
    build_manifest_indexes(output_dir)
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
        sample_cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(manifest_dir)
        self.split = split
        self.episodes = ManifestIndex(self.root / f"{split}.index")
        self.manifest_fingerprint = self.episodes.metadata["manifest_fingerprint"]
        self.available_samples = self.episodes.metadata["anchors"]
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
                episodes = [
                    (
                        i,
                        sorted(
                            row["anchor_indices"],
                            key=lambda frame: _fraction(
                                f"eval:{row['catalog_id']}:{frame}", 0
                            ),
                        ),
                    )
                    for i, row in enumerate(self.episodes)
                    if row["schema_id"] == schema
                ]
                for offset in range(max(len(frames) for _, frames in episodes)):
                    for i, frames in episodes:
                        if offset < len(frames):
                            yield i, frames[offset]

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
        metadata = json.loads((self.root / "normalization.json").read_text())
        # Catalog IDs record preparation provenance, not per-sample transforms.
        self.schemas = {
            name: {
                key: value
                for key, value in schema.items()
                if key != "train_catalog_ids"
            }
            for name, schema in metadata["schemas"].items()
        }
        self.sampling_unit = metadata.get("sampling_unit", "frame")
        self.requires_contract = metadata.get("action_horizon") is not None
        for field, actual in (
            ("action_horizon", action_horizon),
            ("action_dim", action_dim),
        ):
            expected = metadata.get(field)
            if expected is not None and expected != actual:
                raise ValueError(
                    f"Prepared {field}={expected}, requested {actual}; recompute statistics for this contract"
                )
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.max_token_len, self.cache_files = max_token_len, cache_files
        if action_horizon < 1 or action_dim < 1:
            raise ValueError("Action horizon and dimension must be positive")
        self.ends = (
            self.episodes.ends
            if isinstance(self.episodes, ManifestIndex)
            else np.cumsum([len(row["anchor_indices"]) for row in self.episodes])
        )
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError(f"No supervised anchors in {split} manifest")
        self._cache = None
        self._tokenizer = None
        self._video_cache = None
        self._trajectory_cache = None
        for schema_id, schema in self.schemas.items():
            if (
                max(len(schema["action"]["mean"]), len(schema["state"]["mean"]))
                > action_dim
            ):
                raise ValueError(f"Native vector exceeds model dimension: {schema_id}")
        self.sample_cache_dir = (
            Path(sample_cache_dir) if sample_cache_dir is not None else None
        )
        if self.sample_cache_dir is not None:
            metadata = json.loads((self.sample_cache_dir / "metadata.json").read_text())
            if metadata["contract"] != self.sample_cache_contract():
                raise ValueError(
                    "Sample cache requires identical manifests, input dimensions, "
                    "token length and preprocessing code"
                )

    def sample_cache_contract(self) -> dict:
        """Bind materialized samples to selected rows and deterministic preprocessing."""
        inputs = {
            "episodes": [
                {**row, "anchor_indices": list(map(int, row["anchor_indices"]))}
                for row in self.episodes
            ],
            "schemas": self.schemas,
            "split": self.split,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "max_token_len": self.max_token_len,
        }
        source = Path(__file__)
        files = (
            source,
            source.with_name("control_schema.py"),
            source.with_name("realized_trajectory.py"),
            source.parents[1] / "video.py",
            source.parents[1] / "parquet.py",
            source.parents[3] / "models/embodiment/openpi/mixture_io.py",
        )
        return {
            "version": 1,
            "inputs_sha256": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "preprocessing_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
        }

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        (
            state["_cache"],
            state["_tokenizer"],
            state["_video_cache"],
            state["_trajectory_cache"],
        ) = None, None, None, None
        if isinstance(self.episodes, ManifestIndex):
            state["ends"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if isinstance(self.episodes, ManifestIndex):
            self.ends = self.episodes.ends

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
        schema = self.schemas[row["schema_id"]]
        if row.get("target_recipe") is not None:
            from rlinf.data.datasets.openpi.realized_trajectory import (
                RealizedTrajectoryCache,
            )

            if self._trajectory_cache is None:
                self._trajectory_cache = RealizedTrajectoryCache(self._cache)
            sample = self._trajectory_cache.sample(row, frame)
            state = np.zeros(self.action_dim, dtype=np.float32)
            actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
            loss_mask = np.zeros_like(actions, dtype=bool)
            state[:8] = np.where(
                sample["state_mask"], normalize(sample["state"], schema["state"]), 0
            )
            actions[:, :8] = np.where(
                sample["action_mask"], normalize(sample["actions"], schema["action"]), 0
            )
            loss_mask[:, :8] = sample["action_mask"]
            return row, frame, state, actions, loss_mask
        action, state, supervised = self._cache.read(row)
        adapter = ControlAdapter.from_config(schema.get("adapter"))
        chunk, valid = action_chunk(
            action, state, supervised, frame, self.action_horizon, adapter
        )
        native_dim = action.shape[-1]
        actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        actions[:, :native_dim] = normalize(chunk, schema["action"])
        loss_mask = np.zeros_like(actions, dtype=bool)
        loss_mask[:, :native_dim] = valid[:, None]
        normalized_state = np.zeros(self.action_dim, dtype=np.float32)
        normalized_state[: state.shape[-1]] = normalize(
            adapter.encode_state(state[frame]), schema["state"]
        )
        return row, frame, normalized_state, actions, loss_mask

    def __getitem__(self, index: int) -> dict:
        if self.sample_cache_dir is not None:
            with gzip.open(self.sample_cache_dir / f"{index}.pkl.gz", "rb") as stream:
                return pickle.load(stream)
        from openpi.models.tokenizer import PaligemmaTokenizer

        from rlinf.models.embodiment.openpi.mixture_io import (
            build_observation,
            build_prompt,
        )

        row, frame, state, actions, loss_mask = self.numeric_item(index)
        if self._tokenizer is None:
            self._tokenizer = PaligemmaTokenizer(self.max_token_len)
        prompt = build_prompt(
            row["schema_id"],
            self.schemas[row["schema_id"]]["control_mode"],
            row["fps"],
            row["task"],
        )
        images = {}
        for slot in IMAGE_KEYS:
            if slot in row["videos"]:
                video = row["videos"][slot]
                timestamp = video["from_timestamp"] + frame / row["fps"]
                if self._video_cache is None:
                    self._video_cache = VideoFrameCache(self.cache_files)
                images[slot] = self._video_cache.read(
                    video["path"], timestamp, tolerance_s=0.5 / row["fps"] + 1e-5
                )
        return {
            "observation": build_observation(state, images, prompt, self._tokenizer),
            "actions": actions,
            "action_loss_mask": loss_mask,
            "sample_schema": row["schema_id"],
            "sample_group": row["source_group_id"],
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
        groups = {}
        for row in dataset.episodes:
            counts[row["schema_id"]] = counts.get(row["schema_id"], 0) + len(
                row["anchor_indices"]
            )
            key = (row["schema_id"], row["source_group_id"])
            groups[key] = groups.get(key, 0) + len(row["anchor_indices"])
        group_counts = {}
        for schema, _ in groups:
            group_counts[schema] = group_counts.get(schema, 0) + 1
        self.weights = torch.cat(
            [
                torch.full(
                    (len(row["anchor_indices"]),),
                    dataset.schemas[row["schema_id"]]["weight"]
                    / (
                        group_counts[row["schema_id"]]
                        * groups[(row["schema_id"], row["source_group_id"])]
                        if dataset.sampling_unit == "episode"
                        else counts[row["schema_id"]]
                    ),
                    dtype=torch.float64,
                )
                for row in dataset.episodes
            ]
        )
        self._cumulative_weights = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.position = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        total = self.num_samples * self.world_size
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            if self.weights.numel() > 2**24:
                # torch.multinomial rejects more than 2**24 categories.
                if self._cumulative_weights is None:
                    self._cumulative_weights = self.weights.cumsum(0)
                draws = torch.rand(total, dtype=torch.float64, generator=generator)
                draws *= self._cumulative_weights[-1]
                indices = torch.searchsorted(
                    self._cumulative_weights, draws, right=True
                ).tolist()
            else:
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
        state = {"epoch": self.epoch, "position": self.position}
        if self.dataset.manifest_fingerprint is not None:
            state["manifest_fingerprint"] = self.dataset.manifest_fingerprint
            state["seed"] = self.seed
            state["global_samples"] = self.num_samples * self.world_size
            state["world_size"] = self.world_size
            state["rank"] = self.rank
        return state

    def load_state_dict(self, state: dict) -> None:
        if self.dataset.requires_contract and "manifest_fingerprint" not in state:
            raise ValueError("Adapted mixture resume requires a manifest fingerprint")
        if "manifest_fingerprint" in state:
            expected = (
                self.dataset.manifest_fingerprint,
                self.seed,
                self.num_samples * self.world_size,
                self.world_size,
                self.rank,
            )
            actual = (
                state["manifest_fingerprint"],
                state["seed"],
                state["global_samples"],
                state["world_size"],
                state["rank"],
            )
            if actual != expected:
                raise ValueError(
                    "Sampler resume requires identical manifests, seed, sample budget and rank topology"
                )
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
    cache_root = OmegaConf.select(cfg, "data.mixture.sample_cache_dir")
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
        sample_cache_dir=(
            Path(cache_root) / ("val" if eval_dataset else "train")
            if cache_root is not None
            else None
        ),
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

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

"""Prepare only explicitly READY sources from a canonical qtarget manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rlinf.data.datasets.openpi.manifest_index import build_manifest_indexes
from rlinf.data.datasets.openpi.mixture import NumericEpisodeCache, _episode_metadata
from rlinf.data.datasets.openpi.realized_trajectory import MaskedMoments
from rlinf.data.datasets.openpi_official.qtarget import RECIPE, CanonicalQtargetDataset


def prepare(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    action_horizon: int,
    stats_samples: int,
) -> dict:
    """Prepare episode routes and bounded, deterministic train-only statistics."""
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1 or manifest["contract"] != "qtarget_width_v1":
        raise ValueError("Expected qtarget_width_v1 manifest")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = {"train": [], "val": [], "test": []}
    schemas = {}
    names = set()
    numeric_cache = NumericEpisodeCache(4)
    column_cache = {}
    for source in manifest["sources"]:
        if source["status"] != "READY":
            raise ValueError(f"Source is not READY: {source['name']}")
        name = source["name"]
        if name in names or source["weight"] <= 0:
            raise ValueError(
                f"Expected unique source names and positive explicit weights: {name}"
            )
        names.add(name)
        data_root = Path(source["root"])
        info = json.loads((data_root / "meta/info.json").read_text())
        expected_action = [f"q_target_rad_{i}" for i in range(7)] + [
            "gripper_closedness"
        ]
        expected_state = [f"q_measured_rad_{i}" for i in range(7)] + ["gripper_width_m"]
        for feature, expected in (
            ("action", expected_action),
            ("observation.state", expected_state),
            ("observation.state_mask", expected_state),
        ):
            if (
                info["features"][feature]["shape"] != [8]
                or info["features"][feature]["names"] != expected
            ):
                raise ValueError(f"Canonical feature mismatch: {name}/{feature}")
        if "action_mask" in info["features"] and info["features"]["action_mask"][
            "shape"
        ] != [8]:
            raise ValueError(f"Expected canonical action_mask8: {name}")
        cameras = source["camera_map"]
        if (
            set(cameras) - {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
            or not cameras
        ):
            raise ValueError(f"Expected explicit canonical camera slots: {name}")
        for feature in cameras.values():
            if info["features"][feature]["dtype"] != "video":
                raise ValueError(f"Expected a video camera: {name}/{feature}")
        metadata = _episode_metadata(data_root, cameras)
        episode_tasks = {}
        for episode_path in sorted((data_root / "meta/episodes").rglob("*.parquet")):
            for item in pq.read_table(
                episode_path, columns=["episode_index", "tasks"]
            ).to_pylist():
                episode_tasks[int(item["episode_index"])] = item["tasks"]
        split = {"train": "train", "validation": "val"}[source["split"]]
        schema = source.get("normalization_schema", name)
        identity = {
            "robot_type": source["robot_type"],
            "gripper_type": source["gripper_type"],
        }
        if schema in schemas:
            if schemas[schema]["robot_identity"] != identity:
                raise ValueError(
                    "Shared normalization route crosses robot/gripper identity"
                )
        else:
            schemas[schema] = {
                "weight": source["weight"],
                "robot_identity": identity,
                "state": {"mean": [0.0] * 8, "std": [1.0] * 8, "transform": "mean_std"},
                "action": {
                    "mean": [0.0] * 8,
                    "std": [1.0] * 8,
                    "transform": "mean_std",
                },
            }
        for episode, item in metadata.items():
            data_file = data_root / info["data_path"].format(
                chunk_index=item["data/chunk_index"], file_index=item["data/file_index"]
            )
            videos = {
                slot: {
                    "path": str(
                        data_root
                        / info["video_path"].format(
                            video_key=feature,
                            chunk_index=item[f"videos/{feature}/chunk_index"],
                            file_index=item[f"videos/{feature}/file_index"],
                        )
                    ),
                    "from_timestamp": item[f"videos/{feature}/from_timestamp"],
                }
                for slot, feature in cameras.items()
            }
            if data_file not in column_cache:
                column_cache[data_file] = pq.read_schema(data_file).names
            columns = column_cache[data_file]
            task = episode_tasks[episode][0] if episode_tasks[episode] else None
            if task is None:
                raise ValueError(f"Missing original task text: {name}/{episode}")
            row = {
                "catalog_id": f"{name}:{episode}",
                "schema_id": schema,
                "source_group_id": f"{name}:{episode}",
                "robot_identity": identity,
                "qtarget_contract": info["qtarget_contract"],
                "data_file": str(data_file),
                "row_offset": item["row_offset"],
                "frames": item["length"],
                "core_range": [0, item["length"]],
                "anchor_indices": list(range(item["length"])),
                "fps": info["fps"],
                "video_timestamp_feature": "timestamp"
                if "timestamp" in columns
                else None,
                "timestamp_feature": "source.timestamp"
                if "source.timestamp" in columns
                else "timestamp"
                if "timestamp" in columns
                else None,
                "videos": videos,
                "task": task,
            }
            mask_feature = "source.extra.pipeline_supervision_mask"
            mask_columns = []
            if mask_feature in columns:
                row["supervision_mask_feature"] = mask_feature
                mask_columns.append(mask_feature)
            if "action_mask" in columns:
                row["action_mask_feature"] = "action_mask"
                mask_columns.append("action_mask")
            if mask_columns:
                arrays = numeric_cache.read_columns(row, mask_columns)
                valid = np.ones(row["frames"], dtype=bool)
                if mask_feature in arrays:
                    valid &= arrays[mask_feature].reshape(-1).astype(bool)
                if "action_mask" in arrays:
                    if arrays["action_mask"].shape != (row["frames"], 8):
                        raise ValueError(f"Expected action_mask8: {name}/{episode}")
                    valid &= arrays["action_mask"].astype(bool).any(axis=-1)
                row["anchor_indices"] = np.flatnonzero(valid).tolist()
            if row["anchor_indices"]:
                rows[split].append(row)
    train_schemas = {row["schema_id"] for row in rows["train"]}
    if set(schemas) - train_schemas:
        raise ValueError(
            "Validation-only schema requires normalization_schema pointing to its train source"
        )
    normalization = {
        "target_recipe": RECIPE,
        "purpose": manifest.get("purpose", "training"),
        "action_horizon": action_horizon,
        "action_dim": 8,
        "sampling_unit": "episode",
        "schemas": schemas,
        "stats_budget_per_schema": stats_samples,
        "approved_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }

    def write_contract():
        for split, records in rows.items():
            (output_dir / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
        (output_dir / "normalization.json").write_text(
            json.dumps(normalization, indent=2) + "\n"
        )
        names = ["train.jsonl", "val.jsonl", "test.jsonl", "normalization.json"]
        (output_dir / "contract.json").write_text(
            json.dumps(
                {
                    "files": {
                        name: hashlib.sha256(
                            (output_dir / name).read_bytes()
                        ).hexdigest()
                        for name in names
                    }
                },
                indent=2,
            )
        )
        build_manifest_indexes(output_dir)

    write_contract()
    dataset = CanonicalQtargetDataset(
        output_dir, model_path=".", action_horizon=action_horizon
    )
    for schema in schemas:
        candidates = [
            (int(dataset.ends[i - 1]) if i else 0, len(row["anchor_indices"]))
            for i, row in enumerate(dataset.episodes)
            if row["schema_id"] == schema
        ]
        total = sum(count for _, count in candidates)
        selected = np.unique(
            np.linspace(0, total - 1, min(stats_samples, total), dtype=np.int64)
        )
        ends = np.cumsum([count for _, count in candidates])
        state_moments, action_moments = MaskedMoments(), MaskedMoments()
        for local_index in selected:
            episode = int(np.searchsorted(ends, local_index, side="right"))
            start, _ = candidates[episode]
            index = (
                start + int(local_index) - (int(ends[episode - 1]) if episode else 0)
            )
            _, _, state, state_mask, actions, action_mask, _, _ = dataset.raw_item(
                index
            )
            state_moments.add(state[None], state_mask[None])
            action_moments.add(actions, action_mask)
        schemas[schema]["state"] = {**state_moments.result(), "transform": "mean_std"}
        schemas[schema]["action"] = {**action_moments.result(), "transform": "mean_std"}
        schemas[schema]["stats_anchors"] = len(selected)
    write_contract()
    return normalization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--stats-samples-per-schema", type=int, required=True)
    args = parser.parse_args()
    if args.stats_samples_per_schema <= 0:
        parser.error("stats-samples-per-schema must be positive")
    prepare(
        args.manifest,
        args.output,
        action_horizon=args.action_horizon,
        stats_samples=args.stats_samples_per_schema,
    )

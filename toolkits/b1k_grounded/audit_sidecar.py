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

"""Audit a dense grounded-control sidecar before starting formal SFT."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from rlinf.data.b1k_grounded import GroundedControlSpec

_PATH_COLUMNS = (
    "source_data_path",
    "rgb_head_path",
    "rgb_left_wrist_path",
    "rgb_right_wrist_path",
)
_MAX_REPORTED_ERRORS = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_frames(start: int, end: int, stride: int) -> set[int]:
    frames = set(range(start, end, stride))
    frames.add(end - 1)
    return frames


def audit_sidecar(
    dataset_dir: str | Path,
    *,
    expected_tasks: int,
    expected_episodes: int,
    frame_stride: int,
    expected_state_dim: int = 256,
    expected_action_dim: int = 23,
    expected_action_horizon: int = 32,
) -> dict[str, Any]:
    """Validate coverage, tensors, grounding records, and action boundaries."""
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"].get("frame_stride") != frame_stride:
        raise ValueError(
            f"Manifest frame_stride is {manifest['config'].get('frame_stride')}, "
            f"expected {frame_stride}."
        )
    if manifest["config"].get("frame_stride_skills"):
        raise ValueError("frame_stride_skills must be empty for all-skill dense SFT.")
    if manifest["config"].get("selection_mode") != "all":
        raise ValueError("Dense SFT requires selection_mode='all'.")
    if not (dataset_dir / "structural_token_mapping.json").is_file():
        raise FileNotFoundError("structural_token_mapping.json is missing.")

    dataset_root = Path(manifest["dataset_root"])
    interval_frames: dict[tuple[int, int, int, int], set[int]] = (
        collections.defaultdict(set)
    )
    interval_bounds: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    sample_ids: set[str] = set()
    episodes: set[tuple[int, int]] = set()
    tasks: set[int] = set()
    source_paths: set[Path] = set()
    skills: collections.Counter[str] = collections.Counter()
    errors: list[str] = []
    rows = 0
    fully_grounded = 0
    part_samples = 0
    part_grounded = 0

    def record_error(message: str) -> None:
        if len(errors) < _MAX_REPORTED_ERRORS:
            errors.append(message)

    for shard in manifest["shards"]:
        parquet_path = dataset_dir / shard["path"]
        if parquet_path.stat().st_size != shard["bytes"]:
            record_error(f"Shard size differs from manifest: {parquet_path}.")
        if _sha256(parquet_path) != shard["sha256"]:
            record_error(f"Shard SHA-256 differs from manifest: {parquet_path}.")

        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                rows += 1
                sample_id = row["sample_id"]
                if sample_id in sample_ids:
                    record_error(f"Duplicate sample_id: {sample_id}.")
                sample_ids.add(sample_id)

                task_index = row["task_index"]
                episode_index = row["episode_index"]
                frame_index = row["frame_index"]
                interval_start = row["interval_start"]
                interval_end = row["interval_end"]
                key = (
                    task_index,
                    episode_index,
                    row["segment_index"],
                    row["interval_index"],
                )
                bounds = (interval_start, interval_end)
                previous_bounds = interval_bounds.setdefault(key, bounds)
                if previous_bounds != bounds:
                    record_error(
                        f"Interval {key} has inconsistent bounds "
                        f"{previous_bounds} and {bounds}."
                    )
                if not interval_start <= frame_index < interval_end:
                    record_error(
                        f"Sample {sample_id} lies outside [{interval_start}, "
                        f"{interval_end})."
                    )
                interval_frames[key].add(frame_index)
                tasks.add(task_index)
                episodes.add((task_index, episode_index))
                skills[row["skill"]] += 1

                state = np.asarray(row["state"], dtype=np.float32)
                actions = np.asarray(row["actions"], dtype=np.float32)
                action_is_pad = np.asarray(row["action_is_pad"], dtype=np.bool_)
                if state.shape != (expected_state_dim,):
                    record_error(f"Sample {sample_id} has state shape {state.shape}.")
                if actions.shape != (
                    expected_action_horizon,
                    expected_action_dim,
                ):
                    record_error(
                        f"Sample {sample_id} has action shape {actions.shape}."
                    )
                    continue
                if action_is_pad.shape != (expected_action_horizon,):
                    record_error(
                        f"Sample {sample_id} has action_is_pad shape "
                        f"{action_is_pad.shape}."
                    )
                    continue
                if not np.isfinite(state).all() or not np.isfinite(actions).all():
                    record_error(f"Sample {sample_id} contains NaN or infinity.")

                valid_length = min(
                    expected_action_horizon, interval_end - frame_index
                )
                expected_padding = (
                    np.arange(expected_action_horizon) >= valid_length
                )
                if not np.array_equal(action_is_pad, expected_padding):
                    record_error(
                        f"Sample {sample_id} has an incorrect action boundary mask."
                    )
                if valid_length < expected_action_horizon:
                    repeated_tail = np.broadcast_to(
                        actions[valid_length - 1], actions[valid_length:].shape
                    )
                    if not np.array_equal(actions[valid_length:], repeated_tail):
                        record_error(
                            f"Sample {sample_id} does not repeat its final valid action."
                        )

                try:
                    control = GroundedControlSpec.from_json(row["control_json"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    record_error(f"Sample {sample_id} has invalid control_json: {error}")
                else:
                    if control.skill != row["skill"]:
                        record_error(f"Sample {sample_id} has a mismatched skill.")
                    if control.segment_id != row["segment_index"]:
                        record_error(f"Sample {sample_id} has a mismatched segment.")
                    if control.timestep != frame_index:
                        record_error(f"Sample {sample_id} has a mismatched timestep.")

                is_fully_grounded = (
                    row["object_grounding_complete"]
                    and row["part_grounding_complete"]
                )
                if row["fully_grounded"] != is_fully_grounded:
                    record_error(
                        f"Sample {sample_id} has inconsistent grounding flags."
                    )
                fully_grounded += row["fully_grounded"]
                if row["has_part_argument"]:
                    part_samples += 1
                    part_grounded += row["part_grounding_complete"]
                for column in _PATH_COLUMNS:
                    source_paths.add(dataset_root / row[column])

    for key, bounds in interval_bounds.items():
        expected = _expected_frames(*bounds, frame_stride)
        observed = interval_frames[key]
        if observed != expected:
            missing = sorted(expected - observed)[:8]
            unexpected = sorted(observed - expected)[:8]
            record_error(
                f"Interval {key} is not stride-{frame_stride}: "
                f"missing={missing}, unexpected={unexpected}."
            )

    for path in source_paths:
        if not path.is_file():
            record_error(f"Referenced source file is missing: {path}.")

    if len(tasks) != expected_tasks:
        record_error(f"Found {len(tasks)} tasks, expected {expected_tasks}.")
    if len(episodes) != expected_episodes:
        record_error(f"Found {len(episodes)} episodes, expected {expected_episodes}.")
    if rows != manifest["counts"]["samples"]:
        record_error(
            f"Read {rows} rows, manifest reports {manifest['counts']['samples']}."
        )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Sidecar audit failed with {len(errors)} errors:\n{details}")

    return {
        "status": "passed",
        "dataset_dir": str(dataset_dir),
        "tasks": len(tasks),
        "episodes": len(episodes),
        "intervals": len(interval_frames),
        "samples": rows,
        "frame_stride": frame_stride,
        "state_dim": expected_state_dim,
        "action_dim": expected_action_dim,
        "action_horizon": expected_action_horizon,
        "fully_grounded": fully_grounded,
        "fully_grounded_fraction": fully_grounded / rows,
        "part_samples": part_samples,
        "part_grounded": part_grounded,
        "part_grounded_fraction": (
            part_grounded / part_samples if part_samples else 1.0
        ),
        "skill_counts": dict(sorted(skills.items())),
    }


def main() -> None:
    """Run the strict sidecar audit and optionally save its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit_sidecar(
        args.dataset_dir,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        frame_stride=args.frame_stride,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

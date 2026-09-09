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

"""Audit annotation-entity resolution against sampled episode metadata."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from rlinf.data.b1k_grounded import (
    EntityResolver,
    ParseStatus,
    parse_instance_id_mapping,
    parse_skill_annotation,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def _task_goals(dataset_root: Path) -> dict[int, str]:
    goals = {}
    with (dataset_root / "meta" / "tasks.jsonl").open() as file:
        for line in file:
            record = json.loads(line)
            goals[int(record["task_index"])] = record["task"]
    return goals


def _episode_index(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def build_mapping_coverage(
    dataset_root: str | Path,
    *,
    episodes_per_task: int = 1,
) -> dict[str, Any]:
    """Measure mesh-ID resolution on deterministic episode samples.

    Args:
        dataset_root: Root of the B1K 2025 challenge demos.
        episodes_per_task: Number of lexicographically first episodes sampled
            from each task.

    Returns:
        JSON-serializable mapping coverage report.
    """
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    dataset_root = Path(dataset_root)
    goals = _task_goals(dataset_root)
    status_counts: collections.Counter[str] = collections.Counter()
    mesh_count_histogram: collections.Counter[int] = collections.Counter()
    unresolved: collections.Counter[str] = collections.Counter()
    unresolved_parts: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    sampled_episodes = []

    for task_index, goal in sorted(goals.items()):
        annotation_dir = dataset_root / "annotations" / f"task-{task_index:04d}"
        annotation_paths = sorted(annotation_dir.glob("episode_*.json"))[
            :episodes_per_task
        ]
        for annotation_path in annotation_paths:
            episode_index = _episode_index(annotation_path)
            metadata_path = (
                dataset_root
                / "meta"
                / "episodes"
                / f"task-{task_index:04d}"
                / f"episode_{episode_index:08d}.json"
            )
            metadata = _read_json(metadata_path)
            resolver = EntityResolver(
                parse_instance_id_mapping(metadata["ins_id_mapping"])
            )
            annotation = _read_json(annotation_path)
            sampled_episodes.append(
                {
                    "task_index": task_index,
                    "episode_index": episode_index,
                }
            )
            for record in annotation["skill_annotation"]:
                result = parse_skill_annotation(
                    record,
                    goal=goal,
                    episode_id=f"episode_{episode_index:08d}",
                )
                status_counts[result.status.value] += 1
                if result.status is not ParseStatus.VALID:
                    continue
                for argument in result.segment.control.arguments:
                    counts["arguments"] += 1
                    raw_object_id = argument.raw_object_id
                    if raw_object_id in {None, "robot"}:
                        counts["symbolic_arguments"] += 1
                        continue
                    counts["groundable_arguments"] += 1
                    mesh_ids = resolver.resolve(raw_object_id)
                    if mesh_ids:
                        counts["resolved_arguments"] += 1
                        mesh_count_histogram[len(mesh_ids)] += 1
                    else:
                        unresolved[raw_object_id] += 1
                    if argument.part is None:
                        continue
                    counts["part_arguments"] += 1
                    if resolver.resolve_part(raw_object_id, argument.part.name):
                        counts["resolved_part_arguments"] += 1
                    else:
                        unresolved_parts[f"{raw_object_id}/{argument.part.name}"] += 1

    groundable = counts["groundable_arguments"]
    part_count = counts["part_arguments"]
    return {
        "dataset_root": str(dataset_root),
        "sampling": {
            "strategy": "lexicographically_first",
            "episodes_per_task": episodes_per_task,
            "task_count": len(goals),
            "episode_count": len(sampled_episodes),
            "episodes": sampled_episodes,
        },
        "parse_status_counts": dict(sorted(status_counts.items())),
        "arguments": {
            **dict(counts),
            "resolution_fraction": (
                counts["resolved_arguments"] / groundable if groundable else 0.0
            ),
            "part_resolution_fraction": (
                counts["resolved_part_arguments"] / part_count if part_count else 0.0
            ),
        },
        "mesh_count_histogram": {
            str(mesh_count): count
            for mesh_count, count in sorted(mesh_count_histogram.items())
        },
        "unresolved_objects": dict(unresolved.most_common()),
        "unresolved_parts": dict(unresolved_parts.most_common()),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_mapping_coverage(
        args.dataset_root, episodes_per_task=args.episodes_per_task
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report["arguments"], indent=2))


if __name__ == "__main__":
    main()

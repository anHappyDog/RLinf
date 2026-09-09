#!/usr/bin/env python3
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

"""Measure grounded-control parser coverage over B1K annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rlinf.data.b1k_grounded import (
    DEFAULT_SKILL_SIGNATURE_REGISTRY,
    ParseStatus,
    parse_skill_annotation,
)


def _load_tasks(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "meta" / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing task metadata: {path}")
    tasks = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            task_index = task.get("task_index")
            if not isinstance(task_index, int):
                raise ValueError(f"Invalid task_index at {path}:{line_number}.")
            if task_index in tasks:
                raise ValueError(f"Duplicate task_index {task_index} in {path}.")
            tasks[task_index] = task
    return tasks


def _path_indices(path: Path) -> tuple[int, int]:
    try:
        return (
            int(path.parent.name.removeprefix("task-")),
            int(path.stem.removeprefix("episode_")),
        )
    except ValueError as error:
        raise ValueError(f"Unexpected annotation path layout: {path}") from error


def _record_skill_name(record: Any) -> str:
    if not isinstance(record, dict):
        return "<invalid_record>"
    descriptions = record.get("skill_description")
    if (
        not isinstance(descriptions, list)
        or len(descriptions) != 1
        or not isinstance(descriptions[0], str)
    ):
        return "<invalid_description>"
    return " ".join(descriptions[0].strip().lower().split())


def build_annotation_coverage(
    dataset_root: Path,
    *,
    max_issue_examples: int = 10,
) -> dict[str, Any]:
    """Parse every skill annotation and summarize explicit outcomes.

    Args:
        dataset_root: Root containing B1K ``meta`` and ``annotations``.
        max_issue_examples: Maximum raw records retained per issue code.

    Returns:
        A deterministic JSON-serializable coverage report.
    """
    if max_issue_examples <= 0:
        raise ValueError("max_issue_examples must be positive.")
    dataset_root = dataset_root.resolve()
    tasks = _load_tasks(dataset_root)
    annotation_paths = sorted(
        (dataset_root / "annotations").glob("task-*/episode_*.json")
    )
    if not annotation_paths:
        raise FileNotFoundError(
            f"No annotations found under {dataset_root / 'annotations'}."
        )

    status_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    skill_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skill_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issue_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_task_indices = set()
    record_count = 0

    for path in annotation_paths:
        task_index, episode_index = _path_indices(path)
        observed_task_indices.add(task_index)
        task = tasks.get(task_index)
        if task is None:
            raise ValueError(f"No task metadata exists for task index {task_index}.")
        annotation = json.loads(path.read_text(encoding="utf-8"))
        records = annotation.get("skill_annotation")
        if not isinstance(records, list):
            raise ValueError(f"skill_annotation is not a list in {path}.")

        for record in records:
            record_count += 1
            skill = _record_skill_name(record)
            result = parse_skill_annotation(
                record,
                goal=task["task"],
                episode_id=path.stem,
            )
            status_counts[result.status.value] += 1
            skill_status_counts[skill][result.status.value] += 1
            for issue in result.issues:
                issue_counts[issue.code] += 1
                skill_issue_counts[skill][issue.code] += 1
                examples = issue_examples[issue.code]
                if len(examples) < max_issue_examples:
                    examples.append(
                        {
                            "annotation_path": str(path.relative_to(dataset_root)),
                            "task_index": task_index,
                            "episode_index": episode_index,
                            "skill": skill,
                            "issue": issue.message,
                            "record": record,
                        }
                    )

    valid_count = status_counts[ParseStatus.VALID.value]
    return {
        "schema_version": 1,
        "dataset": {
            "root": str(dataset_root),
            "annotation_file_count": len(annotation_paths),
            "task_metadata_count": len(tasks),
            "observed_task_count": len(observed_task_indices),
            "skill_annotation_count": record_count,
        },
        "registry": {
            "signature_count": len(DEFAULT_SKILL_SIGNATURE_REGISTRY),
            "skills": [
                signature.skill
                for signature in DEFAULT_SKILL_SIGNATURE_REGISTRY.signatures()
            ],
        },
        "coverage": {
            "status_counts": dict(sorted(status_counts.items())),
            "valid_fraction": valid_count / record_count,
            "issue_counts": dict(sorted(issue_counts.items())),
        },
        "per_skill": [
            {
                "skill": skill,
                "status_counts": dict(sorted(skill_status_counts[skill].items())),
                "issue_counts": dict(sorted(skill_issue_counts[skill].items())),
            }
            for skill in sorted(skill_status_counts)
        ],
        "issue_examples": dict(sorted(issue_examples.items())),
    }


def write_coverage_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a coverage report atomically as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-issue-examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Build and write full B1K annotation coverage."""
    args = parse_args()
    report = build_annotation_coverage(
        args.dataset_root, max_issue_examples=args.max_issue_examples
    )
    write_coverage_report(report, args.output)
    print(
        json.dumps(
            {
                **report["dataset"],
                **report["coverage"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

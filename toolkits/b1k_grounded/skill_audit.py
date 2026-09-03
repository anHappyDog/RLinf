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

"""Audit canonical skills in BEHAVIOR-1K demonstration annotations.

The audit deliberately preserves annotation evidence instead of assigning
semantic roles. Its output is intended for the manual review that precedes a
frozen ``SkillSignatureRegistry``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_skill(description: str) -> str:
    return " ".join(description.strip().lower().split())


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_strings(item))
        return flattened
    raise ValueError(f"Expected a string or nested list of strings, got {value!r}.")


def _value_shape(value: Any) -> str:
    if not isinstance(value, list):
        return type(value).__name__
    child_shapes = [_value_shape(item) for item in value]
    compressed = []
    for shape in child_shapes:
        if compressed and compressed[-1][0] == shape:
            compressed[-1] = (shape, compressed[-1][1] + 1)
        else:
            compressed.append((shape, 1))
    content = ",".join(
        shape if count == 1 else f"{shape}*{count}" for shape, count in compressed
    )
    return f"list[{content}]"


def _frame_intervals(value: Any) -> list[tuple[int, int]]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        start, end = value
        if end <= start:
            raise ValueError(f"Frame interval must be increasing, got {value!r}.")
        return [(start, end)]
    if isinstance(value, list) and value:
        intervals = []
        for item in value:
            intervals.extend(_frame_intervals(item))
        return intervals
    raise ValueError(
        f"frame_duration must be an interval or nested intervals, got {value!r}."
    )


def _json_counter(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": json.loads(value), "count": count}
        for value, count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _plain_counter(counter: Counter[Any], value_name: str) -> list[dict[str, Any]]:
    return [
        {value_name: value, "count": count}
        for value, count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _select_evenly(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    return [
        items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)
    ]


@dataclasses.dataclass
class _IssueLog:
    max_examples: int
    counts: Counter[str] = dataclasses.field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)

    def add(self, issue: str, example: dict[str, Any]) -> None:
        self.counts[issue] += 1
        issue_examples = self.examples.setdefault(issue, [])
        if len(issue_examples) < self.max_examples:
            issue_examples.append(example)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": sum(self.counts.values()),
            "counts": dict(sorted(self.counts.items())),
            "examples": dict(sorted(self.examples.items())),
        }


@dataclasses.dataclass
class _SkillStats:
    skill: str
    max_examples: int
    max_issue_examples: int
    annotation_count: int = 0
    episode_paths: set[str] = dataclasses.field(default_factory=set)
    task_indices: set[int] = dataclasses.field(default_factory=set)
    raw_descriptions: Counter[str] = dataclasses.field(default_factory=Counter)
    skill_ids: Counter[str] = dataclasses.field(default_factory=Counter)
    skill_types: Counter[str] = dataclasses.field(default_factory=Counter)
    object_arities: Counter[int] = dataclasses.field(default_factory=Counter)
    manipulating_arities: Counter[int] = dataclasses.field(default_factory=Counter)
    object_shapes: Counter[str] = dataclasses.field(default_factory=Counter)
    manipulating_shapes: Counter[str] = dataclasses.field(default_factory=Counter)
    memory_prefixes: Counter[str] = dataclasses.field(default_factory=Counter)
    spatial_prefixes: Counter[str] = dataclasses.field(default_factory=Counter)
    frame_interval_counts: Counter[int] = dataclasses.field(default_factory=Counter)
    frame_duration_shapes: Counter[str] = dataclasses.field(default_factory=Counter)
    examples_by_task: dict[int, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    examples_by_variant: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    issues: _IssueLog = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.issues = _IssueLog(self.max_issue_examples)

    def add_example(
        self,
        task_index: int,
        example: dict[str, Any],
        variant: tuple[Any, ...],
    ) -> None:
        self.examples_by_task.setdefault(task_index, example)
        self.examples_by_variant.setdefault(_json_key(variant), example)

    def as_dict(self) -> dict[str, Any]:
        unique_examples = {
            (example["annotation_path"], example["skill_idx"]): example
            for example in (
                list(self.examples_by_task.values())
                + list(self.examples_by_variant.values())
            )
        }
        ordered_examples = sorted(
            unique_examples.values(),
            key=lambda item: (
                item["task_index"],
                item["annotation_path"],
                item["skill_idx"],
            ),
        )
        return {
            "skill": self.skill,
            "annotation_count": self.annotation_count,
            "episode_count": len(self.episode_paths),
            "task_count": len(self.task_indices),
            "task_indices": sorted(self.task_indices),
            "raw_descriptions": _plain_counter(self.raw_descriptions, "description"),
            "skill_ids": _json_counter(self.skill_ids),
            "skill_types": _json_counter(self.skill_types),
            "object_arity_counts": _plain_counter(self.object_arities, "arity"),
            "manipulating_object_arity_counts": _plain_counter(
                self.manipulating_arities, "arity"
            ),
            "object_structure_counts": _plain_counter(self.object_shapes, "shape"),
            "manipulating_object_structure_counts": _plain_counter(
                self.manipulating_shapes, "shape"
            ),
            "memory_prefix_values": _json_counter(self.memory_prefixes),
            "spatial_prefix_values": _json_counter(self.spatial_prefixes),
            "frame_interval_count_counts": _plain_counter(
                self.frame_interval_counts, "interval_count"
            ),
            "frame_duration_structure_counts": _plain_counter(
                self.frame_duration_shapes, "shape"
            ),
            "issues": self.issues.as_dict(),
            "examples": _select_evenly(ordered_examples, self.max_examples),
        }


def _load_task_metadata(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "meta" / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing BEHAVIOR task metadata: {path}")

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
        task_index = int(path.parent.name.removeprefix("task-"))
        episode_index = int(path.stem.removeprefix("episode_"))
    except ValueError as error:
        raise ValueError(f"Unexpected annotation path layout: {path}") from error
    return task_index, episode_index


def _issue_example(
    annotation_path: str,
    task_index: int,
    episode_index: int,
    skill_index: int | None,
    message: str,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    example = {
        "annotation_path": annotation_path,
        "task_index": task_index,
        "episode_index": episode_index,
        "skill_idx": skill_index,
        "message": message,
    }
    if record is not None:
        example["record"] = record
    return example


def _add_skill_issue(
    issue: str,
    message: str,
    *,
    stats: _SkillStats,
    global_issues: _IssueLog,
    example: dict[str, Any],
) -> None:
    enriched = {**example, "message": message}
    stats.issues.add(issue, enriched)
    global_issues.add(issue, {**enriched, "skill": stats.skill})


def _record_skill(
    record: Any,
    *,
    record_index: int,
    annotation_path: str,
    task_index: int,
    episode_index: int,
    task: dict[str, Any],
    skills: dict[str, _SkillStats],
    global_issues: _IssueLog,
    max_examples_per_skill: int,
    max_issue_examples: int,
) -> bool:
    if not isinstance(record, dict):
        global_issues.add(
            "invalid_skill_record",
            _issue_example(
                annotation_path,
                task_index,
                episode_index,
                record_index,
                "Skill annotation is not an object.",
            ),
        )
        return False

    descriptions = record.get("skill_description")
    if (
        not isinstance(descriptions, list)
        or len(descriptions) != 1
        or not isinstance(descriptions[0], str)
        or not descriptions[0].strip()
    ):
        global_issues.add(
            "invalid_skill_description",
            _issue_example(
                annotation_path,
                task_index,
                episode_index,
                record.get("skill_idx", record_index),
                "skill_description must contain exactly one non-empty string.",
                record,
            ),
        )
        return False

    raw_description = descriptions[0]
    skill = _normalize_skill(raw_description)
    stats = skills.setdefault(
        skill,
        _SkillStats(skill, max_examples_per_skill, max_issue_examples),
    )
    stats.annotation_count += 1
    stats.episode_paths.add(annotation_path)
    stats.task_indices.add(task_index)
    stats.raw_descriptions[raw_description] += 1
    stats.skill_ids[_json_key(record.get("skill_id"))] += 1
    stats.skill_types[_json_key(record.get("skill_type"))] += 1

    example = {
        "annotation_path": annotation_path,
        "task_index": task_index,
        "task_name": task.get("task_name"),
        "task": task.get("task"),
        "episode_index": episode_index,
        "skill_idx": record.get("skill_idx", record_index),
        "skill_id": record.get("skill_id"),
        "skill_description": descriptions,
        "object_id": record.get("object_id"),
        "manipulating_object_id": record.get("manipulating_object_id"),
        "spatial_prefix": record.get("spatial_prefix"),
        "memory_prefix": record.get("memory_prefix"),
        "frame_duration": record.get("frame_duration"),
        "skill_type": record.get("skill_type"),
    }

    object_groups = record.get("object_id")
    object_group: Any = None
    object_ids: list[str] = []
    if not isinstance(object_groups, list) or len(object_groups) != 1:
        _add_skill_issue(
            "invalid_object_group",
            "object_id must contain exactly one object group for a skill.",
            stats=stats,
            global_issues=global_issues,
            example=example,
        )
    else:
        object_group = object_groups[0]
        try:
            object_ids = _flatten_strings(object_group)
        except ValueError as error:
            _add_skill_issue(
                "invalid_object_id",
                str(error),
                stats=stats,
                global_issues=global_issues,
                example=example,
            )
        else:
            stats.object_arities[len(object_ids)] += 1
            stats.object_shapes[_value_shape(object_group)] += 1
            if len(object_ids) != len(set(object_ids)):
                _add_skill_issue(
                    "duplicate_flattened_object_id",
                    "Flattened object_id contains a duplicate value.",
                    stats=stats,
                    global_issues=global_issues,
                    example=example,
                )

    manipulating_group = record.get("manipulating_object_id")
    manipulating_ids: list[str] = []
    if not isinstance(manipulating_group, list):
        _add_skill_issue(
            "invalid_manipulating_object_group",
            "manipulating_object_id must be a list.",
            stats=stats,
            global_issues=global_issues,
            example=example,
        )
    else:
        try:
            manipulating_ids = _flatten_strings(manipulating_group)
        except ValueError as error:
            _add_skill_issue(
                "invalid_manipulating_object_id",
                str(error),
                stats=stats,
                global_issues=global_issues,
                example=example,
            )
        else:
            stats.manipulating_arities[len(manipulating_ids)] += 1
            stats.manipulating_shapes[_value_shape(manipulating_group)] += 1
            missing = sorted(set(manipulating_ids).difference(object_ids))
            if missing:
                _add_skill_issue(
                    "manipulating_object_not_in_object_id",
                    f"Manipulating objects are absent from object_id: {missing}.",
                    stats=stats,
                    global_issues=global_issues,
                    example=example,
                )

    memory_prefix = record.get("memory_prefix")
    if isinstance(memory_prefix, list):
        stats.memory_prefixes[_json_key(memory_prefix)] += 1
    else:
        _add_skill_issue(
            "invalid_memory_prefix",
            "memory_prefix must be a list.",
            stats=stats,
            global_issues=global_issues,
            example=example,
        )

    spatial_prefix = record.get("spatial_prefix")
    if isinstance(spatial_prefix, list):
        stats.spatial_prefixes[_json_key(spatial_prefix)] += 1
    else:
        _add_skill_issue(
            "invalid_spatial_prefix",
            "spatial_prefix must be a list.",
            stats=stats,
            global_issues=global_issues,
            example=example,
        )

    skill_id = record.get("skill_id")
    if not isinstance(skill_id, list) or len(skill_id) != 1:
        _add_skill_issue(
            "invalid_skill_id",
            "skill_id must contain exactly one value for a skill.",
            stats=stats,
            global_issues=global_issues,
            example=example,
        )

    skill_type = record.get("skill_type")
    if (
        not isinstance(skill_type, list)
        or len(skill_type) != 1
        or not isinstance(skill_type[0], str)
    ):
        _add_skill_issue(
            "invalid_skill_type",
            "skill_type must contain exactly one string.",
            stats=stats,
            global_issues=global_issues,
            example=example,
        )

    frame_duration = record.get("frame_duration")
    try:
        intervals = _frame_intervals(frame_duration)
    except ValueError as error:
        _add_skill_issue(
            "invalid_frame_duration",
            str(error),
            stats=stats,
            global_issues=global_issues,
            example=example,
        )
    else:
        stats.frame_interval_counts[len(intervals)] += 1
        stats.frame_duration_shapes[_value_shape(frame_duration)] += 1

    variant = (
        _value_shape(object_group),
        _value_shape(manipulating_group),
        memory_prefix,
        spatial_prefix,
        skill_type,
    )
    stats.add_example(task_index, example, variant)
    return True


def build_skill_audit(
    dataset_root: Path,
    *,
    max_examples_per_skill: int = 20,
    max_issue_examples: int = 5,
) -> dict[str, Any]:
    """Build an evidence-preserving audit of B1K skill annotations.

    Args:
        dataset_root: Root containing ``annotations/`` and ``meta/tasks.jsonl``.
        max_examples_per_skill: Maximum representative records saved per skill.
        max_issue_examples: Maximum records saved for each issue category.

    Returns:
        A deterministic JSON-serializable audit report.

    Raises:
        FileNotFoundError: If required dataset metadata or annotations are absent.
        ValueError: If limits or task metadata are invalid.
    """
    dataset_root = dataset_root.resolve()
    if max_examples_per_skill <= 0:
        raise ValueError("max_examples_per_skill must be positive.")
    if max_issue_examples <= 0:
        raise ValueError("max_issue_examples must be positive.")

    tasks = _load_task_metadata(dataset_root)
    annotation_paths = sorted(
        (dataset_root / "annotations").glob("task-*/episode_*.json")
    )
    if not annotation_paths:
        raise FileNotFoundError(
            f"No BEHAVIOR annotations found under {dataset_root / 'annotations'}."
        )

    skills: dict[str, _SkillStats] = {}
    issues = _IssueLog(max_issue_examples)
    task_indices = set()
    skill_annotation_count = 0
    accepted_skill_annotation_count = 0
    primitive_annotation_count = 0
    primitive_descriptions: Counter[str] = Counter()
    primitive_description_arities: Counter[int] = Counter()

    for path in annotation_paths:
        task_index, episode_index = _path_indices(path)
        task_indices.add(task_index)
        relative_path = str(path.relative_to(dataset_root))
        task = tasks.get(task_index, {})
        if not task:
            issues.add(
                "missing_task_metadata",
                _issue_example(
                    relative_path,
                    task_index,
                    episode_index,
                    None,
                    f"No task metadata exists for task index {task_index}.",
                ),
            )

        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            issues.add(
                "unreadable_annotation",
                _issue_example(
                    relative_path,
                    task_index,
                    episode_index,
                    None,
                    str(error),
                ),
            )
            continue
        if not isinstance(annotation, dict):
            issues.add(
                "invalid_annotation",
                _issue_example(
                    relative_path,
                    task_index,
                    episode_index,
                    None,
                    "Annotation root is not an object.",
                ),
            )
            continue

        records = annotation.get("skill_annotation")
        if not isinstance(records, list):
            issues.add(
                "invalid_skill_annotation_list",
                _issue_example(
                    relative_path,
                    task_index,
                    episode_index,
                    None,
                    "skill_annotation is not a list.",
                ),
            )
            continue
        skill_annotation_count += len(records)
        for record_index, record in enumerate(records):
            accepted_skill_annotation_count += _record_skill(
                record,
                record_index=record_index,
                annotation_path=relative_path,
                task_index=task_index,
                episode_index=episode_index,
                task=task,
                skills=skills,
                global_issues=issues,
                max_examples_per_skill=max_examples_per_skill,
                max_issue_examples=max_issue_examples,
            )

        primitives = annotation.get("primitive_annotation")
        if not isinstance(primitives, list):
            issues.add(
                "invalid_primitive_annotation_list",
                _issue_example(
                    relative_path,
                    task_index,
                    episode_index,
                    None,
                    "primitive_annotation is not a list.",
                ),
            )
            continue
        primitive_annotation_count += len(primitives)
        for primitive in primitives:
            descriptions = (
                primitive.get("primitive_description")
                if isinstance(primitive, dict)
                else None
            )
            if not isinstance(descriptions, list):
                primitive_description_arities[-1] += 1
                continue
            primitive_description_arities[len(descriptions)] += 1
            for description in descriptions:
                if isinstance(description, str) and description.strip():
                    primitive_descriptions[_normalize_skill(description)] += 1

    return {
        "schema_version": 1,
        "source": "BEHAVIOR-1K skill_annotation",
        "dataset": {
            "root": str(dataset_root),
            "annotation_file_count": len(annotation_paths),
            "task_metadata_count": len(tasks),
            "observed_task_count": len(task_indices),
            "observed_task_indices": sorted(task_indices),
            "skill_annotation_count": skill_annotation_count,
            "accepted_skill_annotation_count": accepted_skill_annotation_count,
            "canonical_skill_count": len(skills),
            "primitive_annotation_count": primitive_annotation_count,
        },
        "primitive_summary": {
            "canonical_description_count": len(primitive_descriptions),
            "description_counts": _plain_counter(primitive_descriptions, "description"),
            "description_arity_counts": _plain_counter(
                primitive_description_arities, "arity"
            ),
        },
        "issues": issues.as_dict(),
        "skills": [skills[name].as_dict() for name in sorted(skills)],
    }


def write_skill_audit(report: dict[str, Any], output_path: Path) -> None:
    """Write an audit report atomically as formatted JSON."""
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
    parser.add_argument("--max-examples-per-skill", type=int, default=20)
    parser.add_argument("--max-issue-examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Build and write a BEHAVIOR-1K skill audit."""
    args = parse_args()
    report = build_skill_audit(
        args.dataset_root,
        max_examples_per_skill=args.max_examples_per_skill,
        max_issue_examples=args.max_issue_examples,
    )
    write_skill_audit(report, args.output)
    print(
        json.dumps(
            {
                **report["dataset"],
                "issue_count": report["issues"]["total"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

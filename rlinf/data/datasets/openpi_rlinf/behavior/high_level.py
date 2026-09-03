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

"""High-level text supervision helpers for OpenPI-RLinf on BEHAVIOR-1K."""

from __future__ import annotations

import dataclasses
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclasses.dataclass(frozen=True)
class R0ManifestEntry:
    """One fixed observation-to-subtask example used by the R0 overfit test."""

    task_index: int
    task_name: str
    task: str
    episode_index: int
    frame_index: int
    primitive_index: int
    subtask: str


@dataclasses.dataclass(frozen=True)
class R1ManifestEntry:
    """One episode-split observation-to-subtask example for R1 training."""

    split: str
    task_index: int
    task_name: str
    task: str
    episode_index: int
    frame_index: int
    primitive_index: int
    subtask: str
    target_source: str


@dataclasses.dataclass(frozen=True)
class TokenizedSubtask:
    """Padded PaliGemma tokens and masks for subtask text modeling."""

    tokens: np.ndarray
    input_mask: np.ndarray
    ar_mask: np.ndarray
    loss_mask: np.ndarray


@dataclasses.dataclass(frozen=True)
class PrimitivePromptInterval:
    """Canonical oracle subtask active on a half-open frame interval."""

    start_frame: int
    end_frame: int
    primitive_index: int
    subtask: str
    target_source: str


def build_primitive_prompt_intervals(
    annotation: dict[str, Any],
) -> list[PrimitivePromptInterval]:
    """Build canonical per-frame oracle prompts from one B1K annotation.

    Unlike the R1 manipulation-anchor sampler, action SFT uses the union of all
    skills referenced by a primitive, including navigation. Native primitive
    duration envelopes can overlap when a primitive has disjoint skills, so the
    referenced skill intervals are the authoritative alignment. Frames covered
    by multiple residual intervals are treated as ambiguous and excluded by
    :func:`resolve_primitive_prompt`.
    """
    skills = {
        int(skill.get("skill_idx", index)): skill
        for index, skill in enumerate(annotation.get("skill_annotation") or [])
    }
    intervals = []
    for primitive in annotation.get("primitive_annotation") or []:
        subtask, target_source = _canonicalize_r1_target(primitive, skills)
        primitive_index = int(primitive["primitive_idx"])
        for start_frame, end_frame in _primitive_action_intervals(primitive, skills):
            intervals.append(
                PrimitivePromptInterval(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    primitive_index=primitive_index,
                    subtask=subtask,
                    target_source=target_source,
                )
            )

    intervals.sort(key=lambda item: (item.start_frame, item.end_frame))
    return intervals


def resolve_primitive_prompt(
    intervals: list[PrimitivePromptInterval],
    frame_index: int,
    *,
    action_horizon: int = 1,
) -> PrimitivePromptInterval | None:
    """Resolve a frame when its entire action chunk stays in one primitive."""
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    active = [
        interval
        for interval in intervals
        if interval.start_frame <= frame_index < interval.end_frame
    ]
    if len(active) != 1:
        return None
    interval = active[0]
    return (
        interval if frame_index + action_horizon <= interval.end_frame else None
    )


def canonicalize_object_id(object_id: str) -> str:
    """Convert a BEHAVIOR instance identifier to a readable category name.

    Examples include ``radio_89`` and ``coffee_table_koagbh_0``. The final
    numeric instance suffix is removed, followed by an optional six-letter
    object-model hash.
    """
    parts = [part for part in object_id.strip().lower().split("_") if part]
    numeric_suffixes = []
    while parts and parts[-1].isdigit():
        numeric_suffixes.append(int(parts.pop()))
    if (
        len(parts) >= 2
        and numeric_suffixes
        and numeric_suffixes[0] <= 10
        and re.fullmatch(r"[a-z]{6}", parts[-1])
    ):
        parts.pop()
    if not parts:
        raise ValueError(f"Cannot canonicalize empty object id {object_id!r}.")
    return " ".join(parts)


def canonicalize_primitive(primitive: dict[str, Any]) -> str:
    """Render a structured B1K primitive annotation as canonical text."""
    descriptions = primitive.get("primitive_description")
    if not descriptions:
        raise ValueError(
            f"Primitive {primitive.get('primitive_idx')} has no description."
        )
    object_groups = primitive.get("object_id") or []
    manipulating_groups = primitive.get("manipulating_object_id") or []
    if len(descriptions) != len(object_groups):
        raise ValueError(
            f"Primitive {primitive.get('primitive_idx')} has mismatched descriptions "
            "and object groups."
        )
    if manipulating_groups and len(descriptions) != len(manipulating_groups):
        raise ValueError(
            f"Primitive {primitive.get('primitive_idx')} has mismatched descriptions "
            "and manipulating-object groups."
        )

    steps = []
    for index, description in enumerate(descriptions):
        manipulating_group = manipulating_groups[index] if manipulating_groups else []
        step = canonicalize_action(
            str(description), object_groups[index], manipulating_group
        )
        if not steps or step != steps[-1]:
            steps.append(step)
    return " then ".join(steps)


def canonicalize_action(
    description: str,
    object_group: Any,
    manipulating_group: Any,
) -> str:
    """Render one structured BEHAVIOR skill or primitive action as text."""
    action = " ".join(description.strip().lower().split())
    raw_objects = _flatten_object_ids(object_group)
    raw_manipulating = _flatten_object_ids(manipulating_group)
    raw_manipulating_set = set(raw_manipulating)
    objects = _canonical_names(raw_objects)
    manipulating = _canonical_names(raw_manipulating)
    other_objects = _canonical_names(
        [item for item in raw_objects if item not in raw_manipulating_set]
    )

    subject = _join_names(manipulating or objects[:1])
    primary_other = other_objects[0] if other_objects else ""
    final_other = other_objects[-1] if other_objects else ""
    target = final_other or (objects[-1] if objects else "")

    if action == "move to" and target:
        return f"move to {target}"
    if action == "pick up from" and subject and target:
        return f"pick up {subject} from {target}"
    if action in {"place in", "place on", "place under"} and subject and target:
        return f"place {subject} {action.removeprefix('place ')} {target}"
    if action in {"place in next to", "place on next to"} and subject:
        if len(other_objects) >= 2:
            preposition = action.removeprefix("place ").removesuffix(" next to")
            return (
                f"place {subject} {preposition} {other_objects[-2]} "
                f"next to {other_objects[-1]}"
            )
        if target:
            return f"place {subject} next to {target}"
    if action == "push to" and subject and target:
        return f"push {subject} to {target}"
    if action == "turn to" and subject and target:
        return f"turn {subject} toward {target}"
    if action in {"open door", "close door", "open drawer", "close drawer"}:
        if target:
            verb, part = action.split()
            return f"{verb} {target} {part}"
    if action in {"open lid", "close lid"} and target:
        verb, part = action.split()
        return f"{verb} {target} {part}"
    if (
        action
        in {
            "press",
            "turn on switch",
            "turn off switch",
            "tip over",
            "hold",
            "release",
        }
        and subject
    ):
        verb = action.removesuffix(" switch")
        return f"{verb} {subject}"
    if action == "pour" and subject:
        return f"pour {subject}" + (f" into {target}" if target else "")
    if action == "chop" and (primary_other or subject):
        return f"chop {primary_other or subject}"
    if action == "spray" and subject:
        return f"spray {target} with {subject}" if target else f"spray {subject}"
    if action in {"sweep surface", "sweep off", "wipe hard"} and subject:
        verb = {"sweep surface": "sweep", "sweep off": "sweep", "wipe hard": "wipe"}[
            action
        ]
        return f"{verb} {target} with {subject}" if target else f"{verb} {subject}"
    if action == "insert" and subject and target:
        return f"insert {subject} into {target}"
    if action in {"attach", "hang"} and subject:
        preposition = "to" if action == "attach" else "on"
        return (
            f"{action} {subject} {preposition} {target}"
            if target
            else f"{action} {subject}"
        )
    if action == "ignite" and subject:
        return f"ignite {target} with {subject}" if target else f"ignite {subject}"
    if action == "hand over" and subject:
        return f"hand over {subject}"
    if action in {"pull tray", "push tray"} and target:
        verb = action.split()[0]
        return f"{verb} {target} tray"
    raise ValueError(
        f"Unsupported action annotation: action={action!r}, objects={objects!r}, "
        f"manipulating={manipulating!r}."
    )


def build_r1_manifest(
    dataset_root: str | Path,
    *,
    samples_per_primitive: int = 2,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    task_indices: Iterable[int] | None = None,
) -> tuple[list[R1ManifestEntry], dict[str, Any]]:
    """Build deterministic episode-split R1 manifests from B1K annotations."""
    if samples_per_primitive <= 0:
        raise ValueError("samples_per_primitive must be positive.")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "Validation and test fractions must be non-negative and sum below one."
        )

    dataset_root = Path(dataset_root)
    tasks = {
        int(task["task_index"]): task
        for task in _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    }
    selected_tasks = set(task_indices) if task_indices is not None else set(tasks)
    missing_tasks = selected_tasks.difference(tasks)
    if missing_tasks:
        raise ValueError(f"Unknown task indices: {sorted(missing_tasks)}")

    annotation_paths_by_task = {
        task_index: sorted(
            (dataset_root / "annotations" / f"task-{task_index:04d}").glob(
                "episode_*.json"
            )
        )
        for task_index in sorted(selected_tasks)
    }
    episode_splits = _split_episode_paths(
        annotation_paths_by_task,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    entries = []
    skipped_reasons: Counter[str] = Counter()
    skipped_examples = []
    split_episode_counts: Counter[str] = Counter(episode_splits.values())
    target_sources: Counter[str] = Counter()
    for task_index, annotation_paths in annotation_paths_by_task.items():
        task = tasks[task_index]
        for annotation_path in annotation_paths:
            episode_index = _episode_index(annotation_path)
            if not _has_rgb_videos(dataset_root, task_index, episode_index):
                skipped_reasons["missing_rgb_video"] += 1
                continue
            annotation = json.loads(annotation_path.read_text())
            skills = {
                int(skill.get("skill_idx", index)): skill
                for index, skill in enumerate(annotation.get("skill_annotation") or [])
            }
            for primitive in annotation.get("primitive_annotation") or []:
                try:
                    subtask, target_source = _canonicalize_r1_target(primitive, skills)
                    intervals = _primitive_sampling_intervals(primitive, skills)
                    frame_indices = _sample_interval_frames(
                        intervals, samples_per_primitive
                    )
                except (KeyError, TypeError, ValueError) as error:
                    reason = type(error).__name__
                    skipped_reasons[reason] += 1
                    if len(skipped_examples) < 20:
                        skipped_examples.append(
                            {
                                "annotation": str(annotation_path),
                                "primitive_index": primitive.get("primitive_idx"),
                                "error": str(error),
                            }
                        )
                    continue

                target_sources[target_source] += len(frame_indices)
                for frame_index in frame_indices:
                    entries.append(
                        R1ManifestEntry(
                            split=episode_splits[episode_index],
                            task_index=task_index,
                            task_name=str(task["task_name"]),
                            task=str(task["task"]),
                            episode_index=episode_index,
                            frame_index=frame_index,
                            primitive_index=int(primitive["primitive_idx"]),
                            subtask=subtask,
                            target_source=target_source,
                        )
                    )

    entries.sort(
        key=lambda item: (
            item.task_index,
            item.episode_index,
            item.primitive_index,
            item.frame_index,
        )
    )
    report = {
        "schema_version": 1,
        "seed": seed,
        "samples_per_primitive": samples_per_primitive,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "annotation_episode_count": sum(
            len(paths) for paths in annotation_paths_by_task.values()
        ),
        "split_episode_counts": dict(sorted(split_episode_counts.items())),
        "entry_counts": dict(sorted(Counter(item.split for item in entries).items())),
        "target_source_counts": dict(sorted(target_sources.items())),
        "task_entry_counts": dict(
            sorted(Counter(str(item.task_index) for item in entries).items())
        ),
        "unique_subtask_count": len({item.subtask for item in entries}),
        "skipped_counts": dict(sorted(skipped_reasons.items())),
        "skipped_examples": skipped_examples,
    }
    return entries, report


def write_r1_manifests(
    entries: Iterable[R1ManifestEntry],
    output_dir: str | Path,
    report: dict[str, Any],
) -> None:
    """Write split R1 JSONL manifests and their build report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_by_split: dict[str, list[R1ManifestEntry]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for entry in entries:
        entries_by_split[entry.split].append(entry)
    for split, split_entries in entries_by_split.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as output_file:
            for entry in split_entries:
                output_file.write(json.dumps(dataclasses.asdict(entry)) + "\n")
    (output_dir / "stats.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def read_r1_manifest(manifest_path: str | Path) -> list[R1ManifestEntry]:
    """Read one R1 JSONL split manifest."""
    return [R1ManifestEntry(**item) for item in _load_jsonl(Path(manifest_path))]


def _flatten_object_ids(value: Any) -> list[str]:
    if value is None or value == []:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_object_ids(item))
        return flattened
    raise TypeError(f"Object ids must be strings or lists, got {type(value)!r}.")


def _canonical_names(raw_ids: Iterable[str]) -> list[str]:
    names = []
    for raw_id in raw_ids:
        name = canonicalize_object_id(raw_id)
        if name not in names:
            names.append(name)
    return names


def _join_names(names: list[str]) -> str:
    return " and ".join(names)


def _split_episode_paths(
    annotation_paths_by_task: dict[int, list[Path]],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[int, str]:
    episode_splits = {}
    for task_index, paths in annotation_paths_by_task.items():
        episode_indices = [_episode_index(path) for path in paths]
        random.Random(seed + task_index * 1_000_003).shuffle(episode_indices)
        val_count = round(len(episode_indices) * val_fraction)
        test_count = round(len(episode_indices) * test_fraction)
        train_count = len(episode_indices) - val_count - test_count
        for position, episode_index in enumerate(episode_indices):
            if position < train_count:
                split = "train"
            elif position < train_count + val_count:
                split = "val"
            else:
                split = "test"
            episode_splits[episode_index] = split
    return episode_splits


def _episode_index(annotation_path: Path) -> int:
    return int(annotation_path.stem.removeprefix("episode_"))


def _has_rgb_videos(dataset_root: Path, task_index: int, episode_index: int) -> bool:
    task_dir = dataset_root / "videos" / f"task-{task_index:04d}"
    return all(
        (task_dir / camera / f"episode_{episode_index:08d}.mp4").is_file()
        for camera in (
            "observation.images.rgb.head",
            "observation.images.rgb.left_wrist",
            "observation.images.rgb.right_wrist",
        )
    )


def _canonicalize_r1_target(
    primitive: dict[str, Any], skills: dict[int, dict[str, Any]]
) -> tuple[str, str]:
    if primitive.get("primitive_description"):
        return canonicalize_primitive(primitive), "primitive"

    fallback_skills = [
        skills[int(index)] for index in primitive.get("skill_idxes") or []
    ]
    if not fallback_skills:
        raise ValueError(
            f"Primitive {primitive.get('primitive_idx')} has no description or skills."
        )
    non_navigation = [
        skill
        for skill in fallback_skills
        if "navigation" not in (skill.get("skill_type") or [])
    ]
    selected_skills = non_navigation or fallback_skills
    steps = []
    for skill in selected_skills:
        step = canonicalize_primitive(
            {
                "primitive_idx": primitive.get("primitive_idx"),
                "primitive_description": skill.get("skill_description"),
                "object_id": skill.get("object_id"),
                "manipulating_object_id": skill.get("manipulating_object_id"),
            }
        )
        if not steps or step != steps[-1]:
            steps.append(step)
    return " then ".join(steps), "skill_fallback"


def _primitive_sampling_intervals(
    primitive: dict[str, Any], skills: dict[int, dict[str, Any]]
) -> list[tuple[int, int]]:
    primitive_descriptions = {
        str(description).strip().lower()
        for description in primitive.get("primitive_description") or []
    }
    primitive_skills = [
        skills[int(index)] for index in primitive.get("skill_idxes") or []
    ]
    matching_skills = [
        skill
        for skill in primitive_skills
        if primitive_descriptions.intersection(
            str(description).strip().lower()
            for description in skill.get("skill_description") or []
        )
    ]
    if not matching_skills:
        non_navigation = [
            skill
            for skill in primitive_skills
            if "navigation" not in (skill.get("skill_type") or [])
        ]
        matching_skills = non_navigation or primitive_skills

    intervals = []
    for skill in matching_skills:
        intervals.extend(_frame_intervals(skill.get("frame_duration")))
    if not intervals:
        intervals = _frame_intervals(primitive.get("frame_duration"))
    return sorted(intervals)


def _primitive_action_intervals(
    primitive: dict[str, Any], skills: dict[int, dict[str, Any]]
) -> list[tuple[int, int]]:
    """Return merged frame ranges for every skill owned by a primitive."""
    intervals = []
    for skill_index in primitive.get("skill_idxes") or []:
        frame_duration = skills[int(skill_index)]["frame_duration"]
        intervals.extend(_nonempty_frame_intervals(frame_duration))
    if not intervals:
        intervals = _nonempty_frame_intervals(primitive.get("frame_duration"))
    if not intervals:
        raise ValueError(
            f"Primitive {primitive.get('primitive_idx')} has no positive frame range."
        )

    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _nonempty_frame_intervals(frame_duration: Any) -> list[tuple[int, int]]:
    """Parse frame ranges while ignoring zero-length or reversed annotations."""
    if (
        isinstance(frame_duration, list)
        and len(frame_duration) == 2
        and all(isinstance(value, int) for value in frame_duration)
    ):
        start, end = frame_duration
        return [(start, end)] if end > start else []
    if isinstance(frame_duration, list) and frame_duration:
        intervals = []
        for item in frame_duration:
            intervals.extend(_nonempty_frame_intervals(item))
        return intervals
    raise ValueError(f"Invalid frame duration {frame_duration!r}.")


def _frame_intervals(frame_duration: Any) -> list[tuple[int, int]]:
    if (
        isinstance(frame_duration, list)
        and len(frame_duration) == 2
        and all(isinstance(value, int) for value in frame_duration)
    ):
        start, end = frame_duration
        if end <= start:
            raise ValueError(f"Empty frame interval {frame_duration!r}.")
        return [(start, end)]
    if isinstance(frame_duration, list) and frame_duration:
        intervals = []
        for item in frame_duration:
            intervals.extend(_frame_intervals(item))
        return intervals
    raise ValueError(f"Invalid frame duration {frame_duration!r}.")


def _sample_interval_frames(
    intervals: list[tuple[int, int]], samples_per_primitive: int
) -> list[int]:
    total_duration = sum(end - start for start, end in intervals)
    if total_duration <= 0:
        raise ValueError("Cannot sample an empty set of frame intervals.")

    frame_indices = []
    for sample_index in range(samples_per_primitive):
        offset = int((sample_index + 1) * total_duration / (samples_per_primitive + 1))
        for start, end in intervals:
            duration = end - start
            if offset < duration:
                frame_indices.append(start + offset)
                break
            offset -= duration
        else:
            frame_indices.append(intervals[-1][1] - 1)
    return frame_indices


def build_r0_manifest(
    dataset_root: str | Path,
    *,
    task_index: int = 0,
    episode_index: int | None = None,
    samples_per_primitive: int = 4,
) -> list[R0ManifestEntry]:
    """Build a deterministic microset from one B1K annotated episode."""
    if samples_per_primitive <= 0:
        raise ValueError("samples_per_primitive must be positive.")

    dataset_root = Path(dataset_root)
    tasks = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    task_metadata = next(
        (task for task in tasks if int(task["task_index"]) == task_index), None
    )
    if task_metadata is None:
        raise ValueError(f"Task index {task_index} is absent from meta/tasks.jsonl.")

    annotation_dir = dataset_root / "annotations" / f"task-{task_index:04d}"
    if episode_index is None:
        annotation_paths = sorted(annotation_dir.glob("episode_*.json"))
        if not annotation_paths:
            raise FileNotFoundError(
                f"No episode annotations found in {annotation_dir}."
            )
        annotation_path = annotation_paths[0]
        episode_index = int(annotation_path.stem.removeprefix("episode_"))
    else:
        annotation_path = annotation_dir / f"episode_{episode_index:08d}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"R0 annotation does not exist: {annotation_path}")

    annotation = json.loads(annotation_path.read_text())
    primitives = annotation.get("primitive_annotation") or []
    if not primitives:
        raise ValueError(f"No primitive annotations found in {annotation_path}.")

    entries = []
    for primitive in sorted(primitives, key=lambda item: item["primitive_idx"]):
        frame_duration = primitive.get("frame_duration")
        if not frame_duration or len(frame_duration) != 2:
            raise ValueError(
                f"Primitive {primitive.get('primitive_idx')} has invalid frame_duration."
            )
        start, end = map(int, frame_duration)
        if end <= start:
            raise ValueError(
                f"Primitive {primitive.get('primitive_idx')} has an empty frame range."
            )
        subtask = canonicalize_primitive(primitive)
        duration = end - start
        for sample_index in range(samples_per_primitive):
            fraction = (sample_index + 1) / (samples_per_primitive + 1)
            frame_index = start + min(duration - 1, int(fraction * duration))
            entries.append(
                R0ManifestEntry(
                    task_index=task_index,
                    task_name=str(task_metadata["task_name"]),
                    task=str(task_metadata["task"]),
                    episode_index=episode_index,
                    frame_index=frame_index,
                    primitive_index=int(primitive["primitive_idx"]),
                    subtask=subtask,
                )
            )
    return entries


def write_r0_manifest(
    entries: Iterable[R0ManifestEntry], output_path: str | Path
) -> None:
    """Write R0 manifest entries as JSON Lines."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for entry in entries:
            output_file.write(json.dumps(dataclasses.asdict(entry)) + "\n")


def read_r0_manifest(manifest_path: str | Path) -> list[R0ManifestEntry]:
    """Read an R0 JSON Lines manifest."""
    return [R0ManifestEntry(**item) for item in _load_jsonl(Path(manifest_path))]


class PaligemmaSubtaskTokenizer:
    """Tokenizer for ``Task [... State ...] -> Subtask`` text training."""

    def __init__(self, max_len: int = 200):
        if max_len <= 1:
            raise ValueError("max_len must be greater than one.")
        # Reuse OpenPI's downloader and exact SentencePiece model. The upstream
        # wrapper intentionally exposes only action-prompt tokenization, so this
        # adapter uses its underlying processor for a new Subtask response.
        from openpi.models.tokenizer import PaligemmaTokenizer

        upstream_tokenizer = PaligemmaTokenizer(max_len=max_len)
        self._tokenizer = upstream_tokenizer._tokenizer  # noqa: SLF001
        self.max_len = max_len

    @property
    def eos_token_id(self) -> int:
        """Return the PaliGemma EOS token id."""
        return int(self._tokenizer.eos_id())

    def tokenize(
        self,
        task: str,
        *,
        state: np.ndarray | None = None,
        subtask: str | None = None,
    ) -> TokenizedSubtask:
        """Tokenize a high-level prefix and optional supervised response."""
        cleaned_task = task.strip().replace("_", " ").replace("\n", " ")
        if state is None:
            prefix = f"Task: {cleaned_task};\nSubtask: "
        else:
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_text = " ".join(map(str, discretized_state))
            prefix = f"Task: {cleaned_task}, State: {state_text};\nSubtask: "

        prefix_tokens = self._tokenizer.encode(prefix, add_bos=True)
        response_tokens = (
            self._tokenizer.encode(subtask.strip(), add_eos=True)
            if subtask is not None
            else []
        )
        tokens = prefix_tokens + response_tokens
        if len(tokens) > self.max_len:
            raise ValueError(
                f"High-level sequence has {len(tokens)} tokens, exceeding max_len="
                f"{self.max_len}; target truncation is not allowed."
            )

        valid_length = len(tokens)
        padding_length = self.max_len - valid_length
        return TokenizedSubtask(
            tokens=np.asarray(tokens + [0] * padding_length, dtype=np.int64),
            input_mask=np.asarray(
                [True] * valid_length + [False] * padding_length, dtype=np.bool_
            ),
            ar_mask=np.asarray(
                [False] * len(prefix_tokens)
                + [True] * len(response_tokens)
                + [False] * padding_length,
                dtype=np.bool_,
            ),
            loss_mask=np.asarray(
                [False] * len(prefix_tokens)
                + [True] * len(response_tokens)
                + [False] * padding_length,
                dtype=np.bool_,
            ),
        )

    def decode(self, tokens: Iterable[int]) -> str:
        """Decode generated token ids into normalized text."""
        return self._tokenizer.decode([int(token) for token in tokens]).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]

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

"""Build a deterministic grounded-control SFT sidecar from B1K demos."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.data.b1k_grounded import (
    CameraID,
    EntityResolver,
    GroundedControlSpec,
    GroundingConfig,
    ParseStatus,
    ground_control_spec,
    parse_skill_annotation,
    select_primary_grounding,
)
from toolkits.b1k_grounded.recorded_segmentation import (
    EpisodeSegmentationMetadata,
    episode_metadata_path,
    episode_video_path,
    read_segmentation_video_frames,
)

FORMAT_VERSION = "b1k_grounded_sft_sidecar_v0.2"
CAMERAS = (CameraID.HEAD, CameraID.LEFT_WRIST, CameraID.RIGHT_WRIST)


@dataclasses.dataclass(frozen=True)
class PilotBuildConfig:
    """Parameters that determine pilot sampling and output tensors."""

    episodes_per_task: int = 1
    sample_fractions: tuple[float, ...] = (0.5,)
    frame_stride: int | None = None
    frame_stride_skills: tuple[str, ...] = ()
    selection_mode: str = "all"
    action_horizon: int = 32
    min_visible_pixels: int = 4
    min_component_pixels: int = 64
    min_component_fraction_of_largest: float = 0.005

    def __post_init__(self) -> None:
        if self.episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive.")
        if not self.sample_fractions:
            raise ValueError("sample_fractions must not be empty.")
        if not all(0.0 <= value <= 1.0 for value in self.sample_fractions):
            raise ValueError("sample_fractions must be in [0, 1].")
        if self.frame_stride is not None and self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive when provided.")
        if self.frame_stride_skills and self.frame_stride is None:
            raise ValueError("frame_stride_skills requires frame_stride.")
        if any(not skill.strip() for skill in self.frame_stride_skills):
            raise ValueError("frame_stride_skills must contain non-empty names.")
        if self.frame_stride is not None and self.selection_mode != "all":
            raise ValueError("frame_stride requires selection_mode='all'.")
        if self.selection_mode not in {"all", "best_visibility"}:
            raise ValueError("selection_mode must be 'all' or 'best_visibility'.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")

    def grounding_config(self) -> GroundingConfig:
        """Return the geometry thresholds represented by this build config."""
        return GroundingConfig(
            min_visible_pixels=self.min_visible_pixels,
            min_component_pixels=self.min_component_pixels,
            min_component_fraction_of_largest=(self.min_component_fraction_of_largest),
        )

    def sample_interval(
        self, start: int, end: int, skill: str
    ) -> tuple[tuple[int, float], ...]:
        """Select frames for one skill interval under this sampling policy."""
        use_stride = self.frame_stride is not None and (
            not self.frame_stride_skills or skill in self.frame_stride_skills
        )
        if use_stride:
            return select_interval_frames_by_stride(start, end, self.frame_stride)
        return select_interval_frames(start, end, self.sample_fractions)


@dataclasses.dataclass(frozen=True)
class GroundingAssessment:
    """Explicit eligibility and issue labels for one grounded sample."""

    issues: tuple[str, ...]
    object_grounding_complete: bool
    part_grounding_complete: bool
    visible_arguments: int
    groundable_arguments: int
    primary_cameras: tuple[str | None, ...]
    primary_visible_fraction: float

    @property
    def fully_grounded(self) -> bool:
        """Whether object and part groundings are both complete."""
        return self.object_grounding_complete and self.part_grounding_complete


@dataclasses.dataclass(frozen=True)
class _PendingSample:
    control: GroundedControlSpec
    skill_id: int
    skill_type: str
    memory_prefix: tuple[str, ...]
    interval_index: int
    interval_start: int
    interval_end: int
    sample_fraction: float
    frame_index: int


def select_interval_frames(
    start: int, end: int, fractions: tuple[float, ...]
) -> tuple[tuple[int, float], ...]:
    """Select unique frames from a half-open annotation interval."""
    if end <= start:
        raise ValueError(f"Invalid half-open interval [{start}, {end}).")
    if not fractions or not all(0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("fractions must be non-empty values in [0, 1].")

    selected: dict[int, float] = {}
    last_offset = end - start - 1
    for fraction in fractions:
        frame_index = start + int(last_offset * fraction)
        selected.setdefault(frame_index, fraction)
    return tuple(sorted(selected.items()))


def select_interval_frames_by_stride(
    start: int, end: int, stride: int
) -> tuple[tuple[int, float], ...]:
    """Select regular frames and the final frame of a half-open interval."""
    if end <= start:
        raise ValueError(f"Invalid half-open interval [{start}, {end}).")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    final_frame = end - 1
    frame_indices = set(range(start, end, stride))
    frame_indices.add(final_frame)
    denominator = max(final_frame - start, 1)
    return tuple(
        (frame_index, (frame_index - start) / denominator)
        for frame_index in sorted(frame_indices)
    )


def extract_action_chunk(
    actions: np.ndarray,
    frame_index: int,
    action_horizon: int,
    *,
    end_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a future action chunk clamped to an exclusive boundary.

    ``end_index`` defaults to the episode length. Grounded skill datasets pass
    the current annotation interval's end so a prompt never receives actions
    from the following skill. The final valid action is repeated only to keep
    the model's fixed action horizon; ``is_padding`` excludes those repetitions
    from the flow loss.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("actions must have shape (T, action_dim) with T > 0.")
    if not 0 <= frame_index < actions.shape[0]:
        raise IndexError(f"frame_index {frame_index} is outside the episode.")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if end_index is None:
        end_index = actions.shape[0]
    if not frame_index < end_index <= actions.shape[0]:
        raise ValueError(
            "end_index must be after frame_index and within the episode; "
            f"got frame_index={frame_index}, end_index={end_index}, "
            f"episode_length={actions.shape[0]}."
        )

    indices = frame_index + np.arange(action_horizon)
    is_padding = indices >= end_index
    indices = np.minimum(indices, end_index - 1)
    return actions[indices], is_padding


def assess_grounding(
    control: GroundedControlSpec, resolver: EntityResolver
) -> GroundingAssessment:
    """Report visibility and resolution failures without inventing fallbacks."""
    issues = []
    visible_arguments = 0
    groundable_arguments = 0
    object_complete = True
    part_complete = True
    primary_cameras = []
    primary_visible_fraction = 0.0

    for argument in control.arguments:
        raw_object_id = argument.raw_object_id
        if raw_object_id in {None, "robot"}:
            primary_cameras.append(None)
            continue
        groundable_arguments += 1
        object_ids = resolver.resolve(raw_object_id)
        if not object_ids:
            issues.append(f"unresolved_object:{raw_object_id}")
            object_complete = False
        if not argument.groundings:
            issues.append(f"object_not_visible:{raw_object_id}")
            object_complete = False
        else:
            visible_arguments += 1

        primary = select_primary_grounding(argument.groundings)
        if argument.part is not None:
            part_name = argument.part.name
            part_ids = resolver.resolve_part(raw_object_id, part_name)
            if not part_ids:
                issues.append(f"unresolved_part:{raw_object_id}/{part_name}")
                part_complete = False
            elif not argument.part.groundings:
                issues.append(f"part_not_visible:{raw_object_id}/{part_name}")
                part_complete = False
            else:
                primary = select_primary_grounding(argument.part.groundings)
        primary_cameras.append(None if primary is None else primary.camera.value)
        if primary is not None:
            primary_visible_fraction += primary.visible_fraction

    return GroundingAssessment(
        issues=tuple(issues),
        object_grounding_complete=object_complete,
        part_grounding_complete=part_complete,
        visible_arguments=visible_arguments,
        groundable_arguments=groundable_arguments,
        primary_cameras=tuple(primary_cameras),
        primary_visible_fraction=primary_visible_fraction,
    )


def pilot_arrow_schema(
    *, state_dim: int, action_dim: int, action_horizon: int
) -> pa.Schema:
    """Return the stable Arrow schema for grounded SFT sidecar rows."""
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("task_index", pa.int32(), nullable=False),
            pa.field("task_name", pa.string(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("timestamp", pa.float64(), nullable=False),
            pa.field("segment_index", pa.int32(), nullable=False),
            pa.field("interval_index", pa.int32(), nullable=False),
            pa.field("interval_start", pa.int32(), nullable=False),
            pa.field("interval_end", pa.int32(), nullable=False),
            pa.field("sample_fraction", pa.float32(), nullable=False),
            pa.field("skill_id", pa.int32(), nullable=False),
            pa.field("skill", pa.string(), nullable=False),
            pa.field("skill_type", pa.string(), nullable=False),
            pa.field("goal", pa.string(), nullable=False),
            pa.field("memory_prefix", pa.list_(pa.string()), nullable=False),
            pa.field("control_json", pa.large_string(), nullable=False),
            pa.field("grounding_issues", pa.list_(pa.string()), nullable=False),
            pa.field("object_grounding_complete", pa.bool_(), nullable=False),
            pa.field("has_part_argument", pa.bool_(), nullable=False),
            pa.field("part_grounding_complete", pa.bool_(), nullable=False),
            pa.field("fully_grounded", pa.bool_(), nullable=False),
            pa.field("visible_arguments", pa.int16(), nullable=False),
            pa.field("groundable_arguments", pa.int16(), nullable=False),
            pa.field("primary_cameras", pa.list_(pa.string()), nullable=False),
            pa.field("primary_visible_fraction", pa.float32(), nullable=False),
            pa.field("state", pa.list_(pa.float32(), state_dim), nullable=False),
            pa.field(
                "actions",
                pa.list_(pa.list_(pa.float32(), action_dim), action_horizon),
                nullable=False,
            ),
            pa.field(
                "action_is_pad",
                pa.list_(pa.bool_(), action_horizon),
                nullable=False,
            ),
            pa.field("source_data_path", pa.string(), nullable=False),
            pa.field("rgb_head_path", pa.string(), nullable=False),
            pa.field("rgb_left_wrist_path", pa.string(), nullable=False),
            pa.field("rgb_right_wrist_path", pa.string(), nullable=False),
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def _task_records(dataset_root: Path) -> dict[int, dict[str, Any]]:
    records = {}
    with (dataset_root / "meta" / "tasks.jsonl").open() as file:
        for line in file:
            record = json.loads(line)
            records[int(record["task_index"])] = record
    return records


def _episode_index(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def _selected_episodes(
    dataset_root: Path, task_indices: list[int], episodes_per_task: int
) -> list[tuple[int, int]]:
    selected = []
    for task_index in task_indices:
        annotation_dir = dataset_root / "annotations" / f"task-{task_index:04d}"
        paths = sorted(annotation_dir.glob("episode_*.json"))[:episodes_per_task]
        selected.extend((task_index, _episode_index(path)) for path in paths)
    return selected


def _relative_video_path(task_index: int, episode_index: int, camera: CameraID) -> str:
    return str(
        Path("videos")
        / f"task-{task_index:04d}"
        / f"observation.images.rgb.{camera.value}"
        / f"episode_{episode_index:08d}.mp4"
    )


def _source_data_path(task_index: int, episode_index: int) -> str:
    return str(
        Path("data") / f"task-{task_index:04d}" / f"episode_{episode_index:08d}.parquet"
    )


def _load_episode_arrays(
    dataset_root: Path, task_index: int, episode_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        dataset_root / _source_data_path(task_index, episode_index),
        columns=["timestamp", "observation.state", "action"],
    )
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if timestamps.shape[0] != states.shape[0] or states.shape[0] != actions.shape[0]:
        raise ValueError("Episode parquet columns have inconsistent lengths.")
    return timestamps, states, actions


def _pending_samples(
    annotation: dict[str, Any],
    *,
    goal: str,
    episode_index: int,
    config: PilotBuildConfig,
    parse_status_counts: collections.Counter[str],
) -> list[_PendingSample]:
    pending = []
    for record in annotation["skill_annotation"]:
        result = parse_skill_annotation(
            record,
            goal=goal,
            episode_id=f"episode_{episode_index:08d}",
        )
        parse_status_counts[result.status.value] += 1
        if result.status is not ParseStatus.VALID:
            continue
        segment = result.segment
        for interval_index, (start, end) in enumerate(segment.frame_intervals):
            selected_frames = config.sample_interval(
                start,
                end,
                segment.control.skill,
            )
            for frame_index, fraction in selected_frames:
                pending.append(
                    _PendingSample(
                        control=segment.control,
                        skill_id=segment.skill_id,
                        skill_type=segment.skill_type,
                        memory_prefix=segment.memory_prefix,
                        interval_index=interval_index,
                        interval_start=start,
                        interval_end=end,
                        sample_fraction=fraction,
                        frame_index=frame_index,
                    )
                )
    return pending


def _build_row(
    pending: _PendingSample,
    *,
    task_index: int,
    task_name: str,
    episode_index: int,
    timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    segmentations: dict[CameraID, np.ndarray],
    resolver: EntityResolver,
    config: PilotBuildConfig,
) -> tuple[dict[str, Any], GroundingAssessment]:
    frame_index = pending.frame_index
    if frame_index >= len(timestamps):
        raise IndexError(
            f"Annotated frame {frame_index} exceeds episode {episode_index} "
            f"length {len(timestamps)}."
        )
    grounded = ground_control_spec(
        pending.control,
        segmentations,
        resolver,
        config=config.grounding_config(),
        timestep=frame_index,
    )
    assessment = assess_grounding(grounded, resolver)
    action_chunk, action_is_pad = extract_action_chunk(
        actions,
        frame_index,
        config.action_horizon,
        end_index=pending.interval_end,
    )
    sample_id = (
        f"t{task_index:04d}-e{episode_index:08d}-"
        f"s{grounded.segment_id:03d}-i{pending.interval_index:02d}-f{frame_index:06d}"
    )
    row = {
        "sample_id": sample_id,
        "task_index": task_index,
        "task_name": task_name,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": float(timestamps[frame_index]),
        "segment_index": grounded.segment_id,
        "interval_index": pending.interval_index,
        "interval_start": pending.interval_start,
        "interval_end": pending.interval_end,
        "sample_fraction": pending.sample_fraction,
        "skill_id": pending.skill_id,
        "skill": grounded.skill,
        "skill_type": pending.skill_type,
        "goal": grounded.goal,
        "memory_prefix": list(pending.memory_prefix),
        "control_json": grounded.to_json(),
        "grounding_issues": list(assessment.issues),
        "object_grounding_complete": assessment.object_grounding_complete,
        "has_part_argument": any(
            argument.part is not None for argument in grounded.arguments
        ),
        "part_grounding_complete": assessment.part_grounding_complete,
        "fully_grounded": assessment.fully_grounded,
        "visible_arguments": assessment.visible_arguments,
        "groundable_arguments": assessment.groundable_arguments,
        "primary_cameras": [camera or "none" for camera in assessment.primary_cameras],
        "primary_visible_fraction": assessment.primary_visible_fraction,
        "state": states[frame_index].tolist(),
        "actions": action_chunk.tolist(),
        "action_is_pad": action_is_pad.tolist(),
        "source_data_path": _source_data_path(task_index, episode_index),
        "rgb_head_path": _relative_video_path(task_index, episode_index, CameraID.HEAD),
        "rgb_left_wrist_path": _relative_video_path(
            task_index, episode_index, CameraID.LEFT_WRIST
        ),
        "rgb_right_wrist_path": _relative_video_path(
            task_index, episode_index, CameraID.RIGHT_WRIST
        ),
    }
    return row, assessment


def _candidate_score(
    candidate: tuple[dict[str, Any], GroundingAssessment],
) -> tuple[bool, bool, int, float, float, int]:
    """Rank a temporal candidate by grounding quality and stable tie-breakers."""
    row, assessment = candidate
    return (
        assessment.object_grounding_complete,
        assessment.part_grounding_complete,
        assessment.visible_arguments,
        assessment.primary_visible_fraction,
        -abs(row["sample_fraction"] - 0.5),
        -row["frame_index"],
    )


def _select_episode_candidates(
    candidates: list[tuple[dict[str, Any], GroundingAssessment]],
    selection_mode: str,
) -> list[tuple[dict[str, Any], GroundingAssessment]]:
    """Select all candidates or one best-visible frame per skill interval."""
    if selection_mode == "all":
        return candidates
    if selection_mode != "best_visibility":
        raise ValueError(f"Unsupported selection mode: {selection_mode}.")

    grouped: dict[tuple[int, int], list[tuple[dict[str, Any], GroundingAssessment]]] = (
        collections.defaultdict(list)
    )
    for candidate in candidates:
        row, _ = candidate
        grouped[(row["segment_index"], row["interval_index"])].append(candidate)
    return [max(grouped[key], key=_candidate_score) for key in sorted(grouped)]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pilot_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    config: PilotBuildConfig = PilotBuildConfig(),
    task_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Build a grounded SFT pilot sidecar and return its manifest."""
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _task_records(dataset_root)
    selected_task_indices = sorted(tasks if task_indices is None else task_indices)
    unknown_tasks = sorted(set(selected_task_indices).difference(tasks))
    if unknown_tasks:
        raise ValueError(f"Unknown task indices: {unknown_tasks}.")
    episodes = _selected_episodes(
        dataset_root, selected_task_indices, config.episodes_per_task
    )

    rows = []
    parse_status_counts: collections.Counter[str] = collections.Counter()
    issue_counts: collections.Counter[str] = collections.Counter()
    skill_counts: collections.Counter[str] = collections.Counter()
    camera_counts: collections.Counter[str] = collections.Counter()
    task_sample_counts: collections.Counter[int] = collections.Counter()
    complete_object = 0
    part_samples = 0
    complete_part = 0
    fully_grounded = 0
    boundary_padded_samples = 0
    boundary_padded_steps = 0

    for episode_number, (task_index, episode_index) in enumerate(episodes, start=1):
        annotation_path = (
            dataset_root
            / "annotations"
            / f"task-{task_index:04d}"
            / f"episode_{episode_index:08d}.json"
        )
        annotation = _read_json(annotation_path)
        task = tasks[task_index]
        pending = _pending_samples(
            annotation,
            goal=task["task"],
            episode_index=episode_index,
            config=config,
            parse_status_counts=parse_status_counts,
        )
        frame_indices = tuple(sample.frame_index for sample in pending)
        metadata = EpisodeSegmentationMetadata.from_path(
            episode_metadata_path(dataset_root, task_index, episode_index)
        )
        timestamps, states, actions = _load_episode_arrays(
            dataset_root, task_index, episode_index
        )
        if metadata.n_steps != len(timestamps):
            raise ValueError(
                f"Episode {episode_index} metadata has {metadata.n_steps} steps, "
                f"but parquet has {len(timestamps)} rows."
            )
        segmentation_frames = {
            camera: read_segmentation_video_frames(
                episode_video_path(
                    dataset_root,
                    task_index,
                    episode_index,
                    "seg_instance_id",
                    camera,
                ),
                frame_indices,
                metadata.unique_instance_ids[camera],
            )
            for camera in CAMERAS
        }

        episode_candidates = []
        for sample in pending:
            segmentations = {
                camera: segmentation_frames[camera][sample.frame_index]
                for camera in CAMERAS
            }
            row, assessment = _build_row(
                sample,
                task_index=task_index,
                task_name=task["task_name"],
                episode_index=episode_index,
                timestamps=timestamps,
                states=states,
                actions=actions,
                segmentations=segmentations,
                resolver=metadata.resolver,
                config=config,
            )
            episode_candidates.append((row, assessment))

        selected_candidates = _select_episode_candidates(
            episode_candidates, config.selection_mode
        )
        for row, assessment in selected_candidates:
            rows.append(row)
            issue_counts.update(assessment.issues)
            skill_counts[row["skill"]] += 1
            camera_counts.update(
                camera for camera in assessment.primary_cameras if camera is not None
            )
            task_sample_counts[task_index] += 1
            complete_object += assessment.object_grounding_complete
            if row["has_part_argument"]:
                part_samples += 1
                complete_part += assessment.part_grounding_complete
            fully_grounded += assessment.fully_grounded
            padded_steps = sum(row["action_is_pad"])
            boundary_padded_samples += padded_steps > 0
            boundary_padded_steps += padded_steps

        print(
            f"[{episode_number}/{len(episodes)}] task={task_index:04d} "
            f"episode={episode_index:08d} candidates={len(pending)} "
            f"samples={len(selected_candidates)}",
            flush=True,
        )

    if not rows:
        raise ValueError("Pilot selection produced no valid samples.")
    state_dim = len(rows[0]["state"])
    action_dim = len(rows[0]["actions"][0])
    schema = pilot_arrow_schema(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    data_dir = output_dir / "data"
    data_dir.mkdir()
    parquet_path = data_dir / "part-00000.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    sample_count = len(rows)
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_root": str(dataset_root),
        "config": dataclasses.asdict(config),
        "source_episodes": [
            {"task_index": task_index, "episode_index": episode_index}
            for task_index, episode_index in episodes
        ],
        "schema": {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_horizon": config.action_horizon,
            "camera_order": [camera.value for camera in CAMERAS],
            "bbox_convention": "normalized_half_open_xyxy",
        },
        "counts": {
            "tasks": len(selected_task_indices),
            "episodes": len(episodes),
            "samples": sample_count,
            "object_grounding_complete": complete_object,
            "part_samples": part_samples,
            "part_grounding_complete": complete_part,
            "fully_grounded": fully_grounded,
            "boundary_padded_samples": boundary_padded_samples,
            "boundary_padded_steps": boundary_padded_steps,
            "object_grounding_fraction": complete_object / sample_count,
            "part_grounding_fraction": (
                complete_part / part_samples if part_samples else 1.0
            ),
            "fully_grounded_fraction": fully_grounded / sample_count,
        },
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "issue_counts": dict(issue_counts.most_common()),
        "skill_counts": dict(sorted(skill_counts.items())),
        "primary_camera_counts": dict(sorted(camera_counts.items())),
        "task_sample_counts": {
            str(task_index): count
            for task_index, count in sorted(task_sample_counts.items())
        },
        "shards": [
            {
                "path": str(parquet_path.relative_to(output_dir)),
                "rows": sample_count,
                "bytes": parquet_path.stat().st_size,
                "sha256": _file_sha256(parquet_path),
            }
        ],
    }
    with (output_dir / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def _parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item) for item in value.split(",") if item.strip())
    if not fractions:
        raise argparse.ArgumentTypeError("At least one fraction is required.")
    return fractions


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--sample-fractions", type=_parse_fractions, default=(0.5,))
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="Select every Nth frame in each interval and always include its final frame.",
    )
    parser.add_argument(
        "--frame-stride-skills",
        nargs="*",
        default=(),
        help=(
            "Apply --frame-stride only to these normalized skill names; "
            "other skills retain --sample-fractions. An empty list applies the "
            "stride to every skill for backward compatibility."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("all", "best_visibility"),
        default="all",
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--task-indices", type=int, nargs="*")
    args = parser.parse_args()
    config = PilotBuildConfig(
        episodes_per_task=args.episodes_per_task,
        sample_fractions=args.sample_fractions,
        frame_stride=args.frame_stride,
        frame_stride_skills=tuple(args.frame_stride_skills),
        selection_mode=args.selection_mode,
        action_horizon=args.action_horizon,
    )
    manifest = build_pilot_dataset(
        args.dataset_root,
        args.output_dir,
        config=config,
        task_indices=args.task_indices,
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

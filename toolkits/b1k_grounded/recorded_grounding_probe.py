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

"""Run a mask-grounding feasibility probe on recorded B1K demonstrations."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.b1k_grounded import (
    CameraID,
    EntityArgument,
    GroundedControlSpec,
    GroundingConfig,
    ParseStatus,
    ground_control_spec,
    parse_skill_annotation,
    select_primary_grounding,
)
from toolkits.b1k_grounded.recorded_segmentation import (
    DecodeDiagnostics,
    EpisodeSegmentationMetadata,
    episode_metadata_path,
    episode_video_path,
    read_rgb_video_frame,
    read_segmentation_video_frame,
)


@dataclasses.dataclass(frozen=True)
class ProbeCase:
    """One annotated skill segment selected for the feasibility probe."""

    name: str
    task_index: int
    episode_index: int
    segment_index: int


DEFAULT_CASES = (
    ProbeCase("radio_press", 0, 10, 2),
    ProbeCase("microwave_switch", 40, 400030, 7),
    ProbeCase("trash_place_in", 1, 10010, 4),
    ProbeCase("mousetrap_place_next_to", 5, 50020, 4),
    ProbeCase("fridge_open_door", 41, 410030, 1),
    ProbeCase("cabbage_chop", 41, 410030, 15),
)
CAMERAS = (CameraID.HEAD, CameraID.LEFT_WRIST, CameraID.RIGHT_WRIST)
COLORS = (
    (255, 64, 64),
    (64, 255, 64),
    (64, 128, 255),
    (255, 192, 64),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def _load_goals(dataset_root: Path) -> dict[int, str]:
    goals = {}
    with (dataset_root / "meta" / "tasks.jsonl").open() as file:
        for line in file:
            record = json.loads(line)
            goals[int(record["task_index"])] = record["task"]
    return goals


def _annotation_path(dataset_root: Path, case: ProbeCase) -> Path:
    return (
        dataset_root
        / "annotations"
        / f"task-{case.task_index:04d}"
        / f"episode_{case.episode_index:08d}.json"
    )


def _load_segment(
    dataset_root: Path, case: ProbeCase, goal: str
) -> tuple[dict[str, Any], GroundedControlSpec, tuple[int, int]]:
    annotation = _load_json(_annotation_path(dataset_root, case))
    matches = [
        record
        for record in annotation["skill_annotation"]
        if record["skill_idx"] == case.segment_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one segment {case.segment_index} for {case.name}, got {len(matches)}."
        )
    record = matches[0]
    result = parse_skill_annotation(
        record,
        goal=goal,
        episode_id=f"episode_{case.episode_index:08d}",
    )
    if result.status is not ParseStatus.VALID:
        raise ValueError(f"Could not parse {case.name}: {result.issues}.")
    intervals = result.segment.frame_intervals
    if len(intervals) != 1:
        raise ValueError(f"Probe case {case.name} has non-contiguous frame intervals.")
    return annotation, result.segment.control, intervals[0]


def _candidate_frames(start: int, end: int) -> tuple[int, ...]:
    duration = end - start
    return tuple(
        sorted(
            {
                min(end - 1, start + max(0, round(duration * fraction)))
                for fraction in (0.1, 0.5, 0.9)
            }
        )
    )


def _read_segmentations(
    dataset_root: Path,
    case: ProbeCase,
    metadata: EpisodeSegmentationMetadata,
    frame_index: int,
) -> tuple[dict[CameraID, np.ndarray], dict[CameraID, DecodeDiagnostics]]:
    segmentations = {}
    diagnostics = {}
    for camera in CAMERAS:
        path = episode_video_path(
            dataset_root,
            case.task_index,
            case.episode_index,
            "seg_instance_id",
            camera,
        )
        segmentation, camera_diagnostics = read_segmentation_video_frame(
            path,
            frame_index,
            metadata.unique_instance_ids[camera],
        )
        segmentations[camera] = segmentation
        diagnostics[camera] = camera_diagnostics
    return segmentations, diagnostics


def _argument_primary(argument: EntityArgument):
    part_primary = (
        None
        if argument.part is None
        else select_primary_grounding(argument.part.groundings)
    )
    return part_primary or select_primary_grounding(argument.groundings)


def _grounding_score(control: GroundedControlSpec) -> tuple[int, float]:
    primary = [_argument_primary(argument) for argument in control.arguments]
    visible = [grounding for grounding in primary if grounding is not None]
    return len(visible), sum(item.visible_fraction for item in visible)


def _serialize_diagnostics(value: DecodeDiagnostics) -> dict[str, Any]:
    return dataclasses.asdict(value)


def _argument_report(
    argument: EntityArgument, metadata: EpisodeSegmentationMetadata
) -> dict[str, Any]:
    raw_object_id = argument.raw_object_id
    mesh_ids = () if raw_object_id is None else metadata.resolver.resolve(raw_object_id)
    part_mesh_ids = (
        ()
        if raw_object_id is None or argument.part is None
        else metadata.resolver.resolve_part(raw_object_id, argument.part.name)
    )
    primary = _argument_primary(argument)
    return {
        "role": argument.role.value,
        "raw_object_id": raw_object_id,
        "category_name": argument.category_name,
        "resolved_mesh_ids": list(mesh_ids),
        "resolved_part_mesh_ids": list(part_mesh_ids),
        "groundings": {
            camera.value: grounding.to_dict()
            for camera, grounding in argument.groundings.items()
        },
        "part": None if argument.part is None else argument.part.to_dict(),
        "primary_camera": None if primary is None else primary.camera.value,
    }


def _render_debug_image(
    dataset_root: Path,
    output_path: Path,
    case: ProbeCase,
    control: GroundedControlSpec,
    frame_index: int,
) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Debug rendering requires OpenCV (cv2).") from error

    panels = []
    for camera in CAMERAS:
        rgb = read_rgb_video_frame(
            episode_video_path(
                dataset_root,
                case.task_index,
                case.episode_index,
                "rgb",
                camera,
            ),
            frame_index,
        )
        panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = panel.shape[:2]
        for index, argument in enumerate(control.arguments):
            grounding = (
                None if argument.part is None else argument.part.groundings.get(camera)
            )
            grounding = grounding or argument.groundings.get(camera)
            if grounding is None:
                continue
            color = COLORS[index % len(COLORS)]
            x_min, y_min, x_max, y_max = grounding.bbox_xyxy
            first = (round(x_min * width), round(y_min * height))
            second = (round(x_max * width) - 1, round(y_max * height) - 1)
            cv2.rectangle(panel, first, second, color, 3)
            if grounding.point_xy is not None:
                point = (
                    round(grounding.point_xy[0] * width),
                    round(grounding.point_xy[1] * height),
                )
                cv2.circle(panel, point, 6, color, -1)
            part_suffix = "" if argument.part is None else f"/{argument.part.name}"
            label = f"{argument.role.value}: {argument.raw_object_id}{part_suffix}"
            cv2.putText(
                panel,
                label,
                (first[0], max(20, first[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            camera.value,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(panel, (480, 480), interpolation=cv2.INTER_AREA))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.concatenate(panels, axis=1)):
        raise OSError(f"Could not write debug image {output_path}.")


def run_probe(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    cases: tuple[ProbeCase, ...] = DEFAULT_CASES,
) -> dict[str, Any]:
    """Run all recorded grounding cases and write JSON plus PNG evidence."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    goals = _load_goals(dataset_root)
    case_reports = []

    for case in cases:
        annotation, control, (start, end) = _load_segment(
            dataset_root, case, goals[case.task_index]
        )
        metadata = EpisodeSegmentationMetadata.from_path(
            episode_metadata_path(dataset_root, case.task_index, case.episode_index)
        )
        candidates = []
        for frame_index in _candidate_frames(start, end):
            segmentations, diagnostics = _read_segmentations(
                dataset_root, case, metadata, frame_index
            )
            grounded = ground_control_spec(
                control,
                segmentations,
                metadata.resolver,
                config=GroundingConfig(
                    min_visible_pixels=4,
                    min_component_pixels=64,
                    min_component_fraction_of_largest=0.005,
                ),
                timestep=frame_index,
            )
            candidates.append((grounded, diagnostics, _grounding_score(grounded)))
        grounded, diagnostics, score = max(candidates, key=lambda item: item[2])
        frame_index = grounded.timestep
        image_name = f"{case.name}_frame_{frame_index:05d}.png"
        _render_debug_image(
            dataset_root, output_dir / image_name, case, grounded, frame_index
        )
        case_reports.append(
            {
                "name": case.name,
                "task_index": case.task_index,
                "task_name": annotation["task_name"],
                "episode_index": case.episode_index,
                "segment_index": case.segment_index,
                "skill": grounded.skill,
                "frame_interval": [start, end],
                "selected_frame": frame_index,
                "visible_arguments": score[0],
                "total_arguments": len(grounded.arguments),
                "debug_image": image_name,
                "arguments": [
                    _argument_report(argument, metadata)
                    for argument in grounded.arguments
                ],
                "decode_diagnostics": {
                    camera.value: _serialize_diagnostics(diagnostics[camera])
                    for camera in CAMERAS
                },
            }
        )

    argument_count = sum(case["total_arguments"] for case in case_reports)
    visible_count = sum(case["visible_arguments"] for case in case_reports)
    report = {
        "dataset_root": str(dataset_root),
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "visible_arguments": visible_count,
            "total_arguments": argument_count,
            "visible_argument_fraction": (
                visible_count / argument_count if argument_count else 0.0
            ),
            "max_ambiguous_pixel_fraction": max(
                diagnostic["ambiguous_pixel_fraction"]
                for case in case_reports
                for diagnostic in case["decode_diagnostics"].values()
            ),
            "max_mean_color_error": max(
                diagnostic["mean_color_error"]
                for case in case_reports
                for diagnostic in case["decode_diagnostics"].values()
            ),
            "max_p99_color_error": max(
                diagnostic["p99_color_error"]
                for case in case_reports
                for diagnostic in case["decode_diagnostics"].values()
            ),
        },
    }
    with (output_dir / "grounding_probe.json").open("w") as file:
        json.dump(report, file, indent=2)
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(args.dataset_root, args.output_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

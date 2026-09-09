"""Build deterministic B1K VQA candidates and annotated frame images."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import pyarrow.parquet as pq
from common import (
    assign_group_splits,
    load_config,
    make_choices,
    stable_rng,
    unique_strings,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)
VIDEO_COLUMNS = {
    "head": "rgb_head_path",
    "left_wrist": "rgb_left_wrist_path",
    "right_wrist": "rgb_right_wrist_path",
}
ROLE_LABELS = {
    "manipulated": "object being manipulated",
    "target": "navigation or interaction target",
    "destination": "destination",
    "source": "source surface or container",
    "tool": "tool",
    "reference": "spatial reference object",
    "other": "secondary task object",
}
REQUIRED_COLUMNS = [
    "sample_id",
    "task_index",
    "task_name",
    "episode_index",
    "frame_index",
    "segment_index",
    "skill",
    "goal",
    "control_json",
    "fully_grounded",
    "source_data_path",
    *VIDEO_COLUMNS.values(),
]


def _parquet_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    paths = sorted(path.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return paths


def _read_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = pq.read_table(_parquet_paths(path), columns=REQUIRED_COLUMNS).to_pylist()
    return rows if limit is None else rows[:limit]


def _read_frame(path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
    return frame


def _bbox_pixels(
    bbox: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected xyxy bbox, got {bbox}")
    x1 = max(0, min(width - 1, int(bbox[0] * width)))
    y1 = max(0, min(height - 1, int(bbox[1] * height)))
    x2 = max(x1 + 1, min(width, int(bbox[2] * width + 0.999999)))
    y2 = max(y1 + 1, min(height, int(bbox[3] * height + 0.999999)))
    return x1, y1, x2, y2


def _passes_quality(
    grounding: dict[str, Any],
    bbox_pixels: tuple[int, int, int, int],
    width: int,
    height: int,
    quality: dict[str, Any],
) -> bool:
    x1, y1, x2, y2 = bbox_pixels
    bbox_area = (x2 - x1) * (y2 - y1)
    visible_pixels = int(grounding["visible_pixels"])
    return (
        visible_pixels >= int(quality["min_visible_pixels"])
        and x2 - x1 >= int(quality["min_bbox_side_pixels"])
        and y2 - y1 >= int(quality["min_bbox_side_pixels"])
        and bbox_area / (width * height) <= float(quality["max_bbox_area_fraction"])
        and visible_pixels / bbox_area >= float(quality["min_mask_fill_fraction"])
    )


def _annotate(frame, bbox_pixels: tuple[int, int, int, int]):
    image = frame.copy()
    x1, y1, x2, y2 = bbox_pixels
    thickness = max(3, round(min(image.shape[:2]) / 160))
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), thickness)
    cv2.putText(
        image,
        "QUERY",
        (x1, max(thickness * 5, y1 - thickness * 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


def _best_grounding(groundings: dict[str, Any]) -> dict[str, Any] | None:
    if not groundings:
        return None
    return max(
        groundings.values(),
        key=lambda value: (value["visible_pixels"], value["visible_fraction"]),
    )


def _vocabularies(controls: list[dict[str, Any]]):
    category_by_role: dict[str, set[str]] = defaultdict(set)
    category_by_skill_role: dict[tuple[str, str], set[str]] = defaultdict(set)
    categories = set()
    parts = set()
    skills = set()
    for control in controls:
        skill = control["skill"]
        skills.add(skill)
        for argument in control["arguments"]:
            category = argument["category_name"]
            role = argument["role"]
            categories.add(category)
            category_by_role[role].add(category)
            category_by_skill_role[(skill, role)].add(category)
            if argument.get("part"):
                parts.add(argument["part"]["name"])
    return category_by_role, category_by_skill_role, categories, parts, skills


def _category_pool(
    skill: str,
    role: str,
    correct: str,
    vocabularies,
    record_id: str,
    seed: int,
    pool_size: int,
) -> list[str]:
    category_by_role, category_by_skill_role, categories, _, _ = vocabularies
    values = unique_strings(
        [
            *sorted(category_by_skill_role[(skill, role)]),
            *sorted(category_by_role[role]),
            *sorted(categories),
        ]
    )
    values = [value for value in values if value.casefold() != correct.casefold()]
    stable_rng(record_id, seed).shuffle(values)
    return values[:pool_size]


def _base_record(
    row: dict[str, Any],
    split: str,
    record_id: str,
    question_type: str,
    image_path: Path,
    grounding: dict[str, Any],
    bbox_pixels: tuple[int, int, int, int],
    argument: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "split": split,
        "question_type": question_type,
        "image_path": str(image_path.resolve()),
        "sample_id": row["sample_id"],
        "task_index": row["task_index"],
        "task_name": row["task_name"],
        "episode_index": row["episode_index"],
        "frame_index": row["frame_index"],
        "segment_index": row["segment_index"],
        "camera": grounding["camera"],
        "bbox_xyxy": grounding["bbox_xyxy"],
        "bbox_pixels": list(bbox_pixels),
        "visible_pixels": grounding["visible_pixels"],
        "visible_fraction": grounding["visible_fraction"],
        "role": argument["role"],
        "category_name": argument["category_name"],
        "skill": row["skill"],
        "goal": row["goal"],
        "source_data_path": row["source_data_path"],
    }


def _add_choices(
    record: dict[str, Any],
    question: str,
    correct_label: str,
    distractor_pool: list[str],
    choice_count: int,
    seed: int,
) -> dict[str, Any]:
    choices, correct_answer = make_choices(
        record["id"], correct_label, distractor_pool, choice_count, seed
    )
    record.update(
        {
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "correct_label": correct_label,
            "distractor_pool": distractor_pool,
        }
    )
    return record


def build_candidates(
    config: dict[str, Any],
    output_root: Path,
    limit: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Build candidate JSONL records and annotated images."""
    source_config = config["source"]
    candidate_config = config["candidates"]
    split_config = config["splits"]
    sidecar_path = Path(source_config["sidecar_path"])
    dataset_root = Path(source_config["dataset_root"])
    rows = _read_rows(sidecar_path, limit)
    if not rows:
        raise ValueError("Sidecar contains no rows")

    controls = [json.loads(row["control_json"]) for row in rows]
    vocabularies = _vocabularies(controls)
    group_key = split_config["group_by"]
    split_by_group = assign_group_splits(
        (row[group_key] for row in rows),
        float(split_config["train_fraction"]),
        float(split_config["validation_fraction"]),
        int(config["seed"]),
    )

    candidates_path = output_root / "intermediate" / "candidates.jsonl"
    if candidates_path.exists() and not overwrite:
        raise FileExistsError(
            f"{candidates_path} already exists; pass --overwrite to replace it"
        )
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    enabled_types = set(candidate_config["question_types"])
    choice_count = int(candidate_config["choice_count"])
    pool_size = int(candidate_config["distractor_pool_size"])
    max_per_sample = int(candidate_config["max_questions_per_sample"])
    require_fully_grounded = bool(candidate_config["require_fully_grounded"])
    seed = int(config["seed"])
    quality = config["quality"]
    all_part_names = sorted(vocabularies[3])
    all_skills = sorted(vocabularies[4])
    role_pool = list(ROLE_LABELS.values())

    output_records = []
    skipped = Counter()
    counts = Counter()
    for row, control in zip(rows, controls):
        if require_fully_grounded and not row["fully_grounded"]:
            skipped["not_fully_grounded"] += 1
            continue
        split = split_by_group[row[group_key]]
        frame_cache = {}
        sample_records = []
        skill_anchor = None

        for argument_index, argument in enumerate(control["arguments"]):
            object_grounding = _best_grounding(argument.get("groundings", {}))
            if object_grounding is not None:
                camera = object_grounding["camera"]
                video_path = dataset_root / row[VIDEO_COLUMNS[camera]]
                if camera not in frame_cache:
                    frame_cache[camera] = _read_frame(video_path, row["frame_index"])
                frame = frame_cache[camera]
                height, width = frame.shape[:2]
                bbox_pixels = _bbox_pixels(object_grounding["bbox_xyxy"], width, height)
                if _passes_quality(
                    object_grounding, bbox_pixels, width, height, quality
                ):
                    image_path = (
                        image_dir
                        / f"{row['sample_id']}__arg{argument_index:02d}__{camera}.jpg"
                    )
                    if not cv2.imwrite(
                        str(image_path),
                        _annotate(frame, bbox_pixels),
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    ):
                        raise RuntimeError(f"Failed to write {image_path}")
                    if skill_anchor is None:
                        skill_anchor = (
                            argument,
                            object_grounding,
                            bbox_pixels,
                            image_path,
                        )

                    if "object_recognition" in enabled_types:
                        record_id = f"{row['sample_id']}::object::{argument_index}"
                        pool = _category_pool(
                            row["skill"],
                            argument["role"],
                            argument["category_name"],
                            vocabularies,
                            record_id,
                            seed,
                            pool_size,
                        )
                        if len(pool) >= choice_count - 1:
                            record = _base_record(
                                row,
                                split,
                                record_id,
                                "object_recognition",
                                image_path,
                                object_grounding,
                                bbox_pixels,
                                argument,
                            )
                            sample_records.append(
                                _add_choices(
                                    record,
                                    "What object is marked by the red bounding box?",
                                    argument["category_name"],
                                    pool,
                                    choice_count,
                                    seed,
                                )
                            )

                    if (
                        "role_identification" in enabled_types
                        and len(control["arguments"]) >= 2
                        and argument["role"] in ROLE_LABELS
                    ):
                        record_id = f"{row['sample_id']}::role::{argument_index}"
                        record = _base_record(
                            row,
                            split,
                            record_id,
                            "role_identification",
                            image_path,
                            object_grounding,
                            bbox_pixels,
                            argument,
                        )
                        question = (
                            f"Task: {row['goal']}\n"
                            f"Current primitive: {row['skill']}\n"
                            "What role does the red-boxed object have in this primitive?"
                        )
                        sample_records.append(
                            _add_choices(
                                record,
                                question,
                                ROLE_LABELS[argument["role"]],
                                role_pool,
                                choice_count,
                                seed,
                            )
                        )
                else:
                    skipped["object_quality"] += 1

            part = argument.get("part")
            part_grounding = (
                None if part is None else _best_grounding(part.get("groundings", {}))
            )
            if "part_recognition" not in enabled_types or part_grounding is None:
                continue
            camera = part_grounding["camera"]
            video_path = dataset_root / row[VIDEO_COLUMNS[camera]]
            if camera not in frame_cache:
                frame_cache[camera] = _read_frame(video_path, row["frame_index"])
            frame = frame_cache[camera]
            height, width = frame.shape[:2]
            bbox_pixels = _bbox_pixels(part_grounding["bbox_xyxy"], width, height)
            if not _passes_quality(part_grounding, bbox_pixels, width, height, quality):
                skipped["part_quality"] += 1
                continue
            record_id = f"{row['sample_id']}::part::{argument_index}"
            pool = [name for name in all_part_names if name != part["name"]]
            stable_rng(record_id, seed).shuffle(pool)
            pool = pool[:pool_size]
            if len(pool) < choice_count - 1:
                skipped["part_distractors"] += 1
                continue
            image_path = (
                image_dir
                / f"{row['sample_id']}__arg{argument_index:02d}__part__{camera}.jpg"
            )
            if not cv2.imwrite(
                str(image_path),
                _annotate(frame, bbox_pixels),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"Failed to write {image_path}")
            record = _base_record(
                row,
                split,
                record_id,
                "part_recognition",
                image_path,
                part_grounding,
                bbox_pixels,
                argument,
            )
            record["part_name"] = part["name"]
            sample_records.append(
                _add_choices(
                    record,
                    (
                        f"The red box marks a part of the {argument['category_name']}. "
                        "Which part is it?"
                    ),
                    part["name"],
                    pool,
                    choice_count,
                    seed,
                )
            )

        if "skill_identification" in enabled_types and skill_anchor is not None:
            argument, grounding, bbox_pixels, image_path = skill_anchor
            record_id = f"{row['sample_id']}::skill"
            pool = [skill for skill in all_skills if skill != row["skill"]]
            stable_rng(record_id, seed).shuffle(pool)
            pool = pool[:pool_size]
            if len(pool) >= choice_count - 1:
                record = _base_record(
                    row,
                    split,
                    record_id,
                    "skill_identification",
                    image_path,
                    grounding,
                    bbox_pixels,
                    argument,
                )
                sample_records.append(
                    _add_choices(
                        record,
                        (
                            f"Task: {row['goal']}\n"
                            "Which low-level primitive is the robot currently executing?"
                        ),
                        row["skill"],
                        pool,
                        choice_count,
                        seed,
                    )
                )

        priority = {
            "part_recognition": 0,
            "object_recognition": 1,
            "skill_identification": 2,
            "role_identification": 3,
        }
        sample_records.sort(
            key=lambda record: (priority[record["question_type"]], record["id"])
        )
        if len(sample_records) > max_per_sample:
            skipped["sample_cap"] += len(sample_records) - max_per_sample
            sample_records = sample_records[:max_per_sample]
        output_records.extend(sample_records)
        counts.update(record["question_type"] for record in sample_records)

    written = write_jsonl(candidates_path, output_records)
    manifest = {
        "format_version": "b1k_vqa_candidates_v0.1",
        "source_sidecar": str(sidecar_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "records": written,
        "counts_by_question_type": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "split_groups": {
            split: sorted(
                str(group) for group, value in split_by_group.items() if value == split
            )
            for split in ("train", "validation", "test")
        },
    }
    manifest_path = output_root / "intermediate" / "candidate_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run candidate construction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["output_root"])
    manifest = build_candidates(config, output_root, args.limit, args.overwrite)
    LOGGER.info("Wrote %d candidates to %s", manifest["records"], output_root)
    LOGGER.info("Question counts: %s", manifest["counts_by_question_type"])
    LOGGER.info("Skipped: %s", manifest["skipped"])


if __name__ == "__main__":
    main()

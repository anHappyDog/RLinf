"""Tests for generator validation and Parquet finalization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from common import write_jsonl  # noqa: E402
from finalize_dataset import finalize  # noqa: E402
from run_sglang import _validate_generator, _validate_judge  # noqa: E402
from validate_dataset import validate_dataset  # noqa: E402


def _record(image_path: Path, split: str, task_index: int) -> dict:
    return {
        "id": f"record-{task_index}",
        "split": split,
        "question_type": "object_recognition",
        "image_path": str(image_path),
        "question": "What object is marked by the red box?",
        "choices": ["radio", "speaker", "television", "table"],
        "correct_answer": "A",
        "correct_label": "radio",
        "distractor_pool": ["speaker", "television", "table", "lamp"],
        "sample_id": f"sample-{task_index}",
        "task_index": task_index,
        "task_name": "turning_on_radio",
        "episode_index": task_index * 10000 + 10,
        "frame_index": 132,
        "segment_index": 0,
        "camera": "head",
        "bbox_xyxy": [0.1, 0.2, 0.4, 0.5],
        "role": "target",
        "category_name": "radio",
        "skill": "move to",
        "goal": "Turn on the radio.",
        "source_data_path": "data/task/episode.parquet",
    }


def test_generator_and_judge_validation(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)
    record = _record(image_path, "train", 0)

    generated = _validate_generator(
        record,
        {
            "question": "Which item is inside the red box?",
            "distractors": ["speaker", "television", "table"],
            "visually_answerable": True,
            "ambiguous": False,
            "reason": "clear",
        },
        choice_count=4,
        seed=831,
        restrict_distractors_to_pool=True,
    )
    judged = _validate_judge(
        generated,
        {
            "predicted_answer": generated["correct_answer"],
            "visually_answerable": True,
            "ambiguous": False,
            "reason": "clear",
        },
    )

    assert judged["judge"]["predicted_answer"] == judged["correct_answer"]


def test_generator_can_propose_hard_negatives_outside_pool(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)
    record = _record(image_path, "train", 0)

    generated = _validate_generator(
        record,
        {
            "question": "Which item is inside the red box?",
            "distractors": ["alarm clock", "speaker system", "television set"],
            "visually_answerable": True,
            "ambiguous": False,
            "reason": "clear",
        },
        choice_count=4,
        seed=831,
        restrict_distractors_to_pool=False,
    )

    assert "alarm clock" in generated["choices"]


def test_finalize_and_validate_round_trip(tmp_path: Path) -> None:
    records = []
    for split, task_index in (("train", 0), ("validation", 1), ("test", 2)):
        image_path = tmp_path / f"image-{task_index}.jpg"
        Image.new("RGB", (32, 32), (task_index * 20, 0, 0)).save(image_path)
        records.append(_record(image_path, split, task_index))
    input_path = tmp_path / "candidates.jsonl"
    write_jsonl(input_path, records)
    dataset_dir = tmp_path / "dataset"
    config = {
        "filters": {
            "require_generator_visually_answerable": True,
            "reject_generator_ambiguous": True,
            "require_judge_visually_answerable": True,
            "reject_judge_ambiguous": True,
            "require_judge_correct": True,
        }
    }

    manifest = finalize(config, input_path, dataset_dir, allow_unjudged=True)

    assert manifest["accepted"] == 3
    assert validate_dataset(dataset_dir, "task_index") == []
    loaded_manifest = json.loads((dataset_dir / "manifest.json").read_text())
    assert loaded_manifest["accepted"] == 3

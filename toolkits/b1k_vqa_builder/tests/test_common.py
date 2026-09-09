"""Unit tests for B1K VQA shared utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from common import assign_group_splits, make_choices, parse_json_object  # noqa: E402


def test_make_choices_is_deterministic_and_correct() -> None:
    choices_a, answer_a = make_choices(
        "sample", "radio", ["speaker", "television", "table"], 4, 831
    )
    choices_b, answer_b = make_choices(
        "sample", "radio", ["speaker", "television", "table"], 4, 831
    )

    assert choices_a == choices_b
    assert answer_a == answer_b
    assert choices_a[ord(answer_a) - ord("A")] == "radio"


def test_make_choices_rejects_duplicate_distractors() -> None:
    with pytest.raises(ValueError, match="distractors"):
        make_choices("sample", "radio", ["speaker", "Speaker", "table"], 4, 831)


def test_assign_group_splits_keeps_exact_groups() -> None:
    assignments = assign_group_splits(range(50), 0.8, 0.1, 831)

    counts = {
        split: list(assignments.values()).count(split)
        for split in ("train", "validation", "test")
    }
    assert counts == {"train": 40, "validation": 5, "test": 5}


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": "A"}',
        '```json\n{"answer": "A"}\n```',
        '<think>done</think>\n{"answer": "A"}',
    ],
)
def test_parse_json_object(text: str) -> None:
    assert parse_json_object(text) == {"answer": "A"}

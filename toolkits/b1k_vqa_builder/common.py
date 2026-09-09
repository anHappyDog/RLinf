"""Shared utilities for the B1K VQA construction pipeline."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

CHOICE_LETTERS = "ABCDE"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL records."""
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL and return the number written."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_rng(value: str, seed: int) -> random.Random:
    """Create a deterministic random generator for a record identifier."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def unique_strings(values: Iterable[str]) -> list[str]:
    """Deduplicate non-empty strings while preserving order."""
    result = []
    seen = set()
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def make_choices(
    record_id: str,
    correct_label: str,
    distractors: Iterable[str],
    choice_count: int,
    seed: int,
) -> tuple[list[str], str]:
    """Build deterministic choices and return choices plus answer letter."""
    if not 2 <= choice_count <= len(CHOICE_LETTERS):
        raise ValueError("choice_count must be between 2 and 5")
    candidates = [
        value
        for value in unique_strings(distractors)
        if value.casefold() != correct_label.casefold()
    ]
    if len(candidates) < choice_count - 1:
        raise ValueError(
            f"{record_id} has {len(candidates)} distractors, "
            f"but {choice_count - 1} are required"
        )
    rng = stable_rng(record_id, seed)
    rng.shuffle(candidates)
    choices = [correct_label, *candidates[: choice_count - 1]]
    rng.shuffle(choices)
    correct_index = next(
        index
        for index, value in enumerate(choices)
        if value.casefold() == correct_label.casefold()
    )
    return choices, CHOICE_LETTERS[correct_index]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a Markdown fence or leading reasoning."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response contains no JSON object") from None
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def assign_group_splits(
    groups: Iterable[int | str],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[int | str, str]:
    """Assign complete groups to deterministic train/validation/test splits."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    unique_groups = sorted(set(groups), key=str)
    if len(unique_groups) < 3:
        raise ValueError("At least three groups are required for three-way splitting")
    random.Random(seed).shuffle(unique_groups)
    train_count = max(1, round(len(unique_groups) * train_fraction))
    validation_count = max(1, round(len(unique_groups) * validation_fraction))
    if train_count + validation_count >= len(unique_groups):
        train_count = len(unique_groups) - validation_count - 1

    split_by_group = {}
    for index, group in enumerate(unique_groups):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        split_by_group[group] = split
    return split_by_group


def answer_index(answer: str, choices: list[str]) -> int:
    """Convert an answer letter to a validated zero-based index."""
    letter = answer.strip().upper()
    if len(letter) != 1 or letter not in CHOICE_LETTERS[: len(choices)]:
        raise ValueError(f"Invalid answer {answer!r} for {len(choices)} choices")
    return CHOICE_LETTERS.index(letter)

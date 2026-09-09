"""Validate finalized B1K VQA Parquet files and split isolation."""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from common import answer_index, load_config
from PIL import Image

LOGGER = logging.getLogger(__name__)


def validate_dataset(dataset_dir: Path, group_by: str) -> list[str]:
    """Return validation errors for a finalized dataset directory."""
    errors = []
    seen_ids = set()
    group_splits: dict[object, set[str]] = defaultdict(set)
    sample_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    type_counts = Counter()
    answer_counts = Counter()
    split_counts = Counter()

    for split in ("train", "validation", "test"):
        paths = sorted((dataset_dir / split).glob("*.parquet"))
        if not paths:
            errors.append(f"Missing Parquet shard for split {split}")
            continue
        for row in pq.read_table(paths).to_pylist():
            record_id = row["id"]
            if record_id in seen_ids:
                errors.append(f"Duplicate ID: {record_id}")
            seen_ids.add(record_id)
            if row["split"] != split:
                errors.append(f"{record_id}: stored split does not match directory")
            group_splits[row[group_by]].add(split)
            sample_splits[row["sample_id"]].add(split)
            split_counts[split] += 1
            type_counts[row["question_type"]] += 1
            answer_counts[row["correct_answer"]] += 1

            choices = row["choices"]
            if not 2 <= len(choices) <= 5:
                errors.append(f"{record_id}: invalid choice count {len(choices)}")
                continue
            if len({choice.strip().casefold() for choice in choices}) != len(choices):
                errors.append(f"{record_id}: duplicate choices")
            try:
                correct_index = answer_index(row["correct_answer"], choices)
            except ValueError as error:
                errors.append(f"{record_id}: {error}")
                continue
            if choices[correct_index].casefold() != row["correct_label"].casefold():
                errors.append(f"{record_id}: correct label does not match answer")
            if not row["question"].strip():
                errors.append(f"{record_id}: empty question")
            bbox = row["bbox_xyxy"]
            if (
                len(bbox) != 4
                or not all(0.0 <= value <= 1.0 for value in bbox)
                or bbox[0] >= bbox[2]
                or bbox[1] >= bbox[3]
            ):
                errors.append(f"{record_id}: invalid normalized bbox {bbox}")

            image_bytes = row["image"]["bytes"]
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            image_splits[image_hash].add(split)
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
            except Exception as error:
                errors.append(f"{record_id}: invalid image bytes: {error}")

    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"{group_by}={group!r} crosses splits: {sorted(splits)}")
    for sample_id, splits in sample_splits.items():
        if len(splits) > 1:
            errors.append(f"sample_id={sample_id!r} crosses splits: {sorted(splits)}")
    for image_hash, splits in image_splits.items():
        if len(splits) > 1:
            errors.append(f"image {image_hash[:12]} crosses splits: {sorted(splits)}")

    LOGGER.info("Split counts: %s", dict(split_counts))
    LOGGER.info("Question types: %s", dict(type_counts))
    LOGGER.info("Answer positions: %s", dict(answer_counts))
    return errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run dataset validation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["output_root"])
    dataset_dir = args.dataset_dir or output_root / "dataset"
    errors = validate_dataset(dataset_dir, config["splits"]["group_by"])
    if errors:
        for error in errors[:100]:
            LOGGER.error(error)
        if len(errors) > 100:
            LOGGER.error("... and %d more errors", len(errors) - 100)
        raise SystemExit(f"Validation failed with {len(errors)} errors")
    LOGGER.info("Validation passed: %s", dataset_dir)


if __name__ == "__main__":
    main()

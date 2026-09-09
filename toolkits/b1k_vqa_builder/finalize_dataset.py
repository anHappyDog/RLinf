"""Filter judged B1K VQA records and export RLinf-compatible Parquet splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from common import answer_index, load_config, read_jsonl

LOGGER = logging.getLogger(__name__)


def dataset_schema() -> pa.Schema:
    """Return the stable output schema consumed by RLinf VLM datasets."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field(
                "image",
                pa.struct(
                    [
                        pa.field("bytes", pa.binary(), nullable=False),
                        pa.field("path", pa.string()),
                    ]
                ),
                nullable=False,
            ),
            pa.field("question", pa.string(), nullable=False),
            pa.field("choices", pa.list_(pa.string()), nullable=False),
            pa.field("correct_answer", pa.string(), nullable=False),
            pa.field("correct_label", pa.string(), nullable=False),
            pa.field("solution", pa.string(), nullable=False),
            pa.field("question_type", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("task_index", pa.int32(), nullable=False),
            pa.field("task_name", pa.string(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("segment_index", pa.int32(), nullable=False),
            pa.field("camera", pa.string(), nullable=False),
            pa.field("bbox_xyxy", pa.list_(pa.float32()), nullable=False),
            pa.field("role", pa.string(), nullable=False),
            pa.field("category_name", pa.string(), nullable=False),
            pa.field("part_name", pa.string()),
            pa.field("skill", pa.string(), nullable=False),
            pa.field("goal", pa.string(), nullable=False),
            pa.field("source_data_path", pa.string(), nullable=False),
            pa.field("generator_json", pa.large_string()),
            pa.field("judge_json", pa.large_string()),
        ]
    )


def _rejection_reason(
    record: dict[str, Any], filters: dict[str, Any], allow_unjudged: bool
) -> str | None:
    generator = record.get("generator")
    if generator is None:
        if not allow_unjudged:
            return "missing_generator"
    else:
        if (
            filters["require_generator_visually_answerable"]
            and not generator["visually_answerable"]
        ):
            return "generator_not_answerable"
        if filters["reject_generator_ambiguous"] and generator["ambiguous"]:
            return "generator_ambiguous"

    judge = record.get("judge")
    if judge is None:
        if not allow_unjudged:
            return "missing_judge"
    else:
        if (
            filters["require_judge_visually_answerable"]
            and not judge["visually_answerable"]
        ):
            return "judge_not_answerable"
        if filters["reject_judge_ambiguous"] and judge["ambiguous"]:
            return "judge_ambiguous"
        if filters["require_judge_correct"] and (
            judge["predicted_answer"] != record["correct_answer"]
        ):
            return "judge_incorrect"
    return None


def _output_row(record: dict[str, Any]) -> dict[str, Any]:
    choices = list(record["choices"])
    correct_label = choices[answer_index(record["correct_answer"], choices)]
    if correct_label.casefold() != record["correct_label"].casefold():
        raise ValueError(f"Answer mismatch in {record['id']}")
    return {
        "id": record["id"],
        "image": {"bytes": Path(record["image_path"]).read_bytes(), "path": None},
        "question": record["question"],
        "choices": choices,
        "correct_answer": record["correct_answer"],
        "correct_label": record["correct_label"],
        "solution": record["correct_label"],
        "question_type": record["question_type"],
        "split": record["split"],
        "sample_id": record["sample_id"],
        "task_index": record["task_index"],
        "task_name": record["task_name"],
        "episode_index": record["episode_index"],
        "frame_index": record["frame_index"],
        "segment_index": record["segment_index"],
        "camera": record["camera"],
        "bbox_xyxy": record["bbox_xyxy"],
        "role": record["role"],
        "category_name": record["category_name"],
        "part_name": record.get("part_name"),
        "skill": record["skill"],
        "goal": record["goal"],
        "source_data_path": record["source_data_path"],
        "generator_json": (
            json.dumps(record["generator"], ensure_ascii=False)
            if record.get("generator") is not None
            else None
        ),
        "judge_json": (
            json.dumps(record["judge"], ensure_ascii=False)
            if record.get("judge") is not None
            else None
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(
    config: dict[str, Any],
    input_path: Path,
    dataset_dir: Path,
    allow_unjudged: bool,
) -> dict[str, Any]:
    """Filter records and write one Parquet shard per split."""
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    rejected = Counter()
    accepted_types = Counter()
    seen_ids = set()
    for record in read_jsonl(input_path):
        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate record ID: {record['id']}")
        seen_ids.add(record["id"])
        reason = _rejection_reason(record, config["filters"], allow_unjudged)
        if reason:
            rejected[reason] += 1
            continue
        split = record["split"]
        if split not in rows_by_split:
            raise ValueError(f"Unknown split {split!r} in {record['id']}")
        rows_by_split[split].append(_output_row(record))
        accepted_types[record["question_type"]] += 1

    dataset_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: row["id"])
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / "part-00000.parquet"
        table = pa.Table.from_pylist(rows, schema=dataset_schema())
        pq.write_table(table, path, compression="zstd")
        shards.append(
            {
                "split": split,
                "path": str(path.relative_to(dataset_dir)),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "format_version": "b1k_vqa_rlinf_v0.1",
        "input": str(input_path.resolve()),
        "accepted": sum(len(rows) for rows in rows_by_split.values()),
        "accepted_by_question_type": dict(sorted(accepted_types.items())),
        "rejected": dict(sorted(rejected.items())),
        "shards": shards,
    }
    with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--allow-unjudged",
        action="store_true",
        help="Allow canonical/generated rows without both model stages.",
    )
    return parser.parse_args()


def main() -> None:
    """Export the finalized dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["output_root"])
    input_path = args.input or output_root / "intermediate" / "judged.jsonl"
    dataset_dir = args.dataset_dir or output_root / "dataset"
    manifest = finalize(config, input_path, dataset_dir, args.allow_unjudged)
    LOGGER.info("Accepted %d records into %s", manifest["accepted"], dataset_dir)
    LOGGER.info("Rejected: %s", manifest["rejected"])


if __name__ == "__main__":
    main()

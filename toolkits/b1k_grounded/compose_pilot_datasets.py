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

"""Compose grounded B1K sidecars and repair action chunks at skill boundaries."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolkits.b1k_grounded.build_pilot_dataset import FORMAT_VERSION


def make_action_chunk_boundary_safe(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Clamp one fixed-horizon action chunk to its half-open skill interval."""
    actions = np.asarray(row["actions"], dtype=np.float32)
    action_is_pad = np.asarray(row["action_is_pad"], dtype=np.bool_)
    if actions.ndim != 2 or action_is_pad.shape != actions.shape[:1]:
        raise ValueError(
            f"Sample {row['sample_id']} has incompatible action shapes: "
            f"actions={actions.shape}, action_is_pad={action_is_pad.shape}."
        )
    valid_length = min(actions.shape[0], row["interval_end"] - row["frame_index"])
    if valid_length <= 0:
        raise ValueError(
            f"Sample {row['sample_id']} starts outside its skill interval."
        )

    boundary_padding = np.arange(actions.shape[0]) >= valid_length
    repaired_steps = int(np.count_nonzero(boundary_padding & ~action_is_pad))
    action_is_pad |= boundary_padding
    actions[valid_length:] = actions[valid_length - 1]
    return {
        **row,
        "actions": actions.tolist(),
        "action_is_pad": action_is_pad.tolist(),
    }, repaired_steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_path(input_dirs: list[Path]) -> Path:
    candidates = [
        directory / "structural_token_mapping.json"
        for directory in input_dirs
        if (directory / "structural_token_mapping.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No input dataset contains structural_token_mapping.json."
        )
    mappings = {path.read_text(encoding="utf-8") for path in candidates}
    if len(mappings) != 1:
        raise ValueError("Input datasets use different structural token mappings.")
    return candidates[0]


def _interval_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return the source skill-interval identity for one sample row."""
    return (
        row["task_index"],
        row["episode_index"],
        row["segment_index"],
        row["interval_index"],
    )


def _merge_input_rows(
    rows_by_id: dict[str, dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    replacement_scope: str,
) -> int:
    """Merge one input shard and return the number of interval-replaced rows."""
    replaced_samples = 0
    if replacement_scope == "interval":
        replacement_keys = {_interval_key(row) for row in input_rows}
        replaced_ids = [
            sample_id
            for sample_id, row in rows_by_id.items()
            if _interval_key(row) in replacement_keys
        ]
        replaced_samples = len(replaced_ids)
        for sample_id in replaced_ids:
            del rows_by_id[sample_id]

    for row in input_rows:
        rows_by_id[row["sample_id"]] = row
    return replaced_samples


def compose_pilot_datasets(
    input_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    replacement_scope: str = "sample",
) -> dict[str, Any]:
    """Merge sidecars, with later inputs replacing samples or intervals."""
    if replacement_scope not in {"sample", "interval"}:
        raise ValueError("replacement_scope must be 'sample' or 'interval'.")
    inputs = [Path(directory).resolve() for directory in input_dirs]
    if len(inputs) < 2:
        raise ValueError("At least two input datasets are required.")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}.")

    source_manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in inputs
    ]
    dataset_roots = {manifest["dataset_root"] for manifest in source_manifests}
    if len(dataset_roots) != 1:
        raise ValueError("Input datasets reference different B1K dataset roots.")

    tables = [pq.read_table(path / "data" / "part-00000.parquet") for path in inputs]
    schema = tables[0].schema
    for path, table in zip(inputs[1:], tables[1:], strict=True):
        if table.schema != schema:
            raise ValueError(f"Input sidecar schema differs: {path}.")

    rows_by_id: dict[str, dict[str, Any]] = {}
    input_rows = 0
    replaced_input_samples = 0
    repaired_samples = 0
    repaired_steps = 0
    for table in tables:
        input_rows += table.num_rows
        input_table_rows = []
        for row in table.to_pylist():
            row, repaired = make_action_chunk_boundary_safe(row)
            repaired_samples += repaired > 0
            repaired_steps += repaired
            input_table_rows.append(row)

        replaced_input_samples += _merge_input_rows(
            rows_by_id,
            input_table_rows,
            replacement_scope=replacement_scope,
        )

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            row["task_index"],
            row["episode_index"],
            row["frame_index"],
            row["segment_index"],
            row["interval_index"],
            row["sample_id"],
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir()
    parquet_path = data_dir / "part-00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), parquet_path, compression="zstd"
    )
    shutil.copyfile(_mapping_path(inputs), output / "structural_token_mapping.json")

    skills = collections.Counter(row["skill"] for row in rows)
    tasks = collections.Counter(row["task_index"] for row in rows)
    part_rows = [row for row in rows if row["has_part_argument"]]
    padded_rows = [row for row in rows if any(row["action_is_pad"])]
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_root": dataset_roots.pop(),
        "config": {
            "composition": [str(path) for path in inputs],
            "replacement_scope": replacement_scope,
            "duplicate_policy": "later_input_wins",
            "action_boundary": "repeat_last_valid_and_mask_tail",
        },
        "counts": {
            "tasks": len(tasks),
            "episodes": len(
                {(row["task_index"], row["episode_index"]) for row in rows}
            ),
            "input_samples": input_rows,
            "duplicate_samples": input_rows - len(rows),
            "replaced_input_samples": replaced_input_samples,
            "samples": len(rows),
            "object_grounding_complete": sum(
                row["object_grounding_complete"] for row in rows
            ),
            "part_samples": len(part_rows),
            "part_grounding_complete": sum(
                row["part_grounding_complete"] for row in part_rows
            ),
            "fully_grounded": sum(row["fully_grounded"] for row in rows),
            "boundary_padded_samples": len(padded_rows),
            "boundary_padded_steps": sum(
                sum(row["action_is_pad"]) for row in padded_rows
            ),
            "repaired_input_samples": repaired_samples,
            "repaired_input_steps": repaired_steps,
        },
        "skill_counts": dict(sorted(skills.items())),
        "task_sample_counts": {
            str(task_index): count for task_index, count in sorted(tasks.items())
        },
        "shards": [
            {
                "path": str(parquet_path.relative_to(output)),
                "rows": len(rows),
                "bytes": parquet_path.stat().st_size,
                "sha256": _sha256(parquet_path),
            }
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replacement-scope",
        choices=("sample", "interval"),
        default="sample",
        help=(
            "Replace only duplicate sample IDs, or replace every earlier row "
            "from each skill interval represented by a later input."
        ),
    )
    args = parser.parse_args()
    manifest = compose_pilot_datasets(
        args.input_dir,
        args.output_dir,
        replacement_scope=args.replacement_scope,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

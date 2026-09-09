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

"""Add primitive-derived subgoals to an existing B1K sidecar without resampling."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.data.b1k_grounded import (
    SUBGOAL_RESOLUTION_POLICY,
    GroundedControlSpec,
    ResolvedSubgoal,
    resolve_episode_subgoals,
)

FORMAT_VERSION = "b1k_grounded_sft_sidecar_v0.4"
SUBGOAL_COLUMNS = (
    "subgoal",
    "subgoal_status",
    "subgoal_primitive_indices",
)
_REQUIRED_COLUMNS = {
    "sample_id",
    "task_index",
    "episode_index",
    "segment_index",
    "skill",
    "goal",
    "control_json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _annotation_path(dataset_root: Path, task_index: int, episode_index: int) -> Path:
    return (
        dataset_root
        / "annotations"
        / f"task-{task_index:04d}"
        / f"episode_{episode_index:08d}.json"
    )


def _replace_control_subgoal(row: dict[str, Any], resolution: ResolvedSubgoal) -> str:
    control = GroundedControlSpec.from_json(row["control_json"])
    if control.subgoal is not None:
        raise ValueError(
            f"Sample {row.get('sample_id', '<unknown>')} already has a subgoal."
        )
    if control.goal != row["goal"]:
        raise ValueError(
            f"Sample {row.get('sample_id', '<unknown>')} has a mismatched goal."
        )
    if control.skill != row["skill"]:
        raise ValueError(
            f"Sample {row.get('sample_id', '<unknown>')} has a mismatched skill."
        )
    if control.segment_id != row["segment_index"]:
        raise ValueError(
            f"Sample {row.get('sample_id', '<unknown>')} has a mismatched segment."
        )
    return dataclasses.replace(control, subgoal=resolution.text).to_json()


def _add_subgoal_columns(
    table: pa.Table,
    *,
    dataset_root: Path,
    resolution_cache: dict[tuple[int, int], dict[int, ResolvedSubgoal]],
) -> tuple[pa.Table, collections.Counter[str]]:
    existing_subgoal_columns = set(SUBGOAL_COLUMNS).intersection(table.column_names)
    if existing_subgoal_columns:
        raise ValueError(
            "Input sidecar already contains subgoal metadata columns: "
            f"{sorted(existing_subgoal_columns)}."
        )
    missing_columns = _REQUIRED_COLUMNS.difference(table.column_names)
    if missing_columns:
        raise ValueError(
            f"Input sidecar is missing required columns: {sorted(missing_columns)}."
        )

    control_json = []
    subgoals = []
    statuses = []
    primitive_indices = []
    status_counts: collections.Counter[str] = collections.Counter()
    columns = table.select(sorted(_REQUIRED_COLUMNS)).to_pylist()
    for row in columns:
        episode_key = (row["task_index"], row["episode_index"])
        if episode_key not in resolution_cache:
            annotation = _read_json(_annotation_path(dataset_root, *episode_key))
            resolution_cache[episode_key] = resolve_episode_subgoals(annotation)
        resolution = resolution_cache[episode_key].get(row["segment_index"])
        if resolution is None:
            raise ValueError(
                f"Sample {row.get('sample_id', '<unknown>')} references skill "
                f"{row['segment_index']}, which has no primitive resolution in "
                f"episode {episode_key}."
            )

        control_json.append(_replace_control_subgoal(row, resolution))
        subgoals.append(resolution.text)
        statuses.append(resolution.status.value)
        primitive_indices.append(list(resolution.primitive_indices))
        status_counts[resolution.status.value] += 1

    control_index = table.schema.get_field_index("control_json")
    control_field = table.schema.field(control_index)
    enriched = table.set_column(
        control_index,
        control_field,
        pa.array(control_json, type=control_field.type),
    )
    insert_at = enriched.schema.get_field_index("goal") + 1
    additions = (
        (pa.field("subgoal", pa.string(), nullable=True), subgoals),
        (pa.field("subgoal_status", pa.string(), nullable=False), statuses),
        (
            pa.field(
                "subgoal_primitive_indices",
                pa.list_(pa.int32()),
                nullable=False,
            ),
            primitive_indices,
        ),
    )
    for offset, (field, values) in enumerate(additions):
        enriched = enriched.add_column(
            insert_at + offset,
            field,
            pa.array(values, type=field.type),
        )

    unchanged_columns = [name for name in table.column_names if name != "control_json"]
    if not table.select(unchanged_columns).equals(enriched.select(unchanged_columns)):
        raise RuntimeError("A non-control source column changed during enrichment.")
    return enriched, status_counts


def _validate_written_shard(source: pa.Table, destination_path: Path) -> None:
    written = pq.read_table(destination_path)
    unchanged_columns = [name for name in source.column_names if name != "control_json"]
    if source.num_rows != written.num_rows:
        raise RuntimeError(
            f"Written shard has {written.num_rows} rows; expected {source.num_rows}."
        )
    if not source.select(unchanged_columns).equals(written.select(unchanged_columns)):
        raise RuntimeError(
            f"Written shard changed source values outside control_json: "
            f"{destination_path}."
        )


def enrich_sidecar_subgoals(
    input_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Enrich an existing sidecar while preserving its rows and source columns."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}.")

    manifest_path = input_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    dataset_root = Path(manifest["dataset_root"]).resolve()
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Input manifest must contain at least one shard.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    resolution_cache: dict[tuple[int, int], dict[int, ResolvedSubgoal]] = {}
    status_counts: collections.Counter[str] = collections.Counter()
    output_shards = []
    source_columns: list[str] | None = None
    try:
        for shard in shards:
            relative_path = Path(shard["path"])
            source_path = input_dir / relative_path
            if _sha256(source_path) != shard["sha256"]:
                raise ValueError(
                    f"Source shard SHA-256 differs from its manifest: {source_path}."
                )
            source = pq.read_table(source_path)
            if source_columns is None:
                source_columns = source.column_names
            elif source.column_names != source_columns:
                raise ValueError("Input shards have different column layouts.")
            enriched, shard_status_counts = _add_subgoal_columns(
                source,
                dataset_root=dataset_root,
                resolution_cache=resolution_cache,
            )
            status_counts.update(shard_status_counts)

            destination_path = staging_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(enriched, destination_path, compression="zstd")
            _validate_written_shard(source, destination_path)
            output_shards.append(
                {
                    "path": str(relative_path),
                    "rows": enriched.num_rows,
                    "bytes": destination_path.stat().st_size,
                    "sha256": _sha256(destination_path),
                }
            )

        mapping_source = input_dir / "structural_token_mapping.json"
        if not mapping_source.is_file():
            raise FileNotFoundError(
                f"Missing structural token mapping: {mapping_source}."
            )
        shutil.copyfile(mapping_source, staging_dir / mapping_source.name)

        output_manifest = {
            **manifest,
            "format_version": FORMAT_VERSION,
            "subgoal_resolution_policy": SUBGOAL_RESOLUTION_POLICY,
            "subgoal_status_counts": dict(sorted(status_counts.items())),
            "shards": output_shards,
            "lineage": {
                "operation": "add_primitive_subgoals",
                "source_dataset": str(input_dir),
                "source_format_version": manifest.get("format_version"),
                "source_manifest_sha256": _sha256(manifest_path),
                "preserved_columns": [
                    name for name in source_columns or [] if name != "control_json"
                ],
                "modified_columns": ["control_json"],
                "control_json_modified_fields": ["subgoal"],
                "added_columns": list(SUBGOAL_COLUMNS),
            },
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return output_manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = enrich_sidecar_subgoals(args.input_dir, args.output_dir)
    print(
        json.dumps(
            {
                "format_version": manifest["format_version"],
                "counts": manifest["counts"],
                "subgoal_status_counts": manifest["subgoal_status_counts"],
                "lineage": manifest["lineage"],
                "shards": manifest["shards"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

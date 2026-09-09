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

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.data.b1k_grounded import GroundedControlSpec
from toolkits.b1k_grounded.enrich_sidecar_subgoals import (
    FORMAT_VERSION,
    enrich_sidecar_subgoals,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_sidecar(tmp_path):
    dataset_root = tmp_path / "b1k"
    annotation_path = (
        dataset_root / "annotations" / "task-0000" / "episode_00000010.json"
    )
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "skill_annotation": [{"skill_idx": 0}],
                "primitive_annotation": [
                    {
                        "primitive_idx": 4,
                        "primitive_id": [1],
                        "primitive_description": ["move to"],
                        "object_id": [["radio_89"]],
                        "memory_prefix": [],
                        "spatial_prefix": [],
                        "skill_idxes": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = tmp_path / "source"
    data_dir = source / "data"
    data_dir.mkdir(parents=True)
    rows = []
    for frame_index in (0, 8):
        control = GroundedControlSpec(
            goal="Turn on the radio.",
            subgoal=None,
            skill="move to",
            arguments=(),
            episode_id="episode_00000010",
            segment_id=0,
            timestep=frame_index,
        )
        rows.append(
            {
                "sample_id": f"sample-{frame_index}",
                "task_index": 0,
                "episode_index": 10,
                "frame_index": frame_index,
                "segment_index": 0,
                "goal": "Turn on the radio.",
                "control_json": control.to_json(),
                "skill": "move to",
                "state": [float(frame_index), 2.0],
                "actions": [[float(frame_index), 3.0]],
                "action_is_pad": [False],
                "fully_grounded": True,
            }
        )
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("task_index", pa.int32(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("segment_index", pa.int32(), nullable=False),
            pa.field("goal", pa.string(), nullable=False),
            pa.field("control_json", pa.large_string(), nullable=False),
            pa.field("skill", pa.string(), nullable=False),
            pa.field("state", pa.list_(pa.float32(), 2), nullable=False),
            pa.field(
                "actions",
                pa.list_(pa.list_(pa.float32(), 2), 1),
                nullable=False,
            ),
            pa.field("action_is_pad", pa.list_(pa.bool_(), 1), nullable=False),
            pa.field("fully_grounded", pa.bool_(), nullable=False),
        ]
    )
    parquet_path = data_dir / "part-00000.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), parquet_path)
    manifest = {
        "format_version": "b1k_grounded_sft_sidecar_v0.2",
        "dataset_root": str(dataset_root),
        "config": {
            "composition": ["midpoint", "dense-radio"],
            "replacement_scope": "interval",
            "action_boundary": "repeat_last_valid_and_mask_tail",
        },
        "counts": {"tasks": 1, "episodes": 1, "samples": 2},
        "shards": [
            {
                "path": "data/part-00000.parquet",
                "rows": 2,
                "bytes": parquet_path.stat().st_size,
                "sha256": _sha256(parquet_path),
            }
        ],
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "structural_token_mapping.json").write_text(
        '{"mapping": "frozen"}', encoding="utf-8"
    )
    return source, manifest


def test_enrichment_preserves_latest_sidecar_rows_and_source_values(tmp_path):
    source_dir, source_manifest = _write_source_sidecar(tmp_path)
    source_path = source_dir / "data" / "part-00000.parquet"
    source_hash = _sha256(source_path)
    source = pq.read_table(source_path)
    output_dir = tmp_path / "enriched"

    manifest = enrich_sidecar_subgoals(source_dir, output_dir)

    output = pq.read_table(output_dir / "data" / "part-00000.parquet")
    unchanged_columns = [name for name in source.column_names if name != "control_json"]
    assert output.num_rows == source.num_rows
    assert output.select(unchanged_columns).equals(source.select(unchanged_columns))
    assert output.column_names == [
        "sample_id",
        "task_index",
        "episode_index",
        "frame_index",
        "segment_index",
        "goal",
        "subgoal",
        "subgoal_status",
        "subgoal_primitive_indices",
        "control_json",
        "skill",
        "state",
        "actions",
        "action_is_pad",
        "fully_grounded",
    ]
    assert output["subgoal"].to_pylist() == ["move to radio", "move to radio"]
    assert output["subgoal_status"].to_pylist() == ["unique", "unique"]
    assert output["subgoal_primitive_indices"].to_pylist() == [[4], [4]]
    for old_json, new_json in zip(
        source["control_json"].to_pylist(),
        output["control_json"].to_pylist(),
        strict=True,
    ):
        expected = json.loads(old_json)
        expected["subgoal"] = "move to radio"
        assert json.loads(new_json) == expected

    assert _sha256(source_path) == source_hash
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["config"] == source_manifest["config"]
    assert manifest["counts"] == source_manifest["counts"]
    assert manifest["subgoal_status_counts"] == {"unique": 2}
    assert manifest["lineage"]["source_dataset"] == str(source_dir.resolve())
    assert manifest["lineage"]["modified_columns"] == ["control_json"]
    assert manifest["lineage"]["control_json_modified_fields"] == ["subgoal"]
    assert (output_dir / "structural_token_mapping.json").read_text() == (
        source_dir / "structural_token_mapping.json"
    ).read_text()

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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.b1k_grounded import GroundedControlSpec
from toolkits.b1k_grounded.audit_sidecar import audit_sidecar
from toolkits.b1k_grounded.build_pilot_dataset import pilot_arrow_schema


def _write_dataset(tmp_path, frame_indices=(10, 14, 18, 19)):
    dataset_root = tmp_path / "source"
    source_paths = (
        "data/task-0000/episode_00000001.parquet",
        "videos/task-0000/observation.images.rgb.head/episode_00000001.mp4",
        "videos/task-0000/observation.images.rgb.left_wrist/episode_00000001.mp4",
        "videos/task-0000/observation.images.rgb.right_wrist/episode_00000001.mp4",
    )
    for relative_path in source_paths:
        path = dataset_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    output = tmp_path / "sidecar"
    data_dir = output / "data"
    data_dir.mkdir(parents=True)
    control = GroundedControlSpec(
        goal="test goal",
        subgoal=None,
        skill="pick up from",
        arguments=(),
        episode_id="episode_00000001",
        segment_id=1,
    )
    rows = []
    actions = np.arange(32 * 23, dtype=np.float32).reshape(32, 23)
    for frame_index in frame_indices:
        valid_length = min(32, 20 - frame_index)
        row_actions = actions.copy()
        row_actions[valid_length:] = row_actions[valid_length - 1]
        rows.append(
            {
                "sample_id": f"sample-{frame_index}",
                "task_index": 0,
                "task_name": "test_task",
                "episode_index": 1,
                "frame_index": frame_index,
                "timestamp": frame_index / 30,
                "segment_index": 1,
                "interval_index": 0,
                "interval_start": 10,
                "interval_end": 20,
                "sample_fraction": (frame_index - 10) / 9,
                "skill_id": 1,
                "skill": "pick up from",
                "skill_type": "atomic",
                "goal": "test goal",
                "memory_prefix": [],
                "control_json": GroundedControlSpec(
                    **{
                        **control.__dict__,
                        "timestep": frame_index,
                    }
                ).to_json(),
                "grounding_issues": [],
                "object_grounding_complete": True,
                "has_part_argument": False,
                "part_grounding_complete": True,
                "fully_grounded": True,
                "visible_arguments": 0,
                "groundable_arguments": 0,
                "primary_cameras": [],
                "primary_visible_fraction": 0.0,
                "state": np.zeros(256, dtype=np.float32).tolist(),
                "actions": row_actions.tolist(),
                "action_is_pad": (np.arange(32) >= valid_length).tolist(),
                "source_data_path": source_paths[0],
                "rgb_head_path": source_paths[1],
                "rgb_left_wrist_path": source_paths[2],
                "rgb_right_wrist_path": source_paths[3],
            }
        )

    parquet_path = data_dir / "part-00000.parquet"
    table = pa.Table.from_pylist(
        rows,
        schema=pilot_arrow_schema(
            state_dim=256,
            action_dim=23,
            action_horizon=32,
        ),
    )
    pq.write_table(table, parquet_path)
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_root": str(dataset_root),
        "config": {
            "frame_stride": 4,
            "frame_stride_skills": [],
            "selection_mode": "all",
        },
        "counts": {"samples": len(rows)},
        "shards": [
            {
                "path": "data/part-00000.parquet",
                "rows": len(rows),
                "bytes": parquet_path.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "structural_token_mapping.json").write_text("{}")
    return output


def test_audit_sidecar_accepts_complete_dense_intervals(tmp_path):
    output = _write_dataset(tmp_path)

    report = audit_sidecar(
        output,
        expected_tasks=1,
        expected_episodes=1,
        frame_stride=4,
    )

    assert report["status"] == "passed"
    assert report["samples"] == 4
    assert report["intervals"] == 1


def test_audit_sidecar_rejects_incomplete_stride_coverage(tmp_path):
    output = _write_dataset(tmp_path, frame_indices=(10, 18, 19))

    with pytest.raises(ValueError, match="not stride-4"):
        audit_sidecar(
            output,
            expected_tasks=1,
            expected_episodes=1,
            frame_stride=4,
        )

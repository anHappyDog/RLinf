# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the compatibility view consumed by B1K's subtask evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rlinf.data.b1k_grounded import GroundedControlSpec, Role


def prepare_subtask_evaluation_view(
    source_dataset_dir: str | Path,
    sidecar_path: str | Path,
    output_dir: str | Path,
    *,
    task_index: int,
    task_name: str,
) -> Path:
    """Materialize evaluator annotations while reusing the source parquet data."""
    import pyarrow.parquet as pq

    source_dataset_dir = Path(source_dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source_data_dir = source_dataset_dir / "data"
    if not source_data_dir.is_dir():
        raise FileNotFoundError(f"B1K source data directory not found: {source_data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_link = output_dir / "data"
    if data_link.exists() or data_link.is_symlink():
        if not data_link.is_symlink() or data_link.resolve() != source_data_dir:
            raise ValueError(f"Evaluation data link has an unexpected target: {data_link}")
    else:
        os.symlink(source_data_dir, data_link, target_is_directory=True)

    table = pq.read_table(
        Path(sidecar_path).expanduser(),
        columns=[
            "task_name",
            "episode_index",
            "segment_index",
            "interval_start",
            "interval_end",
            "skill",
            "control_json",
        ],
        filters=[("task_name", "=", task_name)],
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"No sidecar rows found for task {task_name!r}: {sidecar_path}")

    for row in rows:
        episode_index = int(row["episode_index"])
        segment_index = int(row["segment_index"])
        control = GroundedControlSpec.from_json(row["control_json"])
        object_names = [argument.category_name for argument in control.arguments]
        manipulated_ids = [
            argument.raw_object_id
            for argument in control.arguments
            if argument.role is Role.MANIPULATED and argument.raw_object_id is not None
        ]
        description = " ".join([row["skill"], *object_names]).strip()
        payload = {
            "start_frame": int(row["interval_start"]),
            "end_frame": int(row["interval_end"]),
            "skill_description": row["skill"],
            "cot_subtask_description": description,
            "manipulating_object_id": manipulated_ids,
        }
        annotation_dir = (
            output_dir
            / "orchestrators"
            / f"task-{task_index:04d}"
            / f"episode_{episode_index:08d}"
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = annotation_dir / f"subtask_{segment_index}_annotated.json"
        annotation_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output_dir

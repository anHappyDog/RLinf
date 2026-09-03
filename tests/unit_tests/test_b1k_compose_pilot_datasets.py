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

import numpy as np
import pytest

from toolkits.b1k_grounded.compose_pilot_datasets import (
    _interval_key,
    _merge_input_rows,
    make_action_chunk_boundary_safe,
)


def test_make_action_chunk_boundary_safe_repeats_and_masks_tail():
    actions = np.arange(20, dtype=np.float32).reshape(5, 4)
    row = {
        "sample_id": "sample-0",
        "frame_index": 7,
        "interval_end": 10,
        "actions": actions.tolist(),
        "action_is_pad": [False] * 5,
    }

    repaired, repaired_steps = make_action_chunk_boundary_safe(row)

    assert repaired_steps == 2
    assert repaired["actions"] == [
        actions[0].tolist(),
        actions[1].tolist(),
        actions[2].tolist(),
        actions[2].tolist(),
        actions[2].tolist(),
    ]
    assert repaired["action_is_pad"] == [False, False, False, True, True]


def test_make_action_chunk_boundary_safe_preserves_existing_padding():
    row = {
        "sample_id": "sample-0",
        "frame_index": 7,
        "interval_end": 12,
        "actions": np.zeros((5, 4), dtype=np.float32).tolist(),
        "action_is_pad": [False, False, False, True, True],
    }

    repaired, repaired_steps = make_action_chunk_boundary_safe(row)

    assert repaired_steps == 0
    assert repaired["action_is_pad"] == row["action_is_pad"]


def test_make_action_chunk_boundary_safe_rejects_out_of_interval_sample():
    row = {
        "sample_id": "sample-0",
        "frame_index": 7,
        "interval_end": 7,
        "actions": np.zeros((5, 4), dtype=np.float32).tolist(),
        "action_is_pad": [False] * 5,
    }

    with pytest.raises(ValueError, match="outside its skill interval"):
        make_action_chunk_boundary_safe(row)


def test_interval_key_distinguishes_annotation_intervals():
    row = {
        "task_index": 0,
        "episode_index": 10,
        "segment_index": 2,
        "interval_index": 1,
    }

    assert _interval_key(row) == (0, 10, 2, 1)


def test_interval_replacement_removes_midpoint_before_adding_dense_rows():
    midpoint = {
        "sample_id": "midpoint",
        "task_index": 0,
        "episode_index": 10,
        "segment_index": 2,
        "interval_index": 0,
    }
    other_interval = {
        "sample_id": "other",
        "task_index": 0,
        "episode_index": 10,
        "segment_index": 3,
        "interval_index": 0,
    }
    dense_rows = [
        {
            **midpoint,
            "sample_id": f"dense-{frame}",
            "frame_index": frame,
        }
        for frame in (100, 108, 116)
    ]
    rows_by_id = {row["sample_id"]: row for row in (midpoint, other_interval)}

    replaced = _merge_input_rows(
        rows_by_id,
        dense_rows,
        replacement_scope="interval",
    )

    assert replaced == 1
    assert set(rows_by_id) == {"other", "dense-100", "dense-108", "dense-116"}

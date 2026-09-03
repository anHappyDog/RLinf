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

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.data.datasets.openpi_rlinf.behavior.high_level_state import (
    BehaviorStateReader,
    compute_state_norm_stats,
    read_state_norm_stats,
    write_state_norm_stats,
)


@dataclasses.dataclass(frozen=True)
class _Entry:
    episode_index: int
    frame_index: int


def _make_dataset(tmp_path, states: np.ndarray) -> None:
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "task-0000"
    meta_dir.mkdir()
    data_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 10_000,
                "data_path": (
                    "data/task-{episode_chunk:04d}/"
                    "episode_{episode_index:08d}.parquet"
                ),
            }
        )
    )
    pq.write_table(
        pa.table({"observation.state": states.tolist()}),
        data_dir / "episode_00000010.parquet",
    )


def test_behavior_state_reader_extracts_official_23d_order(tmp_path):
    states = np.zeros((2, 256), dtype=np.float64)
    states[0, 253:256] = [1, 2, 3]
    states[0, 236:240] = [4, 5, 6, 7]
    states[0, 158:165] = np.arange(8, 15)
    states[0, 197:204] = np.arange(15, 22)
    states[0, 193:195] = [1.5, 2.5]
    states[0, 232:234] = [3.5, 4.5]
    _make_dataset(tmp_path, states)

    state = BehaviorStateReader(tmp_path).read(10, 0)

    np.testing.assert_allclose(
        state,
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            *range(8, 15),
            *range(15, 22),
            4,
            8,
        ],
    )


def test_state_stats_use_only_supplied_training_entries(tmp_path):
    states = np.zeros((4, 256), dtype=np.float64)
    for frame_index, value in enumerate((0.0, 1.0, 2.0, 1000.0)):
        states[frame_index, 253:256] = value
        states[frame_index, 236:240] = value
        states[frame_index, 158:165] = value
        states[frame_index, 197:204] = value
        states[frame_index, 193:195] = value / 2
        states[frame_index, 232:234] = value / 2
    _make_dataset(tmp_path, states)
    reader = BehaviorStateReader(tmp_path)

    stats = compute_state_norm_stats(
        reader,
        [_Entry(10, 0), _Entry(10, 1), _Entry(10, 2)],
    )

    assert stats.sample_count == 3
    assert np.all(stats.q99 < 3)
    np.testing.assert_allclose(
        stats.normalize(stats.q01), np.full(23, -1.0), atol=2e-5
    )
    np.testing.assert_allclose(
        stats.normalize(stats.q99), np.full(23, 1.0), atol=2e-5
    )


def test_state_stats_json_round_trip(tmp_path):
    states = np.zeros((3, 256), dtype=np.float64)
    states[1] = 1
    states[2] = 2
    _make_dataset(tmp_path, states)
    stats = compute_state_norm_stats(
        BehaviorStateReader(tmp_path),
        [_Entry(10, 0), _Entry(10, 1), _Entry(10, 2)],
    )
    stats_path = tmp_path / "state_norm_stats.json"

    write_state_norm_stats(stats, stats_path)
    restored = read_state_norm_stats(stats_path)

    assert restored.sample_count == stats.sample_count
    np.testing.assert_array_equal(restored.q01, stats.q01)
    np.testing.assert_array_equal(restored.q99, stats.q99)

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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.b1k_grounded import (
    STRUCTURAL_TOKENS,
    ControlProfile,
    GroundedControlSpec,
    ReservedTokenMapping,
    TokenBinding,
)
from rlinf.data.datasets.openpi_rlinf.behavior.grounded_sft_dataset import (
    EpisodeShardedSampler,
    GroundedBehaviorSftDataset,
)


def _write_sidecar(path, *, control_timestep: int = 7) -> None:
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal="Press the power button.",
        skill="press",
        arguments=(),
        episode_id="episode_00000000",
        segment_id=0,
        timestep=control_timestep,
    )
    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "sample-0",
                "frame_index": 7,
                "control_json": control.to_json(),
                "state": np.arange(256, dtype=np.float32).tolist(),
                "actions": np.zeros((32, 23), dtype=np.float32).tolist(),
                "action_is_pad": ([False] * 29) + ([True] * 3),
                "rgb_head_path": "videos/head.mp4",
                "rgb_left_wrist_path": "videos/left.mp4",
                "rgb_right_wrist_path": "videos/right.mp4",
            }
        ]
    )
    pq.write_table(table, path)


def _dataset(tmp_path, *, control_timestep: int = 7):
    sidecar_path = tmp_path / "sidecar.parquet"
    _write_sidecar(sidecar_path, control_timestep=control_timestep)
    return GroundedBehaviorSftDataset(
        dataset_root=tmp_path,
        sidecar_path=sidecar_path,
        token_mapping=ReservedTokenMapping(
            bindings=tuple(
                TokenBinding(token, f"<unused{index}>", index + 7)
                for index, token in enumerate(STRUCTURAL_TOKENS)
            )
        ),
        profile=ControlProfile.P1_SIMPLE_SG,
    )


def test_grounded_sft_dataset_reads_sidecar_and_serializes_prompt(
    tmp_path, monkeypatch
):
    dataset = _dataset(tmp_path)
    reads = []

    def fake_read(path, frame_index):
        reads.append((path.relative_to(tmp_path), frame_index))
        return np.full((8, 12, 3), frame_index, dtype=np.uint8)

    monkeypatch.setattr(dataset, "_read_rgb_frame", fake_read)

    item = dataset[0]

    assert item["sample_id"] == "sample-0"
    assert item["observation.state"].shape == (256,)
    assert item["observation.state"].dtype == np.float32
    assert item["action"].shape == (32, 23)
    assert item["action"].dtype == np.float32
    assert item["action_is_pad"].dtype == np.bool_
    assert item["action_is_pad"].tolist() == ([False] * 29) + ([True] * 3)
    assert item["observation.images.rgb.head"].shape == (8, 12, 3)
    assert item["prompt"].startswith("<unused0> Turn on the radio.")
    assert "<unused1> Press the power button." in item["prompt"]
    assert "<unused2> press" in item["prompt"]
    assert reads == [
        (tmp_path.joinpath("videos/head.mp4").relative_to(tmp_path), 7),
        (tmp_path.joinpath("videos/left.mp4").relative_to(tmp_path), 7),
        (tmp_path.joinpath("videos/right.mp4").relative_to(tmp_path), 7),
    ]


def test_grounded_sft_dataset_rejects_timestep_mismatch(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, control_timestep=8)
    monkeypatch.setattr(
        dataset,
        "_read_rgb_frame",
        lambda path, frame_index: np.zeros((8, 12, 3), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="frame/control timestep mismatch"):
        dataset[0]


def test_episode_sharded_sampler_keeps_episodes_on_one_rank():
    groups = (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11),
        (12, 13),
        (14,),
    )
    samplers = [
        EpisodeShardedSampler(
            groups,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            seed=42,
        )
        for rank in range(2)
    ]
    rank_indices = [list(sampler) for sampler in samplers]

    assert len(samplers[0]) == len(samplers[1])
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert set().union(*rank_indices) == set(range(15))
    for group in groups:
        owners = [
            rank for rank, values in enumerate(rank_indices) if set(group) & set(values)
        ]
        assert len(owners) <= 1


def test_episode_sharded_sampler_shuffles_episode_order_deterministically():
    groups = ((0, 1), (2, 3), (4, 5), (6, 7))
    sampler = EpisodeShardedSampler(
        groups,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
    )

    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    assert set(epoch_zero) == set(epoch_one) == set(range(8))
    for values in (epoch_zero, epoch_one):
        assert all(
            abs(values[index] - values[index + 1]) == 1 for index in range(0, 8, 2)
        )

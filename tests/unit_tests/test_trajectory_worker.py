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

import asyncio
from collections import defaultdict

import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
)
from rlinf.scheduler.channel.trajectory_channel.data import (
    EnvResult,
    RolloutResult,
    TrajectorySegment,
)
from rlinf.scheduler.channel.trajectory_channel.trajectory_worker import (
    TrajectoryWorker,
)


def test_bootstrap_reward_uses_worker_config():
    worker = object.__new__(TrajectoryWorker)
    worker._cfg = OmegaConf.create(
        {"env": {"train": {"auto_reset": True}}, "algorithm": {"gamma": 0.5}}
    )
    worker.env_reward_weight = 1.0
    worker.reward_weight = 1.0
    segment = TrajectorySegment(
        step_id=0,
        epoch_id=0,
        sources=[(0, 0, 1)],
        obs={"states": torch.zeros(1, 1)},
        next_obs={"states": torch.zeros(1, 1)},
        rollout_result=RolloutResult(
            actions=torch.zeros(1, 1),
            forward_inputs={"action": torch.zeros(1, 1)},
            bootstrap_values=torch.tensor([[4.0]]),
        ),
        env_result=EnvResult(
            rewards=torch.tensor([[1.0]]),
            dones=torch.tensor([[True]]),
            truncations=torch.tensor([[True]]),
        ),
    )

    assert torch.equal(worker._rewards(segment), torch.tensor([[3.0]]))


def test_flush_waits_for_every_producer():
    worker = object.__new__(TrajectoryWorker)
    worker._finished_sources = {3: {(0, 0)}}
    worker.producer_count = 2

    asyncio.run(worker._flush(3))

    assert worker._finished_sources == {3: {(0, 0)}}


def test_online_lerobot_flush_emits_one_item_per_shard():
    worker = object.__new__(TrajectoryWorker)
    worker._collectors = {
        (0, 0): EmbodiedLerobotTrajectoryBuilder(
            max_episode_length=4,
            num_envs=2,
            num_action_chunks=1,
            action_dim=1,
        )
    }
    worker._finished_sources = {3: {(0, 0)}}
    worker._output_queues = defaultdict(asyncio.Queue)
    worker.enable_online_lerobot = True
    worker.producer_count = 1
    worker.source_count = 1
    worker.shards_per_source = 2

    asyncio.run(worker._flush(3))

    assert worker._output_queues["default"].qsize() == 2
    assert 3 not in worker._finished_sources


def test_pipeline_flush_routes_micro_batches_and_cleans_epoch_state():
    class _Collector:
        def to_splited_trajectories_by_sizes(self, sizes):
            return [object() for _ in sizes]

    class _Placement:
        def get_world_size(self, component):
            assert component == "actor"
            return 1

    worker = object.__new__(TrajectoryWorker)
    key = (2, 1)
    worker._cfg = OmegaConf.create(
        {
            "env": {"train": {"total_num_envs": 2}},
            "rollout": {"pipeline_stage_num": 1},
            "algorithm": {"normalize_advantages": False},
        }
    )
    worker._placement = _Placement()
    worker._pipeline_collectors = {key: {(0, 0): _Collector()}}
    worker._finished_epoch_sources = {key: {(0, 0)}}
    worker._output_queues = defaultdict(asyncio.Queue)
    worker.producer_count = 1
    worker.source_count = 1
    worker._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    worker._pipeline_micro_batches = lambda batch, actor_rank: [batch]

    asyncio.run(worker._flush_pipeline(key))

    assert worker._output_queues["actor:0"].qsize() == 1
    assert key not in worker._pipeline_collectors
    assert key not in worker._finished_epoch_sources

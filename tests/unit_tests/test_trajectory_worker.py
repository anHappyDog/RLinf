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
from rlinf.data.schema.embodied_types import (
    EmbodiedRolloutResult,
    EnvResult,
    TrajectoryKey,
    TrajectorySource,
)
from rlinf.scheduler.channel.trajectory_channel.data import (
    EnvStepResult,
    PolicyStep,
    TerminalResult,
)
from rlinf.scheduler.channel.trajectory_channel.trajectory_worker import (
    TrajectoryWorker,
    _TrajectoryStep,
)


def test_bootstrap_reward_uses_worker_config():
    worker = object.__new__(TrajectoryWorker)
    worker._cfg = OmegaConf.create(
        {"env": {"train": {"auto_reset": True}}, "algorithm": {"gamma": 0.5}}
    )
    worker.env_reward_weight = 1.0
    worker.reward_weight = 1.0
    segment = _TrajectoryStep(
        step_id=0,
        epoch_id=0,
        obs={"states": torch.zeros(1, 1)},
        next_obs={"states": torch.zeros(1, 1)},
        rollout_result=EmbodiedRolloutResult(
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
    worker._completed_keys = set()
    worker.source_count = 2
    worker.chunk_count = 1
    worker._cfg = OmegaConf.create({"env": {"train": {"rollout_epoch": 1}}})

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
    worker.source_count = 1
    worker.shards_per_source = 2
    worker._completed_keys = {TrajectoryKey(3, 0, 0, 0, 0)}
    worker.chunk_count = 1
    worker._cfg = OmegaConf.create({"env": {"train": {"rollout_epoch": 1}}})

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
    worker.source_count = 1
    worker.chunk_count = 1
    worker._completed_keys = {TrajectoryKey(2, 1, 0, 0, 0)}
    worker._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    worker._pipeline_micro_batches = lambda batch, actor_rank: [batch]

    asyncio.run(worker._flush_pipeline(key))

    assert worker._output_queues["actor:0"].qsize() == 1
    assert key not in worker._pipeline_collectors
    assert key not in worker._finished_epoch_sources


def test_source_fragments_join_across_rollout_workers():
    worker = object.__new__(TrajectoryWorker)
    worker._cfg = OmegaConf.create(
        {"env": {"train": {"total_num_envs": 2, "auto_reset": True}}}
    )
    worker.source_count = 1
    worker.chunk_count = 2
    worker._fragments = defaultdict(list)
    worker._policy_steps = {}
    worker._env_results = {}
    worker._terminal_results = {}
    worker._completed_keys = set()
    worker._initial_results = {}
    worker.enable_online_lerobot = False
    worker.collect_prev_infos = False
    worker.use_training_pipeline = False
    worker._step_collectors = {}
    captured = []
    worker._collectors_for = lambda _: {}
    worker._append_one = lambda _, source, segment: captured.append((source, segment))

    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    worker._initial_results[(0, 0, 0, 0)] = EnvResult(
        dones=torch.zeros(2, 1, dtype=torch.bool)
    )

    def policy_fragment(key, offset, value):
        return PolicyStep(
            sources=[TrajectorySource(key, 1, offset)],
            obs={"states": torch.tensor([[value]])},
            rollout_result=EmbodiedRolloutResult(
                actions=torch.tensor([[value]]),
                forward_inputs={"action": torch.tensor([[value]])},
            ),
        )

    worker._store_event(policy_fragment(key0, 1, 11))
    worker._store_event(policy_fragment(key0, 0, 10))
    worker._store_event(
        EnvStepResult(
            sources=[TrajectorySource(key0, 2)],
            result=EnvResult(rewards=torch.ones(2, 1)),
        )
    )
    worker._store_event(policy_fragment(key1, 0, 20))
    worker._store_event(policy_fragment(key1, 1, 21))

    assert len(captured) == 1
    assert torch.equal(captured[0][1].obs["states"], torch.tensor([[10], [11]]))
    assert torch.equal(captured[0][1].next_obs["states"], torch.tensor([[20], [21]]))


def test_terminal_result_completes_final_transition():
    worker = object.__new__(TrajectoryWorker)
    worker._cfg = OmegaConf.create(
        {"env": {"train": {"total_num_envs": 1, "auto_reset": True}}}
    )
    worker.source_count = 1
    worker.chunk_count = 1
    worker._fragments = defaultdict(list)
    worker._policy_steps = {}
    worker._env_results = {}
    worker._terminal_results = {}
    worker._completed_keys = set()
    worker._initial_results = {(0, 0, 0, 0): EnvResult()}
    worker.enable_online_lerobot = False
    worker.collect_prev_infos = False
    worker.use_training_pipeline = False
    worker._step_collectors = {}
    captured = []
    worker._collectors_for = lambda _: {}
    worker._append_one = lambda _, source, segment: captured.append(segment)
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key, 1)

    worker._store_event(
        PolicyStep(
            sources=[source],
            obs={"states": torch.tensor([[1]])},
            rollout_result=EmbodiedRolloutResult(
                actions=torch.tensor([[2]]),
                forward_inputs={"action": torch.tensor([[2]])},
            ),
        )
    )
    worker._store_event(
        EnvStepResult(sources=[source], result=EnvResult(), needs_terminal=True)
    )
    worker._store_event(
        TerminalResult(
            sources=[source],
            obs={"states": torch.tensor([[3]])},
            bootstrap_values=torch.tensor([[4.0]]),
            forward_inputs={"states": torch.tensor([[3]])},
        )
    )

    assert len(captured) == 1
    assert torch.equal(captured[0].next_obs["states"], torch.tensor([[3]]))
    assert torch.equal(
        captured[0].rollout_result.bootstrap_values, torch.tensor([[4.0]])
    )


def test_online_lerobot_step_does_not_wait_for_terminal_result():
    worker = object.__new__(TrajectoryWorker)
    worker._cfg = OmegaConf.create(
        {"env": {"train": {"total_num_envs": 1, "auto_reset": True}}}
    )
    worker.source_count = 1
    worker.chunk_count = 1
    worker._fragments = defaultdict(list)
    worker._policy_steps = {}
    worker._env_results = {}
    worker._terminal_results = {}
    worker._completed_keys = set()
    worker._initial_results = {(0, 0, 0, 0): EnvResult()}
    worker.enable_online_lerobot = True
    worker.collect_prev_infos = False
    worker.use_training_pipeline = False
    worker._collectors = {}
    captured = []
    worker._collectors_for = lambda _: {}
    worker._append_one = lambda _, source, segment: captured.append(segment)
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key, 1)

    worker._store_event(
        PolicyStep(
            sources=[source],
            obs={"images": torch.zeros(1, 3, 2, 2)},
            rollout_result=EmbodiedRolloutResult(
                actions=torch.zeros(1, 1),
                forward_inputs={"action": torch.zeros(1, 1)},
            ),
        )
    )
    worker._store_event(
        EnvStepResult(
            sources=[source],
            result=EnvResult(
                episode_data={
                    "chunk_actions": torch.zeros(1, 1),
                    "obs_list": [{"states": torch.zeros(1, 1)}],
                    "terminations": torch.zeros(1, dtype=torch.bool),
                    "truncations": torch.zeros(1, dtype=torch.bool),
                    "infos_list": [{}],
                }
            ),
        )
    )

    assert len(captured) == 1
    assert key in worker._completed_keys

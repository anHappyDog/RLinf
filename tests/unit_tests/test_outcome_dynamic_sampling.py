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
import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.subtask import (
    outcome_actor_channel_key,
    outcome_group_is_trainable,
    parallel_outcome_sampling_enabled,
    reduce_first_episode_successes,
    reduce_trajectory_group_ids,
    reduce_trajectory_successes,
)
from rlinf.config import (
    _validate_critic_only,
    _validate_independent_gradient_clipping,
    _validate_outcome_dynamic_sampling,
)
from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
    EnvOutput,
    convert_trajectories_to_batch,
)
from rlinf.runners.embodied_runner import (
    EmbodiedRunner,
    _validate_outcome_reset_metadata,
)
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import _seed_rollout_sampling


class _ImmediateHandle:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


@pytest.mark.parametrize("use_training_pipeline", [False, True])
def test_embodied_runner_closes_environments_after_training(use_training_pipeline):
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"runner": {"use_training_pipeline": use_training_pipeline}}
    )
    runner._finish_run = MagicMock()
    runner._run_synchronous = MagicMock()
    runner.run_pipeline = MagicMock()
    close_handle = _ImmediateHandle()
    runner.env = MagicMock()
    runner.env.close_envs.return_value = close_handle

    runner.run()

    selected_loop = (
        runner.run_pipeline if use_training_pipeline else runner._run_synchronous
    )
    selected_loop.assert_called_once_with()
    runner._finish_run.assert_called_once_with()
    runner.env.close_envs.assert_called_once_with()
    assert close_handle.wait_count == 1


def test_embodied_runner_closes_environments_after_training_failure():
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"use_training_pipeline": False}})
    runner._finish_run = MagicMock()
    runner._run_synchronous = MagicMock(side_effect=RuntimeError("training failed"))
    close_handle = _ImmediateHandle()
    runner.env = MagicMock()
    runner.env.close_envs.return_value = close_handle

    with pytest.raises(RuntimeError, match="training failed"):
        runner.run()

    runner._finish_run.assert_called_once_with()
    runner.env.close_envs.assert_called_once_with()
    assert close_handle.wait_count == 1


def _dynamic_sampling_config():
    return OmegaConf.create(
        {
            "algorithm": {
                "outcome_dynamic_sampling": {
                    "enabled": True,
                    "group_size": 4,
                    "min_successes": 1,
                    "min_failures": 1,
                    "attempt_warning_interval": 2,
                }
            },
            "env": {
                "train": {
                    "rollout_epoch": 1,
                    "total_num_envs": 4,
                    "subpool": {"outcome_group_size": 4},
                }
            },
            "rollout": {"seed": 1234},
        }
    )


@pytest.mark.parametrize(
    ("outcome_shards", "expected_trainable", "expected_counts"),
    [
        ([[False], [True], [False], [True]], True, (2, 2)),
        ([[True], [True], [True], [True]], False, (4, 0)),
        ([[False], [False], [False], [False]], False, (0, 4)),
    ],
)
def test_outcome_group_requires_positive_and_negative_trajectories(
    outcome_shards, expected_trainable, expected_counts
):
    trainable, successes, failures = outcome_group_is_trainable(
        outcome_shards,
        expected_size=4,
        min_successes=1,
        min_failures=1,
    )

    assert trainable is expected_trainable
    assert (successes, failures) == expected_counts


def test_outcome_group_rejects_missing_rank_results():
    with pytest.raises(ValueError, match="Expected 4 rollout outcomes, received 3"):
        outcome_group_is_trainable(
            [[True], [False], [False]],
            expected_size=4,
            min_successes=1,
            min_failures=1,
        )


def test_dynamic_sampling_config_requires_stochastic_openpi_rollouts():
    cfg = _dynamic_sampling_config()
    stochastic_model = OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "openpi": {"noise_method": "flow_sde", "noise_level": 0.4},
        }
    )
    _validate_outcome_dynamic_sampling(cfg, stochastic_model)

    deterministic_model = OmegaConf.merge(
        stochastic_model,
        {"openpi": {"noise_method": "flow_ode", "noise_level": 0.0}},
    )
    with pytest.raises(AssertionError, match="positive flow_sde action noise"):
        _validate_outcome_dynamic_sampling(cfg, deterministic_model)


def test_dynamic_sampling_config_requires_positive_groups_per_update():
    cfg = _dynamic_sampling_config()
    cfg.algorithm.outcome_dynamic_sampling.groups_per_update = 0
    model = OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "openpi": {"noise_method": "flow_sde", "noise_level": 0.4},
        }
    )

    with pytest.raises(AssertionError, match="positive groups_per_update"):
        _validate_outcome_dynamic_sampling(cfg, model)


def test_parallel_dynamic_sampling_config_requires_one_env_per_worker():
    cfg = _dynamic_sampling_config()
    cfg.algorithm.outcome_dynamic_sampling.groups_per_update = 2
    cfg.algorithm.outcome_dynamic_sampling.parallel_groups = True
    cfg.env.train.total_num_envs = 8
    cfg.rollout.pipeline_stage_num = 1
    model = OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "openpi": {"noise_method": "flow_sde", "noise_level": 0.4},
        }
    )

    _validate_outcome_dynamic_sampling(
        cfg,
        model,
        env_world_size=8,
        actor_world_size=4,
    )
    with pytest.raises(AssertionError, match="one environment per EnvWorker"):
        _validate_outcome_dynamic_sampling(
            cfg,
            model,
            env_world_size=4,
            actor_world_size=4,
        )


def test_independent_gradient_clipping_requires_both_positive_limits():
    actor_cfg = OmegaConf.create(
        {
            "model": {"add_value_head": True},
            "optim": {"policy_clip_grad": 5.0, "value_clip_grad": 1.0},
            "fsdp_config": {"strategy": "fsdp", "use_orig_params": True},
        }
    )

    _validate_independent_gradient_clipping(actor_cfg)

    del actor_cfg.optim.value_clip_grad
    with pytest.raises(AssertionError, match="configured together"):
        _validate_independent_gradient_clipping(actor_cfg)


def test_critic_only_requires_value_head_and_actor_critic_loss():
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"add_value_head": True},
                "optim": {"critic_only": True},
                "enable_sft_co_train": False,
            },
            "algorithm": {"loss_type": "actor_critic"},
        }
    )
    _validate_critic_only(cfg)

    cfg.actor.model.add_value_head = False
    with pytest.raises(AssertionError, match="add_value_head"):
        _validate_critic_only(cfg)


def test_rollout_sampling_seed_is_reproducible_and_rank_offset():
    def sample(rank):
        _seed_rollout_sampling(1234, rank)
        return random.random(), np.random.random(), torch.rand(1).item()

    rank_zero = sample(0)

    assert sample(0) == rank_zero
    assert sample(1) != rank_zero


def test_reduce_trajectory_successes_uses_success_once_over_time():
    successes = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, False, True],
        ]
    )

    assert reduce_trajectory_successes(successes) == [True, False, True]


def test_auto_reset_outcomes_use_only_first_complete_episode():
    terminations = torch.tensor(
        [
            [False, False],
            [False, True],
            [True, False],
            [True, False],
        ]
    )
    dones = torch.tensor(
        [
            [[False], [False]],
            [[True], [True]],
            [[False], [False]],
            [[True], [True]],
        ]
    )

    assert reduce_first_episode_successes(terminations, dones) == [False, True]


def test_auto_reset_outcomes_accept_rollout_bootstrap_row():
    terminations = torch.tensor(
        [
            [[False], [False]],
            [[False], [False]],
            [[True], [False]],
            [[False], [False]],
        ]
    )
    dones = torch.tensor(
        [
            [[False], [False]],
            [[False], [False]],
            [[True], [True]],
            [[False], [False]],
        ]
    )

    assert reduce_first_episode_successes(terminations, dones) == [True, False]


def test_auto_reset_outcomes_require_a_complete_first_episode():
    with pytest.raises(ValueError, match="complete first episode"):
        reduce_first_episode_successes(
            torch.tensor([[False], [True]]),
            torch.tensor([[[False]], [[False]]]),
        )


def test_success_outcomes_survive_trajectory_split_and_merge():
    builder = EmbodiedTrajectoryBuilder()
    builder.append_step_result(
        ChunkStepResult(
            rewards=torch.zeros(2, 8),
            successes=torch.tensor([False, False]),
            outcome_group_ids=torch.tensor([0, 1]),
        )
    )
    builder.append_step_result(
        ChunkStepResult(
            rewards=torch.zeros(2, 8),
            successes=torch.tensor([True, False]),
            outcome_group_ids=torch.tensor([0, 1]),
        )
    )

    trajectories = builder.to_splited_trajectories(split_size=2)
    batch = convert_trajectories_to_batch(trajectories)

    assert torch.equal(
        batch["successes"],
        torch.tensor([[False, False], [True, False]]),
    )
    assert reduce_trajectory_successes(batch["successes"]) == [True, False]
    assert reduce_trajectory_group_ids(batch["outcome_group_ids"]) == [0, 1]
    assert outcome_actor_channel_key(3) == "outcome_actor_3"


def test_env_worker_extracts_cumulative_episode_outcomes():
    env_output = EnvOutput(
        obs={},
        env_infos={
            "episode": {
                "success": torch.tensor([False, False]),
                "success_once": torch.tensor([True, False]),
            }
        },
    )

    outcomes = EnvWorker._extract_success_outcomes(env_output)

    assert torch.equal(outcomes, torch.tensor([True, False]))


def test_env_worker_extracts_terminal_outcome_across_auto_reset():
    env_output = EnvOutput(
        obs={},
        env_infos={
            "episode": {"success_once": torch.tensor([False, False])},
            "final_info": {"episode": {"success_once": torch.tensor([True, False])}},
            "_final_info": torch.tensor([[True], [False]]),
        },
    )

    outcomes = EnvWorker._extract_success_outcomes(env_output)

    assert torch.equal(outcomes, torch.tensor([True, False]))


def test_env_worker_routes_parallel_groups_evenly_across_actor_ranks():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "outcome_dynamic_sampling": {
                    "enabled": True,
                    "parallel_groups": True,
                    "group_size": 4,
                }
            }
        }
    )
    worker._rank = 5
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.actor_split_num = 1
    builder = MagicMock()
    trajectory = MagicMock()
    builder.to_splited_trajectories.return_value = [trajectory]
    channel = MagicMock()
    send_trajectories = EnvWorker.send_rollout_trajectories
    while hasattr(send_trajectories, "__wrapped__"):
        send_trajectories = send_trajectories.__wrapped__

    asyncio.run(send_trajectories(worker, builder, channel, stage_id=0))

    channel.put.assert_called_once_with(
        trajectory,
        key="outcome_actor_1",
        async_op=True,
    )
    assert torch.equal(worker._outcome_group_ids(0), torch.tensor([1]))


def test_env_worker_forces_synchronized_reset_into_next_bootstrap():
    class FakeEnv:
        is_start = False

        def __init__(self):
            self.prepared_index = None
            self.prepared_logical_group = None
            self.prepared_update = None

        def prepare_outcome_group_reset(
            self, collection_index, logical_group_index, update_index
        ):
            self.prepared_index = collection_index
            self.prepared_logical_group = logical_group_index
            self.prepared_update = update_index

        @property
        def outcome_group_reset_metadata(self):
            return {
                "sampling_group": 0,
                "snapshot_id": "canonical-episode-50",
                "episode_index": 50,
                "subtask_id": 1,
                "pool_type": "canonical",
            }

        def reset(self):
            return {"states": torch.ones(1, 2)}, {"episode": {}}

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {"algorithm": {"outcome_dynamic_sampling": {"enabled": True, "group_size": 4}}}
    )
    worker._rank = 0
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.model_cfg = OmegaConf.create({"num_action_chunks": 32})
    worker.env_list = [FakeEnv()]
    worker.last_obs_list = []
    worker.last_intervened_info_list = []
    worker._prefetched_train_bootstrap = None
    worker._forced_train_bootstrap = None

    metadata = worker.reset_train_envs_for_outcome_group(7, [3], 11)

    assert worker.env_list[0].prepared_index == 7
    assert worker.env_list[0].prepared_logical_group == 3
    assert worker.env_list[0].prepared_update == 11
    assert metadata[0]["snapshot_id"] == "canonical-episode-50"
    assert metadata[0]["outcome_group_id"] == 0
    assert worker._forced_train_bootstrap[0].obs["states"].eq(1).all()
    assert worker.last_obs_list[0]["states"].eq(1).all()


def test_env_worker_uses_unkeyed_channel_when_sampling_is_disabled():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "outcome_dynamic_sampling": {
                    "enabled": False,
                    "parallel_groups": True,
                    "group_size": 4,
                }
            }
        }
    )
    worker._rank = 0
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.actor_split_num = 1
    builder = MagicMock()
    trajectory = MagicMock()
    builder.to_splited_trajectories.return_value = [trajectory]
    channel = MagicMock()
    send_trajectories = EnvWorker.send_rollout_trajectories
    while hasattr(send_trajectories, "__wrapped__"):
        send_trajectories = send_trajectories.__wrapped__

    asyncio.run(send_trajectories(worker, builder, channel, stage_id=0))

    channel.put.assert_called_once_with(trajectory, async_op=True)


def test_env_worker_propagates_policy_global_step_to_supported_envs():
    class RecordingEnv:
        def __init__(self):
            self.policy_global_step = None

        def set_policy_global_step(self, global_step):
            self.policy_global_step = global_step

    worker = object.__new__(EnvWorker)
    recording_env = RecordingEnv()
    worker.cfg = OmegaConf.create(
        {"env": {"train": {"subpool": {"failure_state_capture": {"enabled": True}}}}}
    )
    worker.env_list = [recording_env, object()]

    worker.set_global_step(34)

    assert recording_env.policy_global_step == 34

    with pytest.raises(ValueError, match="non-negative"):
        worker.set_global_step(-1)


def test_env_worker_skips_policy_step_rpc_when_failure_capture_is_disabled():
    env = MagicMock()
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {"env": {"train": {"subpool": {"failure_state_capture": {"enabled": False}}}}}
    )
    worker.env_list = [env]

    worker.set_global_step(34)

    env.set_policy_global_step.assert_not_called()


def _dynamic_sampling_runner(warning_interval=2, groups_per_update=1):
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "algorithm": {
                "outcome_dynamic_sampling": {
                    "enabled": True,
                    "group_size": 4,
                    "min_successes": 1,
                    "min_failures": 1,
                    "attempt_warning_interval": warning_interval,
                    "groups_per_update": groups_per_update,
                }
            }
        }
    )
    runner.logger = MagicMock()
    runner.actor = MagicMock()
    runner.env = MagicMock()
    reset_handle = MagicMock()
    reset_handle.wait.return_value = [
        [
            {
                "outcome_group_id": 0,
                "snapshot_id": "canonical-episode-50",
                "episode_index": 50,
                "subtask_id": 1,
                "pool_type": "canonical",
            }
        ]
        for _ in range(4)
    ]
    runner.env.reset_train_envs_for_outcome_group.return_value = reset_handle
    runner._outcome_collection_index = 0
    runner.global_step = 9
    return runner


def _expected_snapshot_metrics(candidate_groups, successes, failures, accepted_groups):
    prefix = "dynamic_sampling/snapshot/canonical-episode-50"
    return {
        f"{prefix}/episode_index": 50,
        f"{prefix}/candidate_groups": candidate_groups,
        f"{prefix}/candidate_successes": successes,
        f"{prefix}/candidate_failures": failures,
        f"{prefix}/candidate_success_rate": successes / (successes + failures),
        f"{prefix}/accepted_groups": accepted_groups,
    }


def test_runner_resamples_until_outcome_group_is_mixed():
    runner = _dynamic_sampling_runner()
    rejected_env = MagicMock()
    accepted_env = MagicMock()
    accepted_rollout = MagicMock()
    runner._collect_train_rollout = MagicMock(
        side_effect=[
            (rejected_env, MagicMock(), None, [[False]] * 4),
            (
                accepted_env,
                accepted_rollout,
                None,
                [[False], [True], [False], [True]],
            ),
        ]
    )

    env_handles, rollout_handles, _, metrics = runner._collect_trainable_rollout()

    assert env_handles == [accepted_env]
    assert rollout_handles == [accepted_rollout]
    rejected_env.wait.assert_called_once_with()
    runner.actor.begin_rollout_group_collection.assert_called_once_with()
    runner.actor.accept_rollout_group.assert_called_once_with()
    runner.actor.finalize_rollout_group_collection.assert_called_once_with(1)
    expected_metrics = {
        "dynamic_sampling/groups_per_update": 1,
        "dynamic_sampling/attempts": 2,
        "dynamic_sampling/rejected_groups": 1,
        "dynamic_sampling/successes": 2,
        "dynamic_sampling/failures": 2,
        "dynamic_sampling/candidate_successes": 2,
        "dynamic_sampling/candidate_failures": 6,
    }
    expected_metrics.update(_expected_snapshot_metrics(2, 2, 6, 1))
    assert metrics == expected_metrics
    assert runner.env.reset_train_envs_for_outcome_group.call_count == 2
    assert runner.env.reset_train_envs_for_outcome_group.call_args_list == [
        ((0, [0], 9),),
        ((1, [0], 9),),
    ]


def test_runner_keeps_single_rollout_behavior_when_sampling_is_disabled():
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"algorithm": {"outcome_dynamic_sampling": {"enabled": False}}}
    )
    handles = (MagicMock(), MagicMock(), None, None)
    runner._collect_train_rollout = MagicMock(return_value=handles)

    env_handles, rollout_handles, reward_handles, metrics = (
        runner._collect_trainable_rollout()
    )

    assert env_handles == [handles[0]]
    assert rollout_handles == [handles[1]]
    assert reward_handles == [None]
    assert metrics == {}


def test_disabled_sampling_does_not_enable_parallel_actor_routing():
    sampling_cfg = {
        "enabled": False,
        "parallel_groups": True,
    }

    assert not parallel_outcome_sampling_enabled(sampling_cfg)


def test_runner_collects_each_update_group_independently():
    runner = _dynamic_sampling_runner(groups_per_update=2)
    rejected_env = MagicMock()
    accepted_envs = [MagicMock(), MagicMock()]
    accepted_rollouts = [MagicMock(), MagicMock()]
    runner._collect_train_rollout = MagicMock(
        side_effect=[
            (rejected_env, MagicMock(), None, [[True]] * 4),
            (
                accepted_envs[0],
                accepted_rollouts[0],
                None,
                [[True], [False], [False], [False]],
            ),
            (
                accepted_envs[1],
                accepted_rollouts[1],
                None,
                [[True], [True], [False], [False]],
            ),
        ]
    )

    env_handles, rollout_handles, _, metrics = runner._collect_trainable_rollout()

    assert env_handles == accepted_envs
    assert rollout_handles == accepted_rollouts
    rejected_env.wait.assert_called_once_with()
    assert runner.actor.accept_rollout_group.call_count == 2
    runner.actor.finalize_rollout_group_collection.assert_called_once_with(2)
    expected_metrics = {
        "dynamic_sampling/groups_per_update": 2,
        "dynamic_sampling/attempts": 3,
        "dynamic_sampling/rejected_groups": 1,
        "dynamic_sampling/successes": 3,
        "dynamic_sampling/failures": 5,
        "dynamic_sampling/candidate_successes": 7,
        "dynamic_sampling/candidate_failures": 5,
    }
    expected_metrics.update(_expected_snapshot_metrics(3, 7, 5, 2))
    assert metrics == expected_metrics
    assert runner.env.reset_train_envs_for_outcome_group.call_args_list == [
        ((0, [0], 9),),
        ((1, [0], 9),),
        ((2, [1], 9),),
    ]


def test_runner_continues_sampling_after_warning_interval():
    runner = _dynamic_sampling_runner()
    rejected_envs = [MagicMock(), MagicMock()]
    accepted_env = MagicMock()
    runner._collect_train_rollout = MagicMock(
        side_effect=[
            (rejected_envs[0], MagicMock(), None, [[False]] * 4),
            (rejected_envs[1], MagicMock(), None, [[False]] * 4),
            (
                accepted_env,
                MagicMock(),
                None,
                [[True], [False], [False], [False]],
            ),
        ]
    )

    env_handles, _, _, metrics = runner._collect_trainable_rollout()

    assert env_handles == [accepted_env]
    assert metrics["dynamic_sampling/attempts"] == 3
    for env_handle in rejected_envs:
        env_handle.wait.assert_called_once_with()
    runner.logger.warning.assert_called_once()
    runner.actor.finalize_rollout_group_collection.assert_called_once_with(1)


def test_runner_collects_two_outcome_groups_in_parallel():
    runner = _dynamic_sampling_runner(groups_per_update=2)
    runner.cfg.algorithm.outcome_dynamic_sampling.parallel_groups = True
    runner.env.reset_train_envs_for_outcome_group.return_value.wait.return_value = [
        [
            {
                "outcome_group_id": rank // 4,
                "snapshot_id": "canonical-episode-50",
                "episode_index": 50,
                "subtask_id": 1,
                "pool_type": "canonical",
            }
        ]
        for rank in range(8)
    ]
    env_handles = [MagicMock(), MagicMock()]
    rollout_handles = [MagicMock(), MagicMock()]
    runner._collect_train_rollout = MagicMock(
        side_effect=[
            (
                env_handles[0],
                rollout_handles[0],
                None,
                [
                    {0: [True], 1: [True]},
                    {0: [False], 1: [True]},
                    {0: [False], 1: [True]},
                    {0: [False], 1: [True]},
                ],
            ),
            (
                env_handles[1],
                rollout_handles[1],
                None,
                [
                    {0: [False], 1: [True]},
                    {0: [False], 1: [True]},
                    {0: [False], 1: [False]},
                    {0: [False], 1: [False]},
                ],
            ),
        ]
    )

    accepted_env_handles, accepted_rollout_handles, _, metrics = (
        runner._collect_trainable_rollout()
    )

    assert [handle.group_ids for handle in accepted_env_handles] == [
        frozenset({0}),
        frozenset({1}),
    ]
    assert accepted_rollout_handles == rollout_handles
    assert runner.actor.accept_rollout_groups.call_args_list == [
        (([0],),),
        (([1],),),
    ]
    runner.actor.finalize_rollout_group_collection.assert_called_once_with(2)
    expected_metrics = {
        "dynamic_sampling/groups_per_update": 2,
        "dynamic_sampling/sampling_rounds": 2,
        "dynamic_sampling/attempts": 3,
        "dynamic_sampling/rejected_groups": 1,
        "dynamic_sampling/successes": 3,
        "dynamic_sampling/failures": 5,
        "dynamic_sampling/candidate_successes": 7,
        "dynamic_sampling/candidate_failures": 5,
    }
    expected_metrics.update(_expected_snapshot_metrics(3, 7, 5, 2))
    assert metrics == expected_metrics


def test_outcome_reset_metadata_rejects_desynchronized_group():
    metadata = [
        [
            {
                "outcome_group_id": 0,
                "snapshot_id": f"snapshot-{rank}",
                "episode_index": rank,
                "subtask_id": 1,
                "pool_type": "canonical",
            }
        ]
        for rank in range(4)
    ]

    with pytest.raises(RuntimeError, match="loaded different snapshots"):
        _validate_outcome_reset_metadata(
            metadata,
            expected_group_ids={0},
            group_size=4,
        )


def test_actor_merges_accepted_groups_before_batch_processing():
    actor = object.__new__(EmbodiedFSDPActor)
    actor._process_received_rollout_batch = MagicMock(side_effect=lambda batch: batch)
    actor.begin_rollout_group_collection()

    actor._candidate_rollout_batch = {
        "successes": torch.tensor([[True], [False]]),
        "forward_inputs": {"state": torch.tensor([[[1.0]], [[2.0]]])},
    }
    actor.accept_rollout_group()
    actor._candidate_rollout_batch = {
        "successes": torch.tensor([[False], [True]]),
        "forward_inputs": {"state": torch.tensor([[[3.0]], [[4.0]]])},
    }
    actor.accept_rollout_group()

    actor.finalize_rollout_group_collection(expected_groups=2)

    assert torch.equal(
        actor.rollout_batch["successes"],
        torch.tensor([[True, False], [False, True]]),
    )
    assert actor.rollout_batch["forward_inputs"]["state"].shape == (2, 2, 1)
    assert actor._accepted_rollout_batches is None
    actor._process_received_rollout_batch.assert_called_once()


def test_actor_builds_auto_reset_subtask_loss_mask():
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "reward_type": "subtask_chunk_level",
                "filter_rewards": False,
                "group_size": 1,
            },
            "env": {
                "train": {
                    "rollout_epoch": 1,
                    "auto_reset": True,
                    "ignore_terminations": False,
                }
            },
        }
    )
    executed_action_mask = torch.tensor(
        [
            [[True, True, False, False]],
            [[True, True, True, True]],
        ]
    )

    batch = actor._process_received_rollout_batch(
        {
            "rewards": torch.ones(2, 1, 4),
            "dones": torch.zeros(3, 1, 4, dtype=torch.bool),
            "executed_action_mask": executed_action_mask,
            "subtask_ids": torch.zeros(2, 1, dtype=torch.long),
        }
    )

    assert torch.equal(batch["loss_mask"], torch.ones(2, 1, 1, dtype=torch.bool))
    assert torch.equal(batch["loss_mask_sum"], torch.full((2, 1, 1), 2))
    assert torch.equal(batch["sample_weights"], torch.ones(2, 1, 1))


def test_actor_selects_independent_groups_from_parallel_candidate_batches():
    actor = object.__new__(EmbodiedFSDPActor)
    actor._process_received_rollout_batch = MagicMock(side_effect=lambda batch: batch)
    actor.begin_rollout_group_collection()
    actor._candidate_rollout_batch = {
        "successes": torch.tensor([[True, False], [True, False]]),
        "outcome_group_ids": torch.tensor([[0, 1], [0, 1]]),
        "forward_inputs": {"state": torch.tensor([[[0.0], [1.0]]] * 2)},
    }
    actor.accept_rollout_groups([1])
    actor._candidate_rollout_batch = {
        "successes": torch.tensor([[False, True], [False, True]]),
        "outcome_group_ids": torch.tensor([[0, 1], [0, 1]]),
        "forward_inputs": {"state": torch.tensor([[[2.0], [3.0]]] * 2)},
    }
    actor.accept_rollout_groups([0])

    actor.finalize_rollout_group_collection(expected_groups=2)

    assert torch.equal(
        actor.rollout_batch["outcome_group_ids"],
        torch.tensor([[1, 0], [1, 0]]),
    )
    assert torch.equal(
        actor.rollout_batch["forward_inputs"]["state"],
        torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]]),
    )

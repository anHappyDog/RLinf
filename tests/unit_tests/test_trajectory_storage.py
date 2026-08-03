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
import threading
from collections import defaultdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    EnvResult,
    LeRobotStepResult,
    RewardResult,
    RolloutResult,
    ValueResult,
)
from rlinf.scheduler.channel.trajectory_channel.owner_key import (
    BatchKey,
    LeRobotOwnerKey,
    PipelineBatchKey,
)
from rlinf.scheduler.channel.trajectory_channel.storage import (
    LeRobotEpisodeBatch,
    LeRobotStorageContext,
    LeRobotTrajectoryStorage,
    PipelineStorageContext,
    PipelineTrajectoryStorage,
    RolloutTrajectoryStorage,
    TrajectoryBatchContext,
    create_embodied_progress,
    get_progress_factory,
)
from rlinf.scheduler.channel.trajectory_channel.workers import (
    TrajectoryControllerWorker,
)
from rlinf.scheduler.manager import WorkerAddress
from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_adv


def _record_args(actor_rank: int) -> dict:
    return {
        "global_step": 7,
        "actor_rank": actor_rank,
        "pipeline_stage": 0,
        "rollout_epoch": 0,
        "chunk_step": 0,
        "slot_ids": (0, 1),
    }


@pytest.mark.parametrize(
    "algorithm",
    ("ppo", "grpo", "sac", "dsrl", "dagger", "nft", "rlt_ac", "opd"),
)
def test_algorithm_types_share_the_embodied_progress_factory(algorithm: str) -> None:
    assert get_progress_factory(algorithm) is create_embodied_progress


def test_storage_keeps_actor_batches_separate() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
            ),
        )
        flags = torch.zeros(2, 1, dtype=torch.bool)
        for actor_rank in (3, 5):
            args = _record_args(actor_rank)
            owner_key = BatchKey(
                global_step=args["global_step"],
                actor_rank=actor_rank,
            )
            await storage.record(
                EnvResult(
                    **args,
                    rewards=torch.ones(2, 1),
                    dones=flags,
                    terminations=flags,
                    truncations=flags,
                ),
                owner_key,
            )
            await storage.record(
                RolloutResult(
                    **args,
                    actions=torch.full((2, 1), actor_rank),
                ),
                owner_key,
            )

        batches = (await storage.take(), await storage.take())
        assert {batch.actor_rank for batch in batches} == {3, 5}

    asyncio.run(run())


def test_storage_aggregates_pipeline_stages() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1, 2, 3),
            ),
        )
        owner_key = BatchKey(global_step=7, actor_rank=0)
        flags = torch.zeros(2, 1, dtype=torch.bool)
        for pipeline_stage, slot_ids in enumerate(((0, 1), (2, 3))):
            args = {
                **_record_args(actor_rank=0),
                "pipeline_stage": pipeline_stage,
                "slot_ids": slot_ids,
            }
            await storage.record(
                EnvResult(
                    **args,
                    rewards=torch.ones(2, 1),
                    dones=flags,
                    terminations=flags,
                    truncations=flags,
                ),
                owner_key,
            )
            await storage.record(
                RolloutResult(**args, actions=torch.ones(2, 1)), owner_key
            )

        batch = await storage.take()
        assert batch.slot_ids == (0, 1, 2, 3)
        assert batch.actions is not None
        assert batch.actions.shape[:3] == (1, 1, 4)

    asyncio.run(run())


def test_pipeline_storage_emits_normalized_micro_batches_per_epoch() -> None:
    async def record_epoch(
        storage: PipelineTrajectoryStorage,
        rollout_epoch: int,
    ) -> None:
        owner_key = PipelineBatchKey(
            global_step=7,
            rollout_epoch=rollout_epoch,
            actor_rank=0,
        )
        flags = torch.zeros(2, 1, dtype=torch.bool)
        for slot_ids, reward, marker in (((0, 1), 1.0, 10.0), ((2, 3), 3.0, 20.0)):
            args = {
                **_record_args(actor_rank=0),
                "rollout_epoch": rollout_epoch,
                "slot_ids": slot_ids,
            }
            await storage.record(
                EnvResult(
                    **args,
                    rewards=torch.full((2, 1), reward),
                    dones=flags,
                    terminations=flags,
                    truncations=flags,
                ),
                owner_key,
            )
            await storage.record(
                RolloutResult(
                    **args,
                    actions=torch.full((2, 1), marker),
                    prev_logprobs=torch.zeros(2, 1),
                    state_values=torch.zeros(2, 1),
                ),
                owner_key,
            )
            await storage.record(
                ValueResult(
                    **{**args, "chunk_step": 1},
                    kind="boundary",
                    values=torch.zeros(2, 1),
                ),
                owner_key,
            )

    async def take_epoch(storage: PipelineTrajectoryStorage) -> list:
        return [await storage.take() for _ in range(4)]

    async def run() -> None:
        cfg = OmegaConf.create(
            {
                "runner": {"task_type": "embodied", "use_training_pipeline": True},
                "env": {
                    "train": {
                        "auto_reset": True,
                        "ignore_terminations": False,
                    }
                },
                "algorithm": {
                    "adv_type": "gae",
                    "reward_type": "action_level",
                    "group_size": 1,
                    "normalize_advantages": True,
                    "shuffle_rollout": False,
                },
                "actor": {
                    "seed": 0,
                    "micro_batch_size": 1,
                    "model": {"num_action_chunks": 1},
                },
                "reward": {},
            }
        )
        storage = PipelineTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1, 2, 3),
                collect_values=True,
            ),
            PipelineStorageContext(
                total_num_envs=4,
                actor_world_size=1,
                env_world_size=2,
                stage_num=1,
            ),
            cfg,
        )

        await record_epoch(storage, rollout_epoch=0)
        first_epoch = await take_epoch(storage)
        assert {batch.rollout_epoch for batch in first_epoch} == {0}
        assert [batch.is_last for batch in first_epoch] == [False, False, False, True]

        advantages = {
            batch.batch["actions"].item(): batch.batch["advantages"].item()
            for batch in first_epoch
        }
        assert advantages[10.0] < 0
        assert advantages[20.0] > 0

        await record_epoch(storage, rollout_epoch=1)
        second_epoch = await take_epoch(storage)
        assert {batch.rollout_epoch for batch in second_epoch} == {1}

    asyncio.run(run())


def test_trajectory_batch_converts_to_training_layout() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=2,
                slot_ids=(0, 1),
                collect_values=True,
            ),
        )
        owner_key = BatchKey(global_step=7, actor_rank=0)
        flags = torch.zeros(2, 1, dtype=torch.bool)
        for chunk_step in range(2):
            args = {
                **_record_args(actor_rank=0),
                "chunk_step": chunk_step,
            }
            await storage.record(
                EnvResult(
                    **args,
                    rewards=torch.ones(2, 2),
                    dones=flags,
                    terminations=flags,
                    truncations=flags,
                ),
                owner_key,
            )
            await storage.record(
                RolloutResult(
                    **args,
                    actions=torch.ones(2, 4),
                    forward_inputs={"action": torch.full((2, 3), 2.0)},
                    state_values=torch.ones(2, 4),
                ),
                owner_key,
            )
        await storage.record(
            ValueResult(
                **{**_record_args(actor_rank=0), "chunk_step": 2},
                kind="boundary",
                values=torch.ones(2, 1),
            ),
            owner_key,
        )

        stored_batch = await storage.take()
        cfg = OmegaConf.create(
            {
                "algorithm": {"gamma": 0.99},
                "reward": {},
                "env": {"train": {"max_episode_steps": 2}},
            }
        )
        batch = stored_batch.to_training_batch(cfg)
        assert batch["rewards"].shape == (2, 2, 2)
        assert batch["dones"].shape == (3, 2, 1)
        assert batch["actions"].shape == (2, 2, 3)
        torch.testing.assert_close(batch["actions"], torch.full((2, 2, 3), 2.0))
        assert batch["prev_values"].shape == (3, 2, 4)
        torch.testing.assert_close(batch["prev_values"][-1, :, 0], torch.ones(2))
        torch.testing.assert_close(batch["prev_values"][-1, :, 1:], torch.zeros(2, 3))

        trajectory = stored_batch.to_trajectory(cfg)
        assert trajectory.max_episode_length == 2
        assert trajectory.model_weights_id
        torch.testing.assert_close(trajectory.actions, batch["actions"])

    asyncio.run(run())


def test_storage_does_not_wait_for_truncation_values_without_auto_reset() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0,),
                collect_values=True,
            ),
        )
        args = _record_args(actor_rank=0) | {"slot_ids": (0,)}
        owner_key = BatchKey(global_step=7, actor_rank=0)
        await storage.record(
            EnvResult(
                **args,
                rewards=torch.ones(1, 1),
                dones=torch.ones(1, 1, dtype=torch.bool),
                terminations=torch.zeros(1, 1, dtype=torch.bool),
                truncations=torch.ones(1, 1, dtype=torch.bool),
            ),
            owner_key,
        )
        await storage.record(RolloutResult(**args, actions=torch.ones(1, 1)), owner_key)

        batch = await storage.take()
        assert batch.truncation_values is None

    asyncio.run(run())


def test_storage_records_executed_intervention_actions() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "rlt_ac",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0,),
            ),
        )
        args = _record_args(actor_rank=0) | {"slot_ids": (0,)}
        owner_key = BatchKey(global_step=7, actor_rank=0)
        await storage.record(
            RolloutResult(
                **args,
                actions=torch.ones(1, 4),
                forward_inputs={"action": torch.ones(1, 4)},
                intervene_flags=torch.zeros(1, 2, dtype=torch.bool),
            ),
            owner_key,
        )
        await storage.record(
            EnvResult(
                **args,
                rewards=torch.ones(1, 1),
                dones=torch.zeros(1, 1, dtype=torch.bool),
                terminations=torch.zeros(1, 1, dtype=torch.bool),
                truncations=torch.zeros(1, 1, dtype=torch.bool),
                intervene_actions=torch.full((1, 4), 3.0),
                intervene_flags=torch.tensor([[False, True]]),
            ),
            owner_key,
        )

        batch = await storage.take()
        training_batch = batch.to_training_batch(
            OmegaConf.create({"algorithm": {"gamma": 0.99}, "reward": {}})
        )
        torch.testing.assert_close(
            training_batch["actions"], torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
        )
        torch.testing.assert_close(
            training_batch["forward_inputs"]["action"],
            training_batch["actions"],
        )
        assert training_batch["intervene_flags"].tolist() == [
            [[False, False, True, True]]
        ]

    asyncio.run(run())


def test_storage_preserves_epoch_reward_and_value_boundaries() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=2,
                chunk_steps=2,
                slot_ids=(0,),
                collect_values=True,
            ),
        )
        owner_key = BatchKey(global_step=7, actor_rank=0)
        flags = torch.zeros(1, 1, dtype=torch.bool)
        for rollout_epoch in range(2):
            for chunk_step in range(2):
                marker = 10 * rollout_epoch + chunk_step
                args = {
                    **_record_args(actor_rank=0),
                    "rollout_epoch": rollout_epoch,
                    "chunk_step": chunk_step,
                    "slot_ids": (0,),
                }
                await storage.record(
                    EnvResult(
                        **args,
                        rewards=torch.full((1, 1), marker),
                        dones=flags,
                        terminations=flags,
                        truncations=flags,
                    ),
                    owner_key,
                )
                await storage.record(
                    RolloutResult(
                        **args,
                        actions=torch.full((1, 1), marker),
                        prev_logprobs=torch.full((1, 1), marker),
                        state_values=torch.full((1, 1), marker),
                    ),
                    owner_key,
                )
            await storage.record(
                ValueResult(
                    **{
                        **_record_args(actor_rank=0),
                        "rollout_epoch": rollout_epoch,
                        "chunk_step": 2,
                        "slot_ids": (0,),
                    },
                    kind="boundary",
                    values=torch.full((1, 1), 100 + rollout_epoch),
                ),
                owner_key,
            )

        batch = (await storage.take()).to_training_batch(
            OmegaConf.create({"algorithm": {"gamma": 0.99}, "reward": {}})
        )
        batch = process_nested_dict_for_adv(batch, rollout_epoch=2)

        torch.testing.assert_close(
            batch["rewards"].squeeze(-1), torch.tensor([[0, 10], [1, 11]])
        )
        torch.testing.assert_close(
            batch["prev_values"].squeeze(-1),
            torch.tensor([[0, 10], [1, 11], [100, 101]]),
        )
        assert not batch["dones"].any()

    asyncio.run(run())


def test_storage_preserves_opd_inputs_for_actor_teacher_logprobs() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=2,
                chunk_steps=2,
                slot_ids=(0,),
            ),
        )
        owner_key = BatchKey(global_step=7, actor_rank=0)
        flags = torch.zeros(1, 1, dtype=torch.bool)
        for rollout_epoch in range(2):
            for chunk_step in range(2):
                marker = 10 * rollout_epoch + chunk_step
                args = {
                    **_record_args(actor_rank=0),
                    "rollout_epoch": rollout_epoch,
                    "chunk_step": chunk_step,
                    "slot_ids": (0,),
                }
                await storage.record(
                    EnvResult(
                        **args,
                        rewards=torch.ones(1, 1),
                        dones=flags,
                        terminations=flags,
                        truncations=flags,
                    ),
                    owner_key,
                )
                await storage.record(
                    RolloutResult(
                        **args,
                        actions=torch.full((1, 1, 2), marker),
                        forward_inputs={"action_tokens": torch.full((1, 1, 2), marker)},
                        prev_logprobs=torch.full((1, 1, 2), marker),
                    ),
                    owner_key,
                )

        batch = (await storage.take()).to_training_batch(
            OmegaConf.create({"algorithm": {"gamma": 0.99}, "reward": {}})
        )
        actor_batch = process_nested_dict_for_adv(batch, rollout_epoch=2)
        expected = torch.tensor(
            [
                [[[0, 0]], [[10, 10]]],
                [[[1, 1]], [[11, 11]]],
            ]
        )

        torch.testing.assert_close(actor_batch["prev_logprobs"], expected)
        torch.testing.assert_close(
            actor_batch["forward_inputs"]["action_tokens"], expected
        )

    asyncio.run(run())


def test_storage_serializes_records_for_the_same_owner() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
            ),
            num_record_threads=2,
        )
        original_record = storage._record_sync
        active = 0
        max_active = 0
        lock = threading.Lock()

        def track_record(owner_key, data):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                return original_record(owner_key, data)
            finally:
                with lock:
                    active -= 1

        storage._record_sync = track_record
        args = _record_args(actor_rank=3)
        owner_key = BatchKey(global_step=7, actor_rank=3)
        flags = torch.zeros(2, 1, dtype=torch.bool)
        await storage.record(
            EnvResult(
                **args,
                rewards=torch.ones(2, 1),
                dones=flags,
                terminations=flags,
                truncations=flags,
            ),
            owner_key,
        )
        await storage.record(RolloutResult(**args, actions=torch.ones(2, 1)), owner_key)

        await storage.take()
        assert max_active == 1

    asyncio.run(run())


def test_storage_records_different_owners_in_parallel() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
            ),
            num_record_threads=2,
        )
        original_record = storage._record_sync
        rendezvous = threading.Barrier(2)
        completed = threading.Event()
        completed_count = 0
        lock = threading.Lock()

        def track_record(owner_key, data):
            nonlocal completed_count
            rendezvous.wait(timeout=2)
            result = original_record(owner_key, data)
            with lock:
                completed_count += 1
                if completed_count == 2:
                    completed.set()
            return result

        storage._record_sync = track_record
        flags = torch.zeros(2, 1, dtype=torch.bool)
        for actor_rank in (3, 5):
            owner_key = BatchKey(
                global_step=7,
                actor_rank=actor_rank,
            )
            await storage.record(
                EnvResult(
                    **_record_args(actor_rank),
                    rewards=torch.ones(2, 1),
                    dones=flags,
                    terminations=flags,
                    truncations=flags,
                ),
                owner_key,
            )

        finished = await asyncio.wait_for(
            asyncio.to_thread(completed.wait, 3),
            timeout=4,
        )
        assert finished

    asyncio.run(run())


def test_storage_surfaces_background_writer_failure() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
            ),
        )
        args = _record_args(actor_rank=3)
        flags = torch.zeros(2, 1, dtype=torch.bool)
        record = EnvResult(
            **args,
            rewards=torch.ones(2, 1),
            dones=flags,
            terminations=flags,
            truncations=flags,
        )
        owner_key = BatchKey(global_step=7, actor_rank=3)
        await storage.record(record, owner_key)
        await storage.record(record, owner_key)

        with pytest.raises(
            ValueError,
            match="Received more record positions than expected",
        ):
            await storage.take()

    asyncio.run(run())


def test_terminal_reward_expectation_is_derived_from_env_results() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
                reward_mode="terminal",
            ),
        )
        args = _record_args(actor_rank=3)
        owner_key = BatchKey(global_step=7, actor_rank=3)
        dones = torch.tensor([[True], [False]])
        flags = torch.zeros(2, 1, dtype=torch.bool)
        await storage.record(
            EnvResult(
                **args,
                rewards=torch.ones(2, 1),
                dones=dones,
                terminations=dones,
                truncations=flags,
            ),
            owner_key,
        )
        await storage.record(RolloutResult(**args, actions=torch.ones(2, 1)), owner_key)
        await storage.record(
            RewardResult(
                **{**args, "slot_ids": (0,)},
                mode="terminal",
                rewards=torch.ones(1, 1),
            ),
            owner_key,
        )

        batch = await storage.take()
        assert batch.reward_mask is not None
        assert batch.reward_mask.sum().item() == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("reward_mode", "expected"),
    (
        ("per_step", torch.tensor([[[2.5, 2.5], [4.5, 4.5]]])),
        ("terminal", torch.tensor([[[0.5, 2.5], [1.0, 1.0]]])),
    ),
)
def test_trajectory_batch_merges_external_rewards(reward_mode, expected) -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
                reward_mode=reward_mode,
                reward_steps=() if reward_mode == "terminal" else (0,),
            ),
        )
        args = _record_args(actor_rank=0)
        owner_key = BatchKey(global_step=7, actor_rank=0)
        dones = torch.tensor([[False, True], [False, False]])
        await storage.record(
            EnvResult(
                **args,
                rewards=torch.ones(2, 2),
                dones=dones,
                terminations=dones,
                truncations=torch.zeros_like(dones),
            ),
            owner_key,
        )
        await storage.record(RolloutResult(**args, actions=torch.ones(2, 1)), owner_key)
        reward_args = args
        rewards = torch.tensor([[1.0], [2.0]])
        if reward_mode == "terminal":
            reward_args = {**args, "slot_ids": (0,)}
            rewards = rewards[:1]
        await storage.record(
            RewardResult(
                **reward_args,
                mode=reward_mode,
                rewards=rewards,
            ),
            owner_key,
        )

        batch = await storage.take()
        cfg = OmegaConf.create(
            {
                "algorithm": {"gamma": 0.99},
                "reward": {"env_reward_weight": 0.5, "reward_weight": 2.0},
            }
        )
        torch.testing.assert_close(batch.to_training_batch(cfg)["rewards"], expected)

    asyncio.run(run())


def test_history_buffer_rewards_are_assigned_to_previous_steps() -> None:
    async def run() -> None:
        storage = RolloutTrajectoryStorage(
            "ppo",
            TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=3,
                slot_ids=(0, 1),
                reward_mode="history_buffer",
                reward_steps=(0, 1, 2),
            ),
        )
        owner_key = BatchKey(global_step=7, actor_rank=0)
        flags = torch.zeros(2, 1, dtype=torch.bool)
        external_rewards = ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0))
        history_lengths = ((1, 1), (2, 1), (3, 2))
        for chunk_step in range(3):
            args = {**_record_args(actor_rank=0), "chunk_step": chunk_step}
            await storage.record(
                EnvResult(
                    **args,
                    rewards=torch.zeros(2, 1),
                    dones=flags,
                    terminations=flags,
                    truncations=flags,
                ),
                owner_key,
            )
            await storage.record(
                RolloutResult(**args, actions=torch.ones(2, 1)), owner_key
            )
            await storage.record(
                RewardResult(
                    **args,
                    mode="history_buffer",
                    rewards=torch.tensor(external_rewards[chunk_step]).unsqueeze(-1),
                    history_lengths=torch.tensor(history_lengths[chunk_step]),
                ),
                owner_key,
            )

        batch = await storage.take()
        cfg = OmegaConf.create(
            {
                "algorithm": {"gamma": 0.99},
                "reward": {
                    "env_reward_weight": 0.0,
                    "reward_weight": 1.0,
                    "history_reward_assign": True,
                },
            }
        )
        expected = torch.tensor([[[6.0], [10.0]], [[5.0], [50.0]], [[3.0], [30.0]]])
        torch.testing.assert_close(batch.to_training_batch(cfg)["rewards"], expected)

    asyncio.run(run())


def test_controller_balances_batch_owners() -> None:
    controller = object.__new__(TrajectoryControllerWorker)
    controller._owners = {}
    controller._active_owners = defaultdict(int)
    controller._owner_cursor = 0
    workers = tuple(WorkerAddress("storage", rank) for rank in range(2))
    keys = tuple(BatchKey(global_step=step, actor_rank=0) for step in range(3))

    first = controller.claim_storage_worker(keys[0], (), workers)
    second = controller.claim_storage_worker(keys[1], (), workers)
    assert first.rank != second.rank
    assert controller.claim_storage_worker(keys[0], (), workers).rank == first.rank

    controller.release_storage_worker(keys[0])
    third = controller.claim_storage_worker(keys[2], (), workers)
    assert third.rank == first.rank


def test_lerobot_episode_batch_round_trip() -> None:
    episodes = [
        [
            {
                "state": np.arange(4, dtype=np.float32),
                "actions": np.ones(2, dtype=np.float32),
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "task": "pick",
                "done": np.array([True]),
            }
        ]
    ]
    batch = LeRobotEpisodeBatch.from_episodes(
        global_step=3,
        actor_rank=1,
        episodes=episodes,
    )
    skeleton, tensors = batch.flatten()
    restored = LeRobotEpisodeBatch.from_flattened(skeleton, tensors)
    restored_episodes = restored.to_numpy_episodes()

    assert restored.actor_rank == 1
    assert np.array_equal(restored_episodes[0][0]["state"], episodes[0][0]["state"])
    assert restored_episodes[0][0]["task"] == "pick"


def test_lerobot_storage_keeps_stream_across_global_steps() -> None:
    async def run() -> None:
        storage = LeRobotTrajectoryStorage(
            LeRobotStorageContext(
                only_success=False,
                num_action_chunks=1,
                action_dim=2,
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0,),
            )
        )
        owner_key = LeRobotOwnerKey(actor_rank=1)

        for global_step, terminated in ((3, False), (4, True)):
            await storage.record(
                LeRobotStepResult(
                    global_step=global_step,
                    actor_rank=1,
                    pipeline_stage=0,
                    env_rank=0,
                    rollout_epoch=0,
                    chunk_step=0,
                    slot_ids=(0,),
                    chunk_actions=torch.ones(1, 1, 2),
                    observations=[
                        {
                            "states": torch.ones(1, 3),
                            "task_descriptions": ["pick"],
                        }
                    ],
                    terminations=torch.tensor([[terminated]]),
                    truncations=torch.zeros(1, 1, dtype=torch.bool),
                    env_infos=[{}],
                ),
                owner_key,
            )

            batch = await storage.take()
            assert batch.global_step == global_step
            if not terminated:
                assert batch.episodes == []

        assert len(batch.episodes) == 1
        assert len(batch.episodes[0]) == 2

    asyncio.run(run())


def test_lerobot_storage_aggregates_pipeline_stages() -> None:
    async def run() -> None:
        storage = LeRobotTrajectoryStorage(
            LeRobotStorageContext(
                only_success=False,
                num_action_chunks=1,
                action_dim=2,
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
            )
        )
        owner_key = LeRobotOwnerKey(actor_rank=0)
        for pipeline_stage, slot_id in enumerate((0, 1)):
            await storage.record(
                LeRobotStepResult(
                    global_step=3,
                    actor_rank=0,
                    pipeline_stage=pipeline_stage,
                    env_rank=0,
                    rollout_epoch=0,
                    chunk_step=0,
                    slot_ids=(slot_id,),
                    chunk_actions=torch.ones(1, 1, 2),
                    observations=[
                        {
                            "states": torch.ones(1, 3),
                            "task_descriptions": ["pick"],
                        }
                    ],
                    terminations=torch.zeros(1, 1, dtype=torch.bool),
                    truncations=torch.zeros(1, 1, dtype=torch.bool),
                    env_infos=[{}],
                ),
                owner_key,
            )

        batch = await storage.take()
        assert batch.global_step == 3
        assert batch.actor_rank == 0
        assert batch.episodes == []

    asyncio.run(run())

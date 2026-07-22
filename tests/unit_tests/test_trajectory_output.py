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

import asyncio
import time

import pytest
import torch

from rlinf.data.trajectory import EnvResult, RolloutResult
from rlinf.models.embodiment.openpi.forward_inputs import OpenPILiberoForwardInputs
from rlinf.workers.trajectory import (
    RoutePlan,
    StorageConfig,
    TrajectoryBatch,
    TrajectoryReader,
    TrajectoryStorage,
    TrajectoryStorageWorker,
    TransportAck,
    WorkerLayout,
    WorkerState,
    flatten_trajectory,
    merge_trajectory_batches,
    restore_trajectory,
    select_trajectory_batch,
)
from rlinf.workers.trajectory.transport import ReceiveResult


def _batch(slot_ids: tuple[int, ...] = (0, 1, 2, 3)) -> TrajectoryBatch:
    batch_size = len(slot_ids)
    values = torch.tensor(slot_ids, dtype=torch.float32).reshape(1, 1, batch_size, 1)
    flat = values.flatten(0, 2)
    inputs = OpenPILiberoForwardInputs(
        chains=flat[:, None, None, :].expand(batch_size, 5, 2, 1).clone(),
        denoise_inds=torch.arange(4).expand(batch_size, 4).clone(),
        tokenized_prompt=torch.arange(3).expand(batch_size, 3).clone(),
        tokenized_prompt_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        action=flat.expand(batch_size, 7).clone(),
        model_action=(flat + 1).expand(batch_size, 7).clone(),
        image=torch.zeros(batch_size, 2, 2, 3, dtype=torch.uint8),
        wrist_image=torch.ones(batch_size, 2, 2, 3, dtype=torch.uint8),
        state=flat.expand(batch_size, 8).clone(),
    )
    return TrajectoryBatch(
        global_step=3,
        slot_ids=slot_ids,
        env_rewards=values,
        dones=torch.zeros_like(values, dtype=torch.bool),
        terminations=torch.zeros_like(values, dtype=torch.bool),
        truncations=torch.zeros_like(values, dtype=torch.bool),
        actions=values.expand(1, 1, batch_size, 7).clone(),
        observations={
            "states": values.expand(1, 1, batch_size, 8).clone(),
            "task_descriptions": [[[f"slot {slot}" for slot in slot_ids]]],
        },
        forward_inputs=inputs,
        state_values=values / 10,
        tail_values=values[:, 0] / 100,
        tail_mask=torch.ones(1, batch_size, dtype=torch.bool),
    )


def test_actor_shards_select_and_merge_in_requested_slot_order() -> None:
    complete = _batch()
    storage_zero = select_trajectory_batch(complete, (0, 1))
    storage_one = select_trajectory_batch(complete, (2, 3))
    contributions = (
        select_trajectory_batch(storage_one, (0,)),
        select_trajectory_batch(storage_zero, (1,)),
    )

    actor = merge_trajectory_batches(contributions, (1, 2))

    assert actor.slot_ids == (1, 2)
    assert actor.actions[0, 0, :, 0].tolist() == [1, 2]
    assert actor.observations["task_descriptions"] == [[["slot 1", "slot 2"]]]
    assert actor.forward_inputs is not None
    assert actor.forward_inputs.action[:, 0].tolist() == [1, 2]
    assert torch.allclose(actor.tail_values[:, :, 0], torch.tensor([[0.01, 0.02]]))


def test_trajectory_tensor_dict_round_trip_has_tensor_free_skeleton() -> None:
    expected = _batch()

    tensors, skeleton = flatten_trajectory(expected)
    actual = restore_trajectory(tensors, skeleton)

    assert tensors
    assert all(isinstance(key, str) for key in tensors)
    assert all(isinstance(value, torch.Tensor) for value in tensors.values())
    _assert_tensor_free(skeleton)
    assert actual.slot_ids == expected.slot_ids
    assert torch.equal(actual.actions, expected.actions)
    assert torch.equal(actual.observations["states"], expected.observations["states"])
    assert (
        actual.observations["task_descriptions"]
        == expected.observations["task_descriptions"]
    )
    assert type(actual.forward_inputs) is type(expected.forward_inputs)
    for (_, actual_value), (_, expected_value) in zip(
        actual.forward_inputs.tensor_fields(),
        expected.forward_inputs.tensor_fields(),
        strict=True,
    ):
        assert torch.equal(actual_value, expected_value)


def test_merge_rejects_overlap_and_incomplete_actor_coverage() -> None:
    batch = _batch((0, 1))
    with pytest.raises(ValueError, match="overlapping"):
        merge_trajectory_batches((batch, batch), (0, 1))
    with pytest.raises(ValueError, match="exactly cover"):
        merge_trajectory_batches((batch,), (0, 1, 2))


def test_storage_worker_exports_only_requested_actor_slots() -> None:
    storage = TrajectoryStorage(
        StorageConfig(
            global_step=3,
            rollout_epochs=1,
            chunk_steps=1,
            slot_ids=(0, 1),
        )
    )
    storage.write(
        EnvResult(
            global_step=3,
            rollout_epoch=0,
            chunk_step=0,
            slot_ids=(0, 1),
            rewards=torch.tensor([[0.0], [1.0]]),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            terminations=torch.zeros(2, 1, dtype=torch.bool),
            truncations=torch.zeros(2, 1, dtype=torch.bool),
        )
    )
    storage.write(
        RolloutResult(
            global_step=3,
            rollout_epoch=0,
            chunk_step=0,
            slot_ids=(0, 1),
            actions=torch.tensor([[0.0], [1.0]]),
        )
    )
    worker = object.__new__(TrajectoryStorageWorker)
    worker._state = WorkerState.READY
    worker._storage = storage
    worker._route_plan = RoutePlan(2, {"storage": 1, "actor": 2})
    worker._participant = "storage"
    worker._logical_rank = 0
    worker._pending_consumers = set()

    shard = worker.pull_actor_shard_via_ray(actor_rank=1)

    assert shard.slot_ids == (1,)
    assert shard.actions[0, 0, 0, 0].item() == 1


def test_received_frames_are_not_pullable_until_ingested() -> None:
    storage = TrajectoryStorage(
        StorageConfig(
            global_step=3,
            rollout_epochs=1,
            chunk_steps=1,
            slot_ids=(0,),
        )
    )
    env = EnvResult(
        global_step=3,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(0,),
        rewards=torch.tensor([[1.0]]),
        dones=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
    )
    rollout = RolloutResult(
        global_step=3,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(0,),
        actions=torch.tensor([[2.0]]),
    )
    worker = object.__new__(TrajectoryStorageWorker)
    worker._state = WorkerState.READY
    worker._storage = storage
    worker._route_plan = RoutePlan(1, {"storage": 1, "actor": 1})
    worker._participant = "storage"
    worker._logical_rank = 0
    worker._failure = None
    worker._pending_consumers = set()
    worker._generation_buffers = []
    worker._active_frames = 2
    worker._capacity_changed = asyncio.Condition()
    worker._metrics = {
        "frames_ingested": 0,
        "frames_rejected": 0,
        "ingest_seconds": 0.0,
    }

    async def run() -> None:
        worker._ingest_queue = asyncio.Queue(2)
        await worker._ingest_queue.put(
            (
                ReceiveResult(env, TransportAck(1, 0), duplicate=False),
                None,
                (1, "env:0"),
                None,
                0.0,
                time.monotonic(),
            )
        )
        await worker._ingest_queue.put(
            (
                ReceiveResult(rollout, TransportAck(2, 0), duplicate=False),
                None,
                (2, "rollout:0"),
                None,
                0.0,
                time.monotonic(),
            )
        )
        with pytest.raises(RuntimeError, match="not ready"):
            worker.pull_actor_shard_via_ray(actor_rank=0)

        task = asyncio.create_task(worker._ingest_loop())
        await worker._ingest_queue.join()
        shard = worker.pull_actor_shard_via_ray(actor_rank=0)
        assert shard.actions[0, 0, 0, 0].item() == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


class _RemoteCall:
    def __init__(self, value: TrajectoryBatch) -> None:
        self._value = value

    def remote(self, *_args):
        return self._value


class _StorageActor:
    def __init__(self, value: TrajectoryBatch) -> None:
        self.pull_actor_shard_via_ray = _RemoteCall(value)


def test_reader_pulls_multiple_storage_shards_and_merges(monkeypatch) -> None:
    complete = _batch()
    actors = {
        2: _StorageActor(select_trajectory_batch(complete, (0, 1))),
        5: _StorageActor(select_trajectory_batch(complete, (2, 3))),
    }
    monkeypatch.setattr("rlinf.workers.trajectory.output.ray.get", lambda value: value)
    reader = TrajectoryReader(
        actors=actors,
        storage_group_name="storage",
        route_plan=RoutePlan(4, {"storage": 2, "actor": 1}),
        storage_layout=WorkerLayout((2, 5)),
        actor_layout=WorkerLayout((0,)),
    )

    batch = reader.pull_via_ray(actor_rank=0)

    assert batch.slot_ids == (0, 1, 2, 3)
    assert batch.actions[0, 0, :, 0].tolist() == [0, 1, 2, 3]


def _assert_tensor_free(value) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for child in value.values():
            _assert_tensor_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_tensor_free(child)

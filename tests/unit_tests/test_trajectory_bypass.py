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

import pytest
import torch

from rlinf.data.trajectory import EnvResult, RewardResult, RolloutResult
from rlinf.scheduler import WorkerAddress
from rlinf.workers.trajectory import (
    EndpointSchema,
    RoutePlan,
    StorageConfig,
    StorageWorkerConfig,
    SubmitStatus,
    TrajectoryChannelWorker,
    TrajectoryStorage,
    TrajectoryStorageWorker,
    TrajectoryWriter,
    TransportAck,
    TransportEndpoint,
    WorkerLayout,
    WorkerState,
    ingest_storage_data,
    select_storage_data,
)


def _env(slot_ids: tuple[int, ...] = (0, 1)) -> EnvResult:
    batch = len(slot_ids)
    return EnvResult(
        global_step=8,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=slot_ids,
        rewards=torch.arange(batch, dtype=torch.float32).reshape(batch, 1),
        dones=torch.zeros(batch, 1, dtype=torch.bool),
        terminations=torch.zeros(batch, 1, dtype=torch.bool),
        truncations=torch.zeros(batch, 1, dtype=torch.bool),
    )


def _rollout(slot_ids: tuple[int, ...] = (0, 1)) -> RolloutResult:
    batch = len(slot_ids)
    return RolloutResult(
        global_step=8,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=slot_ids,
        actions=torch.arange(batch * 5 * 7, dtype=torch.float32).reshape(batch, 5, 7),
    )


def _reward(slot_ids: tuple[int, ...] = (0, 1)) -> RewardResult:
    batch = len(slot_ids)
    return RewardResult(
        global_step=8,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=slot_ids,
        rewards=torch.arange(batch, dtype=torch.float32).reshape(batch, 1) + 10,
        mode="per_step",
    )


def _storage() -> TrajectoryStorage:
    return TrajectoryStorage(
        StorageConfig(
            global_step=8,
            rollout_epochs=1,
            chunk_steps=1,
            slot_ids=(0, 1),
            reward_mode="per_step",
            reward_steps=(0,),
        )
    )


def _receive(
    receiver: TransportEndpoint,
    prepared,
):
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)
    for source, destination in zip(
        prepared.payloads, receiver.payload_views(buffers), strict=True
    ):
        destination.copy_(source)
    return receiver.decode(buffers)


def test_three_record_types_assemble_in_any_arrival_order():
    storage = _storage()
    records = (_reward(), _rollout(), _env())

    for schema_id, record in enumerate(records, start=1):
        schema = EndpointSchema.from_example(schema_id, 2, record)
        sender = TransportEndpoint(schema)
        receiver = TransportEndpoint(schema)
        ack = ingest_storage_data(storage, _receive(receiver, sender.encode(record)))
        assert ack.status is SubmitStatus.INGESTED
        assert ack.trajectory_ready is (record is records[-1])
        assert sender.acknowledge(ack.transport_ack)

    assert storage.ready
    batch = storage.export()
    assert torch.equal(batch.actions, _rollout().actions.unsqueeze(0).unsqueeze(0))
    assert torch.equal(
        batch.external_rewards, _reward().rewards.unsqueeze(0).unsqueeze(0)
    )


def test_retry_ack_is_idempotent_and_conflict_is_rejected():
    storage = _storage()
    record = _env()
    schema = EndpointSchema.from_example(1, 2, record)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)
    prepared = sender.encode(record)

    first = ingest_storage_data(storage, _receive(receiver, prepared))
    retry = ingest_storage_data(storage, _receive(receiver, prepared))
    assert first.sequence_id == retry.sequence_id == 0
    assert first.inserted
    assert not retry.inserted
    assert sender.acknowledge(first.transport_ack)
    assert not sender.acknowledge(retry.transport_ack)

    conflicting = _env()
    conflicting.rewards.add_(100)
    conflict_sender = TransportEndpoint(schema)
    conflict_prepared = conflict_sender.encode(conflicting)
    with pytest.raises(ValueError, match="Conflicting content"):
        ingest_storage_data(storage, _receive(receiver, conflict_prepared))


def test_receive_sequence_is_independent_per_source_lane():
    storage = _storage()
    schema = EndpointSchema.from_example(1, 1, _env((0,)))
    source_zero = TransportEndpoint(schema)
    source_one = TransportEndpoint(schema)
    lane_zero = TransportEndpoint(schema)
    lane_one = TransportEndpoint(schema)

    zero = ingest_storage_data(
        storage, _receive(lane_zero, source_zero.encode(_env((0,))))
    )
    one = ingest_storage_data(
        storage, _receive(lane_one, source_one.encode(_env((1,))))
    )
    assert zero.sequence_id == one.sequence_id == 0
    assert zero.inserted and one.inserted


def test_storage_worker_submit_receives_fixed_buffers_and_returns_strong_ack():
    record = _env()
    schema = EndpointSchema.from_example(5, 2, record)
    sender = TransportEndpoint(schema)
    worker = object.__new__(TrajectoryStorageWorker)
    worker._state = WorkerState.READY
    worker._schemas = {schema.schema_id: schema}
    worker._receive_endpoints = {}
    worker._storage = _storage()
    worker._failure = None
    worker._config = StorageWorkerConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(2, {"storage": 1}),
        storage=worker._storage.config,
        endpoints=(schema,),
    )
    worker._reservations = {}
    worker._next_reservation_id = 0
    worker._active_frames = 0
    worker._resident_bytes = 0
    worker._buffer_pool = {}
    worker._generation_buffers = []
    worker._metrics = {
        "frames_received": 0,
        "frames_ingested": 0,
        "frames_rejected": 0,
        "reservations_expired": 0,
        "generation_releases": 0,
        "bytes_received": 0,
        "backpressure_events": 0,
        "backpressure_seconds": 0.0,
        "receive_seconds": 0.0,
        "ingest_seconds": 0.0,
        "queue_depth_max": 0,
        "active_frames_max": 0,
        "resident_bytes_max": 0,
    }
    source = WorkerAddress("env", 3)

    async def submit(prepared, wait_for):
        transmitted = list(prepared.buffers)

        def recv_tensor(target, *_args, **_kwargs):
            target.copy_(transmitted.pop(0))

        worker.recv_tensor = recv_tensor
        reservation_id = await worker.reserve(source.get_name(), schema.schema_id)
        ack = await worker.submit(
            source, schema.schema_id, reservation_id, wait_for=wait_for
        )
        assert not transmitted
        return ack

    async def run():
        worker._ingest_queue = asyncio.Queue(2)
        worker._capacity_changed = asyncio.Condition()
        task = asyncio.create_task(worker._ingest_loop())
        prepared = sender.encode(record)
        first = await submit(prepared, SubmitStatus.RECEIVED)
        retry = await submit(prepared, SubmitStatus.INGESTED)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return first, retry

    first, retry = asyncio.run(run())
    assert first.status is SubmitStatus.RECEIVED
    assert first.inserted is None
    assert first.trajectory_ready is None
    assert not retry.inserted
    assert first.sequence_id == retry.sequence_id == 0
    assert tuple(worker._receive_endpoints) == ((5, "env:3"),)


def test_writer_routes_only_its_record_type_directly_to_storage():
    env = _env((0, 1, 2, 3))
    schema = EndpointSchema.from_example(1, 4, env)
    writer = TrajectoryWriter(
        actors={},
        storage_group_name="storage",
        route_plan=RoutePlan(4, {"env": 1, "storage": 2}),
        source_participant="env",
        source_layout=WorkerLayout((2,)),
        storage_layout=WorkerLayout((4, 7)),
        schemas_by_rank={0: (schema,)},
    )

    pending = writer._prepare(env, source_rank=0)
    assert tuple(item.destination_rank for item in pending) == (4, 7)
    assert tuple(item.prepared.header[7].item() for item in pending) == (2, 2)
    assert select_storage_data(env, (3, 1)).slot_ids == (3, 1)
    with pytest.raises(RuntimeError, match="in-flight"):
        writer.drain()
    for item in pending:
        item.endpoint.acknowledge(
            TransportAck(item.endpoint.schema.schema_id, item.prepared.sequence_id)
        )
    writer.drain()
    with pytest.raises(TypeError, match="cannot submit RolloutResult"):
        writer._prepare(_rollout(), source_rank=0)


def test_channel_worker_has_no_record_submission_api():
    assert "submit" not in TrajectoryChannelWorker.__dict__
    assert "register_endpoint" not in TrajectoryStorageWorker.__dict__

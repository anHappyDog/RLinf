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
from dataclasses import replace

import pytest
import torch

from rlinf.data.trajectory import EnvResult
from rlinf.workers.trajectory import (
    EndpointSchema,
    RoutePlan,
    StorageConfig,
    StorageWorkerConfig,
    TrajectoryStorage,
    TrajectoryStorageWorker,
    TransportEndpoint,
    WorkerLayout,
    WorkerState,
)
from rlinf.workers.trajectory.workers import _schema_buffer_bytes


def _result() -> EnvResult:
    return EnvResult(
        global_step=0,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(0,),
        rewards=torch.zeros(1, 1),
        dones=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
    )


def _worker(
    *,
    max_inflight_frames: int = 1,
    max_resident_bytes: int | None = None,
    backpressure_timeout_s: float = 0.02,
    reservation_timeout_s: float = 1.0,
    drain_timeout_s: float = 0.02,
    actor_participant: str | None = None,
) -> tuple[TrajectoryStorageWorker, EndpointSchema]:
    schema = EndpointSchema.from_example(1, 1, _result())
    storage = StorageConfig(
        global_step=0,
        rollout_epochs=1,
        chunk_steps=1,
        slot_ids=(0,),
    )
    world_sizes = {"storage": 1}
    if actor_participant is not None:
        world_sizes[actor_participant] = 1
    route_plan = RoutePlan(1, world_sizes)
    config = StorageWorkerConfig(
        layout=WorkerLayout((0,)),
        route_plan=route_plan,
        storage=storage,
        endpoints=(schema,),
        actor_participant=actor_participant,
        max_inflight_frames=max_inflight_frames,
        max_resident_bytes=(
            _schema_buffer_bytes(schema)
            if max_resident_bytes is None
            else max_resident_bytes
        ),
        backpressure_timeout_s=backpressure_timeout_s,
        reservation_timeout_s=reservation_timeout_s,
        drain_timeout_s=drain_timeout_s,
    )
    worker = object.__new__(TrajectoryStorageWorker)
    worker._rank = 0
    worker._state = WorkerState.READY
    worker._logical_rank = 0
    worker._failure = None
    worker._storage = TrajectoryStorage(storage)
    worker._route_plan = route_plan
    worker._participant = "storage"
    worker._schemas = {1: schema}
    worker._receive_endpoints = {}
    worker._ingest_queue = asyncio.Queue(config.ingest_queue_size)
    worker._ingest_task = None
    worker._config = config
    worker._capacity_changed = asyncio.Condition()
    worker._reservations = {}
    worker._next_reservation_id = 0
    worker._active_frames = 0
    worker._resident_bytes = 0
    worker._buffer_pool = {}
    worker._generation_buffers = []
    worker._pending_consumers = {0} if actor_participant else set()
    worker._metrics = {
        "frames_received": 0,
        "frames_ingested": 0,
        "frames_rejected": 0,
        "reservations_expired": 0,
        "buffers_evicted": 0,
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
    return worker, schema


def test_reserve_applies_backpressure_before_receive() -> None:
    async def run() -> None:
        worker, _ = _worker()
        first = await worker.reserve("env:0", 1)
        with pytest.raises(TimeoutError, match="capacity"):
            await worker.reserve("env:0", 1)
        assert worker.metrics()["active_frames"] == 1
        assert worker.metrics()["frames_rejected"] == 1
        assert worker.metrics()["backpressure_events"] == 1
        assert worker.metrics()["backpressure_seconds"] >= 0.02

        worker._expire_reservation(first)
        await asyncio.sleep(0)
        second = await worker.reserve("env:0", 1)
        assert worker.metrics()["resident_bytes"] <= worker._config.max_resident_bytes
        worker._expire_reservation(second)

    asyncio.run(run())


def test_waiting_reserve_is_rejected_when_drain_starts() -> None:
    async def run() -> None:
        worker, _ = _worker(drain_timeout_s=0.1)
        first = await worker.reserve("env:0", 1)
        waiting = asyncio.create_task(worker.reserve("env:0", 1))
        await asyncio.sleep(0)
        draining = asyncio.create_task(worker.drain())
        await asyncio.sleep(0)
        worker._expire_reservation(first)

        with pytest.raises(RuntimeError, match="stopped accepting"):
            await waiting
        health = await draining
        assert health.state is WorkerState.DRAINING

    asyncio.run(run())


def test_abandoned_reservation_expires_and_reuses_buffer() -> None:
    async def run() -> None:
        worker, _ = _worker(reservation_timeout_s=0.01)
        await worker.reserve("env:0", 1)
        await asyncio.sleep(0.03)

        metrics = worker.metrics()
        assert metrics["active_frames"] == 0
        assert metrics["active_reservations"] == 0
        assert metrics["reservations_expired"] == 1
        assert metrics["pooled_buffers"] == 1

    asyncio.run(run())


def test_pool_eviction_keeps_resident_bytes_bounded() -> None:
    async def run() -> None:
        worker, first_schema = _worker()
        second_schema = replace(first_schema, schema_id=2)
        worker._schemas[2] = second_schema

        first = await worker.reserve("env:0", 1)
        worker._expire_reservation(first)
        await asyncio.sleep(0)
        second = await worker.reserve("rollout:0", 2)

        metrics = worker.metrics()
        assert metrics["buffers_evicted"] == 1
        assert metrics["resident_bytes"] <= worker._config.max_resident_bytes
        worker._expire_reservation(second)

    asyncio.run(run())


def test_generation_release_prevents_overwrite_and_allows_next_step() -> None:
    async def run() -> None:
        worker, schema = _worker(actor_participant="actor")
        with pytest.raises(RuntimeError, match="active"):
            worker.begin_generation(replace(worker._storage.config, global_step=1))

        buffers = TransportEndpoint(schema).allocate_receive_buffers()
        worker._generation_buffers.append(((1, "env:0"), buffers))
        worker._resident_bytes = _schema_buffer_bytes(schema)
        worker._mark_consumed(0)

        assert worker._storage is None
        assert worker.metrics()["generation_releases"] == 1
        assert worker.metrics()["pooled_buffers"] == 1
        health = worker.begin_generation(replace(worker._config.storage, global_step=1))
        assert health.state is WorkerState.READY
        assert worker._storage.config.global_step == 1

    asyncio.run(run())


def test_drain_timeout_enters_failed_state() -> None:
    async def run() -> None:
        worker, _ = _worker()
        await worker.reserve("env:0", 1)
        with pytest.raises(TimeoutError, match="drain"):
            await worker.drain()
        assert worker.health().state is WorkerState.FAILED
        assert "TimeoutError" in worker.health().detail

    asyncio.run(run())

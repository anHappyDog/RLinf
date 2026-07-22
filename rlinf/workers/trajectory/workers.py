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
import importlib
import os
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import Enum

import torch

from rlinf.data.trajectory import (
    EnvResult,
    PolicyInput,
    PolicyOutput,
    RewardResult,
    RolloutResult,
    ValueRequest,
    ValueResult,
)
from rlinf.scheduler import ChannelWorker, WeightedItem, Worker, WorkerAddress
from rlinf.workers.trajectory.bypass import (
    SubmitAck,
    SubmitStatus,
    ingest_storage_data,
)
from rlinf.workers.trajectory.compression import (
    CompressionConfig,
    CompressionPipeline,
)
from rlinf.workers.trajectory.live import (
    PolicyInputLayout,
    pack_policy_data,
    pack_value_request,
    select_policy_data,
    select_value_request,
    unpack_policy_data,
    unpack_value_request,
)
from rlinf.workers.trajectory.output import flatten_trajectory
from rlinf.workers.trajectory.route_plan import Route, RoutePlan
from rlinf.workers.trajectory.storage import (
    StorageConfig,
    TrajectoryBatch,
    TrajectoryStorage,
    select_trajectory_batch,
)
from rlinf.workers.trajectory.transport import EndpointSchema, TransportEndpoint


class WorkerState(str, Enum):
    """Lifecycle states shared by the trajectory workers."""

    CREATED = "created"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkerLayout:
    """Ordered physical ranks that own independent data shards."""

    data_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.data_ranks or len(set(self.data_ranks)) != len(self.data_ranks):
            raise ValueError("data_ranks must be non-empty and unique.")
        if any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
            for rank in self.data_ranks
        ):
            raise ValueError("data_ranks must contain non-negative integers.")

    def logical_rank(self, worker_rank: int) -> int:
        """Return a physical worker's data-owning rank."""
        try:
            return self.data_ranks.index(worker_rank)
        except ValueError as error:
            raise ValueError(
                f"worker rank {worker_rank} is not a data-owning rank"
            ) from error


@dataclass(frozen=True)
class ChannelConfig:
    """Control-plane configuration for one live channel worker."""

    layout: WorkerLayout
    route_plan: RoutePlan
    env_layout: WorkerLayout
    rollout_layout: WorkerLayout
    env_participant: str = "env"
    rollout_participant: str = "rollout"
    env_group_name: str = ""
    rollout_group_name: str = ""
    policy_input_layout: PolicyInputLayout | None = None

    def __post_init__(self) -> None:
        for participant in (self.env_participant, self.rollout_participant):
            if participant not in self.route_plan.world_sizes:
                raise ValueError(f"unknown live route participant {participant!r}")
        expected = {
            self.env_participant: len(self.env_layout.data_ranks),
            self.rollout_participant: len(self.rollout_layout.data_ranks),
        }
        for participant, world_size in expected.items():
            if self.route_plan.world_sizes[participant] != world_size:
                raise ValueError(
                    f"route participant {participant!r} world size does not match "
                    "its data-owning ranks"
                )
        if self.policy_input_layout is not None:
            if not self.env_group_name or not self.rollout_group_name:
                raise ValueError("direct policy input requires Env/Rollout group names")
            for rank in range(len(self.env_layout.data_ranks)):
                env_range = self.route_plan.slot_range(self.env_participant, rank)
                if env_range[1] - env_range[0] > self.policy_input_layout.batch_size:
                    raise ValueError(
                        "direct policy input Env batch exceeds the registered capacity"
                    )

    def direct_policy_routes(self, source_rank: int) -> tuple[Route, ...]:
        """Return fixed policy fragments emitted by one Env rank."""
        if self.policy_input_layout is None:
            raise RuntimeError("direct policy input is not configured")
        return self.route_plan.routes(
            self.env_participant,
            source_rank,
            self.rollout_participant,
        )

    def direct_policy_sources(
        self, destination_rank: int
    ) -> tuple[tuple[int, Route], ...]:
        """Return Env source ranks and fragments consumed by one Rollout rank."""
        if self.policy_input_layout is None:
            raise RuntimeError("direct policy input is not configured")
        incoming = []
        for source_rank in range(len(self.env_layout.data_ranks)):
            for route in self.direct_policy_routes(source_rank):
                if route.destination_rank == destination_rank:
                    incoming.append((source_rank, route))
        return tuple(incoming)


@dataclass(frozen=True)
class StorageWorkerConfig:
    """Control-plane configuration for one trajectory storage worker."""

    layout: WorkerLayout
    route_plan: RoutePlan
    storage: StorageConfig
    endpoints: tuple[EndpointSchema, ...] = ()
    registry_modules: tuple[str, ...] = ()
    participant: str = "storage"
    actor_participant: str | None = None
    ingest_queue_size: int = 8
    max_inflight_frames: int = 8
    max_resident_bytes: int = 4 * 1024**3
    backpressure_timeout_s: float = 30.0
    reservation_timeout_s: float = 30.0
    drain_timeout_s: float = 30.0
    compression: CompressionConfig = CompressionConfig()

    def __post_init__(self) -> None:
        if not self.participant:
            raise ValueError("participant must be a non-empty string.")
        schema_ids = [schema.schema_id for schema in self.endpoints]
        if len(set(schema_ids)) != len(schema_ids):
            raise ValueError("endpoint schema_ids must be unique.")
        if any(not module for module in self.registry_modules):
            raise ValueError("registry_modules must contain non-empty names.")
        if self.ingest_queue_size < 1:
            raise ValueError("ingest_queue_size must be positive.")
        if self.max_inflight_frames < 1:
            raise ValueError("max_inflight_frames must be positive.")
        if self.max_resident_bytes < 1:
            raise ValueError("max_resident_bytes must be positive.")
        for name in (
            "backpressure_timeout_s",
            "reservation_timeout_s",
            "drain_timeout_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if (
            self.actor_participant is not None
            and self.actor_participant not in self.route_plan.world_sizes
        ):
            raise ValueError(f"unknown actor participant {self.actor_participant!r}")


@dataclass(frozen=True)
class WorkerHealth:
    """Small serializable health snapshot returned by worker control APIs."""

    state: WorkerState
    process_id: int
    worker_rank: int
    logical_rank: int | None
    detail: str | None = None


class TrajectoryChannelWorker(ChannelWorker):
    """Live-path queue worker isolated from trajectory storage work."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._state = WorkerState.CREATED
        self._logical_rank: int | None = None
        self._failure: str | None = None
        self._route_plan: RoutePlan | None = None
        self._env_participant: str | None = None
        self._rollout_participant: str | None = None

    def configure(self, config: ChannelConfig) -> WorkerHealth:
        """Bind this physical actor to one logical live data rank."""
        if self._state is not WorkerState.CREATED:
            raise RuntimeError(f"cannot configure channel worker in {self._state}")
        try:
            self._logical_rank = config.layout.logical_rank(self._rank)
            self._route_plan = config.route_plan
            self._env_participant = config.env_participant
            self._rollout_participant = config.rollout_participant
            self._state = WorkerState.READY
        except Exception as error:
            self._fail(error)
            raise
        return self.health()

    def ready(self) -> bool:
        """Return whether this worker accepts live-path operations."""
        return self._state is WorkerState.READY

    def health(self) -> WorkerHealth:
        """Return the current lifecycle state without contacting storage."""
        return WorkerHealth(
            state=self._state,
            process_id=os.getpid(),
            worker_rank=self._rank,
            logical_rank=self._logical_rank,
            detail=self._failure,
        )

    async def publish_policy_input(
        self,
        src_addr: WorkerAddress,
        source_rank: int,
    ) -> None:
        """Receive and route one Env policy request over worker collectives."""
        data = unpack_policy_data(
            self.recv(src_addr.root_group_name, src_addr.rank_path)
        )
        await self.publish_policy_input_via_ray(data, source_rank)

    async def publish_policy_input_via_ray(
        self,
        data: PolicyInput,
        source_rank: int,
    ) -> None:
        """Route one Env policy request received through Ray."""
        await self._publish(
            data,
            source_rank=source_rank,
            source=self._env_participant,
            destination=self._rollout_participant,
            kind="policy_input",
            expected_type=PolicyInput,
        )

    async def take_policy_input(
        self,
        dst_addr: WorkerAddress,
        destination_rank: int,
    ) -> None:
        """Send the next routed policy request to a Rollout worker."""
        data = await self.take_policy_input_via_ray(destination_rank)
        self.send(pack_policy_data(data), dst_addr.root_group_name, dst_addr.rank_path)

    async def take_policy_input_via_ray(
        self,
        destination_rank: int,
    ) -> PolicyInput:
        """Return the next routed policy request through Ray."""
        return await self._take(
            "policy_input",
            self._rollout_participant,
            destination_rank,
            PolicyInput,
        )

    async def publish_policy_output(
        self,
        src_addr: WorkerAddress,
        source_rank: int,
    ) -> None:
        """Receive and route executable actions over worker collectives."""
        data = unpack_policy_data(
            self.recv(src_addr.root_group_name, src_addr.rank_path)
        )
        await self.publish_policy_output_via_ray(data, source_rank)

    async def publish_policy_output_via_ray(
        self,
        data: PolicyOutput,
        source_rank: int,
    ) -> None:
        """Route executable actions received through Ray."""
        await self._publish(
            data,
            source_rank=source_rank,
            source=self._rollout_participant,
            destination=self._env_participant,
            kind="policy_output",
            expected_type=PolicyOutput,
        )

    async def take_policy_output(
        self,
        dst_addr: WorkerAddress,
        destination_rank: int,
    ) -> None:
        """Send the next routed actions to an Env worker."""
        data = await self.take_policy_output_via_ray(destination_rank)
        self.send(pack_policy_data(data), dst_addr.root_group_name, dst_addr.rank_path)

    async def take_policy_output_via_ray(
        self,
        destination_rank: int,
    ) -> PolicyOutput:
        """Return the next routed actions through Ray."""
        return await self._take(
            "policy_output",
            self._env_participant,
            destination_rank,
            PolicyOutput,
        )

    async def publish_value_request(
        self,
        src_addr: WorkerAddress,
        source_rank: int,
    ) -> None:
        """Receive and route sparse boundary observations."""
        request = unpack_value_request(
            self.recv(src_addr.root_group_name, src_addr.rank_path)
        )
        await self.publish_value_request_via_ray(request, source_rank)

    async def publish_value_request_via_ray(
        self,
        request: ValueRequest,
        source_rank: int,
    ) -> None:
        """Route one sparse value request received through Ray."""
        self._require_ready()
        assert self._route_plan is not None
        assert self._env_participant is not None
        assert self._rollout_participant is not None
        routes = self._route_plan.route_slots(
            self._env_participant,
            source_rank,
            request.slot_ids,
            self._rollout_participant,
        )
        for route in routes:
            key = ("value_request", route.destination_rank)
            self.create_queue(key, self.maxsize())
            item = select_value_request(request, route.source_indices)
            await self._queue_map[key].put(WeightedItem(weight=0, item=item))

    async def take_value_request(
        self,
        dst_addr: WorkerAddress,
        destination_rank: int,
    ) -> None:
        """Send one sparse boundary request to a Rollout worker."""
        request = await self.take_value_request_via_ray(destination_rank)
        self.send(
            pack_value_request(request),
            dst_addr.root_group_name,
            dst_addr.rank_path,
        )

    async def take_value_request_via_ray(
        self,
        destination_rank: int,
    ) -> ValueRequest:
        """Return the next sparse request for one Rollout logical rank."""
        self._require_ready()
        assert self._route_plan is not None
        assert self._rollout_participant is not None
        self._route_plan.slot_range(self._rollout_participant, destination_rank)
        key = ("value_request", destination_rank)
        self.create_queue(key, self.maxsize())
        item = (await self._queue_map[key].get()).item
        if not isinstance(item, ValueRequest):
            raise RuntimeError(f"invalid value request {type(item).__name__}")
        return item

    def value_request_count(self, destination_rank: int) -> int:
        """Return queued boundary requests for one Rollout logical rank."""
        self._require_ready()
        assert self._route_plan is not None
        assert self._rollout_participant is not None
        self._route_plan.slot_range(self._rollout_participant, destination_rank)
        key = ("value_request", destination_rank)
        self.create_queue(key, self.maxsize())
        return self._queue_map[key].qsize()

    async def _publish(
        self,
        data: PolicyInput | PolicyOutput,
        *,
        source_rank: int,
        source: str | None,
        destination: str | None,
        kind: str,
        expected_type: type[PolicyInput] | type[PolicyOutput],
    ) -> None:
        self._require_ready()
        if type(data) is not expected_type:
            raise TypeError(f"{kind} requires {expected_type.__name__}")
        assert self._route_plan is not None
        assert source is not None and destination is not None
        routes = self._route_plan.route_slots(
            source, source_rank, data.slot_ids, destination
        )
        for route in routes:
            key = (kind, route.destination_rank)
            self.create_queue(key, self.maxsize())
            item = select_policy_data(data, route.source_indices)
            await self._queue_map[key].put(WeightedItem(weight=0, item=item))

    async def _take(
        self,
        kind: str,
        participant: str | None,
        destination_rank: int,
        expected_type: type[PolicyInput] | type[PolicyOutput],
    ) -> PolicyInput | PolicyOutput:
        self._require_ready()
        assert self._route_plan is not None and participant is not None
        self._route_plan.slot_range(participant, destination_rank)
        key = (kind, destination_rank)
        self.create_queue(key, self.maxsize())
        item = (await self._queue_map[key].get()).item
        if type(item) is not expected_type:
            raise RuntimeError(f"invalid {kind} queue item {type(item).__name__}")
        return item

    async def drain(self) -> WorkerHealth:
        """Enter draining state after verifying all live queues are empty."""
        if self._state is WorkerState.STOPPED:
            return self.health()
        if self._state is not WorkerState.READY:
            raise RuntimeError(f"cannot drain channel worker in {self._state}")
        self._state = WorkerState.DRAINING
        pending = sum(queue.qsize() for queue in self._queue_map.values())
        if pending:
            self._state = WorkerState.READY
            raise RuntimeError(
                f"cannot drain channel worker with {pending} queued items"
            )
        return self.health()

    async def shutdown(self) -> WorkerHealth:
        """Stop background maintenance after a successful drain."""
        if self._state is WorkerState.STOPPED:
            return self.health()
        if self._state is WorkerState.READY:
            await self.drain()
        if self._state not in (WorkerState.CREATED, WorkerState.DRAINING):
            raise RuntimeError(f"cannot shut down channel worker in {self._state}")
        self._mem_cleaner_task.cancel()
        try:
            await self._mem_cleaner_task
        except asyncio.CancelledError:
            pass
        self._state = WorkerState.STOPPED
        return self.health()

    def _fail(self, error: Exception) -> None:
        self._state = WorkerState.FAILED
        self._failure = f"{type(error).__name__}: {error}"

    def _require_ready(self) -> None:
        if self._state is not WorkerState.READY:
            raise RuntimeError(f"channel worker is not ready: {self._state}")


class TrajectoryStorageWorker(Worker):
    """Own trajectory assembly and transport state in an independent process."""

    def __init__(self) -> None:
        super().__init__()
        self._state = WorkerState.CREATED
        self._logical_rank: int | None = None
        self._failure: str | None = None
        self._storage: TrajectoryStorage | None = None
        self._route_plan: RoutePlan | None = None
        self._participant: str | None = None
        self._schemas: dict[int, EndpointSchema] = {}
        self._receive_endpoints: dict[tuple[int, str], TransportEndpoint] = {}
        self._ingest_queue: asyncio.Queue | None = None
        self._ingest_task: asyncio.Task | None = None
        self._config: StorageWorkerConfig | None = None
        self._capacity_changed: asyncio.Condition | None = None
        self._reservations: dict[int, tuple] = {}
        self._next_reservation_id = 0
        self._active_frames = 0
        self._resident_bytes = 0
        self._buffer_pool: dict[tuple[int, str], list] = {}
        self._generation_buffers: list[tuple[tuple[int, str], object]] = []
        self._pending_consumers: set[int] = set()
        self._metrics: dict[str, int | float] = {}
        self._compression_pipeline: CompressionPipeline | None = None

    def configure(self, config: StorageWorkerConfig) -> WorkerHealth:
        """Load schemas and initialize this actor's local trajectory shard."""
        if self._state is not WorkerState.CREATED:
            raise RuntimeError(f"cannot configure storage worker in {self._state}")
        try:
            logical_rank = config.layout.logical_rank(self._rank)
            expected_world_size = len(config.layout.data_ranks)
            actual_world_size = config.route_plan.world_sizes.get(config.participant)
            if actual_world_size != expected_world_size:
                raise ValueError(
                    f"route participant {config.participant!r} has world size "
                    f"{actual_world_size}, expected {expected_world_size}"
                )
            start, end = config.route_plan.slot_range(config.participant, logical_rank)
            expected_slots = tuple(range(start, end))
            if config.storage.slot_ids != expected_slots:
                raise ValueError(
                    f"storage slots {config.storage.slot_ids} do not match "
                    f"{config.participant!r} logical rank {logical_rank}: "
                    f"{expected_slots}"
                )
            for module in config.registry_modules:
                importlib.import_module(module)
            self._schemas = {schema.schema_id: schema for schema in config.endpoints}
            self._route_plan = config.route_plan
            self._participant = config.participant
            self._config = config
            if config.compression.enabled:
                self._compression_pipeline = CompressionPipeline(config.compression)
            self._capacity_changed = asyncio.Condition()
            self._ingest_queue = asyncio.Queue(config.ingest_queue_size)
            self._ingest_task = asyncio.create_task(self._ingest_loop())
            self._logical_rank = logical_rank
            self._metrics = {
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
                "compression_raw_bytes": 0,
                "compression_wire_bytes": 0,
                "compression_compressed_blocks": 0,
                "compression_raw_blocks": 0,
                "compression_workspace_allocations": 0,
                "compression_workspace_bytes": 0,
            }
            self._start_generation(config.storage)
            self._state = WorkerState.READY
        except Exception as error:
            self._fail(error)
            raise
        return self.health()

    def ready(self) -> bool:
        """Return whether configuration completed successfully."""
        return self._state is WorkerState.READY

    def health(self) -> WorkerHealth:
        """Return the current storage lifecycle state."""
        return WorkerHealth(
            state=self._state,
            process_id=os.getpid(),
            worker_rank=self._rank,
            logical_rank=self._logical_rank,
            detail=self._failure,
        )

    def trajectory_ready(self) -> bool:
        """Return whether the configured local trajectory is complete."""
        self._require_state(WorkerState.READY)
        return self._storage is not None and self._storage.ready

    def metrics(self) -> dict[str, int | float]:
        """Return bounded-buffer, queue, latency, and throughput counters."""
        queue_depth = self._ingest_queue.qsize() if self._ingest_queue else 0
        return {
            **self._metrics,
            "queue_depth": queue_depth,
            "queue_capacity": self._config.ingest_queue_size if self._config else 0,
            "active_frames": self._active_frames,
            "active_reservations": len(self._reservations),
            "resident_bytes": self._resident_bytes,
            "pooled_buffers": sum(map(len, self._buffer_pool.values())),
            "generation_active": int(self._storage is not None),
        }

    def begin_generation(self, storage: StorageConfig) -> WorkerHealth:
        """Start the next generation after every Actor consumer released the prior."""
        self._require_state(WorkerState.READY)
        if self._storage is not None:
            raise RuntimeError("cannot replace an active trajectory generation")
        assert self._config is not None
        previous = self._config.storage
        if (
            storage.slot_ids != previous.slot_ids
            or storage.rollout_epochs != previous.rollout_epochs
            or storage.chunk_steps != previous.chunk_steps
            or storage.env_fields != previous.env_fields
            or storage.rollout_fields != previous.rollout_fields
            or storage.value_fields != previous.value_fields
            or storage.reward_mode != previous.reward_mode
            or storage.reward_steps != previous.reward_steps
            or storage.boundary_values != previous.boundary_values
        ):
            raise ValueError("generation schema cannot change after configure")
        if storage.global_step <= previous.global_step:
            raise ValueError("generation global_step must increase")
        self._config = dataclass_replace(self._config, storage=storage)
        self._start_generation(storage)
        return self.health()

    async def reserve(self, source: str, schema_id: int) -> int:
        """Reserve bounded receive capacity before the producer sends tensors."""
        self._require_state(WorkerState.READY)
        if self._storage is None:
            raise RuntimeError("no active trajectory generation")
        if not source:
            raise ValueError("source must be non-empty")
        try:
            schema = self._schemas[schema_id]
        except KeyError as error:
            raise ValueError(f"unknown endpoint schema {schema_id}") from error
        assert self._config is not None
        assert self._capacity_changed is not None
        lane = (schema_id, source)
        started = time.monotonic()
        deadline = started + self._config.backpressure_timeout_s
        blocked = False
        async with self._capacity_changed:
            while True:
                if self._state is not WorkerState.READY:
                    raise RuntimeError(
                        f"storage worker stopped accepting reservations: {self._state}"
                    )
                pooled = self._buffer_pool.get(lane)
                required_bytes = 0 if pooled else _schema_buffer_bytes(schema)
                if (
                    self._active_frames < self._config.max_inflight_frames
                    and self._resident_bytes + required_bytes
                    > self._config.max_resident_bytes
                ):
                    self._evict_pooled_buffers(required_bytes, preserve=lane)
                    pooled = self._buffer_pool.get(lane)
                    required_bytes = 0 if pooled else _schema_buffer_bytes(schema)
                if (
                    self._active_frames < self._config.max_inflight_frames
                    and self._resident_bytes + required_bytes
                    <= self._config.max_resident_bytes
                ):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._metrics["frames_rejected"] += 1
                    self._metrics["backpressure_events"] += 1
                    self._metrics["backpressure_seconds"] += time.monotonic() - started
                    raise TimeoutError("trajectory receive capacity timed out")
                try:
                    blocked = True
                    await asyncio.wait_for(
                        self._capacity_changed.wait(), timeout=remaining
                    )
                except TimeoutError:
                    self._metrics["frames_rejected"] += 1
                    self._metrics["backpressure_events"] += 1
                    self._metrics["backpressure_seconds"] += time.monotonic() - started
                    raise TimeoutError(
                        "trajectory receive capacity timed out"
                    ) from None

            if pooled:
                buffers = pooled.pop()
                if not pooled:
                    self._buffer_pool.pop(lane)
            else:
                buffers = TransportEndpoint(schema).allocate_receive_buffers()
                self._resident_bytes += required_bytes
            self._active_frames += 1
            self._metrics["active_frames_max"] = max(
                self._metrics["active_frames_max"], self._active_frames
            )
            self._metrics["resident_bytes_max"] = max(
                self._metrics["resident_bytes_max"], self._resident_bytes
            )
            reservation_id = self._next_reservation_id
            self._next_reservation_id += 1
            waited = time.monotonic() - started
            timer = asyncio.get_running_loop().call_later(
                self._config.reservation_timeout_s,
                self._expire_reservation,
                reservation_id,
            )
            self._reservations[reservation_id] = (lane, buffers, timer, waited)
            if blocked:
                self._metrics["backpressure_events"] += 1
            self._metrics["backpressure_seconds"] += waited
            return reservation_id

    def pull_actor_shard_via_ray(
        self,
        actor_rank: int,
        actor_participant: str = "actor",
    ) -> TrajectoryBatch:
        """Return an uncompressed shard through the diagnostic Ray path."""
        result = self._actor_shard(actor_rank, actor_participant)
        self._mark_consumed(actor_rank)
        return result

    def _actor_shard(
        self,
        actor_rank: int,
        actor_participant: str,
    ) -> TrajectoryBatch:
        """Build this Storage rank's contribution to one Actor shard."""
        self._require_state(WorkerState.READY)
        assert self._storage is not None
        assert self._route_plan is not None
        assert self._participant is not None
        if not self._storage.ready:
            raise RuntimeError(
                "Trajectory is not ready: " + "; ".join(self._storage.missing())
            )
        self._route_plan.slot_range(actor_participant, actor_rank)
        batch = self._storage.export()
        indices = tuple(
            index
            for index, slot_id in enumerate(batch.slot_ids)
            if self._route_plan.owner(actor_participant, slot_id) == actor_rank
        )
        if not indices:
            raise ValueError(
                f"storage rank {self._logical_rank} has no slots for "
                f"{actor_participant!r} rank {actor_rank}"
            )
        if indices == tuple(range(len(batch.slot_ids))):
            result = batch
        else:
            result = select_trajectory_batch(batch, indices)
        return result

    def pull_actor_shard(
        self,
        dst_addr: WorkerAddress,
        actor_rank: int,
        actor_participant: str = "actor",
    ) -> None:
        """Push a requested Actor shard contribution over worker transport."""
        batch = self._actor_shard(actor_rank, actor_participant)
        tensors, skeleton = flatten_trajectory(batch)
        assert self._config is not None
        compression = self._config.compression
        if compression.enabled:
            assert self._compression_pipeline is not None
            tensors, metadata, stats = self._compression_pipeline.compress(tensors)
            skeleton["compression"] = metadata
            for key, value in stats.items():
                self._metrics[f"compression_{key}"] += value
        self.send(
            tensors,
            dst_addr.root_group_name,
            dst_addr.rank_path,
            piggyback_payload=skeleton,
        )
        self._mark_consumed(actor_rank)

    def endpoint_ids(self) -> tuple[int, ...]:
        """Return configured transport schema identifiers."""
        self._require_state(WorkerState.READY)
        return tuple(self._schemas)

    async def submit(
        self,
        src_addr: WorkerAddress,
        schema_id: int,
        reservation_id: int,
        wait_for: SubmitStatus = SubmitStatus.RECEIVED,
    ) -> SubmitAck:
        """Receive one fixed frame and ingest it into local trajectory storage."""
        self._require_state(WorkerState.READY)
        if wait_for not in (SubmitStatus.RECEIVED, SubmitStatus.INGESTED):
            raise ValueError("submit can wait for RECEIVED or INGESTED")
        try:
            lane, buffers, timer, waited = self._reservations.pop(reservation_id)
        except KeyError as error:
            raise ValueError(
                f"unknown or expired reservation {reservation_id}"
            ) from error
        expected_lane = (schema_id, src_addr.get_name())
        if lane != expected_lane:
            self._return_buffer(lane, buffers)
            self._active_frames -= 1
            await self._notify_capacity()
            raise ValueError("reservation does not match submit source/schema")
        timer.cancel()
        schema = self._schemas[schema_id]
        if schema.record_type not in {
            EnvResult.__name__,
            RolloutResult.__name__,
            RewardResult.__name__,
            ValueResult.__name__,
        }:
            raise TypeError(f"cannot submit {schema.record_type}")
        endpoint = self._receive_endpoints.setdefault(lane, TransportEndpoint(schema))
        receive_started = time.monotonic()
        try:
            self.recv_tensor(
                buffers.header,
                src_addr.root_group_name,
                src_addr.rank_path,
            )
            for payload in endpoint.payload_views(buffers):
                self.recv_tensor(
                    payload,
                    src_addr.root_group_name,
                    src_addr.rank_path,
                )
            received = endpoint.decode(buffers)
        except Exception:
            self._return_buffer(lane, buffers)
            self._active_frames -= 1
            await self._notify_capacity()
            raise
        self._metrics["frames_received"] += 1
        self._metrics["bytes_received"] += sum(
            tensor.numel() * tensor.element_size()
            for tensor in (buffers.header, *endpoint.payload_views(buffers))
        )
        self._metrics["receive_seconds"] += time.monotonic() - receive_started
        assert self._ingest_queue is not None
        future = (
            asyncio.get_running_loop().create_future()
            if wait_for is SubmitStatus.INGESTED
            else None
        )
        await self._ingest_queue.put(
            (received, future, lane, buffers, waited, time.monotonic())
        )
        self._metrics["queue_depth_max"] = max(
            self._metrics["queue_depth_max"], self._ingest_queue.qsize()
        )
        if wait_for is SubmitStatus.INGESTED:
            assert future is not None
            return await future
        return SubmitAck(
            status=SubmitStatus.RECEIVED,
            schema_id=received.ack.schema_id,
            sequence_id=received.ack.sequence_id,
            inserted=None,
            trajectory_ready=None,
            backpressure_seconds=waited,
        )

    async def drain(self) -> WorkerHealth:
        """Stop accepting submissions after all transport sends are settled."""
        if self._state is WorkerState.STOPPED:
            return self.health()
        self._require_state(WorkerState.READY)
        self._state = WorkerState.DRAINING
        await self._notify_capacity()
        assert self._ingest_queue is not None
        assert self._config is not None
        try:
            await asyncio.wait_for(
                self._wait_until_quiescent(), timeout=self._config.drain_timeout_s
            )
        except TimeoutError as error:
            self._fail(error)
            raise TimeoutError("trajectory storage drain timed out") from None
        if self._failure is not None:
            raise RuntimeError(self._failure)
        return self.health()

    async def shutdown(self) -> WorkerHealth:
        """Release storage-owned state after a successful drain."""
        if self._state is WorkerState.STOPPED:
            return self.health()
        if self._state is WorkerState.READY:
            await self.drain()
        if self._state not in (WorkerState.CREATED, WorkerState.DRAINING):
            raise RuntimeError(f"cannot shut down storage worker in {self._state}")
        if self._ingest_task is not None:
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except asyncio.CancelledError:
                pass
        self._receive_endpoints.clear()
        self._schemas.clear()
        self._storage = None
        self._route_plan = None
        self._participant = None
        self._ingest_queue = None
        self._ingest_task = None
        self._config = None
        self._capacity_changed = None
        self._reservations.clear()
        self._buffer_pool.clear()
        self._generation_buffers.clear()
        self._resident_bytes = 0
        self._active_frames = 0
        if self._compression_pipeline is not None:
            self._compression_pipeline.close()
            self._compression_pipeline = None
        self._state = WorkerState.STOPPED
        return self.health()

    def _require_state(self, expected: WorkerState) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"storage worker must be {expected}, currently {self._state}"
            )

    def _fail(self, error: Exception) -> None:
        self._state = WorkerState.FAILED
        self._failure = f"{type(error).__name__}: {error}"

    async def _ingest_loop(self) -> None:
        assert self._ingest_queue is not None
        assert self._storage is not None
        while True:
            (
                received,
                future,
                lane,
                buffers,
                waited,
                queued_at,
            ) = await self._ingest_queue.get()
            failure: Exception | None = None
            try:
                ack = ingest_storage_data(self._storage, received)
            except Exception as error:
                self._fail(error)
                failure = error
                self._metrics["frames_rejected"] += 1
                self._return_buffer(lane, buffers)
                if future is not None:
                    future.set_exception(error)
            else:
                self._metrics["frames_ingested"] += 1
                self._metrics["ingest_seconds"] += time.monotonic() - queued_at
                if ack.inserted:
                    self._generation_buffers.append((lane, buffers))
                else:
                    self._return_buffer(lane, buffers)
                if future is not None:
                    future.set_result(
                        dataclass_replace(ack, backpressure_seconds=waited)
                    )
            finally:
                self._active_frames -= 1
                self._ingest_queue.task_done()
                await self._notify_capacity()
            if failure is not None:
                self._reject_queued(failure)
                await self._notify_capacity()
                return

    def _start_generation(self, storage: StorageConfig) -> None:
        self._storage = TrajectoryStorage(storage)
        self._generation_buffers = []
        self._pending_consumers = set()
        if self._config is not None and self._config.actor_participant is not None:
            assert self._route_plan is not None
            self._pending_consumers = {
                self._route_plan.owner(self._config.actor_participant, slot_id)
                for slot_id in storage.slot_ids
            }

    def _mark_consumed(self, actor_rank: int) -> None:
        if actor_rank not in self._pending_consumers:
            return
        self._pending_consumers.remove(actor_rank)
        if self._pending_consumers:
            return
        self._storage = None
        for lane, buffers in self._generation_buffers:
            self._return_buffer(lane, buffers)
        self._generation_buffers.clear()
        self._metrics["generation_releases"] += 1
        if self._capacity_changed is not None:
            asyncio.get_running_loop().create_task(self._notify_capacity())

    def _expire_reservation(self, reservation_id: int) -> None:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return
        lane, buffers, _timer, _waited = reservation
        self._return_buffer(lane, buffers)
        self._active_frames -= 1
        self._metrics["reservations_expired"] += 1
        asyncio.get_running_loop().create_task(self._notify_capacity())

    def _return_buffer(self, lane: tuple[int, str], buffers) -> None:
        self._buffer_pool.setdefault(lane, []).append(buffers)

    def _evict_pooled_buffers(
        self, required_bytes: int, *, preserve: tuple[int, str]
    ) -> None:
        assert self._config is not None
        for lane in tuple(self._buffer_pool):
            if lane == preserve:
                continue
            pool = self._buffer_pool[lane]
            while (
                pool
                and self._resident_bytes + required_bytes
                > self._config.max_resident_bytes
            ):
                buffers = pool.pop()
                self._resident_bytes -= _buffers_nbytes(buffers)
                self._metrics["buffers_evicted"] += 1
            if not pool:
                self._buffer_pool.pop(lane)
            if self._resident_bytes + required_bytes <= self._config.max_resident_bytes:
                return

    async def _notify_capacity(self) -> None:
        if self._capacity_changed is None:
            return
        async with self._capacity_changed:
            self._capacity_changed.notify_all()

    async def _wait_until_quiescent(self) -> None:
        assert self._ingest_queue is not None
        await self._ingest_queue.join()
        assert self._capacity_changed is not None
        async with self._capacity_changed:
            await self._capacity_changed.wait_for(
                lambda: not self._reservations and self._active_frames == 0
            )

    def _reject_queued(self, error: Exception) -> None:
        assert self._ingest_queue is not None
        while True:
            try:
                _received, future, lane, buffers, _waited, _queued_at = (
                    self._ingest_queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                return
            self._return_buffer(lane, buffers)
            self._active_frames -= 1
            self._metrics["frames_rejected"] += 1
            if future is not None:
                future.set_exception(error)
            self._ingest_queue.task_done()


def _schema_buffer_bytes(schema: EndpointSchema) -> int:
    header = schema.header_size * torch.empty((), dtype=torch.int64).element_size()
    payloads = sum(layout.nbytes(schema.max_batch_size) for layout in schema.tensors)
    return header + payloads


def _buffers_nbytes(buffers) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (buffers.header, *buffers.payloads)
    )

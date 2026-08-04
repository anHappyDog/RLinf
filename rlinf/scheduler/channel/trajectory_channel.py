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

"""Trajectory collection channel for embodied training."""

import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ray

from rlinf.data.embodied_io_struct import (
    EmbodiedLerobotRolloutResult,
    EmbodiedRolloutResult,
    assign_history_reward,
)
from rlinf.utils.data_iter_utils import split_list

from ..cluster import Cluster
from ..collective import AsyncChannelCommWork, AsyncChannelWork, AsyncWork
from ..placement import NodePlacementStrategy, PlacementStrategy
from ..worker import Worker, WorkerAddress, WorkerGroup
from .channel import Channel


@dataclass(frozen=True)
class TrajectoryEvent:
    """One ordered mutation to a source trajectory collector."""

    source_id: int
    sequence: int
    kind: "TrajectoryEventType"
    payload: dict[str, Any]


class TrajectoryEventType(str, Enum):
    """The closed set of mutations accepted by a trajectory collector."""

    OPEN = "open"
    APPEND_STEP = "append_step"
    UPDATE_LAST_ACTIONS = "update_last_actions"
    APPEND_TRANSITIONS = "append_transitions"
    MARK_INTERVENE_FLAGS = "mark_intervene_flags"
    ASSIGN_HISTORY_REWARD = "assign_history_reward"
    APPEND_EPISODE_DATA = "append_episode_data"
    CLEAR_REWARDS = "clear_rewards"
    RESET_EPISODE_BUFFERS = "reset_episode_buffers"
    FLUSH = "flush"


@dataclass(frozen=True)
class TrajectoryFailure:
    """A collector error delivered to every actor assigned to a source."""

    source_id: int
    message: str


def _trajectory_group_name(name: str) -> str:
    """Return the dedicated worker-group name for a trajectory channel."""
    return f"{name}_trajectory"


class AsyncTrajectoryTakeWork(AsyncWork):
    """Receive a ready storage rank, then receive its trajectory through P2P."""

    def __init__(self, channel: "TrajectoryChannel"):
        """Request one completed item for the current actor worker."""
        self._channel = channel
        self._query_id = uuid.uuid4().int
        coordinator = channel._trajectory_workers_by_rank[0]
        coordinator.reserve.remote(
            dst_addr=channel._current_worker.worker_address,
            query_id=self._query_id,
            actor_rank=channel._current_worker._rank,
        )
        received = channel._current_worker.recv(
            channel._trajectory_group_name, 0, async_op=True
        )
        self._reserve_work = AsyncChannelCommWork(
            async_comm_work=received,
            query_id=self._query_id,
            channel_actor=coordinator,
        )

    async def async_wait(self) -> Any:
        """Wait asynchronously for the completed item."""
        storage_rank = await self._reserve_work.async_wait()
        item = await self._channel._take_from_storage(
            storage_rank, async_op=True
        ).async_wait()
        return self._channel._unwrap_item(item)

    def wait(self) -> Any:
        """Wait synchronously for the completed item."""
        storage_rank = self._reserve_work.wait()
        return self._channel._unwrap_item(
            self._channel._take_from_storage(storage_rank)
        )

    def done(self) -> bool:
        """Return whether the storage reservation is ready."""
        return self._reserve_work.done()


class TrajectoryWorker(Worker):
    """Own source-affine rollout collectors for one TrajectoryChannel."""

    def __init__(self, collector_config: dict[str, Any], maxsize: int = 0):
        """Initialize source collectors and completed-item queues."""
        super().__init__()
        self._collector_config = collector_config
        self._maxsize = maxsize
        self._collectors: dict[int, EmbodiedRolloutResult] = {}
        self._routes: dict[int, list[tuple[int, int]]] = {}
        self._next_sequences: dict[int, int] = {}
        self._completed: dict[int, asyncio.Queue[Any]] = {}
        self._ready: dict[int, asyncio.Queue[int]] = {}

    def maxsize(self) -> int:
        """Return the configured completed-queue capacity."""
        return self._maxsize

    def _completed_queue(self, actor_rank: int) -> asyncio.Queue[Any]:
        if actor_rank not in self._completed:
            self._completed[actor_rank] = asyncio.Queue(maxsize=self._maxsize)
        return self._completed[actor_rank]

    def _ready_queue(self, actor_rank: int) -> asyncio.Queue[int]:
        if actor_rank not in self._ready:
            self._ready[actor_rank] = asyncio.Queue(maxsize=self._maxsize)
        return self._ready[actor_rank]

    def _new_collector(self) -> EmbodiedRolloutResult:
        if self._collector_config["type"] == "lerobot":
            return EmbodiedLerobotRolloutResult(
                max_episode_length=self._collector_config["max_episode_length"],
                num_envs=self._collector_config["num_envs"],
                only_success=self._collector_config["only_success"],
                num_action_chunks=self._collector_config["num_action_chunks"],
                action_dim=self._collector_config["action_dim"],
            )
        return EmbodiedRolloutResult(
            max_episode_length=self._collector_config["max_episode_length"]
        )

    async def consume_event(self, src_addr: WorkerAddress) -> None:
        """Receive and apply one ordered collector mutation through Worker P2P."""
        event = self.recv(src_addr.root_group_name, src_addr.rank_path)
        try:
            expected = self._next_sequences.get(event.source_id, 0)
            if event.sequence != expected:
                raise RuntimeError(
                    f"Unexpected trajectory event sequence for source {event.source_id}: "
                    f"expected {expected}, got {event.sequence}."
                )

            closed = await self._apply_event(event)
            if closed:
                self._next_sequences.pop(event.source_id, None)
            else:
                self._next_sequences[event.source_id] = expected + 1
        except Exception as error:
            await self._send_failure(event, error)
            raise

    async def _apply_event(self, event: TrajectoryEvent) -> bool:
        source_id = event.source_id
        if event.kind is TrajectoryEventType.OPEN:
            if source_id in self._collectors:
                raise RuntimeError(f"Trajectory source {source_id} is already open.")
            self._collectors[source_id] = self._new_collector()
            self._routes[source_id] = event.payload["routes"]
            return False

        collector = self._collectors[source_id]
        if event.kind is TrajectoryEventType.APPEND_STEP:
            collector.append_step_result(event.payload["step"])
        elif event.kind is TrajectoryEventType.UPDATE_LAST_ACTIONS:
            collector.update_last_actions(
                event.payload["actions"], event.payload["flags"]
            )
        elif event.kind is TrajectoryEventType.APPEND_TRANSITIONS:
            collector.append_transitions(
                event.payload["curr_obs"], event.payload["next_obs"]
            )
        elif event.kind is TrajectoryEventType.MARK_INTERVENE_FLAGS:
            collector.mark_last_step_with_intervene_flags(event.payload["flags"])
        elif event.kind is TrajectoryEventType.ASSIGN_HISTORY_REWARD:
            assign_history_reward(
                collector.rewards,
                event.payload["reward_output"],
                event.payload["assign_lengths"],
                self._collector_config["reward_weight"],
            )
        elif event.kind is TrajectoryEventType.APPEND_EPISODE_DATA:
            assert isinstance(collector, EmbodiedLerobotRolloutResult)
            collector.append_chunk_episode_data(**event.payload)
        elif event.kind is TrajectoryEventType.CLEAR_REWARDS:
            collector.rewards.clear()
        elif event.kind is TrajectoryEventType.RESET_EPISODE_BUFFERS:
            assert isinstance(collector, EmbodiedLerobotRolloutResult)
            collector.reset_episode_buffers()
        elif event.kind is TrajectoryEventType.FLUSH:
            close = event.payload["close"]
            await self._flush(source_id, close)
            return close
        else:
            raise ValueError(f"Unknown trajectory event: {event.kind}.")
        return False

    async def _flush(self, source_id: int, close: bool) -> None:
        collector = self._collectors[source_id]
        routes = self._routes[source_id]
        if isinstance(collector, EmbodiedLerobotRolloutResult):
            episodes = collector.drain_episodes()
            if episodes:
                chunks = split_list(
                    episodes, len(routes), enforce_divisible_batch=False
                )
                for (actor_rank, _), chunk in zip(routes, chunks):
                    if chunk:
                        await self._enqueue(actor_rank, chunk)
        elif close:
            split_sizes = [size for _, size in routes]
            trajectories = collector.to_splited_trajectories_by_sizes(split_sizes)
            for (actor_rank, _), trajectory in zip(routes, trajectories):
                await self._enqueue(actor_rank, trajectory)

        if close:
            del self._collectors[source_id]
            del self._routes[source_id]

    async def _send_failure(self, event: TrajectoryEvent, error: Exception) -> None:
        """Deliver one terminal error to each actor assigned to the source."""
        routes = self._routes.get(event.source_id, event.payload.get("routes", []))
        failure = TrajectoryFailure(event.source_id, str(error))
        for actor_rank in {actor_rank for actor_rank, _ in routes}:
            await self._enqueue(actor_rank, failure)

    async def _enqueue(self, actor_rank: int, item: Any) -> None:
        await self._completed_queue(actor_rank).put(item)
        if self._rank == 0:
            await self._ready_queue(actor_rank).put(0)
            return

        coordinator = ray.get_actor(
            WorkerAddress(self._group_name, 0).get_name(), namespace=Cluster.NAMESPACE
        )
        coordinator.announce_ready.remote(src_addr=self.worker_address)
        self.send(actor_rank, self._group_name, 0, async_op=True)

    async def announce_ready(self, src_addr: WorkerAddress) -> None:
        """Receive a P2P ready notification from another storage worker."""
        actor_rank = self.recv(src_addr.root_group_name, src_addr.rank_path)
        await self._ready_queue(actor_rank).put(src_addr.rank)

    async def reserve(
        self,
        dst_addr: WorkerAddress,
        query_id: int,
        actor_rank: int,
        nowait: bool = False,
    ) -> None:
        """Reserve one completed item and send its storage rank through P2P."""
        queue = self._ready_queue(actor_rank)
        if nowait and queue.empty():
            self.send(
                None,
                dst_addr.root_group_name,
                dst_addr.rank_path,
                async_op=True,
                piggyback_payload=query_id,
            )
            return
        storage_rank = queue.get_nowait() if nowait else await queue.get()
        self.send(
            storage_rank,
            dst_addr.root_group_name,
            dst_addr.rank_path,
            async_op=True,
            piggyback_payload=query_id,
        )

    async def take(
        self, dst_addr: WorkerAddress, query_id: int, actor_rank: int
    ) -> None:
        """Send one reserved completed item directly to its actor worker."""
        item = await self._completed_queue(actor_rank).get()
        self.send(
            item,
            dst_addr.root_group_name,
            dst_addr.rank_path,
            async_op=True,
            piggyback_payload=query_id,
        )


class TrajectoryChannel(Channel):
    """A Channel with a source-affine collector plane for full trajectories.

    Inherited ``put``/``get`` operations keep the ordinary Channel semantics.
    Env workers use the trajectory methods below only to append collector events;
    actor workers use ``take`` or ``try_take`` to receive completed items.
    """

    @classmethod
    def create(
        cls,
        name: str,
        *,
        source_routes: dict[int, list[tuple[int, int]]],
        collector_config: dict[str, Any],
        maxsize: int = 0,
        placement_strategy: PlacementStrategy | None = None,
    ) -> "TrajectoryChannel":
        """Create ordinary Channel and placement-independent collector workers."""
        normal_channel = Channel.create(name, maxsize=maxsize)
        trajectory_name = _trajectory_group_name(name)
        cluster = Cluster()
        placement = placement_strategy or NodePlacementStrategy(node_ranks=[0])
        try:
            worker_group = TrajectoryWorker.create_group(
                collector_config=collector_config, maxsize=maxsize
            ).launch(
                cluster=cluster,
                name=trajectory_name,
                placement_strategy=placement,
                max_concurrency=2**31 - 1,
            )
        except ValueError:
            Worker.logger.warning(
                f"Trajectory worker group {trajectory_name} already exists, connecting to it."
            )
            return cls.connect(
                name,
                Worker.current_worker,
                source_routes=source_routes,
                collector_config=collector_config,
            )

        channel = cls()
        channel.__dict__.update(normal_channel.__dict__)
        channel._initialize_trajectory(trajectory_name, worker_group, source_routes)
        return channel

    @classmethod
    def connect(
        cls,
        name: str,
        current_worker: Worker,
        *,
        source_routes: dict[int, list[tuple[int, int]]],
        collector_config: dict[str, Any],
    ) -> "TrajectoryChannel":
        """Connect to existing ordinary and trajectory worker groups."""
        del collector_config
        normal_channel = Channel.connect(name, current_worker)
        trajectory_name = _trajectory_group_name(name)
        worker_group = WorkerGroup.from_group_name(TrajectoryWorker, trajectory_name)
        channel = cls()
        channel.__dict__.update(normal_channel.__dict__)
        channel._initialize_trajectory(trajectory_name, worker_group, source_routes)
        return channel

    def _initialize_trajectory(
        self,
        trajectory_name: str,
        worker_group: WorkerGroup[TrajectoryWorker],
        source_routes: dict[int, list[tuple[int, int]]],
    ) -> None:
        self._trajectory_group_name = trajectory_name
        self._trajectory_worker_group = worker_group
        self._trajectory_workers_by_rank = {
            worker.rank: worker.worker for worker in worker_group.worker_info_list
        }
        self._source_routes = source_routes
        self._next_sequences: dict[int, int] = {}
        self._open_sources: set[int] = set()
        self._actor_item_counts = self._count_actor_items(source_routes)

    @staticmethod
    def _count_actor_items(
        source_routes: dict[int, list[tuple[int, int]]],
    ) -> dict[int, int]:
        """Count the shards that each actor must receive in one raw rollout."""
        counts: dict[int, int] = {}
        for routes in source_routes.values():
            for actor_rank, _ in routes:
                counts[actor_rank] = counts.get(actor_rank, 0) + 1
        return counts

    def _storage_rank(self, source_id: int) -> int:
        return source_id % len(self._trajectory_workers_by_rank)

    def _current_source(self, source_id: int) -> Worker:
        if self._current_worker is None:
            raise RuntimeError("TrajectoryChannel requires a Worker context.")
        if source_id not in self._source_routes:
            raise ValueError(f"Unknown trajectory source: {source_id}.")
        return self._current_worker

    def _send_event(
        self,
        source_id: int,
        kind: TrajectoryEventType,
        payload: dict[str, Any],
        *,
        async_op: bool,
    ) -> AsyncWork | None:
        worker = self._current_source(source_id)
        if kind is not TrajectoryEventType.OPEN and source_id not in self._open_sources:
            raise RuntimeError(f"Trajectory source {source_id} is not open.")
        sequence = self._next_sequences.get(source_id, 0)
        event = TrajectoryEvent(source_id, sequence, kind, payload)
        storage_rank = self._storage_rank(source_id)
        storage = self._trajectory_workers_by_rank[storage_rank]
        work = AsyncChannelWork(
            channel_name=self._trajectory_group_name,
            channel_key=str(source_id),
            channel_actor=storage,
            method="consume_event",
            src_addr=worker.worker_address,
        )
        worker.send(event, self._trajectory_group_name, storage_rank, async_op=True)
        if kind is TrajectoryEventType.FLUSH and payload["close"]:
            self._next_sequences.pop(source_id, None)
            self._open_sources.discard(source_id)
        else:
            self._next_sequences[source_id] = sequence + 1
        if async_op:
            return work
        work.wait()
        return None

    def open_trajectory(
        self, source_id: int, *, async_op: bool = True
    ) -> AsyncWork | None:
        """Open the collector stream for one EnvWorker stage."""
        if source_id in self._open_sources:
            return self.clear_rewards(source_id, async_op=async_op)
        self._open_sources.add(source_id)
        return self._send_event(
            source_id,
            TrajectoryEventType.OPEN,
            {"routes": self._source_routes[source_id]},
            async_op=async_op,
        )

    def append_step(
        self, source_id: int, step: Any, *, async_op: bool = True
    ) -> AsyncWork | None:
        """Append one existing ``ChunkStepResult`` to a source collector."""
        return self._send_event(
            source_id,
            TrajectoryEventType.APPEND_STEP,
            {"step": step},
            async_op=async_op,
        )

    def update_last_actions(
        self,
        source_id: int,
        actions: Any,
        flags: Any,
        *,
        async_op: bool = True,
    ) -> AsyncWork | None:
        """Patch the previous source action with environment intervention data."""
        return self._send_event(
            source_id,
            TrajectoryEventType.UPDATE_LAST_ACTIONS,
            {"actions": actions, "flags": flags},
            async_op=async_op,
        )

    def append_transitions(
        self,
        source_id: int,
        curr_obs: Any,
        next_obs: Any,
        *,
        async_op: bool = True,
    ) -> AsyncWork | None:
        """Append one RLT or environment transition to a source collector."""
        return self._send_event(
            source_id,
            TrajectoryEventType.APPEND_TRANSITIONS,
            {"curr_obs": curr_obs, "next_obs": next_obs},
            async_op=async_op,
        )

    def mark_last_step_with_intervene_flags(
        self, source_id: int, flags: Any, *, async_op: bool = True
    ) -> AsyncWork | None:
        """Attach rollout intervention flags to the previous source action."""
        return self._send_event(
            source_id,
            TrajectoryEventType.MARK_INTERVENE_FLAGS,
            {"flags": flags},
            async_op=async_op,
        )

    def assign_history_reward(
        self,
        source_id: int,
        reward_output: Any,
        assign_lengths: list[int],
        *,
        async_op: bool = True,
    ) -> AsyncWork | None:
        """Apply history-buffer reward to previous entries in a source collector."""
        return self._send_event(
            source_id,
            TrajectoryEventType.ASSIGN_HISTORY_REWARD,
            {"reward_output": reward_output, "assign_lengths": assign_lengths},
            async_op=async_op,
        )

    def append_episode_data(
        self, source_id: int, *, async_op: bool = True, **payload: Any
    ) -> AsyncWork | None:
        """Append one Online LeRobot chunk to a source collector."""
        return self._send_event(
            source_id,
            TrajectoryEventType.APPEND_EPISODE_DATA,
            payload,
            async_op=async_op,
        )

    def clear_rewards(
        self, source_id: int, *, async_op: bool = True
    ) -> AsyncWork | None:
        """Clear the previous round's reward history without closing a collector."""
        return self._send_event(
            source_id, TrajectoryEventType.CLEAR_REWARDS, {}, async_op=async_op
        )

    def reset_episode_buffers(
        self, source_id: int, *, async_op: bool = True
    ) -> AsyncWork | None:
        """Discard in-progress Online LeRobot episodes after an env reset."""
        return self._send_event(
            source_id,
            TrajectoryEventType.RESET_EPISODE_BUFFERS,
            {},
            async_op=async_op,
        )

    def flush_trajectory(
        self, source_id: int, *, close: bool, async_op: bool = True
    ) -> AsyncWork | None:
        """Flush completed data and optionally close a source collector."""
        return self._send_event(
            source_id,
            TrajectoryEventType.FLUSH,
            {"close": close},
            async_op=async_op,
        )

    def actor_item_count(self) -> int:
        """Return the exact number of source shards assigned to this actor rank."""
        if self._current_worker is None:
            raise RuntimeError(
                "TrajectoryChannel.actor_item_count() requires a Worker context."
            )
        return self._actor_item_counts.get(self._current_worker._rank, 0)

    def take(self, async_op: bool = False) -> AsyncWork | Any:
        """Take one completed trajectory item assigned to the current actor."""
        if self._current_worker is None:
            raise RuntimeError("TrajectoryChannel.take() requires a Worker context.")
        work = AsyncTrajectoryTakeWork(self)
        return work if async_op else work.wait()

    def try_take(self) -> Any | None:
        """Return one completed item for the current actor, if one is ready."""
        if self._current_worker is None:
            raise RuntimeError(
                "TrajectoryChannel.try_take() requires a Worker context."
            )
        query_id = uuid.uuid4().int
        coordinator = self._trajectory_workers_by_rank[0]
        coordinator.reserve.remote(
            dst_addr=self._current_worker.worker_address,
            query_id=query_id,
            actor_rank=self._current_worker._rank,
            nowait=True,
        )
        storage_rank, received_query_id = self._current_worker.recv(
            self._trajectory_group_name, 0
        )
        if received_query_id != query_id:
            raise RuntimeError("Received an unexpected trajectory reservation.")
        if storage_rank is None:
            return None
        return self._unwrap_item(self._take_from_storage(storage_rank))

    @staticmethod
    def _unwrap_item(item: Any) -> Any:
        if isinstance(item, TrajectoryFailure):
            raise RuntimeError(
                f"Trajectory source {item.source_id} failed in storage: {item.message}"
            )
        return item

    def _take_from_storage(
        self, storage_rank: int, *, async_op: bool = False
    ) -> AsyncWork | Any:
        query_id = uuid.uuid4().int
        storage = self._trajectory_workers_by_rank[storage_rank]
        storage.take.remote(
            dst_addr=self._current_worker.worker_address,
            query_id=query_id,
            actor_rank=self._current_worker._rank,
        )
        received = self._current_worker.recv(
            self._trajectory_group_name, storage_rank, async_op=True
        )
        work = AsyncChannelCommWork(
            async_comm_work=received,
            query_id=query_id,
            channel_actor=storage,
        )
        return work if async_op else work.wait()

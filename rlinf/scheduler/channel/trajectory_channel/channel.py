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

from __future__ import annotations

import asyncio
import copy
import threading
import uuid
from collections.abc import Hashable
from typing import Any, Generic, TypeVar

import ray.actor
from omegaconf import DictConfig
from torch.futures import Future

from rlinf.data.embodied_io_struct import TrajectoryData
from rlinf.scheduler import Cluster, ComponentPlacement, NodePlacementStrategy
from rlinf.scheduler.channel.trajectory_channel.compression import (
    CompressionConfigDict,
    CompressionLease,
    CompressionMetadata,
    TensorCompressor,
    get_compression_configs,
)
from rlinf.scheduler.channel.trajectory_channel.data_route import (
    DataRoute,
    DataRouteDict,
    Participant,
    RouteVia,
    get_data_routes,
)
from rlinf.scheduler.channel.trajectory_channel.owner_key import OwnerKey
from rlinf.scheduler.channel.trajectory_channel.workers import (
    PackedTrajectoryData,
    QueueKey,
    TrajectoryChannelWorker,
    TrajectoryControllerWorker,
    TrajectoryStorageWorker,
)
from rlinf.scheduler.collective.async_work import AsyncWork
from rlinf.scheduler.manager import WorkerAddress
from rlinf.scheduler.worker.worker import Worker
from rlinf.scheduler.worker.worker_group import WorkerGroup

T = TypeVar("T", bound=TrajectoryData)

_CONTROLLER_GROUP_NAME = "TrajectoryControllerGroup"
_CHANNEL_WORKER_GROUP_NAME = "TrajectoryChannelGroup"
_STORAGE_WORKER_GROUP_NAME = "TrajectoryStorageGroup"
_MAX_ACTOR_CONCURRENCY = 2**31 - 1


class AsyncPublishWork(AsyncWork):
    """Track completion of a packed-data send and its worker publication."""

    def __init__(
        self,
        publish_ref: ray.ObjectRef,
        send_work: AsyncWork,
        buffer_lease: CompressionLease | None = None,
    ):
        """Initialize work for one channel publication."""
        super().__init__()
        self._publish_ref = publish_ref
        self._send_work = send_work
        self._buffer_lease = buffer_lease
        self._completed: bool = False

    async def async_wait(self) -> None:
        """Wait asynchronously for both publication and transport."""
        if self._completed:
            return

        results = await asyncio.gather(
            self._send_work.async_wait(),
            self._publish_ref,
            return_exceptions=True,
        )
        self._complete()

        for result in results:
            if isinstance(result, BaseException):
                raise result

    def wait(self) -> None:
        """Wait synchronously for both publication and transport."""
        if self._completed:
            return

        publish_done: bool = False
        send_done: bool = False
        errors: list[BaseException] = []
        while not (publish_done and send_done):
            if not publish_done:
                ready, _ = ray.wait([self._publish_ref], timeout=0.01)
                if ready:
                    try:
                        ray.get(self._publish_ref)
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        publish_done = True
            if not send_done and self._send_work.done():
                try:
                    self._send_work.wait()
                except BaseException as error:
                    errors.append(error)
                finally:
                    send_done = True
        self._complete()

        if errors:
            raise errors[0]

    def done(self) -> bool:
        """Return whether both publication and transport have completed."""
        if self._completed:
            return True
        ready, _ = ray.wait([self._publish_ref], timeout=0)
        return self._send_work.done() and bool(ready)

    def _complete(self) -> None:
        if self._completed:
            return
        if self._buffer_lease is not None:
            self._buffer_lease.release()
        self._completed = True


class AsyncStoragePublishWork(AsyncWork):
    """Resolve a dynamic StorageWorker owner before sending a record."""

    def __init__(
        self,
        owner_ref: ray.ObjectRef,
        channel: TrajectoryChannel,
        packed_data: PackedTrajectoryData,
        queue_key: QueueKey,
        buffer_lease: CompressionLease | None,
    ) -> None:
        """Initialize deferred publication after owner resolution."""
        super().__init__()
        self._owner_ref = owner_ref
        self._channel = channel
        self._packed_data = packed_data
        self._queue_key = queue_key
        self._buffer_lease = buffer_lease
        self._publish_work: AsyncPublishWork | None = None

    async def async_wait(self) -> None:
        """Resolve the owner and wait asynchronously for publication."""
        if self._publish_work is None:
            try:
                owner = await self._owner_ref
                self._publish_work = self._channel._publish_to_storage_owner(
                    owner,
                    self._packed_data,
                    self._queue_key,
                    self._buffer_lease,
                )
            except BaseException:
                self._release_buffer()
                raise
        await self._publish_work.async_wait()

    def wait(self) -> None:
        """Resolve the owner and wait synchronously for publication."""
        if self._publish_work is None:
            try:
                owner = ray.get(self._owner_ref)
                self._publish_work = self._channel._publish_to_storage_owner(
                    owner,
                    self._packed_data,
                    self._queue_key,
                    self._buffer_lease,
                )
            except BaseException:
                self._release_buffer()
                raise
        self._publish_work.wait()

    def done(self) -> bool:
        """Return whether the resolved publication has completed."""
        return self._publish_work is not None and self._publish_work.done()

    def _release_buffer(self) -> None:
        if self._buffer_lease is not None:
            self._buffer_lease.release()
            self._buffer_lease = None


class AsyncTakeWork(AsyncWork, Generic[T]):
    """Receive and decode one record reserved from the channel."""

    _result_store: dict[int, Future] = {}
    _store_lock = threading.Lock()

    def __init__(
        self,
        reserve_ref: ray.ObjectRef,
        channel: "TrajectoryChannel",
        queue_key: QueueKey,
    ):
        """Initialize work for a controller reservation."""
        super().__init__()
        self._reserve_ref = reserve_ref
        self._queue_key = queue_key
        self._channel = channel
        self._completed = False
        self._result_future: Future[T] | None = Future()
        self._query_id = uuid.uuid4().int
        self._result: T = None
        with self._store_lock:
            self._result_store[self._query_id] = self._result_future

    async def async_wait(self) -> T:
        """Wait asynchronously for the reserved record."""
        if self._completed:
            return self._result
        try:
            comm_worker_address: WorkerAddress = await self._reserve_ref
            take_ref, recv_work = self._transfer(comm_worker_address)
            recv_result, _ = await asyncio.gather(
                recv_work.async_wait(),
                take_ref,
            )
            packed_data: PackedTrajectoryData
            query_id: int
            packed_data, query_id = recv_result
            self._handle_received_data(query_id, packed_data)

            if not self._result_future.done():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._result_future.wait)

            packed_result: PackedTrajectoryData = self._result_future.value()
            self._result = self._channel._decode(
                self._queue_key.data_type, packed_result
            )
            self._completed = True
            self._result_future = None
            return self._result
        finally:
            self._remove_result_future()

    def wait(self) -> T:
        """Wait synchronously for the reserved record."""
        if self._completed:
            return self._result
        try:
            comm_worker_address: WorkerAddress = ray.get(self._reserve_ref)
            take_ref, recv_work = self._transfer(comm_worker_address)
            recv_result = recv_work.wait()
            ray.get(take_ref)

            packed_data: PackedTrajectoryData
            packed_data, query_id = recv_result
            self._handle_received_data(query_id, packed_data)
            self._result_future.wait()
            packed_result: PackedTrajectoryData = self._result_future.value()
            self._result = self._channel._decode(
                self._queue_key.data_type, packed_result
            )
            self._completed = True
            self._result_future = None
            return self._result
        finally:
            self._remove_result_future()

    def done(self) -> bool:
        """Return whether the record has been decoded."""
        return self._completed or self._result_future.done()

    def _remove_result_future(self) -> None:
        with self._store_lock:
            self._result_store.pop(self._query_id, None)

    @classmethod
    def _handle_received_data(
        cls, query_id: int, packed_data: PackedTrajectoryData
    ) -> None:
        with cls._store_lock:
            future = cls._result_store.get(query_id)

        if future is None:
            raise ValueError(f"No future found for query_id {query_id}.")
        future.set_result(packed_data)

    def _transfer(self, address: WorkerAddress) -> tuple[ray.ObjectRef, AsyncWork]:
        comm_worker, group_name, rank = self._channel._resolve_take_source(
            address, self._queue_key
        )
        take_ref: ray.ObjectRef = comm_worker.take.remote(
            self._channel._current_worker.worker_address,
            self._queue_key,
            self._query_id,
        )
        recv_work: AsyncWork = self._channel._current_worker.recv(
            src_group_name=group_name,
            src_rank=rank,
            async_op=True,
        )

        return take_ref, recv_work


class TrajectoryChannel:
    """Typed transport for embodied trajectory records between participants."""

    def __init__(
        self,
        *,
        routes: DataRouteDict,
        compressions: CompressionConfigDict,
        controller_worker_group: WorkerGroup["TrajectoryControllerWorker"],
        channel_worker_group: WorkerGroup["TrajectoryChannelWorker"],
        storage_worker_group: WorkerGroup["TrajectoryStorageWorker"],
    ):
        """Initialize a channel backed by controller, transport, and storage workers."""
        self._controller_worker_group = controller_worker_group
        self._channel_worker_group = channel_worker_group
        self._storage_worker_group = storage_worker_group
        self._channel_worker_by_rank = {
            info.rank: info.worker for info in channel_worker_group.worker_info_list
        }
        self._storage_worker_by_rank = {
            info.rank: info.worker for info in storage_worker_group.worker_info_list
        }
        self._local_channel_worker_by_rank = {}
        self._local_storage_worker_by_rank = {}
        self._routes = routes
        self._compressions = compressions
        self._tensor_compressors: dict[type[TrajectoryData], TensorCompressor] = {}
        self._publish_cursors: dict[RouteVia, int] = {}
        self._current_worker: Worker | None = Worker.current_worker
        for route in self._routes.values():
            if route.via not in self._publish_cursors:
                self._publish_cursors[route.via] = 0

    @classmethod
    def create(
        cls,
        *,
        cfg: DictConfig,
        cluster: Cluster,
        placement: ComponentPlacement,
    ) -> "TrajectoryChannel":
        """Create the worker groups and a channel from the training configuration."""
        algorithm = str(cfg.algorithm.name)
        routes = get_data_routes(algorithm)
        compressions = get_compression_configs()

        channel_cfg = cfg.get("trajectory_channel", {})
        max_queue_size = int(channel_cfg.get("max_queue_size", 0))
        num_record_threads = int(channel_cfg.get("num_record_threads", 4))

        controller_worker_group = TrajectoryControllerWorker.create_group(
            routes
        ).launch(
            cluster,
            name=_CONTROLLER_GROUP_NAME,
            placement_strategy=NodePlacementStrategy([0]),
            max_concurrency=_MAX_ACTOR_CONCURRENCY,
            isolate_gpu=False,
            disable_distributed_log=True,
        )
        channel_worker_group = TrajectoryChannelWorker.create_group(
            max_queue_size=max_queue_size
        ).launch(
            cluster,
            name=_CHANNEL_WORKER_GROUP_NAME,
            placement_strategy=placement.get_strategy("trajectory_channel"),
            max_concurrency=_MAX_ACTOR_CONCURRENCY,
            isolate_gpu=False,
            disable_distributed_log=True,
        )
        storage_worker_group = TrajectoryStorageWorker.create_group(
            max_queue_size=max_queue_size,
            num_record_threads=num_record_threads,
        ).launch(
            cluster,
            name=_STORAGE_WORKER_GROUP_NAME,
            placement_strategy=placement.get_strategy("trajectory_storage"),
            max_concurrency=_MAX_ACTOR_CONCURRENCY,
            isolate_gpu=False,
            disable_distributed_log=True,
        )

        channel_worker_group.configure(routes, controller_worker_group).wait()
        storage_worker_group.configure(
            routes,
            compressions,
            controller_worker_group,
            algorithm,
            cfg,
            placement.get_world_size("actor"),
        ).wait()
        return cls(
            controller_worker_group=controller_worker_group,
            channel_worker_group=channel_worker_group,
            storage_worker_group=storage_worker_group,
            routes=routes,
            compressions=compressions,
        )

    def for_participant(self, participant: Participant) -> "TrajectoryChannel":
        """Return a route-restricted channel view for one participant."""
        channel = copy.copy(self)
        channel._routes = {
            data_type: route
            for data_type, route in self._routes.items()
            if route.src == participant or route.dst == participant
        }
        channel._compressions = {
            data_type: config
            for data_type, config in self._compressions.items()
            if data_type in channel._routes
        }
        return channel

    def publish(self, data: TrajectoryData, async_op: bool = False) -> AsyncWork | None:
        """Publish a typed record, optionally returning asynchronous work."""
        if self._current_worker is None:
            raise RuntimeError(
                "Current worker is not set while using TrajectoryChannel'S publish."
            )
        data_type = type(data)
        route = self._routes[data_type]
        queue_key = self._get_queue_key(route, data_type, data)
        packed_data, lease = self._encode(data)
        if route.owner_key is not None:
            work = AsyncStoragePublishWork(
                owner_ref=self._claim_storage_owner(route, data),
                channel=self,
                packed_data=packed_data,
                queue_key=queue_key,
                buffer_lease=lease,
            )
            if async_op:
                return work
            work.wait()
            return None

        comm_worker, group_name, rank = self._select_publish_target(route.via)
        publish_ref = comm_worker.publish.remote(self._current_worker.worker_address)
        send_work = self._current_worker.send(
            object=packed_data,
            dst_group_name=group_name,
            dst_rank=rank,
            async_op=True,
            piggyback_payload=queue_key,
        )
        async_publish_work = AsyncPublishWork(
            publish_ref=publish_ref,
            send_work=send_work,
            buffer_lease=lease,
        )
        if async_op:
            return async_publish_work
        async_publish_work.wait()
        return None

    def take(
        self,
        data_type: type[T],
        async_op: bool = False,
        *,
        partition: Hashable | None = None,
    ) -> AsyncTakeWork[T] | T:
        """Take a typed record, optionally returning asynchronous work."""
        if self._current_worker is None:
            raise RuntimeError(
                "Current worker is not set while using TrajectoryChannel'S take."
            )
        route = self._routes[data_type]
        queue_key = self._get_queue_key(route, data_type, partition=partition)
        controller_worker = self._controller_worker_group.worker_info_list[0].worker
        work = AsyncTakeWork[T](
            reserve_ref=controller_worker.reserve.remote(queue_key),
            queue_key=queue_key,
            channel=self,
        )
        return work if async_op else work.wait()

    def _encode(
        self, data: TrajectoryData
    ) -> tuple[PackedTrajectoryData, CompressionLease | None]:
        skeleton, tensors = data.flatten()
        metadata: CompressionMetadata | None = None
        lease: CompressionLease | None = None
        data_type = type(data)
        if data_type in self._tensor_compressors:
            compressor = self._tensor_compressors[data_type]
            compression_output = compressor.compress(tensors)
            metadata = compression_output.metadata
            lease = compression_output.lease
            tensors = compression_output.tensors
        packed_data = PackedTrajectoryData(
            skeleton=skeleton,
            tensors=tensors,
            compression_metadata=metadata,
        )
        return packed_data, lease

    def _decode(
        self,
        data_type: type[T],
        packed_data: PackedTrajectoryData,
    ) -> TrajectoryData:
        if not packed_data.compression_metadata:
            return data_type.from_flattened(packed_data.skeleton, packed_data.tensors)
        if data_type not in self._tensor_compressors:
            raise ValueError(
                f"No tensor compressor found for data type {data_type.__name__}."
            )
        decompressed_tensors = self._tensor_compressors[data_type].decompress(
            packed_data.tensors, packed_data.compression_metadata
        )
        return data_type.from_flattened(packed_data.skeleton, decompressed_tensors)

    def _select_publish_target(
        self, via: RouteVia
    ) -> tuple[ray.actor.ActorHandle, str, int]:
        if via == "channel_worker":
            worker_group = self._channel_worker_group
            local_workers = self._local_channel_worker_by_rank
            all_workers = self._channel_worker_by_rank
        elif via == "storage_worker":
            worker_group = self._storage_worker_group
            local_workers = self._local_storage_worker_by_rank
            all_workers = self._storage_worker_by_rank
        else:
            raise ValueError(f"unknown route via: {via!r}")

        workers = local_workers or all_workers
        if not workers:
            raise RuntimeError(f"no available workers for {via!r}")

        candidates = tuple(workers.items())
        cursor = self._publish_cursors.get(via, 0)
        rank, actor = candidates[cursor % len(candidates)]
        self._publish_cursors[via] = cursor + 1

        return actor, worker_group.worker_group_name, rank

    def _resolve_take_source(
        self, address: WorkerAddress, queue_key: QueueKey
    ) -> tuple[ray.actor.ActorHandle, str, int]:
        route = self._get_data_route(queue_key)
        if route.via == "channel_worker":
            group_name = self._channel_worker_group.worker_group_name
            comm_worker = self._channel_worker_by_rank[address.rank]
        elif route.via == "storage_worker":
            group_name = self._storage_worker_group.worker_group_name
            comm_worker = self._storage_worker_by_rank[address.rank]
        else:
            raise ValueError(f"Unknown route.via: {route.via}")
        return comm_worker, group_name, address.rank

    def _get_data_route(self, queue_key: QueueKey) -> DataRoute:
        data_type = queue_key.data_type
        if data_type not in self._routes:
            raise ValueError(f"No route defined for data type {data_type.__name__}.")
        route = self._routes[data_type]
        if route.src != queue_key.src or route.dst != queue_key.dst:
            raise ValueError(
                f"QueueKey {queue_key} does not match the defined route for data type {data_type.__name__}."
            )
        return route

    def _get_queue_key(
        self,
        route: DataRoute,
        data_type: type[TrajectoryData],
        data: TrajectoryData | None = None,
        partition: Hashable | None = None,
    ) -> QueueKey:
        if partition is not None:
            extra_key = partition
        elif route.extra_key is not None:
            if data is None:
                raise ValueError(
                    f"Route for {data_type.__name__} requires an explicit partition when taking data."
                )
            extra_key = getattr(data, route.extra_key)
        elif route.owner_key is not None and data is not None:
            if self._current_worker is None:
                raise RuntimeError("Current worker is unavailable.")
            extra_key = route.owner_key(data, self._current_worker.worker_address)
        else:
            extra_key = None
        return QueueKey(
            data_type=data_type,
            src=route.src,
            dst=route.dst,
            extra_key=extra_key,
        )

    def _claim_storage_owner(
        self, route: DataRoute, data: TrajectoryData
    ) -> ray.ObjectRef:
        if route.owner_key is None or self._current_worker is None:
            raise RuntimeError("Storage owner routing is unavailable.")
        owner_key: OwnerKey = route.owner_key(data, self._current_worker.worker_address)
        worker_group = self._storage_worker_group

        def addresses(
            workers: dict[int, ray.actor.ActorHandle],
        ) -> tuple[WorkerAddress, ...]:
            return tuple(
                WorkerAddress(worker_group.worker_group_name, rank) for rank in workers
            )

        controller = self._controller_worker_group.worker_info_list[0].worker
        return controller.claim_storage_worker.remote(
            owner_key,
            addresses(self._local_storage_worker_by_rank),
            addresses(self._storage_worker_by_rank),
        )

    def _publish_to_storage_owner(
        self,
        owner: WorkerAddress,
        packed_data: PackedTrajectoryData,
        queue_key: QueueKey,
        buffer_lease: CompressionLease | None,
    ) -> AsyncPublishWork:
        if self._current_worker is None:
            raise RuntimeError("Current worker is unavailable.")
        worker_group = self._storage_worker_group
        comm_worker = self._storage_worker_by_rank[owner.rank]
        publish_ref = comm_worker.publish.remote(self._current_worker.worker_address)
        send_work = self._current_worker.send(
            object=packed_data,
            dst_group_name=worker_group.worker_group_name,
            dst_rank=owner.rank,
            async_op=True,
            piggyback_payload=queue_key,
        )
        return AsyncPublishWork(
            publish_ref=publish_ref,
            send_work=send_work,
            buffer_lease=buffer_lease,
        )

    def _split_worker_group(
        self,
        worker_group: WorkerGroup[
            "TrajectoryChannelWorker" | "TrajectoryStorageWorker"
        ],
    ) -> tuple[WorkerGroup.WorkerRank, ...] | None:
        current_worker = self._current_worker
        if current_worker is None:
            return None

        current_node_rank = current_worker.worker_info.cluster_node_rank
        manager = current_worker.manager_proxy

        local_workers = []

        for worker in worker_group.worker_info_list:
            address = WorkerAddress(
                root_group_name=worker_group.worker_group_name,
                ranks=worker.rank,
            )
            worker_info = manager.get_worker_info(address)
            if worker_info.cluster_node_rank == current_node_rank:
                local_workers.append(worker)
        return tuple(local_workers)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore worker-local state after serialization."""
        self.__dict__.update(state)
        self._current_worker = Worker.current_worker
        self._tensor_compressors = {}
        self._publish_cursors = {}
        # setup tensor compressors for each data type
        for data_type, config in self._compressions.items():
            self._tensor_compressors[data_type] = TensorCompressor(config)

        # setup local worker mappings for channel and storage workers
        local_channel_worker_ranks = self._split_worker_group(
            self._channel_worker_group
        )
        self._local_channel_worker_by_rank = (
            {info.rank: info.worker for info in local_channel_worker_ranks}
            if local_channel_worker_ranks is not None
            else {}
        )
        local_storage_worker_ranks = self._split_worker_group(
            self._storage_worker_group
        )
        self._local_storage_worker_by_rank = (
            {info.rank: info.worker for info in local_storage_worker_ranks}
            if local_storage_worker_ranks is not None
            else {}
        )
        # used for worker selection in round-robin manner
        for route in self._routes.values():
            self._publish_cursors[route.via] = 0

    def __getstate__(self) -> dict[str, Any]:
        """Drop worker-local compressors before serialization."""
        state = self.__dict__.copy()
        state.pop("_tensor_compressors", None)
        state.pop("_local_channel_worker_by_rank", None)
        state.pop("_local_storage_worker_by_rank", None)
        state.pop("_publish_cursors", None)
        return state

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
from asyncio import Queue
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable

import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    TensorPath,
    TrajectoryData,
    TrajectoryRecord,
)
from rlinf.scheduler.channel.trajectory_channel.compression import (
    CompressionConfigDict,
    CompressionLease,
    CompressionMetadata,
    TensorCompressor,
)
from rlinf.scheduler.channel.trajectory_channel.data_route import (
    DataRouteDict,
    Participant,
)
from rlinf.scheduler.channel.trajectory_channel.owner_key import (
    BatchKey,
    LeRobotOwnerKey,
    OwnerKey,
)
from rlinf.scheduler.channel.trajectory_channel.storage import (
    LeRobotEpisodeBatch,
    TrajectoryBatch,
    TrajectoryStorage,
    create_trajectory_storage,
)
from rlinf.scheduler.worker.worker import Worker, WorkerAddress
from rlinf.scheduler.worker.worker_group import WorkerGroup


@dataclass(kw_only=True, frozen=True)
class QueueKey:
    """Identify one typed, directed queue and optional partition."""

    data_type: type[TrajectoryData]
    src: Participant | None
    dst: Participant | None = None
    extra_key: Hashable | None = None


@dataclass(kw_only=True, frozen=True)
class PackedTrajectoryData:
    """Store flattened record metadata and its tensor payload."""

    tensors: dict[TensorPath, torch.Tensor]
    skeleton: dict[str, Any]
    compression_metadata: CompressionMetadata | None = None


@dataclass(kw_only=True)
class _StoredData:
    packed: PackedTrajectoryData
    lease: CompressionLease | None = None
    release_owner_key: OwnerKey | None = None


class TrajectoryControllerWorker(Worker):
    """Coordinate queue availability and storage-worker ownership."""

    def __init__(
        self,
        routes: DataRouteDict,
    ):
        """Initialize routing, availability, and ownership state."""
        Worker.__init__(self)
        self._routes = routes
        self._availability_queues: dict[QueueKey, asyncio.Queue[WorkerAddress]] = (
            defaultdict(asyncio.Queue)
        )
        self._failure: BaseException | None = None
        self._failure_event = asyncio.Event()
        self._owners: dict[OwnerKey, WorkerAddress] = {}
        self._active_owners: dict[str, int] = defaultdict(int)
        self._owner_cursor = 0

    def claim_storage_worker(
        self,
        owner_key: OwnerKey,
        preferred_workers: tuple[WorkerAddress, ...],
        all_workers: tuple[WorkerAddress, ...],
    ) -> WorkerAddress:
        """Choose and pin the storage worker for one record owner."""
        if owner := self._owners.get(owner_key):
            return owner

        candidates = preferred_workers or all_workers
        if not candidates:
            raise RuntimeError("No StorageWorker is available.")
        start = self._owner_cursor % len(candidates)
        ordered = candidates[start:] + candidates[:start]
        owner = min(
            ordered,
            key=lambda worker: self._active_owners[worker.get_name()],
        )
        self._owner_cursor += 1
        self._owners[owner_key] = owner
        self._active_owners[owner.get_name()] += 1
        return owner

    def release_storage_worker(self, owner_key: OwnerKey) -> None:
        """Release an owner assignment after the actor receives its batch."""
        owner = self._owners.pop(owner_key)
        owner_name = owner.get_name()
        self._active_owners[owner_name] -= 1
        if self._active_owners[owner_name] == 0:
            del self._active_owners[owner_name]

    async def notify_available(
        self, queue_key: QueueKey, worker_address: WorkerAddress
    ) -> None:
        """Advertise that a worker can serve one queue key."""
        self._validate(queue_key)
        self._raise_if_failed()
        await self._availability_queues[queue_key].put(worker_address)

    async def reserve(self, queue_key: QueueKey) -> WorkerAddress:
        """Wait for and reserve a worker that can serve a queue key."""
        self._validate(queue_key)
        self._raise_if_failed()
        queue_get = asyncio.create_task(self._availability_queues[queue_key].get())
        failure_wait = asyncio.create_task(self._failure_event.wait())
        try:
            done, _ = await asyncio.wait(
                (queue_get, failure_wait),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_wait in done:
                self._raise_if_failed()
            return queue_get.result()
        finally:
            for task in (queue_get, failure_wait):
                if not task.done():
                    task.cancel()
            await asyncio.gather(queue_get, failure_wait, return_exceptions=True)

    def report_failure(self, error: BaseException) -> None:
        """Make storage failures visible to all waiting channel consumers."""
        if self._failure is None:
            self._failure = error
            self._failure_event.set()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "Trajectory channel storage worker failed."
            ) from self._failure

    def _validate(self, queue_key: QueueKey) -> None:
        data_type = queue_key.data_type
        if data_type not in self._routes:
            raise ValueError(f"No route defined for data type {data_type.__name__}.")
        route = self._routes[data_type]
        if route.src != queue_key.src or route.dst != queue_key.dst:
            raise ValueError(
                f"QueueKey {queue_key} does not match the defined route for data type {data_type.__name__}."
            )


class TrajectoryStorageWorker(Worker):
    """Persist trajectory records and publish completed training batches."""

    def __init__(self, max_queue_size: int = 0, num_record_threads: int = 4):
        """Initialize bounded queues and storage configuration placeholders."""
        Worker.__init__(self)
        self._routes: DataRouteDict = {}
        self._queues: dict[QueueKey, Queue[_StoredData]] = {}
        self._tensor_compressors: dict[type[TrajectoryData], TensorCompressor] = {}
        self._storage: TrajectoryStorage[TrajectoryData] | None = None
        self._output_task: asyncio.Task[None] | None = None
        self._max_queue_size = max_queue_size
        self._num_record_threads = num_record_threads

    async def configure(
        self,
        routes: DataRouteDict,
        compression_configs: CompressionConfigDict,
        controller_worker_group: WorkerGroup[TrajectoryControllerWorker],
        algorithm_name: str,
        cfg: DictConfig,
        actor_world_size: int,
    ) -> None:
        """Configure storage routes, compressors, and the output publisher."""
        self._routes = {
            data_type: route
            for data_type, route in routes.items()
            if route.via == "storage_worker"
        }
        self._tensor_compressors = {
            data_type: TensorCompressor(config)
            for data_type, config in compression_configs.items()
            if data_type in self._routes
        }
        self._storage = create_trajectory_storage(
            algorithm_name,
            cfg,
            actor_world_size,
            max_queue_size=self._max_queue_size,
            num_record_threads=self._num_record_threads,
        )
        self._queues = {}
        for data_type, route in self._routes.items():
            key = QueueKey(data_type=data_type, src=route.src, dst=route.dst)
            self._queues[key] = Queue(maxsize=self._max_queue_size)
        self._controller_worker = controller_worker_group.worker_info_list[0].worker
        self._output_task = asyncio.create_task(self._publish_completed_outputs())

    def _get_or_create_queue(self, key: QueueKey) -> Queue[_StoredData]:
        route = self._routes.get(key.data_type)
        if route is None:
            raise ValueError(
                f"No route defined for data type {key.data_type.__name__}."
            )
        if route.src != key.src or route.dst != key.dst:
            raise ValueError(
                f"QueueKey {key} does not match the defined route for data type {key.data_type.__name__}."
            )

        if key in self._queues:
            return self._queues[key]
        else:
            self._queues[key] = Queue(maxsize=self._max_queue_size)
            return self._queues[key]

    async def publish(self, src_addr: WorkerAddress) -> None:
        """Receive a packed record and route it to storage or a destination queue."""
        item, queue_key = self.recv(src_addr.root_group_name, src_addr.rank_path)
        route = self._routes[queue_key.data_type]
        if route.dst is None:
            if self._storage is None:
                raise RuntimeError("TrajectoryStorageWorker is not configured.")
            record = await self._decode(queue_key.data_type, item)
            if not isinstance(record, TrajectoryRecord):
                raise TypeError(
                    f"Storage input must be TrajectoryRecord, got {type(record).__name__}."
                )
            owner_key = queue_key.extra_key
            if not isinstance(owner_key, (BatchKey, LeRobotOwnerKey)):
                raise ValueError("Storage record has no valid owner key.")
            await self._storage.record(record, owner_key)
            return

        queue = self._get_or_create_queue(queue_key)
        await queue.put(_StoredData(packed=item))
        await self._controller_worker.notify_available.remote(
            queue_key, self.worker_address
        )

    async def take(
        self, dst_addr: WorkerAddress, queue_key: QueueKey, query_id: int
    ) -> None:
        """Send one queued packed record to its requesting worker."""
        stored = await self._queues[queue_key].get()
        send_work = self.send(
            object=stored.packed,
            dst_group_name=dst_addr.root_group_name,
            dst_rank=dst_addr.rank_path,
            async_op=True,
            piggyback_payload=query_id,
        )
        if stored.lease is None and stored.release_owner_key is None:
            return
        try:
            await send_work.async_wait()
        finally:
            if stored.lease is not None:
                stored.lease.release()
        if stored.release_owner_key is not None:
            await self._controller_worker.release_storage_worker.remote(
                stored.release_owner_key
            )

    async def _publish_completed_outputs(self) -> None:
        if self._storage is None:
            raise RuntimeError("TrajectoryStorageWorker is not configured.")
        try:
            while True:
                output = await self._storage.take()
                if not isinstance(output, (TrajectoryBatch, LeRobotEpisodeBatch)):
                    raise TypeError(
                        f"Unsupported storage output {type(output).__name__}."
                    )
                packed, lease = await self._encode(output)
                route = self._routes[type(output)]
                queue_key = QueueKey(
                    data_type=type(output),
                    src=route.src,
                    dst=route.dst,
                )
                queue = self._get_or_create_queue(queue_key)
                release_owner_key = None
                if isinstance(output, TrajectoryBatch):
                    release_owner_key = BatchKey(
                        global_step=output.global_step,
                        actor_rank=output.actor_rank,
                    )
                await queue.put(
                    _StoredData(
                        packed=packed,
                        lease=lease,
                        release_owner_key=release_owner_key,
                    )
                )
                await self._controller_worker.notify_available.remote(
                    queue_key, self.worker_address
                )
        except Exception as error:
            await self._controller_worker.report_failure.remote(error)
            raise

    async def _decode(
        self,
        data_type: type[TrajectoryData],
        packed: PackedTrajectoryData,
    ) -> TrajectoryData:
        tensors = packed.tensors
        if packed.compression_metadata:
            compressor = self._tensor_compressors.get(data_type)
            if compressor is None:
                raise ValueError(
                    f"No tensor compressor configured for {data_type.__name__}."
                )
            tensors = await compressor.decompress_async(
                tensors, packed.compression_metadata
            )
        return data_type.from_flattened(packed.skeleton, tensors)

    async def _encode(
        self, data: TrajectoryData
    ) -> tuple[PackedTrajectoryData, CompressionLease | None]:
        skeleton, tensors = data.flatten()
        compressor = self._tensor_compressors.get(type(data))
        if compressor is None:
            return PackedTrajectoryData(skeleton=skeleton, tensors=tensors), None
        output = await compressor.compress_async(tensors)
        return (
            PackedTrajectoryData(
                skeleton=skeleton,
                tensors=output.tensors,
                compression_metadata=output.metadata,
            ),
            output.lease,
        )


class TrajectoryChannelWorker(Worker):
    """Relay packed policy traffic between channel participants."""

    def __init__(self, max_queue_size: int = 0):
        """Initialize bounded relay queues."""
        Worker.__init__(self)
        self._routes: DataRouteDict = {}
        self._max_queue_size = max_queue_size
        self._queues: dict[QueueKey, Queue[PackedTrajectoryData]] = {}

    def configure(
        self,
        routes: DataRouteDict,
        controller_worker_group: WorkerGroup[TrajectoryControllerWorker],
    ) -> None:
        """Configure the relay routes and availability controller."""
        self._routes = {
            data_type: route
            for data_type, route in routes.items()
            if route.via == "channel_worker"
        }
        self._queues = {}
        for data_type, route in self._routes.items():
            key = QueueKey(data_type=data_type, src=route.src, dst=route.dst)
            self._queues[key] = Queue(maxsize=self._max_queue_size)

        self._controller_worker = controller_worker_group.worker_info_list[0].worker

    def _get_or_create_queue(self, key: QueueKey) -> Queue:
        route = self._routes.get(key.data_type)
        if route is None:
            raise ValueError(
                f"No route defined for data type {key.data_type.__name__}."
            )
        if route.src != key.src or route.dst != key.dst:
            raise ValueError(
                f"QueueKey {key} does not match the defined route for data type {key.data_type.__name__}."
            )
        if key in self._queues:
            return self._queues[key]
        else:
            self._queues[key] = Queue(maxsize=self._max_queue_size)
            return self._queues[key]

    async def publish(self, src_addr: WorkerAddress) -> None:
        """Receive and enqueue one packed policy record."""
        item, key = self.recv(src_addr.root_group_name, src_addr.rank_path)
        queue = self._get_or_create_queue(key)
        await queue.put(item)
        await self._controller_worker.notify_available.remote(key, self.worker_address)

    async def take(
        self, dst_addr: WorkerAddress, queue_key: QueueKey, query_id: int
    ) -> None:
        """Send one queued packed policy record to its destination."""
        packed_data = await self._queues[queue_key].get()
        self.send(
            object=packed_data,
            dst_group_name=dst_addr.root_group_name,
            dst_rank=dst_addr.rank_path,
            async_op=True,
            piggyback_payload=query_id,
        )

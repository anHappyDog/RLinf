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

"""Trajectory channel worker orchestration."""

import asyncio
from collections import defaultdict

from omegaconf import DictConfig

from rlinf.scheduler.channel.trajectory_channel.assembler import (
    TrajectoryEventAssembler,
)
from rlinf.scheduler.channel.trajectory_channel.collectors import (
    create_trajectory_collector,
)
from rlinf.scheduler.channel.trajectory_channel.data import TrajectoryData
from rlinf.scheduler.worker.worker import Worker, WorkerAddress


class TrajectoryWorker(Worker):
    """Receive trajectory events and serve collector outputs."""

    def __init__(self, cfg: DictConfig, max_size: int = 0):
        """Initialize event assembly, collection, and output queues."""
        super().__init__()
        self._cfg = cfg
        self._event_lock = asyncio.Lock()
        self._output_queues: dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_size)
        )

        from rlinf.scheduler.cluster import Cluster
        from rlinf.utils.metric_utils import compute_split_num
        from rlinf.utils.placement import HybridComponentPlacement

        placement = HybridComponentPlacement(cfg, Cluster())
        env_world_size = placement.get_world_size("env")
        actor_world_size = placement.get_world_size("actor")
        self.source_count = env_world_size * cfg.rollout.pipeline_stage_num
        self.chunk_count = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.output_count = (
            compute_split_num(self.source_count, actor_world_size) * actor_world_size
        )
        if self.output_count % self.source_count:
            raise ValueError(
                "Trajectory routing requires each rollout source to have an equal "
                "number of actor shards."
            )
        self.shards_per_source = self.output_count // self.source_count

        self._assembler = TrajectoryEventAssembler(
            source_batch_size=cfg.env.train.total_num_envs // self.source_count
        )
        self._collector = create_trajectory_collector(
            cfg,
            source_count=self.source_count,
            chunk_count=self.chunk_count,
            shards_per_source=self.shards_per_source,
            actor_world_size=actor_world_size,
        )

    async def publish(self, worker_address: WorkerAddress) -> None:
        """Receive and apply one trajectory event from a producer."""
        data = await self.recv(
            src_group_name=worker_address.root_group_name,
            src_rank=worker_address.rank,
            async_op=True,
        ).async_wait()
        async with self._event_lock:
            await self._apply_event(data)

    async def _apply_event(self, data: TrajectoryData) -> None:
        """Assemble one event and enqueue any completed collector outputs."""
        for chunk in self._assembler.push(data):
            outputs = self._collector.collect(chunk)
            self._assembler.acknowledge(chunk.key)
            for output in outputs:
                await self._output_queues[output.queue_key].put(output.data)

    async def subscribe(
        self, worker_address: WorkerAddress, queue_key: str, query_id: int
    ) -> None:
        """Send the next queued item to a subscriber."""
        data = await self._output_queues[queue_key].get()
        await self.send(
            object=data,
            dst_group_name=worker_address.root_group_name,
            dst_rank=worker_address.rank,
            piggyback_payload=query_id,
            async_op=True,
        ).async_wait()

    async def try_subscribe(
        self, worker_address: WorkerAddress, queue_key: str, query_id: int
    ) -> bool:
        """Submit a send if an assembled item is immediately available."""
        try:
            data = self._output_queues[queue_key].get_nowait()
        except asyncio.QueueEmpty:
            return False

        self.send(
            object=data,
            dst_group_name=worker_address.root_group_name,
            dst_rank=worker_address.rank,
            piggyback_payload=query_id,
            async_op=True,
        )
        return True

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

import os
import signal
import time

import pytest
import ray

from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.workers.trajectory import (
    ChannelConfig,
    RoutePlan,
    StorageConfig,
    StorageWorkerConfig,
    TrajectoryChannelWorker,
    TrajectoryStorageWorker,
    WorkerLayout,
    WorkerState,
)


@pytest.fixture(scope="module")
def cluster():
    cluster = Cluster(num_nodes=1)
    yield cluster


def _storage_config(slot_ids: tuple[int, ...] = (0, 1)) -> StorageWorkerConfig:
    return StorageWorkerConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(total_slots=2, world_sizes={"storage": 1}),
        storage=StorageConfig(
            global_step=3,
            rollout_epochs=1,
            chunk_steps=1,
            slot_ids=slot_ids,
        ),
        registry_modules=("rlinf.models.embodiment.openpi.forward_inputs",),
    )


def _channel_config() -> ChannelConfig:
    return ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(2, {"env": 1, "rollout": 1}),
        env_layout=WorkerLayout((0,)),
        rollout_layout=WorkerLayout((0,)),
    )


def _actor(group):
    return group.worker_info_list[0].worker


def test_worker_layout_maps_physical_to_logical_ranks():
    layout = WorkerLayout((2, 5, 9))

    assert layout.logical_rank(2) == 0
    assert layout.logical_rank(9) == 2
    with pytest.raises(ValueError, match="not a data-owning rank"):
        layout.logical_rank(3)


def test_storage_config_rejects_route_and_slot_mismatches(cluster):
    placement = NodePlacementStrategy([0])
    group = TrajectoryStorageWorker.create_group().launch(
        cluster,
        placement,
        name="trajectory_storage_invalid",
        catch_system_failure=False,
    )
    actor = _actor(group)

    invalid = _storage_config(slot_ids=(1,))
    with pytest.raises(ray.exceptions.RayTaskError, match="do not match"):
        ray.get(actor.configure.remote(invalid))
    health = ray.get(actor.health.remote())
    assert health.state is WorkerState.FAILED
    assert "ValueError" in health.detail

    ray.kill(actor)
    group.worker_info_list.clear()


def test_independent_workers_configure_drain_and_shutdown(cluster):
    placement = NodePlacementStrategy([0])
    channel = TrajectoryChannelWorker.create_group(maxsize=4).launch(
        cluster,
        placement,
        name="trajectory_channel_lifecycle",
        max_concurrency=4,
        catch_system_failure=False,
    )
    storage = TrajectoryStorageWorker.create_group().launch(
        cluster,
        placement,
        name="trajectory_storage_lifecycle",
        catch_system_failure=False,
    )

    channel_health = channel.configure(_channel_config()).wait()[0]
    storage_health = storage.configure(_storage_config()).wait()[0]
    assert channel_health.state is WorkerState.READY
    assert storage_health.state is WorkerState.READY
    assert storage.endpoint_ids().wait() == [()]
    assert storage.trajectory_ready().wait() == [False]

    assert channel.drain().wait()[0].state is WorkerState.DRAINING
    assert storage.drain().wait()[0].state is WorkerState.DRAINING
    assert channel.shutdown().wait()[0].state is WorkerState.STOPPED
    assert storage.shutdown().wait()[0].state is WorkerState.STOPPED

    channel._close()
    storage._close()


def test_blocked_or_dead_storage_does_not_block_channel(cluster):
    placement = NodePlacementStrategy([0])
    channel = TrajectoryChannelWorker.create_group().launch(
        cluster,
        placement,
        name="trajectory_channel_isolation",
        max_concurrency=4,
        catch_system_failure=False,
    )
    storage = TrajectoryStorageWorker.create_group().launch(
        cluster,
        placement,
        name="trajectory_storage_blocking",
        catch_system_failure=False,
    )
    channel.configure(_channel_config()).wait()
    storage.configure(_storage_config()).wait()
    channel_actor = _actor(channel)
    storage_actor = _actor(storage)

    storage_health = ray.get(storage_actor.health.remote())
    os.kill(storage_health.process_id, signal.SIGSTOP)
    try:
        started = time.monotonic()
        health = ray.get(channel_actor.health.remote(), timeout=1.0)
        assert health.state is WorkerState.READY
        assert time.monotonic() - started < 1.0
    finally:
        os.kill(storage_health.process_id, signal.SIGCONT)

    ray.kill(storage_actor)
    storage.worker_info_list.clear()
    health = ray.get(channel_actor.health.remote(), timeout=1.0)
    assert health.state is WorkerState.READY

    channel.shutdown().wait()
    channel._close()

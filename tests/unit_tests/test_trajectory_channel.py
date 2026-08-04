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

"""Unit tests for TrajectoryChannel collection semantics."""

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch.futures import Future

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedLerobotRolloutResult,
    assign_history_reward,
)
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel, TrajectoryChannel
from rlinf.scheduler.channel.channel_worker import ChannelWorker
from rlinf.scheduler.channel.trajectory_channel import (
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryFailure,
    TrajectoryWorker,
)
from rlinf.scheduler.collective.async_work import AsyncChannelWork


def _raw_collector_config() -> dict:
    """Return the smallest valid raw collector configuration."""
    return {
        "type": "raw",
        "max_episode_length": 8,
        "num_envs": 2,
        "only_success": False,
        "num_action_chunks": 1,
        "action_dim": 1,
        "reward_weight": 0.5,
    }


def _worker(collector_config: dict | None = None) -> TrajectoryWorker:
    """Build an in-memory TrajectoryWorker without launching a Ray worker."""
    worker = object.__new__(TrajectoryWorker)
    worker._collector_config = collector_config or _raw_collector_config()
    worker._maxsize = 0
    worker._collectors = {}
    worker._routes = {}
    worker._next_sequences = {}
    worker._completed = {}
    worker._ready = {}
    worker._rank = 0
    return worker


def test_trajectory_event_contains_source_order_and_operation() -> None:
    """Events are explicit ordered mutations, not trajectory segments."""
    event = TrajectoryEvent(3, 7, TrajectoryEventType.APPEND_STEP, {"step": "payload"})

    assert event.source_id == 3
    assert event.sequence == 7
    assert event.kind is TrajectoryEventType.APPEND_STEP
    assert event.payload == {"step": "payload"}


def test_worker_collects_then_materializes_once_at_close() -> None:
    """Raw collectors stay mutable until the final event materializes shards."""
    worker = _worker()
    source_id = 2
    routes = [(0, 1), (1, 1)]
    step = ChunkStepResult(
        actions=torch.tensor([[1.0], [2.0]]),
        rewards=torch.tensor([[3.0], [4.0]]),
        dones=torch.tensor([[False], [False]]),
    )

    async def collect() -> None:
        await worker._apply_event(
            TrajectoryEvent(source_id, 0, TrajectoryEventType.OPEN, {"routes": routes})
        )
        await worker._apply_event(
            TrajectoryEvent(
                source_id, 1, TrajectoryEventType.APPEND_STEP, {"step": step}
            )
        )
        await worker._apply_event(
            TrajectoryEvent(source_id, 2, TrajectoryEventType.FLUSH, {"close": False})
        )
        assert worker._completed == {}
        await worker._apply_event(
            TrajectoryEvent(source_id, 3, TrajectoryEventType.FLUSH, {"close": True})
        )

    asyncio.run(collect())

    assert source_id not in worker._collectors
    assert worker._completed[0].get_nowait().actions.tolist() == [[[1.0]]]
    assert worker._completed[1].get_nowait().actions.tolist() == [[[2.0]]]


def test_history_reward_helper_preserves_existing_assignment_semantics() -> None:
    """History reward only updates prior entries, never the latest reward."""
    rewards = [
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
    ]

    assign_history_reward(
        rewards,
        torch.tensor([[2.0], [4.0]]),
        assign_lengths=[3, 2],
        reward_weight=0.5,
    )

    assert rewards[0].tolist() == [[1.0], [0.0]]
    assert rewards[1].tolist() == [[1.0], [2.0]]
    assert rewards[2].tolist() == [[0.0], [0.0]]


def test_storage_routing_prefers_the_source_node() -> None:
    """A storage worker on the source node keeps every shard within one hop."""
    storage_ranks = TrajectoryChannel.plan_storage_ranks(
        source_nodes={0: 0, 1: 1, 2: 0},
        source_routes={0: [(0, 2)], 1: [(1, 2)], 2: [(1, 2)]},
        actor_nodes={0: 1, 1: 0},
        storage_nodes={0: 0, 1: 1},
    )

    assert storage_ranks == {0: 0, 1: 1, 2: 0}


def test_storage_routing_avoids_a_second_cross_node_hop() -> None:
    """Without source-local storage, actor-local storage is the next best path."""
    storage_ranks = TrajectoryChannel.plan_storage_ranks(
        source_nodes={0: 0},
        source_routes={0: [(0, 4)]},
        actor_nodes={0: 1},
        storage_nodes={0: 1, 1: 2},
    )

    assert storage_ranks == {0: 0}


def test_actor_item_count_matches_comm_mapper_routes() -> None:
    """Actors consume the exact number of shards they are assigned."""
    routes = {
        0: [(0, 2)],
        1: [(0, 1), (1, 1)],
        2: [(1, 2)],
    }
    channel = object.__new__(TrajectoryChannel)
    channel._actor_item_counts = TrajectoryChannel._count_actor_items(routes)
    channel._current_worker = Mock(_rank=0)

    assert channel.actor_item_count() == 2
    channel._current_worker._rank = 1
    assert channel.actor_item_count() == 2


def test_lerobot_flush_keeps_in_progress_episodes_until_close() -> None:
    """Online LeRobot keeps unfinished episodes across rollout rounds."""
    config = _raw_collector_config()
    config["type"] = "lerobot"
    worker = _worker(config)
    source_id = 0

    async def flush() -> None:
        await worker._apply_event(
            TrajectoryEvent(
                source_id, 0, TrajectoryEventType.OPEN, {"routes": [(0, 2)]}
            )
        )
        collector = worker._collectors[source_id]
        assert isinstance(collector, EmbodiedLerobotRolloutResult)
        collector._env_buffers[0].append({"frame": 1})
        collector.rewards.append(torch.ones(2, 1))

        await worker._apply_event(
            TrajectoryEvent(source_id, 1, TrajectoryEventType.FLUSH, {"close": False})
        )
        assert collector._env_buffers[0] == [{"frame": 1}]

        await worker._apply_event(
            TrajectoryEvent(source_id, 2, TrajectoryEventType.CLEAR_REWARDS, {})
        )
        assert collector.rewards == []

        await worker._apply_event(
            TrajectoryEvent(source_id, 3, TrajectoryEventType.RESET_EPISODE_BUFFERS, {})
        )
        assert collector._env_buffers == [[], []]

        await worker._apply_event(
            TrajectoryEvent(source_id, 4, TrajectoryEventType.FLUSH, {"close": True})
        )

    asyncio.run(flush())
    assert source_id not in worker._collectors


def test_worker_failure_is_queued_for_each_assigned_actor() -> None:
    """Collector failures reach consumers instead of leaving them blocked."""
    worker = _worker()
    event = TrajectoryEvent(
        1, 0, TrajectoryEventType.OPEN, {"routes": [(0, 1), (1, 1)]}
    )

    asyncio.run(worker._send_failure(event, ValueError("bad step")))

    for actor_rank in (0, 1):
        item = worker._completed[actor_rank].get_nowait()
        assert item == TrajectoryFailure(1, "bad step")
        assert worker._ready[actor_rank].get_nowait() == 0


def test_async_channel_work_propagates_rpc_failure() -> None:
    """A failed RPC completes its work and prevents a silent wait forever."""
    work = object.__new__(AsyncChannelWork)
    work._future = Future()
    rpc = ConcurrentFuture()
    rpc.set_exception(ValueError("bad RPC"))

    work._complete(rpc)

    with pytest.raises(ValueError, match="bad RPC"):
        work.wait()


def test_async_channel_work_stops_a_source_chain_after_failure() -> None:
    """A failed event prevents later events for the same source from running."""
    work = object.__new__(AsyncChannelWork)
    work._future = Future()
    work._execute = Mock()
    work._on_execute = Mock()
    previous = Future()
    previous.set_exception(ValueError("bad event"))

    work._execute_after(previous)

    work._execute.assert_not_called()
    work._on_execute.assert_not_called()
    with pytest.raises(ValueError, match="bad event"):
        work.wait()


def test_trajectory_worker_is_separate_from_channel_worker() -> None:
    """Trajectory placement does not alter ordinary ChannelWorker placement."""
    assert not issubclass(TrajectoryWorker, ChannelWorker)


def test_trajectory_channel_exposes_events_not_segment_publish() -> None:
    """The public trajectory API cannot reintroduce Env-side materialization."""
    assert not hasattr(TrajectoryChannel, "publish")
    assert hasattr(TrajectoryChannel, "append_step")
    assert hasattr(TrajectoryChannel, "flush_trajectory")


def test_runner_registers_all_source_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner derives source routes once and passes them to TrajectoryChannel."""
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {},
            "algorithm": {},
            "rollout": {"pipeline_stage_num": 2},
            "actor": {"model": {"num_action_chunks": 1, "action_dim": 7}},
            "env": {"train": {"total_num_envs": 16, "max_episode_steps": 8}},
            "cluster": {"component_placement": {"trajectory": "all"}},
        }
    )

    placement = Mock()
    placement.get_world_size.side_effect = lambda name: {"env": 2, "actor": 4}[name]
    strategy = Mock()
    strategy.get_placement.side_effect = lambda _: [
        SimpleNamespace(rank=rank, cluster_node_rank=rank // 2) for rank in range(4)
    ]
    placement.get_strategy.return_value = strategy
    monkeypatch.setattr(
        "rlinf.runners.embodied_runner.ComponentPlacement", Mock(return_value=placement)
    )
    monkeypatch.setattr("rlinf.runners.embodied_runner.Cluster", Mock())
    channel_create = Mock()
    trajectory_create = Mock(return_value=object())
    monkeypatch.setattr(Channel, "create", channel_create)
    monkeypatch.setattr(TrajectoryChannel, "create", trajectory_create)

    runner._create_actor_channel()

    routes = trajectory_create.call_args.kwargs["source_routes"]
    assert routes == {
        0: [(0, 4)],
        1: [(1, 4)],
        2: [(2, 4)],
        3: [(3, 4)],
    }
    assert trajectory_create.call_args.kwargs["source_storage_ranks"] == {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
    }
    assert trajectory_create.call_args.kwargs["collector_config"]["type"] == "raw"
    assert not channel_create.called

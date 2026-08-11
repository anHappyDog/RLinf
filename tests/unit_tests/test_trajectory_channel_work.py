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

import asyncio
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.channel.trajectory_channel.trajectory_channel import (
    TrajectoryChannel,
)
from rlinf.scheduler.channel.trajectory_channel.work import AsyncSubscribeWork


class _CompletedRecv:
    def __init__(self, result):
        self.result = result

    def wait(self):
        return self.result

    def done(self):
        return True


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.remote = Mock(return_value=object())


class _TrajectoryWorkerActor:
    def __init__(self, result):
        self.try_subscribe = _RemoteMethod(result)


def test_sync_subscribe_matches_piggyback_query_id():
    query_id = 7
    work = AsyncSubscribeWork(
        subscribe_ref=object(),
        recv_work=_CompletedRecv(("trajectory", query_id)),
        query_id=query_id,
    )

    with patch("ray.get", return_value=None):
        assert work.wait() == "trajectory"

    assert query_id not in AsyncSubscribeWork._data_store


def test_publish_and_subscribe_require_worker_context():
    channel = object.__new__(TrajectoryChannel)
    channel._current_worker = None

    with pytest.raises(RuntimeError, match="within a Worker"):
        channel.publish(object())
    with pytest.raises(RuntimeError, match="within a Worker"):
        channel.subscribe()
    with pytest.raises(RuntimeError, match="within a Worker"):
        channel.try_subscribe()


def test_try_subscribe_empty_does_not_start_recv():
    channel = object.__new__(TrajectoryChannel)
    channel._current_worker = Mock()
    channel._trajectory_workers = {0: _TrajectoryWorkerActor(False)}

    with patch("ray.get", return_value=False), pytest.raises(asyncio.QueueEmpty):
        channel.try_subscribe()

    channel._current_worker.recv.assert_not_called()


def test_try_subscribe_reuses_control_ref_for_recv_work():
    channel = object.__new__(TrajectoryChannel)
    channel._current_worker = Mock()
    channel._current_worker.worker_address = "actor-address"
    channel._current_worker.recv.return_value = _CompletedRecv(("trajectory", 7))
    channel._trajectory_worker_group = Mock(worker_group_name="trajectory-worker")
    actor = _TrajectoryWorkerActor(True)
    channel._trajectory_workers = {0: actor}

    with (
        patch("uuid.uuid4", return_value=Mock(int=7)),
        patch("ray.get", side_effect=[True, True]),
    ):
        assert channel.try_subscribe() == "trajectory"

    actor.try_subscribe.remote.assert_called_once_with("actor-address", "default", 7)
    channel._current_worker.recv.assert_called_once_with(
        src_group_name="trajectory-worker", src_rank=0, async_op=True
    )


def test_create_rejects_invalid_trajectory_node_rank_before_launch():
    cfg = OmegaConf.create({"cluster": {"num_nodes": 2}})

    with pytest.raises(ValueError, match="trajectory_node_rank"):
        TrajectoryChannel.create("test", cfg, trajectory_node_rank=2)

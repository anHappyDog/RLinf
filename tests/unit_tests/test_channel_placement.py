# Copyright 2025 The RLinf Authors.
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

"""Placement resolution for non-distributed channels.

A channel that defaults to node 0 puts itself a network hop away from both ends
of its traffic whenever the workers it serves run elsewhere. These tests cover
the resolution that picks a node from the channel's own producers and consumers.
"""

from types import SimpleNamespace

import pytest

from rlinf.scheduler.cluster import resolve_colocation_node_rank


class _FakeGroup:
    """Stands in for a WorkerGroup with one worker per rank."""

    def __init__(self, name, node_ranks):
        self.worker_group_name = name
        self.worker_info_list = [
            SimpleNamespace(cluster_node_rank=n, rank=i)
            for i, n in enumerate(node_ranks)
        ]


@pytest.fixture
def fake_worker_group(monkeypatch):
    """Make the helper's isinstance check accept _FakeGroup."""
    import rlinf.scheduler.worker as worker_mod

    monkeypatch.setattr(worker_mod, "WorkerGroup", _FakeGroup)
    return _FakeGroup


def test_resolves_to_rank_zero_node(fake_worker_group):
    """The node of rank 0 is the one chosen, not the lowest node in the group."""
    group = _FakeGroup("Rollout", [2, 0, 1])
    assert resolve_colocation_node_rank(group) == 2


def test_prefers_producer_over_consumer(fake_worker_group):
    """Producers win: the collector runs channel-side, before the consumer hop."""
    producer = _FakeGroup("Rollout", [1])
    consumer = _FakeGroup("Actor", [3])
    assert resolve_colocation_node_rank(producer, consumer) == 1


def test_falls_back_to_consumer(fake_worker_group):
    """With no producer given, the consumer still pins the channel."""
    consumer = _FakeGroup("Actor", [3])
    assert resolve_colocation_node_rank(None, consumer) == 3


def test_accepts_iterables_and_skips_empty_groups(fake_worker_group):
    """A group that has not launched yet is skipped rather than chosen."""
    empty = _FakeGroup("NotLaunched", [])
    real = _FakeGroup("Rollout", [2])
    assert resolve_colocation_node_rank([empty, real]) == 2


def test_returns_none_when_nothing_resolves(fake_worker_group):
    """No producers and no consumers leaves the caller to pick a default."""
    assert resolve_colocation_node_rank(None, None) is None


def test_resolves_group_name_via_worker_manager(monkeypatch):
    """A group passed by name is looked up through the worker manager."""

    class _Proxy:
        def get_worker_info(self, address):
            assert address.root_group_name == "Rollout"
            return SimpleNamespace(cluster_node_rank=5)

    import rlinf.scheduler.manager as manager_mod

    monkeypatch.setattr(
        manager_mod.WorkerManager, "get_proxy", staticmethod(lambda: _Proxy())
    )
    assert resolve_colocation_node_rank("Rollout") == 5


def test_unregistered_group_name_is_skipped(monkeypatch):
    """An unknown name yields None instead of raising, so creation can proceed."""

    class _Proxy:
        def get_worker_info(self, address):
            return None

    import rlinf.scheduler.manager as manager_mod

    monkeypatch.setattr(
        manager_mod.WorkerManager, "get_proxy", staticmethod(lambda: _Proxy())
    )
    assert resolve_colocation_node_rank("Missing") is None

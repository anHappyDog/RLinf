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

import pickle

import pytest

from rlinf.workers.trajectory import RoutePlan


@pytest.fixture
def plan() -> RoutePlan:
    return RoutePlan(
        total_slots=10,
        world_sizes={
            "env": 3,
            "rollout": 2,
            "reward": 1,
            "storage": 4,
            "actor": 6,
        },
    )


def test_balanced_ranges_cover_every_slot_once(plan: RoutePlan) -> None:
    assert [plan.slot_range("env", rank) for rank in range(3)] == [
        (0, 4),
        (4, 7),
        (7, 10),
    ]
    assert [plan.slot_range("storage", rank) for rank in range(4)] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]

    covered = [
        slot
        for rank in range(plan.world_sizes["storage"])
        for slot in range(*plan.slot_range("storage", rank))
    ]
    assert covered == list(range(10))


def test_owner_and_local_index_are_inverse(plan: RoutePlan) -> None:
    for participant, world_size in plan.world_sizes.items():
        for slot_id in range(plan.total_slots):
            rank = plan.owner(participant, slot_id)
            local_index = plan.local_index(participant, rank, slot_id)
            assert 0 <= rank < world_size
            assert plan.global_slot(participant, rank, local_index) == slot_id


def test_source_batch_can_cross_multiple_storage_ranks(plan: RoutePlan) -> None:
    routes = plan.routes("env", 0, "storage")

    assert [route.destination_rank for route in routes] == [0, 1]
    assert [route.source_indices for route in routes] == [(0, 1, 2), (3,)]
    assert [route.destination_indices for route in routes] == [(0, 1, 2), (0,)]
    assert [route.slot_ids for route in routes] == [(0, 1, 2), (3,)]


def test_one_source_rank_can_span_every_storage_rank() -> None:
    plan = RoutePlan(
        total_slots=11,
        world_sizes={"env": 1, "storage": 4, "actor": 3},
    )

    routes = plan.routes("env", 0, "storage")

    assert [route.destination_rank for route in routes] == [0, 1, 2, 3]
    assert tuple(slot for route in routes for slot in route.slot_ids) == tuple(
        range(11)
    )


def test_sparse_routes_preserve_enough_information_to_restore_order(
    plan: RoutePlan,
) -> None:
    slot_ids = (7, 1, 9, 4)
    values = ("slot-7", "slot-1", "slot-9", "slot-4")

    routes = plan.route_slots("reward", 0, slot_ids, "storage")
    restored = [None] * len(values)
    for route in routes:
        shard = [values[index] for index in route.source_indices]
        for source_index, value in zip(route.source_indices, shard, strict=True):
            restored[source_index] = value

    assert [route.destination_rank for route in routes] == [0, 1, 2, 3]
    assert tuple(restored) == values
    assert (
        tuple(slot for route in routes for slot in route.slot_ids) != slot_ids
    )  # destination order differs, source indices recover it


def test_storage_to_actor_uses_the_same_mapping(plan: RoutePlan) -> None:
    routes = plan.routes("storage", 1, "actor")

    assert [route.slot_ids for route in routes] == [(3,), (4, 5)]
    assert [route.destination_rank for route in routes] == [1, 2]
    assert [route.destination_indices for route in routes] == [(1,), (0, 1)]


def test_empty_shard_has_no_routes() -> None:
    plan = RoutePlan(total_slots=3, world_sizes={"storage": 5, "actor": 2})

    assert plan.slot_range("storage", 3) == (3, 3)
    assert plan.slot_range("storage", 4) == (3, 3)
    assert plan.routes("storage", 3, "actor") == ()


def test_reference_profile_assigns_eight_slots_per_storage_rank() -> None:
    plan = RoutePlan(
        total_slots=32,
        world_sizes={"env": 4, "rollout": 4, "storage": 4, "actor": 4},
    )

    assert [plan.slot_range("storage", rank) for rank in range(4)] == [
        (0, 8),
        (8, 16),
        (16, 24),
        (24, 32),
    ]
    for rank in range(4):
        route = plan.routes("rollout", rank, "storage")
        assert len(route) == 1
        assert route[0].destination_rank == rank
        assert route[0].source_indices == tuple(range(8))
        assert route[0].destination_indices == tuple(range(8))


def test_route_plan_can_cross_process_boundaries(plan: RoutePlan) -> None:
    restored = pickle.loads(pickle.dumps(plan))

    assert restored.total_slots == plan.total_slots
    assert restored.world_sizes == plan.world_sizes
    assert restored.routes("env", 0, "storage") == plan.routes("env", 0, "storage")


def test_world_sizes_cannot_mutate_the_constructed_plan(plan: RoutePlan) -> None:
    world_sizes = plan.world_sizes
    world_sizes["env"] = 99

    assert plan.world_sizes["env"] == 3
    assert plan.slot_range("env", 2) == (7, 10)


def test_partition_and_route_invariants_hold_for_small_topologies() -> None:
    for total_slots in range(1, 13):
        for source_size in range(1, 7):
            for destination_size in range(1, 7):
                plan = RoutePlan(
                    total_slots=total_slots,
                    world_sizes={
                        "source": source_size,
                        "destination": destination_size,
                    },
                )
                routed_slots = []
                for source_rank in range(source_size):
                    source_start, source_end = plan.slot_range("source", source_rank)
                    routes = plan.routes("source", source_rank, "destination")
                    source_indices = tuple(
                        index for route in routes for index in route.source_indices
                    )
                    assert sorted(source_indices) == list(
                        range(source_end - source_start)
                    )
                    for route in routes:
                        assert list(route.slot_ids) == sorted(route.slot_ids)
                        for destination_index, slot_id in zip(
                            route.destination_indices,
                            route.slot_ids,
                            strict=True,
                        ):
                            assert (
                                plan.global_slot(
                                    "destination",
                                    route.destination_rank,
                                    destination_index,
                                )
                                == slot_id
                            )
                    routed_slots.extend(
                        slot_id for route in routes for slot_id in route.slot_ids
                    )
                assert sorted(routed_slots) == list(range(total_slots))


def test_route_plan_rejects_invalid_coordinates(plan: RoutePlan) -> None:
    with pytest.raises(ValueError, match="unknown route participant"):
        plan.slot_range("missing", 0)
    with pytest.raises(ValueError, match="invalid rank"):
        plan.slot_range("env", 3)
    with pytest.raises(ValueError, match="slot_id"):
        plan.owner("storage", 10)
    with pytest.raises(ValueError, match="not owned"):
        plan.local_index("storage", 0, 4)
    with pytest.raises(ValueError, match="slot_id"):
        plan.local_index("storage", 0, True)
    with pytest.raises(ValueError, match="local index"):
        plan.global_slot("storage", 0, 3)
    with pytest.raises(ValueError, match="duplicates"):
        plan.route_slots("env", 0, (1, 1), "storage")
    with pytest.raises(ValueError, match="not owned"):
        plan.route_slots("env", 0, (4,), "storage")

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

from __future__ import annotations

import collections

import pytest

from rlinf.data.b1k_grounded import (
    SubgoalStatus,
    render_primitive_subgoal,
    resolve_episode_subgoals,
)
from toolkits.b1k_grounded.build_pilot_dataset import (
    PilotBuildConfig,
    _pending_samples,
)


def _primitive(
    primitive_index,
    descriptions,
    primitive_ids,
    object_groups,
    skill_indices,
    *,
    memory_prefixes=None,
    spatial_prefixes=None,
):
    return {
        "primitive_idx": primitive_index,
        "primitive_id": primitive_ids,
        "primitive_description": descriptions,
        "object_id": object_groups,
        "manipulating_object_id": [],
        "memory_prefix": memory_prefixes or [],
        "spatial_prefix": spatial_prefixes or [],
        "frame_duration": [0, 20],
        "skill_idxes": skill_indices,
    }


def _skill(skill_index, description="move to", skill_id=1):
    return {
        "skill_idx": skill_index,
        "skill_id": [skill_id],
        "skill_description": [description],
        "object_id": [["radio_89"]],
        "manipulating_object_id": [],
        "memory_prefix": [],
        "spatial_prefix": [],
        "frame_duration": [0, 10],
        "mp_ef": [],
        "skill_type": ["navigation"],
    }


def test_render_primitive_subgoal_composes_operations_and_annotation_prefixes():
    primitive = _primitive(
        3,
        ["pick up from", "place on"],
        [2, 3],
        [
            ["radio_89", "coffee_table_koagbh_0"],
            ["radio_89", "coffee_table_koagbh_0"],
        ],
        [0, 1],
        memory_prefixes=["the other", "back"],
        spatial_prefixes=[[], [[], ["right"]]],
    )

    assert render_primitive_subgoal(primitive) == (
        "pick up the other radio from coffee table, then "
        "place radio back on coffee table (right)"
    )


def test_render_primitive_subgoal_preserves_qualifier_alignment():
    primitive = _primitive(
        1,
        ["pick up from"],
        [2],
        [["sandal_76", ["floor_1", "hall_tree_2"]]],
        [0],
        spatial_prefixes=[[[], ["", "high_level"]]],
    )

    assert render_primitive_subgoal(primitive) == (
        "pick up sandal from floor and hall tree (high level)"
    )


def test_resolve_episode_subgoals_reuses_unique_primitive_for_child_skills():
    annotation = {
        "skill_annotation": [_skill(0), _skill(1)],
        "primitive_annotation": [
            _primitive(
                0,
                ["pick up from"],
                [2],
                [["radio_89", "coffee_table_koagbh_0"]],
                [0, 1],
            )
        ],
    }

    resolved = resolve_episode_subgoals(annotation)

    assert resolved[0] == resolved[1]
    assert resolved[0].status is SubgoalStatus.UNIQUE
    assert resolved[0].primitive_indices == (0,)
    assert resolved[0].text == "pick up radio from coffee table"


def test_resolve_episode_subgoals_combines_shared_skill_deterministically():
    annotation = {
        "skill_annotation": [_skill(11)],
        "primitive_annotation": [
            _primitive(
                4,
                ["pick up from", "place in"],
                [2, 4],
                [
                    ["tennis_racket_80", "coffee_table_koagbh_0"],
                    ["tennis_racket_80", "car_ssxsje_0"],
                ],
                [11],
            ),
            _primitive(
                3,
                ["pick up from", "place in"],
                [2, 4],
                [
                    ["digital_camera_79", "coffee_table_koagbh_0"],
                    ["digital_camera_79", "toy_box_81"],
                ],
                [11],
            ),
        ],
    }

    resolved = resolve_episode_subgoals(annotation)[11]

    assert resolved.status is SubgoalStatus.COMPOSITE
    assert resolved.primitive_indices == (3, 4)
    assert resolved.text == (
        "pick up digital camera from coffee table, then place digital camera "
        "in toy box; also pick up tennis racket from coffee table, then place "
        "tennis racket in car"
    )


def test_resolve_episode_subgoals_renders_empty_primitive_from_atomic_skills():
    annotation = {
        "skill_annotation": [
            _skill(5),
            _skill(6, description="open door", skill_id=10),
        ],
        "primitive_annotation": [_primitive(4, [], [], [], [5, 6])],
    }

    resolved = resolve_episode_subgoals(annotation)

    assert resolved[5] == resolved[6]
    assert resolved[5].status is SubgoalStatus.UNIQUE
    assert resolved[5].text == "move to radio, then open door on radio"
    assert resolved[5].primitive_indices == (4,)


def test_resolve_episode_subgoals_keeps_unowned_skill_missing():
    annotation = {
        "skill_annotation": [_skill(5), _skill(6)],
        "primitive_annotation": [_primitive(4, ["move to"], [1], [["radio_89"]], [6])],
    }

    resolved = resolve_episode_subgoals(annotation)[5]

    assert resolved.status is SubgoalStatus.MISSING
    assert resolved.text is None
    assert resolved.primitive_indices == ()


def test_resolve_episode_subgoals_rejects_cross_episode_style_unknown_reference():
    annotation = {
        "skill_annotation": [_skill(0)],
        "primitive_annotation": [_primitive(0, ["move to"], [1], [["radio_89"]], [1])],
    }

    with pytest.raises(ValueError, match="unknown skills"):
        resolve_episode_subgoals(annotation)


def test_pending_samples_populates_primitive_subgoal_and_provenance():
    annotation = {
        "skill_annotation": [_skill(0)],
        "primitive_annotation": [
            _primitive(
                0,
                ["pick up from"],
                [2],
                [["radio_89", "coffee_table_koagbh_0"]],
                [0],
            )
        ],
    }

    pending, repairs = _pending_samples(
        annotation,
        goal="Turn on the radio.",
        episode_index=10,
        episode_length=10,
        config=PilotBuildConfig(),
        parse_status_counts=collections.Counter(),
    )

    assert repairs == []
    assert len(pending) == 1
    assert pending[0].control.subgoal == "pick up radio from coffee table"
    assert pending[0].subgoal_status is SubgoalStatus.UNIQUE
    assert pending[0].subgoal_primitive_indices == (0,)

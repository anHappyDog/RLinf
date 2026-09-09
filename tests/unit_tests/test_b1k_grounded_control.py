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

import json

import pytest

from rlinf.data.b1k_grounded import (
    DEFAULT_SKILL_SIGNATURE_REGISTRY,
    CameraID,
    EntityArgument,
    GroundedControlSpec,
    Grounding2D,
    ParseStatus,
    PartArgument,
    Role,
    parse_skill_annotation,
)
from rlinf.data.b1k_grounded.annotation_parser import (
    canonicalize_entity_name,
)
from toolkits.b1k_grounded.annotation_coverage import build_annotation_coverage


def _skill_record(
    description,
    skill_id,
    object_group,
    manipulating,
    *,
    spatial_prefix=None,
    memory_prefix=None,
    frame_duration=None,
    skill_index=0,
    skill_type="uncoordinated",
):
    return {
        "skill_idx": skill_index,
        "skill_id": [skill_id],
        "skill_description": [description],
        "object_id": [object_group],
        "manipulating_object_id": manipulating,
        "memory_prefix": memory_prefix or [],
        "spatial_prefix": spatial_prefix or [],
        "frame_duration": frame_duration or [10, 20],
        "skill_type": [skill_type],
    }


def _parse(record):
    return parse_skill_annotation(
        record,
        goal="Complete the household task.",
        episode_id="episode_00000001",
    )


def test_grounded_control_schema_json_round_trip():
    head = Grounding2D(
        camera=CameraID.HEAD,
        bbox_xyxy=(0.1, 0.2, 0.5, 0.7),
        visible_pixels=120,
        visible_fraction=0.2,
        point_xy=(0.3, 0.4),
    )
    wrist = Grounding2D(
        camera=CameraID.RIGHT_WRIST,
        bbox_xyxy=(0.2, 0.1, 0.6, 0.8),
        visible_pixels=80,
        visible_fraction=0.1,
    )
    spec = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal="Press the power button on the radio.",
        skill="press",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="radio",
                instance_id="radio_89",
                qualifier=None,
                groundings={CameraID.HEAD: head},
                part=PartArgument(
                    name="power button",
                    groundings={CameraID.RIGHT_WRIST: wrist},
                ),
                raw_object_id="radio_89",
            ),
        ),
        episode_id="episode_00000001",
        segment_id=2,
        timestep=1162,
    )

    restored = GroundedControlSpec.from_json(spec.to_json())

    assert restored == spec
    assert json.loads(spec.to_json())["arguments"][0]["role"] == "target"


def test_grounding_rejects_invalid_normalized_coordinates():
    with pytest.raises(ValueError, match="normalized"):
        Grounding2D(
            camera=CameraID.HEAD,
            bbox_xyxy=(-0.1, 0.0, 0.5, 0.5),
            visible_pixels=1,
            visible_fraction=0.1,
        )

    with pytest.raises(ValueError, match="two coordinates"):
        Grounding2D(
            camera=CameraID.HEAD,
            bbox_xyxy=(0.0, 0.0, 0.5, 0.5),
            visible_pixels=1,
            visible_fraction=0.1,
            point_xy=(0.1, 0.2, 0.3),
        )


def test_frozen_registry_contains_all_audited_skills_and_tool_overrides():
    assert len(DEFAULT_SKILL_SIGNATURE_REGISTRY) == 34
    assert DEFAULT_SKILL_SIGNATURE_REGISTRY.get("pick up from").roles == (
        Role.MANIPULATED,
        Role.SOURCE,
    )
    assert DEFAULT_SKILL_SIGNATURE_REGISTRY.get("chop").roles == (
        Role.TOOL,
        Role.TARGET,
    )
    assert DEFAULT_SKILL_SIGNATURE_REGISTRY.get("spray").roles == (
        Role.TOOL,
        Role.TARGET,
    )


def test_parse_pick_up_from_and_nested_frame_intervals():
    result = _parse(
        _skill_record(
            "pick up from",
            2,
            ["radio_89", "coffee_table_koagbh_0"],
            ["radio_89"],
            spatial_prefix=[["", "left"]],
            memory_prefix=["back"],
            frame_duration=[[10, 20], [30, 40]],
        )
    )

    assert result.status is ParseStatus.VALID
    assert result.segment.frame_intervals == ((10, 20), (30, 40))
    assert result.segment.memory_prefix == ("back",)
    arguments = result.segment.control.arguments
    assert [(argument.role, argument.category_name) for argument in arguments] == [
        (Role.MANIPULATED, "radio"),
        (Role.SOURCE, "coffee table"),
    ]
    assert arguments[1].qualifier == "left"


def test_parse_part_and_tool_signatures_from_registry():
    door = _parse(
        _skill_record(
            "open door",
            10,
            ["fridge_petcxr_0"],
            [],
            spatial_prefix=[["right_door"]],
        )
    )
    chop = _parse(
        _skill_record(
            "chop",
            34,
            ["carving_knife_209", "head_cabbage_212"],
            ["carving_knife_209"],
        )
    )

    assert door.segment.control.arguments[0].part.name == "right door"
    assert door.segment.control.arguments[0].qualifier is None
    assert [argument.role for argument in chop.segment.control.arguments] == [
        Role.TOOL,
        Role.TARGET,
    ]


def test_parse_grouped_pour_and_sweep_off_arguments():
    pour = _parse(
        _skill_record(
            "pour",
            28,
            [["bacon_209", "bacon_210"], "tray_208", "frying_pan_207"],
            ["tray_208"],
            spatial_prefix=[["", "", "right"]],
        )
    )
    sweep = _parse(
        _skill_record(
            "sweep off",
            102,
            [["half_log_176_0", "half_log_176_1"], "driveway_umalys_0"],
            [["half_log_176_0", "half_log_176_1"]],
        )
    )

    assert [argument.role for argument in pour.segment.control.arguments] == [
        Role.OTHER,
        Role.OTHER,
        Role.MANIPULATED,
        Role.DESTINATION,
    ]
    assert pour.segment.control.arguments[-1].qualifier == "right"
    assert [argument.role for argument in sweep.segment.control.arguments] == [
        Role.MANIPULATED,
        Role.MANIPULATED,
        Role.SOURCE,
    ]


def test_parse_hand_over_as_one_qualified_entity():
    result = _parse(
        _skill_record(
            "hand over",
            5,
            ["hinged_jar_235", "Right ", "\nLEFT"],
            ["hinged_jar_235"],
        )
    )

    arguments = result.segment.control.arguments
    assert len(arguments) == 1
    assert arguments[0].role is Role.MANIPULATED
    assert arguments[0].qualifier == "from right hand to left hand"


@pytest.mark.parametrize(
    ("record", "status", "issue_code"),
    [
        (
            _skill_record("unknown action", 999, ["object_1"], []),
            ParseStatus.UNSUPPORTED,
            "unsupported_skill",
        ),
        (
            _skill_record(
                "place in",
                4,
                ["cup_1", "cup_1"],
                ["cup_1"],
            ),
            ParseStatus.AMBIGUOUS,
            "duplicate_flattened_object_id",
        ),
        (
            _skill_record(
                "move to",
                1,
                ["radio_1"],
                [],
                frame_duration=[20, 10],
            ),
            ParseStatus.AMBIGUOUS,
            "invalid_frame_duration",
        ),
        (
            _skill_record(
                "hand over",
                5,
                ["cup_1", "left", "riqht"],
                ["cup_1"],
            ),
            ParseStatus.AMBIGUOUS,
            "invalid_hand_over_hands",
        ),
    ],
)
def test_parse_returns_explicit_non_valid_status(record, status, issue_code):
    result = _parse(record)

    assert result.status is status
    assert result.segment is None
    assert result.issues[0].code == issue_code


def test_canonicalize_entity_names():
    assert canonicalize_entity_name("radio_89") == "radio"
    assert canonicalize_entity_name("coffee_table_koagbh_0") == "coffee table"
    assert canonicalize_entity_name("half_head_cabbage_212_1") == "half head cabbage"
    assert canonicalize_entity_name("diced__vidalia_onion") == "diced vidalia onion"
    assert canonicalize_entity_name("robot") == "robot"


def test_annotation_coverage_reports_valid_and_ambiguous_records(tmp_path):
    meta = tmp_path / "meta" / "tasks.jsonl"
    meta.parent.mkdir()
    meta.write_text(
        json.dumps({"task_index": 0, "task_name": "test", "task": "Test the parser."})
        + "\n"
    )
    annotation = tmp_path / "annotations" / "task-0000" / "episode_00000001.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "skill_annotation": [
                    _skill_record("move to", 1, ["radio_1"], []),
                    _skill_record("place in", 4, ["cup_1", "cup_1"], ["cup_1"]),
                ]
            }
        )
    )

    report = build_annotation_coverage(tmp_path)

    assert report["coverage"]["status_counts"] == {"ambiguous": 1, "valid": 1}
    assert report["coverage"]["issue_counts"] == {"duplicate_flattened_object_id": 1}

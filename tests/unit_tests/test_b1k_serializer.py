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

import dataclasses
import math
import re

import numpy as np
import pytest

from rlinf.data.b1k_grounded import (
    STRUCTURAL_TOKENS,
    CameraID,
    ControlProfile,
    ControlSerializer,
    EntityArgument,
    GroundedControlSpec,
    Grounding2D,
    PartArgument,
    ReservedTokenAllocator,
    ReservedTokenMapping,
    Role,
    SerializerOptions,
    bbox_location_tokens,
    quantize_coordinate,
)
from toolkits.b1k_grounded.tokenizer_audit import extract_behavior_pi05_state


class _FakeSentencePiece:
    """Minimal PaliGemma-like vocabulary for allocator unit tests."""

    def vocab_size(self) -> int:
        return 1300

    def id_to_piece(self, token_id: int) -> str:
        if 7 <= token_id <= 105:
            return f"<unused{token_id - 7}>"
        if 200 <= token_id <= 1223:
            return f"<loc{token_id - 200:04d}>"
        if token_id == 3:
            return "<unk>"
        return f"piece_{token_id}"

    def piece_to_id(self, piece: str) -> int:
        if match := re.fullmatch(r"<unused(\d+)>", piece):
            index = int(match.group(1))
            return 7 + index if 0 <= index < 99 else 3
        if match := re.fullmatch(r"<loc(\d{4})>", piece):
            index = int(match.group(1))
            return 200 + index if 0 <= index < 1024 else 3
        return 3

    def encode(self, text: str) -> list[int]:
        token_id = self.piece_to_id(text)
        return [token_id] if token_id != 3 else [3]


def _token_mapping() -> ReservedTokenMapping:
    return ReservedTokenAllocator(_FakeSentencePiece()).allocate()


def _control_spec() -> GroundedControlSpec:
    head = Grounding2D(
        camera=CameraID.HEAD,
        bbox_xyxy=(0.1, 0.2, 0.5, 0.7),
        visible_pixels=100,
        visible_fraction=0.1,
        point_xy=(0.2, 0.3),
    )
    wrist = Grounding2D(
        camera=CameraID.RIGHT_WRIST,
        bbox_xyxy=(0.0, 0.25, 0.5, 1.0),
        visible_pixels=200,
        visible_fraction=0.2,
        point_xy=(0.4, 0.6),
    )
    return GroundedControlSpec(
        goal="Turn_on the radio.",
        subgoal="Press the power button.",
        skill="press",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="radio",
                instance_id="radio_89",
                qualifier="on the left",
                groundings={CameraID.HEAD: head, CameraID.RIGHT_WRIST: wrist},
                part=PartArgument(
                    name="power button",
                    groundings={CameraID.HEAD: head},
                ),
                raw_object_id="radio_89",
            ),
            EntityArgument(
                role=Role.REFERENCE,
                category_name="coffee table",
                instance_id="coffee_table_1",
                qualifier=None,
                raw_object_id="coffee_table_1",
            ),
        ),
    )


def test_reserved_allocator_uses_atomic_unused_tokens_and_round_trips():
    tokenizer = _FakeSentencePiece()
    allocator = ReservedTokenAllocator(tokenizer, model_vocab_size=1300)

    capabilities = allocator.inspect()
    mapping = allocator.allocate()
    restored = ReservedTokenMapping.from_json(mapping.to_json())

    assert capabilities.location_token_start == 200
    assert capabilities.location_token_end == 1223
    assert capabilities.unused_token_count == 99
    assert len(mapping.bindings) == len(STRUCTURAL_TOKENS) == 23
    assert mapping.piece("<goal>") == "<unused0>"
    assert mapping.token_id("<end_control>") == 29
    assert restored == mapping
    allocator.validate(restored)


def test_reserved_allocator_rejects_model_vocabulary_mismatch():
    allocator = ReservedTokenAllocator(_FakeSentencePiece(), model_vocab_size=1299)

    with pytest.raises(ValueError, match="smaller than tokenizer"):
        allocator.inspect()


def test_location_quantization_and_bbox_order():
    assert quantize_coordinate(-1.0) == 0
    assert quantize_coordinate(0.0) == 0
    assert quantize_coordinate(1.0) == 1023
    assert quantize_coordinate(2.0) == 1023
    assert bbox_location_tokens((0.0, 0.25, 0.5, 1.0)) == (
        "<loc0256>",
        "<loc0000>",
        "<loc1023>",
        "<loc0512>",
    )
    with pytest.raises(ValueError, match="finite"):
        quantize_coordinate(math.nan)


def test_p0_p1_p2_profiles_have_strictly_layered_information():
    serializer = ControlSerializer(_token_mapping())
    control = _control_spec()

    p0 = serializer.serialize(control, ControlProfile.P0_DIRECT)
    p1 = serializer.serialize(control, ControlProfile.P1_SIMPLE_SG)
    p2 = serializer.serialize(control, ControlProfile.P2_GROUND_SG)

    assert p0 == "<unused0> Turn on the radio. <unused22>"
    assert "press" not in p0
    assert "press" in p1
    assert "radio" in p1
    assert "power button" in p1
    assert "<loc" not in p1
    assert "<unused21>" not in p1
    assert "<unused15> <unused18>" in p2
    assert "<loc0205> <loc0102> <loc0716> <loc0512>" in p2
    assert "<unused21>" in p2
    assert (
        "<unused14> power button <unused15> <unused19> "
        "<loc0205> <loc0102> <loc0716> <loc0512>"
    ) in p2


def test_p2_optional_point_uses_yx_order_and_requires_a_point():
    serializer = ControlSerializer(
        _token_mapping(), options=SerializerOptions(include_object_point=True)
    )
    control = _control_spec()
    argument = control.arguments[0]
    primary = argument.groundings[CameraID.HEAD]
    missing_point = dataclasses.replace(primary, point_xy=None)
    invalid_control = dataclasses.replace(
        control,
        arguments=(
            dataclasses.replace(
                argument,
                groundings={
                    **argument.groundings,
                    CameraID.HEAD: missing_point,
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires a point"):
        serializer.serialize(invalid_control, ControlProfile.P2_GROUND_SG)

    one_argument = GroundedControlSpec(
        goal=control.goal,
        subgoal=control.subgoal,
        skill=control.skill,
        arguments=(control.arguments[0],),
    )
    serialized = serializer.serialize(one_argument, ControlProfile.P2_GROUND_SG)

    assert "<unused20> <loc0307> <loc0205>" in serialized


def test_serializer_requires_profile_enum():
    with pytest.raises(TypeError, match="ControlProfile"):
        ControlSerializer(_token_mapping()).serialize(_control_spec(), "p0_direct")


def test_tokenizer_audit_extracts_behavior_pi05_state_order():
    proprio = np.arange(256, dtype=np.float32)

    state = extract_behavior_pi05_state(proprio)

    assert state.tolist() == [
        *range(253, 256),
        *range(236, 240),
        *range(158, 165),
        *range(197, 204),
        387,
        465,
    ]

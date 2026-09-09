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

import pytest

from rlinf.data.b1k_grounded import (
    STRUCTURAL_TOKENS,
    ControlProfile,
    ControlSerializer,
    GroundedControlSpec,
    GroundedPromptController,
    ReservedTokenMapping,
    TokenBinding,
)
from rlinf.data.b1k_grounded.tokens import TOKEN_MAPPING_VERSION


def _serializer() -> ControlSerializer:
    mapping = ReservedTokenMapping(
        version=TOKEN_MAPPING_VERSION,
        bindings=tuple(
            TokenBinding(token, f"<unused{index}>", index + 7)
            for index, token in enumerate(STRUCTURAL_TOKENS)
        ),
    )
    return ControlSerializer(mapping)


def test_p0_runtime_prompt_uses_structural_delimiters():
    controller = GroundedPromptController(
        _serializer(), ControlProfile.P0_DIRECT, "Turn on the radio."
    )

    assert controller.prompt({}) == "<unused0> Turn on the radio. <unused22>"


def test_oracle_runtime_prompt_requires_and_serializes_control_json():
    controller = GroundedPromptController(
        _serializer(), ControlProfile.P1_SIMPLE_SG, "Turn on the radio."
    )
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal=None,
        skill="press",
        arguments=(),
    )

    prompt = controller.prompt({"grounded_control_json": control.to_json()})

    assert prompt == ("<unused0> Turn on the radio. <unused2> press <unused22>")
    with pytest.raises(KeyError, match="grounded_control_json"):
        controller.prompt({})

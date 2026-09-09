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

"""Deterministic P0/P1/P2 serialization of grounded control conditions."""

from __future__ import annotations

import dataclasses
import enum
import math

from .grounding import select_primary_grounding
from .schema import CameraID, EntityArgument, GroundedControlSpec, Role
from .tokens import ReservedTokenMapping


class ControlProfile(str, enum.Enum):
    """Condition profiles used by the first grounded-control Oracle study."""

    P0_DIRECT = "p0_direct"
    P1_SIMPLE_SG = "p1_simple_sg"
    P2_GROUND_SG = "p2_ground_sg"


@dataclasses.dataclass(frozen=True)
class SerializerOptions:
    """Optional P2 geometry fields; bbox serialization is always enabled."""

    include_object_point: bool = False


_ROLE_TOKENS = {
    Role.TARGET: "<role_target>",
    Role.MANIPULATED: "<role_manipulated>",
    Role.SOURCE: "<role_source>",
    Role.DESTINATION: "<role_destination>",
    Role.REFERENCE: "<role_reference>",
    Role.TOOL: "<role_tool>",
    Role.OTHER: "<role_other>",
}
_VIEW_TOKENS = {
    CameraID.HEAD: "<view_head>",
    CameraID.LEFT_WRIST: "<view_left_wrist>",
    CameraID.RIGHT_WRIST: "<view_right_wrist>",
}


def quantize_coordinate(value: float) -> int:
    """Quantize a normalized coordinate to the PaliGemma 0..1023 grid."""
    if not math.isfinite(value):
        raise ValueError("Location coordinates must be finite.")
    return int(round(1023 * min(1.0, max(0.0, value))))


def location_token(value: float) -> str:
    """Return one PaliGemma location piece for a normalized coordinate."""
    return f"<loc{quantize_coordinate(value):04d}>"


def bbox_location_tokens(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[str, str, str, str]:
    """Serialize xyxy as PaliGemma's ymin, xmin, ymax, xmax token order."""
    x_min, y_min, x_max, y_max = bbox_xyxy
    return (
        location_token(y_min),
        location_token(x_min),
        location_token(y_max),
        location_token(x_max),
    )


class ControlSerializer:
    """Serialize one structured condition into tokenizer-ready PaliGemma text."""

    def __init__(
        self,
        token_mapping: ReservedTokenMapping,
        *,
        options: SerializerOptions = SerializerOptions(),
    ) -> None:
        self._token_mapping = token_mapping
        self._options = options

    def serialize(self, control: GroundedControlSpec, profile: ControlProfile) -> str:
        """Serialize a control condition using one frozen Oracle profile."""
        if not isinstance(profile, ControlProfile):
            raise TypeError("profile must be a ControlProfile.")
        pieces = [self._token("<goal>"), self._text(control.goal)]
        if profile is not ControlProfile.P0_DIRECT:
            if control.subgoal is not None:
                pieces.extend([self._token("<subgoal>"), self._text(control.subgoal)])
            if control.skill is not None:
                pieces.extend([self._token("<skill>"), self._text(control.skill)])
            for argument in control.arguments:
                pieces.extend(self._serialize_argument(argument, profile))
        pieces.append(self._token("<end_control>"))
        return " ".join(pieces)

    def _serialize_argument(
        self, argument: EntityArgument, profile: ControlProfile
    ) -> list[str]:
        pieces = [
            self._token("<arg>"),
            self._token(_ROLE_TOKENS[argument.role]),
            self._token("<object>"),
            self._text(argument.category_name),
        ]
        if argument.qualifier is not None:
            pieces.extend([self._token("<qualifier>"), self._text(argument.qualifier)])
        if profile is ControlProfile.P2_GROUND_SG:
            pieces.extend(self._serialize_object_grounding(argument))
        if argument.part is not None:
            pieces.extend([self._token("<part>"), self._text(argument.part.name)])
            if profile is ControlProfile.P2_GROUND_SG:
                pieces.extend(self._serialize_part_grounding(argument))
        pieces.append(self._token("<end_arg>"))
        return pieces

    def _serialize_object_grounding(self, argument: EntityArgument) -> list[str]:
        primary_part = (
            None
            if argument.part is None
            else select_primary_grounding(argument.part.groundings)
        )
        primary = (
            argument.groundings.get(primary_part.camera)
            if primary_part is not None
            else None
        )
        if primary is None:
            primary = select_primary_grounding(argument.groundings)
        if primary is None:
            return [self._token("<no_grounding>")]
        pieces = [
            self._token(_VIEW_TOKENS[primary.camera]),
            self._token("<object_bbox>"),
            *bbox_location_tokens(primary.bbox_xyxy),
        ]
        if self._options.include_object_point:
            if primary.point_xy is None:
                raise ValueError(
                    "include_object_point=True requires a point on every selected "
                    "object grounding."
                )
            pieces.extend(
                [
                    self._token("<point>"),
                    location_token(primary.point_xy[1]),
                    location_token(primary.point_xy[0]),
                ]
            )
        return pieces

    def _serialize_part_grounding(self, argument: EntityArgument) -> list[str]:
        if argument.part is None:
            raise ValueError("Part grounding requires an argument part.")
        primary = select_primary_grounding(argument.part.groundings)
        if primary is None:
            return [self._token("<no_grounding>")]
        return [
            self._token(_VIEW_TOKENS[primary.camera]),
            self._token("<part_bbox>"),
            *bbox_location_tokens(primary.bbox_xyxy),
        ]

    def _token(self, logical_token: str) -> str:
        return self._token_mapping.piece(logical_token)

    @staticmethod
    def _text(value: str) -> str:
        return " ".join(value.strip().replace("_", " ").split())

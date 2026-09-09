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

"""Data schema shared by grounded-control preprocessing and model inputs."""

from __future__ import annotations

import dataclasses
import enum
import json
from typing import Any


class Role(str, enum.Enum):
    """Semantic role of an entity in an atomic skill invocation."""

    TARGET = "target"
    MANIPULATED = "manipulated"
    SOURCE = "source"
    DESTINATION = "destination"
    REFERENCE = "reference"
    TOOL = "tool"
    OTHER = "other"


class CameraID(str, enum.Enum):
    """Camera identifiers supported by the BEHAVIOR action policy."""

    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


@dataclasses.dataclass(frozen=True)
class Grounding2D:
    """Visible 2D grounding of one entity or part in one camera."""

    camera: CameraID
    bbox_xyxy: tuple[float, float, float, float]
    visible_pixels: int
    visible_fraction: float
    point_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraID):
            raise TypeError("camera must be a CameraID.")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four coordinates.")
        xmin, ymin, xmax, ymax = self.bbox_xyxy
        if not all(0.0 <= value <= 1.0 for value in self.bbox_xyxy):
            raise ValueError("bbox_xyxy coordinates must be normalized to [0, 1].")
        if xmax < xmin or ymax < ymin:
            raise ValueError("bbox_xyxy must use xmin, ymin, xmax, ymax order.")
        if self.visible_pixels <= 0:
            raise ValueError("visible_pixels must be positive for a grounding.")
        if not 0.0 < self.visible_fraction <= 1.0:
            raise ValueError("visible_fraction must be in (0, 1].")
        if self.point_xy is not None:
            if len(self.point_xy) != 2:
                raise ValueError("point_xy must contain two coordinates.")
            if not all(0.0 <= value <= 1.0 for value in self.point_xy):
                raise ValueError("point_xy coordinates must be normalized to [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "camera": self.camera.value,
            "bbox_xyxy": list(self.bbox_xyxy),
            "visible_pixels": self.visible_pixels,
            "visible_fraction": self.visible_fraction,
            "point_xy": list(self.point_xy) if self.point_xy is not None else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Grounding2D:
        """Construct a grounding from its serialized representation."""
        point = value.get("point_xy")
        return cls(
            camera=CameraID(value["camera"]),
            bbox_xyxy=tuple(value["bbox_xyxy"]),
            visible_pixels=value["visible_pixels"],
            visible_fraction=value["visible_fraction"],
            point_xy=tuple(point) if point is not None else None,
        )


def _validate_groundings(groundings: dict[CameraID, Grounding2D]) -> None:
    for camera, grounding in groundings.items():
        if not isinstance(camera, CameraID):
            raise TypeError("Grounding keys must be CameraID values.")
        if grounding.camera is not camera:
            raise ValueError(
                f"Grounding key {camera.value!r} does not match its camera "
                f"{grounding.camera.value!r}."
            )


def _groundings_to_dict(
    groundings: dict[CameraID, Grounding2D],
) -> dict[str, dict[str, Any]]:
    return {
        camera.value: grounding.to_dict()
        for camera, grounding in sorted(
            groundings.items(), key=lambda item: item[0].value
        )
    }


def _groundings_from_dict(
    values: dict[str, dict[str, Any]],
) -> dict[CameraID, Grounding2D]:
    return {
        CameraID(camera): Grounding2D.from_dict(grounding)
        for camera, grounding in values.items()
    }


@dataclasses.dataclass(frozen=True)
class PartArgument:
    """Named object part with optional per-camera groundings."""

    name: str
    groundings: dict[CameraID, Grounding2D] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Part name must not be empty.")
        _validate_groundings(self.groundings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "groundings": _groundings_to_dict(self.groundings),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PartArgument:
        """Construct a part from its serialized representation."""
        return cls(
            name=value["name"],
            groundings=_groundings_from_dict(value["groundings"]),
        )


@dataclasses.dataclass(frozen=True)
class EntityArgument:
    """One typed entity argument supplied to the action expert."""

    role: Role
    category_name: str
    instance_id: str | None
    qualifier: str | None
    groundings: dict[CameraID, Grounding2D] = dataclasses.field(default_factory=dict)
    part: PartArgument | None = None
    raw_object_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise TypeError("role must be a Role.")
        if not self.category_name.strip():
            raise ValueError("category_name must not be empty.")
        if self.qualifier is not None and not self.qualifier.strip():
            raise ValueError("qualifier must be None or a non-empty string.")
        _validate_groundings(self.groundings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "role": self.role.value,
            "category_name": self.category_name,
            "instance_id": self.instance_id,
            "qualifier": self.qualifier,
            "groundings": _groundings_to_dict(self.groundings),
            "part": self.part.to_dict() if self.part is not None else None,
            "raw_object_id": self.raw_object_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EntityArgument:
        """Construct an entity argument from its serialized representation."""
        part = value.get("part")
        return cls(
            role=Role(value["role"]),
            category_name=value["category_name"],
            instance_id=value.get("instance_id"),
            qualifier=value.get("qualifier"),
            groundings=_groundings_from_dict(value["groundings"]),
            part=PartArgument.from_dict(part) if part is not None else None,
            raw_object_id=value.get("raw_object_id"),
        )


@dataclasses.dataclass(frozen=True)
class GroundedControlSpec:
    """Structured semantic condition for one action-policy timestep."""

    goal: str
    subgoal: str | None
    skill: str | None
    arguments: tuple[EntityArgument, ...]
    episode_id: str | None = None
    segment_id: int | None = None
    timestep: int | None = None

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must not be empty.")
        if self.subgoal is not None and not self.subgoal.strip():
            raise ValueError("subgoal must be None or a non-empty string.")
        if self.skill is not None and not self.skill.strip():
            raise ValueError("skill must be None or a non-empty string.")
        if not isinstance(self.arguments, tuple):
            raise TypeError("arguments must be a tuple for deterministic ordering.")
        if not all(isinstance(argument, EntityArgument) for argument in self.arguments):
            raise TypeError("arguments must contain only EntityArgument values.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "goal": self.goal,
            "subgoal": self.subgoal,
            "skill": self.skill,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "episode_id": self.episode_id,
            "segment_id": self.segment_id,
            "timestep": self.timestep,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GroundedControlSpec:
        """Construct a control specification from serialized data."""
        return cls(
            goal=value["goal"],
            subgoal=value.get("subgoal"),
            skill=value.get("skill"),
            arguments=tuple(
                EntityArgument.from_dict(argument) for argument in value["arguments"]
            ),
            episode_id=value.get("episode_id"),
            segment_id=value.get("segment_id"),
            timestep=value.get("timestep"),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this control specification to JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, value: str) -> GroundedControlSpec:
        """Deserialize a control specification from JSON."""
        return cls.from_dict(json.loads(value))

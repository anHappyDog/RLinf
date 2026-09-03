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

"""Convert raw B1K skill annotations into typed grounded-control segments."""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any

from .schema import EntityArgument, GroundedControlSpec, PartArgument, Role
from .skill_registry import (
    DEFAULT_SKILL_SIGNATURE_REGISTRY,
    ArgumentLayout,
    SkillSignature,
    SkillSignatureRegistry,
)


class ParseStatus(str, enum.Enum):
    """Outcome of parsing one raw skill annotation."""

    VALID = "valid"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


@dataclasses.dataclass(frozen=True)
class ParseIssue:
    """Machine-readable explanation for an unparsed annotation."""

    code: str
    message: str


@dataclasses.dataclass(frozen=True)
class ParsedSkillSegment:
    """A valid control condition and its source segment metadata."""

    control: GroundedControlSpec
    skill_id: int
    skill_type: str
    frame_intervals: tuple[tuple[int, int], ...]
    memory_prefix: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AnnotationParseResult:
    """Explicit valid, ambiguous, or unsupported parse result."""

    status: ParseStatus
    segment: ParsedSkillSegment | None
    issues: tuple[ParseIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.status is ParseStatus.VALID and self.segment is None:
            raise ValueError("A valid parse result must contain a segment.")
        if self.status is not ParseStatus.VALID and self.segment is not None:
            raise ValueError("A non-valid parse result must not contain a segment.")
        if self.status is not ParseStatus.VALID and not self.issues:
            raise ValueError("A non-valid parse result must explain its issues.")

    @property
    def is_valid(self) -> bool:
        """Whether the annotation produced a training-safe segment."""
        return self.status is ParseStatus.VALID


class _AnnotationFormatError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_qualifier(value: str) -> str | None:
    normalized = _normalize_text(value.replace("_", " "))
    return normalized or None


def canonicalize_entity_name(object_id: str) -> str:
    """Convert a B1K instance or system identifier to readable text."""
    parts = [part for part in object_id.strip().lower().split("_") if part]
    numeric_suffixes = []
    while parts and parts[-1].isdigit():
        numeric_suffixes.append(int(parts.pop()))
    if (
        len(parts) >= 2
        and numeric_suffixes
        and numeric_suffixes[0] <= 10
        and re.fullmatch(r"[a-z]{6}", parts[-1])
    ):
        parts.pop()
    if not parts:
        raise ValueError(f"Cannot canonicalize empty object id {object_id!r}.")
    return " ".join(parts)


def _flatten_object_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_object_ids(item))
        return flattened
    raise _AnnotationFormatError(
        "invalid_object_id",
        f"Expected a string or nested object list, got {value!r}.",
    )


def _parse_frame_intervals(value: Any) -> tuple[tuple[int, int], ...]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        start, end = value
        if end <= start:
            raise _AnnotationFormatError(
                "invalid_frame_duration",
                f"Frame interval must be increasing, got {value!r}.",
            )
        return ((start, end),)
    if isinstance(value, list) and value:
        intervals = []
        for item in value:
            intervals.extend(_parse_frame_intervals(item))
        return tuple(intervals)
    raise _AnnotationFormatError(
        "invalid_frame_duration",
        f"Expected an interval or nested intervals, got {value!r}.",
    )


def _parse_string_list(record: dict[str, Any], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _AnnotationFormatError(
            f"invalid_{field}", f"{field} must be a list of strings."
        )
    return tuple(value)


def _parse_spatial_qualifiers(value: Any, raw_arity: int) -> tuple[str | None, ...]:
    if value == []:
        return (None,) * raw_arity
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], list)
        or len(value[0]) != raw_arity
        or not all(isinstance(item, str) for item in value[0])
    ):
        raise _AnnotationFormatError(
            "invalid_spatial_prefix",
            f"spatial_prefix must align with {raw_arity} raw arguments, got {value!r}.",
        )
    return tuple(_normalize_qualifier(item) for item in value[0])


def _part_name(base_name: str, qualifier: str | None) -> str:
    if qualifier is None:
        return base_name
    if base_name in qualifier.split():
        return qualifier
    return f"{qualifier} {base_name}"


def _entity_argument(
    role: Role,
    raw_object_id: str,
    qualifier: str | None,
    *,
    part_name: str | None = None,
) -> EntityArgument:
    part = None
    entity_qualifier = qualifier
    if part_name is not None:
        part = PartArgument(name=_part_name(part_name, qualifier))
        entity_qualifier = None
    return EntityArgument(
        role=role,
        category_name=canonicalize_entity_name(raw_object_id),
        instance_id=None if raw_object_id == "robot" else raw_object_id,
        qualifier=entity_qualifier,
        raw_object_id=raw_object_id,
        part=part,
    )


def _parse_positional_arguments(
    signature: SkillSignature,
    raw_group: list[Any],
    qualifiers: tuple[str | None, ...],
) -> tuple[EntityArgument, ...]:
    if not all(isinstance(item, str) for item in raw_group):
        raise _AnnotationFormatError(
            "invalid_object_layout",
            f"Skill {signature.skill!r} requires flat positional object IDs.",
        )
    return tuple(
        _entity_argument(
            role,
            raw_object_id,
            qualifier,
            part_name=signature.part_name,
        )
        for role, raw_object_id, qualifier in zip(
            signature.roles, raw_group, qualifiers, strict=True
        )
    )


def _parse_repeat_first_arguments(
    signature: SkillSignature,
    raw_group: list[Any],
    qualifiers: tuple[str | None, ...],
) -> tuple[EntityArgument, ...]:
    repeated = raw_group[0]
    if (
        not isinstance(repeated, list)
        or not repeated
        or not all(isinstance(item, str) for item in repeated)
    ):
        raise _AnnotationFormatError(
            "invalid_object_layout",
            f"Skill {signature.skill!r} requires a non-empty first object group.",
        )
    if not all(isinstance(item, str) for item in raw_group[1:]):
        raise _AnnotationFormatError(
            "invalid_object_layout",
            f"Skill {signature.skill!r} has invalid trailing object IDs.",
        )

    arguments = [
        _entity_argument(signature.roles[0], raw_object_id, qualifiers[0])
        for raw_object_id in repeated
    ]
    arguments.extend(
        _entity_argument(role, raw_object_id, qualifier)
        for role, raw_object_id, qualifier in zip(
            signature.roles[1:], raw_group[1:], qualifiers[1:], strict=True
        )
    )
    return tuple(arguments)


def _parse_hand_over_arguments(
    signature: SkillSignature,
    raw_group: list[Any],
) -> tuple[EntityArgument, ...]:
    if not all(isinstance(item, str) for item in raw_group):
        raise _AnnotationFormatError(
            "invalid_object_layout", "hand over requires object and two hand names."
        )
    raw_object_id, source_hand, destination_hand = raw_group
    source_hand = _normalize_text(source_hand)
    destination_hand = _normalize_text(destination_hand)
    if source_hand not in {"left", "right"} or destination_hand not in {
        "left",
        "right",
    }:
        raise _AnnotationFormatError(
            "invalid_hand_over_hands",
            f"Unexpected hand transfer {source_hand!r} -> {destination_hand!r}.",
        )
    if source_hand == destination_hand:
        raise _AnnotationFormatError(
            "duplicate_flattened_object_id",
            "hand over source and destination hands must differ.",
        )
    qualifier = f"from {source_hand} hand to {destination_hand} hand"
    return (_entity_argument(signature.roles[0], raw_object_id, qualifier),)


def _parse_arguments(
    signature: SkillSignature,
    raw_group: list[Any],
    qualifiers: tuple[str | None, ...],
) -> tuple[EntityArgument, ...]:
    if signature.layout is ArgumentLayout.POSITIONAL:
        return _parse_positional_arguments(signature, raw_group, qualifiers)
    if signature.layout is ArgumentLayout.REPEAT_FIRST:
        return _parse_repeat_first_arguments(signature, raw_group, qualifiers)
    if signature.layout is ArgumentLayout.HAND_OVER:
        return _parse_hand_over_arguments(signature, raw_group)
    raise AssertionError(f"Unhandled argument layout {signature.layout!r}.")


def _extract_skill(record: dict[str, Any]) -> str:
    descriptions = record.get("skill_description")
    if (
        not isinstance(descriptions, list)
        or len(descriptions) != 1
        or not isinstance(descriptions[0], str)
        or not descriptions[0].strip()
    ):
        raise _AnnotationFormatError(
            "invalid_skill_description",
            "skill_description must contain one non-empty string.",
        )
    return _normalize_text(descriptions[0])


def _extract_skill_id(record: dict[str, Any], signature: SkillSignature) -> int:
    skill_ids = record.get("skill_id")
    if (
        not isinstance(skill_ids, list)
        or len(skill_ids) != 1
        or not isinstance(skill_ids[0], int)
    ):
        raise _AnnotationFormatError(
            "invalid_skill_id", "skill_id must contain exactly one integer."
        )
    if skill_ids[0] != signature.skill_id:
        raise _AnnotationFormatError(
            "skill_id_mismatch",
            f"Skill {signature.skill!r} expects ID {signature.skill_id}, "
            f"got {skill_ids[0]}.",
        )
    return skill_ids[0]


def _extract_skill_type(record: dict[str, Any]) -> str:
    skill_types = record.get("skill_type")
    if (
        not isinstance(skill_types, list)
        or len(skill_types) != 1
        or not isinstance(skill_types[0], str)
    ):
        raise _AnnotationFormatError(
            "invalid_skill_type", "skill_type must contain exactly one string."
        )
    return skill_types[0]


def _extract_raw_group(record: dict[str, Any], signature: SkillSignature) -> list[Any]:
    object_groups = record.get("object_id")
    if not isinstance(object_groups, list) or len(object_groups) != 1:
        raise _AnnotationFormatError(
            "invalid_object_group",
            "object_id must contain exactly one group for a skill annotation.",
        )
    raw_group = object_groups[0]
    if (
        not isinstance(raw_group, list)
        or len(raw_group) != signature.expected_raw_arity
    ):
        raise _AnnotationFormatError(
            "invalid_object_arity",
            f"Skill {signature.skill!r} expects {signature.expected_raw_arity} "
            f"top-level objects, got {raw_group!r}.",
        )
    flattened = _flatten_object_ids(raw_group)
    if len(flattened) != len(set(flattened)):
        raise _AnnotationFormatError(
            "duplicate_flattened_object_id",
            "Flattened object_id contains a duplicate value.",
        )
    return raw_group


def _validate_manipulating_objects(
    record: dict[str, Any],
    signature: SkillSignature,
    raw_group: list[Any],
) -> None:
    manipulating_group = record.get("manipulating_object_id")
    if not isinstance(manipulating_group, list):
        raise _AnnotationFormatError(
            "invalid_manipulating_object_id",
            "manipulating_object_id must be a list.",
        )
    manipulating_ids = _flatten_object_ids(manipulating_group)
    controlled_ids = set(
        _flatten_object_ids(raw_group[signature.controlled_top_level_index])
    )
    unexpected = sorted(set(manipulating_ids).difference(controlled_ids))
    if unexpected:
        raise _AnnotationFormatError(
            "manipulating_object_mismatch",
            f"Controlled annotation objects do not match the registry: {unexpected}.",
        )


def _parse_known_skill(
    record: dict[str, Any],
    signature: SkillSignature,
    *,
    goal: str,
    subgoal: str | None,
    episode_id: str | None,
    timestep: int | None,
) -> ParsedSkillSegment:
    skill_id = _extract_skill_id(record, signature)
    skill_index = record.get("skill_idx")
    if not isinstance(skill_index, int):
        raise _AnnotationFormatError(
            "invalid_skill_index", "skill_idx must be an integer."
        )
    raw_group = _extract_raw_group(record, signature)
    _validate_manipulating_objects(record, signature, raw_group)
    memory_prefix = _parse_string_list(record, "memory_prefix")
    qualifiers = _parse_spatial_qualifiers(
        record.get("spatial_prefix"), signature.expected_raw_arity
    )
    frame_intervals = _parse_frame_intervals(record.get("frame_duration"))
    skill_type = _extract_skill_type(record)
    arguments = _parse_arguments(signature, raw_group, qualifiers)

    return ParsedSkillSegment(
        control=GroundedControlSpec(
            goal=goal,
            subgoal=subgoal,
            skill=signature.skill,
            arguments=arguments,
            episode_id=episode_id,
            segment_id=skill_index,
            timestep=timestep,
        ),
        skill_id=skill_id,
        skill_type=skill_type,
        frame_intervals=frame_intervals,
        memory_prefix=memory_prefix,
    )


def parse_skill_annotation(
    record: Any,
    *,
    goal: str,
    subgoal: str | None = None,
    episode_id: str | None = None,
    timestep: int | None = None,
    registry: SkillSignatureRegistry = DEFAULT_SKILL_SIGNATURE_REGISTRY,
) -> AnnotationParseResult:
    """Parse one B1K skill annotation without silently repairing bad data.

    Args:
        record: Raw entry from ``skill_annotation``.
        goal: Episode-level natural-language goal.
        subgoal: Optional independently annotated macro subgoal.
        episode_id: Optional episode identifier for provenance.
        timestep: Optional frame associated with this condition.
        registry: Frozen canonical skill signature registry.

    Returns:
        An explicit valid, ambiguous, or unsupported parse result.
    """
    if not isinstance(record, dict):
        return AnnotationParseResult(
            status=ParseStatus.AMBIGUOUS,
            segment=None,
            issues=(
                ParseIssue("invalid_skill_record", "Skill record is not an object."),
            ),
        )
    try:
        skill = _extract_skill(record)
    except _AnnotationFormatError as error:
        return AnnotationParseResult(
            status=ParseStatus.AMBIGUOUS,
            segment=None,
            issues=(ParseIssue(error.code, str(error)),),
        )

    signature = registry.get(skill)
    if signature is None:
        return AnnotationParseResult(
            status=ParseStatus.UNSUPPORTED,
            segment=None,
            issues=(
                ParseIssue("unsupported_skill", f"No signature exists for {skill!r}."),
            ),
        )

    try:
        segment = _parse_known_skill(
            record,
            signature,
            goal=goal,
            subgoal=subgoal,
            episode_id=episode_id,
            timestep=timestep,
        )
    except _AnnotationFormatError as error:
        return AnnotationParseResult(
            status=ParseStatus.AMBIGUOUS,
            segment=None,
            issues=(ParseIssue(error.code, str(error)),),
        )
    return AnnotationParseResult(status=ParseStatus.VALID, segment=segment)

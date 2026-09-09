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

"""Deterministically render B1K primitive annotations as macro subgoals."""

from __future__ import annotations

import collections
import dataclasses
import enum
from collections.abc import Mapping
from typing import Any

from .annotation_parser import canonicalize_entity_name
from .schema import Role
from .skill_registry import (
    DEFAULT_SKILL_SIGNATURE_REGISTRY,
    SkillSignature,
    SkillSignatureRegistry,
)

COMPOSITE_SUBGOAL_SEPARATOR = "; also "
SUBGOAL_RESOLUTION_POLICY = "primitive_or_referenced_atomic_sequence_v1"


class SubgoalStatus(str, enum.Enum):
    """How a skill's primitive-derived subgoal was resolved."""

    MISSING = "missing"
    UNIQUE = "unique"
    COMPOSITE = "composite"


@dataclasses.dataclass(frozen=True)
class ResolvedSubgoal:
    """Rendered subgoal and its episode-local primitive provenance."""

    text: str | None
    status: SubgoalStatus
    primitive_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status is SubgoalStatus.MISSING:
            if self.text is not None:
                raise ValueError("A missing subgoal must not contain text.")
            return
        if self.text is None or not self.text.strip():
            raise ValueError("A resolved subgoal must contain non-empty text.")
        if self.status is SubgoalStatus.UNIQUE and len(self.primitive_indices) != 1:
            raise ValueError("A unique subgoal requires exactly one primitive index.")
        if self.status is SubgoalStatus.COMPOSITE and len(self.primitive_indices) < 2:
            raise ValueError(
                "A composite subgoal requires at least two primitive indices."
            )


_OPERATION_TEMPLATES = {
    "attach": "attach {0} to {1}",
    "chop": "chop {1} with {0}",
    "close door": "close door on {0}",
    "close drawer": "close drawer on {0}",
    "close lid": "close lid on {0}",
    "hand over": "hand {0} over from {1} hand to {2} hand",
    "hang": "hang {0} on {1}",
    "hold": "hold {0}",
    "ignite": "ignite {1} with {0}",
    "insert": "insert {0} into {1}",
    "move to": "move to {0}",
    "open door": "open door on {0}",
    "open drawer": "open drawer on {0}",
    "open lid": "open lid on {0}",
    "pick up from": "pick up {0} from {1}",
    "place in": "place {0} in {1}",
    "place in next to": "place {0} in {1} next to {2}",
    "place on": "place {0} on {1}",
    "place on next to": "place {0} on {1} next to {2}",
    "place under": "place {0} under {1}",
    "pour": "pour {0} from {1} into {2}",
    "press": "press {0}",
    "pull tray": "pull tray from {0}",
    "push to": "push {0} to {1}",
    "push tray": "push tray into {0}",
    "release": "release {0}",
    "spray": "spray {1} with {0}",
    "sweep off": "sweep {0} off {1}",
    "sweep surface": "sweep {1} with {0}",
    "tip over": "tip over {0}",
    "turn off switch": "turn off switch on {0}",
    "turn on switch": "turn on switch on {0}",
    "turn to": "turn {0} toward {1}",
    "wipe hard": "wipe {1} hard with {0}",
}

_BACK_TEMPLATES = {
    "move to": "move back to {0}",
    "pick up from": "pick {0} back up from {1}",
    "place in": "place {0} back in {1}",
    "place in next to": "place {0} back in {1} next to {2}",
    "place on": "place {0} back on {1}",
    "place on next to": "place {0} back on {1} next to {2}",
    "place under": "place {0} back under {1}",
    "push to": "push {0} back to {1}",
}


def _list_field(record: Mapping[str, Any], field: str) -> list[Any]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"Primitive {field} must be a list.")
    return value


def _flatten_strings(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_strings(item, field))
        return tuple(flattened)
    raise ValueError(f"Primitive {field} must contain only strings and lists.")


def _normalized_qualifiers(value: Any) -> tuple[str, ...]:
    return tuple(
        " ".join(item.strip().lower().replace("_", " ").split())
        for item in _flatten_strings(value, "spatial_prefix")
    )


def _format_argument(value: Any, qualifiers: Any) -> str:
    names = tuple(
        canonicalize_entity_name(item) for item in _flatten_strings(value, "object_id")
    )
    if not names:
        raise ValueError("Primitive object arguments must not be empty.")
    normalized_qualifiers = _normalized_qualifiers(qualifiers)
    if not any(normalized_qualifiers):
        return " and ".join(names)
    if len(normalized_qualifiers) == len(names):
        return " and ".join(
            name if not qualifier else f"{name} ({qualifier})"
            for name, qualifier in zip(names, normalized_qualifiers, strict=True)
        )
    qualifier_text = " and ".join(
        qualifier for qualifier in normalized_qualifiers if qualifier
    )
    return f"{' and '.join(names)} ({qualifier_text})"


def _primary_argument_index(signature: SkillSignature) -> int:
    for role in (Role.MANIPULATED, Role.TARGET):
        if role in signature.roles:
            return signature.roles.index(role)
    return 0


def _render_operation(
    operation: str,
    signature: SkillSignature,
    object_group: Any,
    spatial_group: Any,
    memory_prefix: str,
) -> str:
    if not isinstance(object_group, list) or (
        len(object_group) != signature.expected_raw_arity
    ):
        raise ValueError(
            f"Primitive {operation!r} expects {signature.expected_raw_arity} "
            f"object arguments, got {object_group!r}."
        )
    if spatial_group == []:
        spatial_group = [[] for _ in object_group]
    if not isinstance(spatial_group, list) or len(spatial_group) != len(object_group):
        raise ValueError(
            f"Primitive {operation!r} spatial_prefix must align with object_id."
        )

    arguments = [
        _format_argument(value, qualifiers)
        for value, qualifiers in zip(object_group, spatial_group, strict=True)
    ]
    if memory_prefix in {"the other", "the same"}:
        primary_index = _primary_argument_index(signature)
        arguments[primary_index] = f"{memory_prefix} {arguments[primary_index]}"
    elif memory_prefix not in {"", "back"}:
        raise ValueError(f"Unsupported primitive memory_prefix {memory_prefix!r}.")

    template = (
        _BACK_TEMPLATES.get(operation)
        if memory_prefix == "back"
        else _OPERATION_TEMPLATES[operation]
    )
    if template is None:
        return f"{_OPERATION_TEMPLATES[operation].format(*arguments)} (back)"
    return template.format(*arguments)


def render_primitive_subgoal(
    record: Mapping[str, Any],
    *,
    registry: SkillSignatureRegistry = DEFAULT_SKILL_SIGNATURE_REGISTRY,
) -> str | None:
    """Render one raw ``primitive_annotation`` entry as stable text.

    Empty primitive annotations are an explicit B1K condition and produce no
    text. Non-empty annotations are validated against the same frozen skill
    signatures used for atomic controls.
    """
    descriptions = _list_field(record, "primitive_description")
    primitive_ids = _list_field(record, "primitive_id")
    object_groups = _list_field(record, "object_id")
    memory_prefixes = _list_field(record, "memory_prefix")
    spatial_prefixes = _list_field(record, "spatial_prefix")

    if not descriptions:
        if primitive_ids or object_groups or memory_prefixes or spatial_prefixes:
            raise ValueError("An empty primitive description has non-empty fields.")
        return None
    operation_count = len(descriptions)
    if len(primitive_ids) != operation_count or len(object_groups) != operation_count:
        raise ValueError(
            "Primitive descriptions, IDs, and object groups must have equal length."
        )
    if not memory_prefixes:
        memory_prefixes = [""] * operation_count
    if len(memory_prefixes) != operation_count or not all(
        isinstance(value, str) for value in memory_prefixes
    ):
        raise ValueError("Primitive memory_prefix must align with its operations.")
    if not spatial_prefixes:
        spatial_prefixes = [[] for _ in descriptions]
    if len(spatial_prefixes) != operation_count:
        raise ValueError("Primitive spatial_prefix must align with its operations.")

    clauses = []
    for description, primitive_id, object_group, spatial_group, memory_prefix in zip(
        descriptions,
        primitive_ids,
        object_groups,
        spatial_prefixes,
        memory_prefixes,
        strict=True,
    ):
        if not isinstance(description, str):
            raise ValueError("Primitive descriptions must be strings.")
        operation = " ".join(description.strip().lower().split())
        signature = registry.get(operation)
        if signature is None:
            raise ValueError(f"Unsupported primitive operation {operation!r}.")
        if primitive_id != signature.skill_id:
            raise ValueError(
                f"Primitive {operation!r} expects ID {signature.skill_id}, "
                f"got {primitive_id!r}."
            )
        clauses.append(
            _render_operation(
                operation,
                signature,
                object_group,
                spatial_group,
                " ".join(memory_prefix.strip().lower().split()),
            )
        )
    return ", then ".join(clauses)


def _render_referenced_atomic_sequence(
    referenced_skill_indices: list[int],
    skill_records: Mapping[int, Mapping[str, Any]],
    *,
    registry: SkillSignatureRegistry,
) -> str:
    """Render an empty primitive from its ordered child skill annotations."""
    clauses = []
    for skill_index in referenced_skill_indices:
        skill_record = skill_records[skill_index]
        text = render_primitive_subgoal(
            {
                "primitive_description": _list_field(skill_record, "skill_description"),
                "primitive_id": _list_field(skill_record, "skill_id"),
                "object_id": _list_field(skill_record, "object_id"),
                "memory_prefix": _list_field(skill_record, "memory_prefix"),
                "spatial_prefix": _list_field(skill_record, "spatial_prefix"),
            },
            registry=registry,
        )
        if text is None:
            raise ValueError(
                f"Referenced atomic skill {skill_index} has no description."
            )
        clauses.append(text)
    return ", then ".join(clauses)


def resolve_episode_subgoals(
    annotation: Mapping[str, Any],
    *,
    registry: SkillSignatureRegistry = DEFAULT_SKILL_SIGNATURE_REGISTRY,
) -> dict[int, ResolvedSubgoal]:
    """Resolve every skill index to its primitive subgoal within one episode."""
    skill_records = _list_field(annotation, "skill_annotation")
    primitive_records = _list_field(annotation, "primitive_annotation")
    skill_index_list = [
        record["skill_idx"]
        for record in skill_records
        if isinstance(record, Mapping) and isinstance(record.get("skill_idx"), int)
    ]
    skill_indices = set(skill_index_list)
    if len(skill_indices) != len(skill_index_list):
        raise ValueError("Skill annotations contain duplicate skill_idx values.")
    indexed_skill_records = {
        record["skill_idx"]: record
        for record in skill_records
        if isinstance(record, Mapping) and isinstance(record.get("skill_idx"), int)
    }
    owners: dict[int, list[tuple[int, str | None]]] = collections.defaultdict(list)
    primitive_indices = set()
    for record in primitive_records:
        if not isinstance(record, Mapping):
            raise ValueError("Primitive annotations must be objects.")
        primitive_index = record.get("primitive_idx")
        if not isinstance(primitive_index, int):
            raise ValueError("Primitive primitive_idx must be an integer.")
        if primitive_index in primitive_indices:
            raise ValueError(f"Duplicate primitive_idx {primitive_index}.")
        primitive_indices.add(primitive_index)
        referenced_skills = _list_field(record, "skill_idxes")
        if not referenced_skills or not all(
            isinstance(skill_index, int) for skill_index in referenced_skills
        ):
            raise ValueError("Primitive skill_idxes must contain integers.")
        if len(referenced_skills) != len(set(referenced_skills)):
            raise ValueError(
                f"Primitive {primitive_index} contains duplicate skill references."
            )
        unknown_skills = sorted(set(referenced_skills).difference(skill_indices))
        if unknown_skills:
            raise ValueError(
                f"Primitive {primitive_index} references unknown skills "
                f"{unknown_skills}."
            )
        text = render_primitive_subgoal(record, registry=registry)
        if text is None:
            text = _render_referenced_atomic_sequence(
                referenced_skills,
                indexed_skill_records,
                registry=registry,
            )
        for skill_index in referenced_skills:
            owners[skill_index].append((primitive_index, text))

    resolved = {}
    for skill_index in skill_indices:
        candidates = sorted(owners.get(skill_index, ()))
        indices = tuple(primitive_index for primitive_index, _ in candidates)
        texts = tuple(text for _, text in candidates if text is not None)
        if not texts:
            resolved[skill_index] = ResolvedSubgoal(
                text=None,
                status=SubgoalStatus.MISSING,
                primitive_indices=indices,
            )
            continue
        if len(texts) != len(candidates):
            raise ValueError(
                f"Skill {skill_index} mixes empty and non-empty primitive owners."
            )
        status = (
            SubgoalStatus.UNIQUE if len(candidates) == 1 else SubgoalStatus.COMPOSITE
        )
        resolved[skill_index] = ResolvedSubgoal(
            text=COMPOSITE_SUBGOAL_SEPARATOR.join(texts),
            status=status,
            primitive_indices=indices,
        )
    return resolved

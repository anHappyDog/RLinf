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

"""Frozen v0.1 skill signatures derived from B1K annotation evidence."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Iterator

from .schema import Role


class ArgumentLayout(str, enum.Enum):
    """How raw ``object_id`` entries map to typed entity arguments."""

    POSITIONAL = "positional"
    REPEAT_FIRST = "repeat_first"
    HAND_OVER = "hand_over"


@dataclasses.dataclass(frozen=True)
class SkillSignature:
    """Typed interpretation of one canonical B1K skill annotation."""

    skill: str
    skill_id: int
    roles: tuple[Role, ...]
    layout: ArgumentLayout = ArgumentLayout.POSITIONAL
    raw_arity: int | None = None
    controlled_top_level_index: int = 0
    part_name: str | None = None

    def __post_init__(self) -> None:
        if not self.skill.strip():
            raise ValueError("skill must not be empty.")
        if not self.roles:
            raise ValueError(f"Skill {self.skill!r} must define at least one role.")
        raw_arity = self.raw_arity if self.raw_arity is not None else len(self.roles)
        if raw_arity <= 0:
            raise ValueError("raw_arity must be positive.")
        if not 0 <= self.controlled_top_level_index < raw_arity:
            raise ValueError(
                f"Invalid controlled argument index for skill {self.skill!r}."
            )

    @property
    def expected_raw_arity(self) -> int:
        """Return the expected number of top-level raw object entries."""
        return self.raw_arity if self.raw_arity is not None else len(self.roles)


class SkillSignatureRegistry:
    """Immutable lookup table for canonical B1K skill signatures."""

    def __init__(self, signatures: Iterable[SkillSignature]) -> None:
        by_skill: dict[str, SkillSignature] = {}
        by_id: dict[int, SkillSignature] = {}
        for signature in signatures:
            if signature.skill in by_skill:
                raise ValueError(f"Duplicate skill signature {signature.skill!r}.")
            if signature.skill_id in by_id:
                raise ValueError(f"Duplicate skill id {signature.skill_id}.")
            by_skill[signature.skill] = signature
            by_id[signature.skill_id] = signature
        self._by_skill = by_skill
        self._by_id = by_id

    def __len__(self) -> int:
        return len(self._by_skill)

    def __iter__(self) -> Iterator[SkillSignature]:
        return iter(self.signatures())

    def get(self, skill: str) -> SkillSignature | None:
        """Return a signature by canonical skill name, if registered."""
        return self._by_skill.get(skill)

    def get_by_id(self, skill_id: int) -> SkillSignature | None:
        """Return a signature by B1K skill ID, if registered."""
        return self._by_id.get(skill_id)

    def signatures(self) -> tuple[SkillSignature, ...]:
        """Return signatures in deterministic skill-name order."""
        return tuple(self._by_skill[name] for name in sorted(self._by_skill))


def _signature(
    skill: str,
    skill_id: int,
    *roles: Role,
    layout: ArgumentLayout = ArgumentLayout.POSITIONAL,
    raw_arity: int | None = None,
    controlled_index: int = 0,
    part_name: str | None = None,
) -> SkillSignature:
    return SkillSignature(
        skill=skill,
        skill_id=skill_id,
        roles=roles,
        layout=layout,
        raw_arity=raw_arity,
        controlled_top_level_index=controlled_index,
        part_name=part_name,
    )


DEFAULT_SKILL_SIGNATURE_REGISTRY = SkillSignatureRegistry(
    (
        _signature("attach", 19, Role.MANIPULATED, Role.DESTINATION),
        _signature("chop", 34, Role.TOOL, Role.TARGET),
        _signature("close door", 12, Role.TARGET, part_name="door"),
        _signature("close drawer", 11, Role.TARGET, part_name="drawer"),
        _signature("close lid", 14, Role.TARGET, part_name="lid"),
        _signature(
            "hand over",
            5,
            Role.MANIPULATED,
            layout=ArgumentLayout.HAND_OVER,
            raw_arity=3,
        ),
        _signature("hang", 61, Role.MANIPULATED, Role.DESTINATION),
        _signature("hold", 94, Role.MANIPULATED),
        _signature("ignite", 88, Role.TOOL, Role.TARGET),
        _signature("insert", 6, Role.MANIPULATED, Role.DESTINATION),
        _signature("move to", 1, Role.TARGET),
        _signature("open door", 10, Role.TARGET, part_name="door"),
        _signature("open drawer", 9, Role.TARGET, part_name="drawer"),
        _signature("open lid", 13, Role.TARGET, part_name="lid"),
        _signature("pick up from", 2, Role.MANIPULATED, Role.SOURCE),
        _signature("place in", 4, Role.MANIPULATED, Role.DESTINATION),
        _signature(
            "place in next to",
            92,
            Role.MANIPULATED,
            Role.DESTINATION,
            Role.REFERENCE,
        ),
        _signature("place on", 3, Role.MANIPULATED, Role.DESTINATION),
        _signature(
            "place on next to",
            91,
            Role.MANIPULATED,
            Role.DESTINATION,
            Role.REFERENCE,
        ),
        _signature("place under", 98, Role.MANIPULATED, Role.REFERENCE),
        _signature(
            "pour",
            28,
            Role.OTHER,
            Role.MANIPULATED,
            Role.DESTINATION,
            layout=ArgumentLayout.REPEAT_FIRST,
            controlled_index=1,
        ),
        _signature("press", 67, Role.TARGET),
        _signature("pull tray", 101, Role.TARGET, part_name="tray"),
        _signature("push to", 90, Role.MANIPULATED, Role.DESTINATION),
        _signature("push tray", 100, Role.TARGET, part_name="tray"),
        _signature("release", 8, Role.MANIPULATED),
        _signature("spray", 95, Role.TOOL, Role.TARGET),
        _signature(
            "sweep off",
            102,
            Role.MANIPULATED,
            Role.SOURCE,
            layout=ArgumentLayout.REPEAT_FIRST,
        ),
        _signature("sweep surface", 50, Role.TOOL, Role.TARGET),
        _signature("tip over", 99, Role.MANIPULATED),
        _signature("turn off switch", 70, Role.TARGET),
        _signature("turn on switch", 69, Role.TARGET),
        _signature("turn to", 93, Role.MANIPULATED, Role.REFERENCE),
        _signature("wipe hard", 46, Role.TOOL, Role.TARGET),
    )
)

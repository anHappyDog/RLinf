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

"""Resolve B1K annotation entities to OmniGibson visual-mesh IDs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping


def parse_instance_id_mapping(
    value: str | Mapping[str | int, str],
) -> dict[int, str]:
    """Parse the serialized instance-ID mapping stored in episode metadata.

    Args:
        value: JSON string or mapping from segmentation IDs to prim paths.

    Returns:
        A mapping with integer keys in deterministic insertion order.
    """
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping):
        raise TypeError("Instance-ID mapping must be a JSON object or mapping.")

    result: dict[int, str] = {}
    for instance_id, prim_path in parsed.items():
        try:
            numeric_id = int(instance_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid segmentation ID {instance_id!r}.") from error
        is_special_label = numeric_id in {0, 1} and prim_path in {
            "background",
            "unlabelled",
        }
        if not isinstance(prim_path, str) or not (
            prim_path.startswith("/") or is_special_label
        ):
            raise ValueError(f"Invalid prim path for instance {instance_id!r}.")
        result[numeric_id] = prim_path
    return result


def object_name_from_prim_path(prim_path: str) -> str | None:
    """Extract the scene-object name from an OmniGibson visual prim path."""
    components = [component for component in prim_path.split("/") if component]
    if len(components) < 3 or components[0] != "World":
        return None
    return components[2]


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _part_aliases(part_name: str) -> tuple[str, ...]:
    compact = _compact_name(part_name)
    aliases = [compact]
    if compact == "door":
        aliases.append("leaf")
    return tuple(aliases)


_BUTTON_PART_ALIASES = ("button", "switch")


class EntityResolver:
    """Resolve a symbolic B1K object ID to all of its visual-mesh IDs.

    Matching is performed on the exact scene-object path component. This avoids
    substring collisions such as ``mousetrap_4`` matching ``mousetrap_47``.
    """

    def __init__(self, instance_id_mapping: Mapping[int, str]) -> None:
        self._paths = dict(instance_id_mapping)
        self._object_names = {
            instance_id: object_name_from_prim_path(path)
            for instance_id, path in self._paths.items()
        }

    @property
    def instance_id_mapping(self) -> dict[int, str]:
        """Return a copy of the underlying ID-to-prim-path mapping."""
        return dict(self._paths)

    def resolve(
        self,
        raw_object_id: str,
        *,
        visible_instance_ids: Iterable[int] | None = None,
    ) -> tuple[int, ...]:
        """Return all visual-mesh IDs belonging to one annotated object.

        ``robot`` is an agent symbol in annotations rather than a scene-object
        identifier, so it intentionally has no visual grounding.
        """
        if raw_object_id == "robot":
            return ()
        visible = (
            None
            if visible_instance_ids is None
            else {int(instance_id) for instance_id in visible_instance_ids}
        )
        exact_matches = tuple(
            sorted(
                instance_id
                for instance_id, object_name in self._object_names.items()
                if object_name == raw_object_id
                and (visible is None or instance_id in visible)
            )
        )
        if exact_matches:
            return exact_matches

        particle_pattern = re.compile(
            rf"{re.escape(_compact_name(raw_object_id))}particle\d+"
        )
        return tuple(
            sorted(
                instance_id
                for instance_id, path in self._paths.items()
                if (visible is None or instance_id in visible)
                and any(
                    particle_pattern.fullmatch(_compact_name(component))
                    for component in path.split("/")
                )
            )
        )

    def resolve_part(
        self,
        raw_object_id: str,
        part_name: str,
        *,
        visible_instance_ids: Iterable[int] | None = None,
    ) -> tuple[int, ...]:
        """Return object mesh IDs whose suffix identifies the requested part.

        Part annotations are textual (for example, ``right door``), whereas
        prim paths use names such as ``rightdoor``. Comparison therefore
        ignores separators. Generic doors also accept OmniGibson's ``leaf``
        link name.
        """
        object_ids = self.resolve(
            raw_object_id, visible_instance_ids=visible_instance_ids
        )
        aliases = _part_aliases(part_name)
        matches = []
        for instance_id in object_ids:
            path = self._paths[instance_id]
            components = [component for component in path.split("/") if component]
            suffix = _compact_name("/".join(components[3:]))
            if any(alias in suffix for alias in aliases):
                matches.append(instance_id)
        if matches:
            return tuple(matches)

        generic_part = _compact_name(part_name)
        if generic_part not in {"door", "drawer", "lid", "tray"}:
            return ()
        return tuple(
            instance_id
            for instance_id in object_ids
            if "baselink" not in _compact_name(self._paths[instance_id])
            and not any(
                alias in _compact_name(self._paths[instance_id])
                for alias in _BUTTON_PART_ALIASES
            )
        )

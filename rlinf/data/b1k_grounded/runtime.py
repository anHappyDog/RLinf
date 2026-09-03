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

"""Runtime prompt selection for grounded-control policy serving."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from pathlib import Path

from .schema import GroundedControlSpec
from .serializer import ControlProfile, ControlSerializer

GROUND_CONTROL_JSON_KEY = "grounded_control_json"
_EPISODE_PATTERN = re.compile(r"episode_(\d+)$")


def _runtime_control(control: GroundedControlSpec) -> GroundedControlSpec:
    """Remove frame-specific geometry from a reusable online control template."""
    arguments = []
    for argument in control.arguments:
        part = argument.part
        if part is not None:
            part = dataclasses.replace(part, groundings={})
        arguments.append(dataclasses.replace(argument, groundings={}, part=part))
    return dataclasses.replace(control, arguments=tuple(arguments), timestep=None)


class SidecarControlIndex:
    """Index one grounded-control sidecar by episode and segment."""

    def __init__(
        self,
        controls: Mapping[tuple[int, int], GroundedControlSpec],
        intervals: Mapping[tuple[int, int], tuple[tuple[int, int], ...]] | None = None,
    ) -> None:
        self._controls = dict(controls)
        self._intervals = {} if intervals is None else dict(intervals)

    @classmethod
    def from_parquet(cls, path: str | Path, task_name: str) -> "SidecarControlIndex":
        """Load only the columns needed for online Oracle conditioning."""
        import pyarrow.parquet as pq

        table = pq.read_table(
            Path(path).expanduser(),
            columns=[
                "task_name",
                "episode_index",
                "segment_index",
                "interval_start",
                "interval_end",
                "control_json",
            ],
            filters=[("task_name", "=", task_name)],
        )
        controls = {}
        intervals: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for row in table.to_pylist():
            key = (int(row["episode_index"]), int(row["segment_index"]))
            control = _runtime_control(
                GroundedControlSpec.from_json(row["control_json"])
            )
            previous = controls.setdefault(key, control)
            if previous != control:
                raise ValueError(
                    f"Conflicting control records for episode/segment {key}."
                )
            interval = (int(row["interval_start"]), int(row["interval_end"]))
            intervals.setdefault(key, set()).add(interval)
        if not controls:
            raise ValueError(
                f"No sidecar controls found for task {task_name!r}: {path}"
            )
        return cls(
            controls,
            {key: tuple(sorted(values)) for key, values in intervals.items()},
        )

    def get(self, episode_index: int, segment_index: int) -> GroundedControlSpec:
        """Return one exact episode/segment condition."""
        key = (episode_index, segment_index)
        try:
            return self._controls[key]
        except KeyError as error:
            raise KeyError(
                f"No grounded control record for episode/segment {key}."
            ) from error

    def segment_at_start(self, episode_index: int, start_frame: int) -> int:
        """Resolve the segment whose annotated interval starts at ``start_frame``."""
        matches = [
            segment_index
            for (indexed_episode, segment_index), intervals in self._intervals.items()
            for start, _ in intervals
            if indexed_episode == episode_index and start == start_frame
        ]
        if len(matches) != 1:
            raise KeyError(
                "Expected one segment for episode/start frame "
                f"({episode_index}, {start_frame}), found {len(matches)}."
            )
        return matches[0]

    def interval_at_start(
        self, episode_index: int, start_frame: int
    ) -> tuple[int, int, int]:
        """Return ``(segment_index, start, end)`` for one exact interval start."""
        segment_index = self.segment_at_start(episode_index, start_frame)
        intervals = self._intervals[(episode_index, segment_index)]
        return next(
            (segment_index, start, end)
            for start, end in intervals
            if start == start_frame
        )

    def intervals_for_episode(
        self, episode_index: int
    ) -> tuple[tuple[int, int, int], ...]:
        """Return sorted ``(segment_index, start, end)`` episode intervals."""
        return tuple(
            sorted(
                (segment_index, start, end)
                for (
                    indexed_episode,
                    segment_index,
                ), intervals in self._intervals.items()
                if indexed_episode == episode_index
                for start, end in intervals
            )
        )


def episode_index_from_annotation_dir(path: str | Path) -> int:
    """Extract the demo episode index from an orchestrator annotation path."""
    match = _EPISODE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(
            f"Cannot parse episode index from annotation directory: {path}"
        )
    return int(match.group(1))


class GroundedPromptController:
    """Produce the exact P0/P1/P2 prompt format used during action SFT."""

    def __init__(
        self,
        serializer: ControlSerializer,
        profile: ControlProfile,
        goal: str,
    ) -> None:
        self._serializer = serializer
        self._profile = profile
        self._p0_prompt = serializer.serialize(
            GroundedControlSpec(
                goal=goal,
                subgoal=None,
                skill=None,
                arguments=(),
            ),
            ControlProfile.P0_DIRECT,
        )

    def prompt(self, observation: Mapping[str, object]) -> str:
        """Return a static P0 prompt or serialize the supplied Oracle condition."""
        if self._profile is ControlProfile.P0_DIRECT:
            return self._p0_prompt

        control_json = observation[GROUND_CONTROL_JSON_KEY]
        if not isinstance(control_json, str):
            raise TypeError(f"{GROUND_CONTROL_JSON_KEY} must be a JSON string.")
        control = GroundedControlSpec.from_json(control_json)
        return self._serializer.serialize(control, self._profile)

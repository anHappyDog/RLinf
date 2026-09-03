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

"""Uniform-scale potential rewards for BEHAVIOR subtask rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PotentialTerm:
    """One scalar progress metric used to construct a state potential."""

    key: str
    scale: float
    direction: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PotentialTerm":
        """Parse and validate a potential term."""
        term = cls(
            key=str(value["key"]),
            scale=float(value.get("scale", 1.0)),
            direction=str(value.get("direction", "increase")),
        )
        if not term.key:
            raise ValueError("Potential term key must not be empty.")
        if term.scale < 0:
            raise ValueError("Potential term scale must be non-negative.")
        if term.direction not in ("increase", "decrease"):
            raise ValueError(
                "Potential term direction must be 'increase' or 'decrease'."
            )
        return term

    def value(self, stage_info: Mapping[str, Any]) -> float:
        """Evaluate this term from a stage-info mapping."""
        current: Any = stage_info
        for component in self.key.split("."):
            if not isinstance(current, Mapping) or component not in current:
                raise KeyError(
                    f"Reward metric {self.key!r} is missing from stage info."
                )
            current = current[component]
        scalar = float(current)
        sign = 1.0 if self.direction == "increase" else -1.0
        return sign * self.scale * scalar


@dataclass(frozen=True)
class SubtaskRewardSpec:
    """Common reward scale shared by every trained subtask."""

    potential_terms: tuple[PotentialTerm, ...]
    success_bonus: float = 10.0
    timeout_penalty: float = -2.0
    step_penalty: float = -0.01
    progress_clip: float = 1.0
    max_steps: int = 256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SubtaskRewardSpec":
        """Parse a manifest reward specification."""
        terms: Sequence[Mapping[str, Any]] = value.get("potential_terms", ())
        spec = cls(
            potential_terms=tuple(PotentialTerm.from_mapping(term) for term in terms),
            success_bonus=float(value.get("success_bonus", 10.0)),
            timeout_penalty=float(value.get("timeout_penalty", -2.0)),
            step_penalty=float(value.get("step_penalty", -0.01)),
            progress_clip=float(value.get("progress_clip", 1.0)),
            max_steps=int(value.get("max_steps", 256)),
        )
        if spec.success_bonus <= 0:
            raise ValueError("success_bonus must be positive.")
        if spec.timeout_penalty > 0:
            raise ValueError("timeout_penalty must be non-positive.")
        if spec.step_penalty > 0:
            raise ValueError("step_penalty must be non-positive.")
        if spec.progress_clip <= 0:
            raise ValueError("progress_clip must be positive.")
        if spec.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        return spec


@dataclass(frozen=True)
class SubtaskRewardOutcome:
    """Reward and terminal status for one primitive simulator step."""

    reward: float
    success: bool
    timeout: bool
    potential: float
    progress: float


class SubtaskRewardTracker:
    """Stateful potential-difference reward for one subtask episode."""

    def __init__(self, spec: SubtaskRewardSpec) -> None:
        self.spec = spec
        self.steps = 0
        self.previous_potential: float | None = None

    def step(self, stage_info: Mapping[str, Any]) -> SubtaskRewardOutcome:
        """Evaluate one stage-info record on a task-independent reward scale."""
        self.steps += 1
        potential = sum(term.value(stage_info) for term in self.spec.potential_terms)
        if self.previous_potential is None:
            progress = 0.0
        else:
            raw_progress = potential - self.previous_potential
            progress = max(
                -self.spec.progress_clip,
                min(self.spec.progress_clip, raw_progress),
            )
        self.previous_potential = potential

        success = bool(stage_info.get("completed", False))
        timeout = self.steps >= self.spec.max_steps and not success
        reward = self.spec.step_penalty + progress
        if success:
            reward += self.spec.success_bonus
        elif timeout:
            reward += self.spec.timeout_penalty
        return SubtaskRewardOutcome(
            reward=reward,
            success=success,
            timeout=timeout,
            potential=potential,
            progress=progress,
        )


def validate_demo_horizon(
    spec: SubtaskRewardSpec,
    *,
    start_frame: int,
    end_frame: int,
) -> None:
    """Require the configured timeout to admit its audited GT suffix."""
    primitive_steps = end_frame - start_frame
    if primitive_steps <= 0:
        raise ValueError(
            f"Demo horizon must be positive, got [{start_frame}, {end_frame})."
        )
    if spec.max_steps < primitive_steps:
        raise ValueError(
            f"Reward max_steps={spec.max_steps} is shorter than the audited GT "
            f"suffix ({primitive_steps} primitive steps, frames "
            f"[{start_frame}, {end_frame}))."
        )


def get_stage_info(info: Mapping[str, Any], stage_index: int) -> Mapping[str, Any]:
    """Extract one active sequential-reward stage with strict validation."""
    try:
        stage_infos = info["reward"]["task_specific"]["stage_infos"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "BEHAVIOR info is missing reward.task_specific.stage_infos."
        ) from exc
    if not isinstance(stage_infos, Mapping):
        raise TypeError("stage_infos must be a mapping.")
    names = tuple(stage_infos)
    if not 0 <= stage_index < len(names):
        raise IndexError(
            f"stage_index={stage_index} is outside {len(names)} reward stages."
        )
    stage_info = stage_infos[names[stage_index]]
    if not isinstance(stage_info, Mapping):
        raise TypeError("Selected stage info must be a mapping.")
    return stage_info


__all__ = [
    "PotentialTerm",
    "SubtaskRewardOutcome",
    "SubtaskRewardSpec",
    "SubtaskRewardTracker",
    "get_stage_info",
    "validate_demo_horizon",
]

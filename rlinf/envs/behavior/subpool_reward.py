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

import math
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
    potential_discount: float = 1.0
    prime_potential_at_reset: bool = False
    zero_terminal_potential: bool = False

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
            potential_discount=float(value.get("potential_discount", 1.0)),
            prime_potential_at_reset=bool(value.get("prime_potential_at_reset", False)),
            zero_terminal_potential=bool(value.get("zero_terminal_potential", False)),
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
        if not 0.0 <= spec.potential_discount <= 1.0:
            raise ValueError("potential_discount must be in [0, 1].")
        return spec


@dataclass(frozen=True)
class SubtaskRewardOutcome:
    """Reward and terminal status for one primitive simulator step."""

    reward: float
    success: bool
    timeout: bool
    potential: float
    continuation_potential: float
    progress: float
    step_penalty: float
    terminal_reward: float
    cumulative_progress: float
    cumulative_step_penalty: float
    cumulative_terminal_reward: float


class SubtaskRewardTracker:
    """Stateful potential-difference reward for one subtask episode."""

    def __init__(self, spec: SubtaskRewardSpec) -> None:
        self.spec = spec
        self.steps = 0
        self.previous_potential: float | None = None
        self.initial_potential = 0.0
        self.cumulative_progress = 0.0
        self.cumulative_step_penalty = 0.0
        self.cumulative_terminal_reward = 0.0

    def prime(self, stage_info: Mapping[str, Any]) -> float:
        """Record the initial state potential before the first action."""
        if self.steps != 0 or self.previous_potential is not None:
            raise RuntimeError("Reward potential can only be primed once at reset.")
        self.previous_potential = self._potential(stage_info)
        self.initial_potential = self.previous_potential
        return self.previous_potential

    def _potential(self, stage_info: Mapping[str, Any]) -> float:
        return sum(term.value(stage_info) for term in self.spec.potential_terms)

    def step(self, stage_info: Mapping[str, Any]) -> SubtaskRewardOutcome:
        """Evaluate one stage-info record on a task-independent reward scale."""
        self.steps += 1
        potential = self._potential(stage_info)
        success = bool(stage_info.get("completed", False))
        timeout = self.steps >= self.spec.max_steps and not success
        continuation_potential = (
            0.0
            if self.spec.zero_terminal_potential and (success or timeout)
            else potential
        )
        if self.previous_potential is None:
            progress = 0.0
        else:
            raw_progress = (
                self.spec.potential_discount * continuation_potential
                - self.previous_potential
            )
            progress = max(
                -self.spec.progress_clip,
                min(self.spec.progress_clip, raw_progress),
            )
        self.previous_potential = potential

        terminal_reward = 0.0
        if success:
            terminal_reward = self.spec.success_bonus
        elif timeout:
            terminal_reward = self.spec.timeout_penalty
        reward = self.spec.step_penalty + progress + terminal_reward
        self.cumulative_progress += progress
        self.cumulative_step_penalty += self.spec.step_penalty
        self.cumulative_terminal_reward += terminal_reward
        return SubtaskRewardOutcome(
            reward=reward,
            success=success,
            timeout=timeout,
            potential=potential,
            continuation_potential=continuation_potential,
            progress=progress,
            step_penalty=self.spec.step_penalty,
            terminal_reward=terminal_reward,
            cumulative_progress=self.cumulative_progress,
            cumulative_step_penalty=self.cumulative_step_penalty,
            cumulative_terminal_reward=self.cumulative_terminal_reward,
        )


def compute_pickup_potential_v2(
    *,
    eef_distance: float,
    target_contact: bool,
    in_hand: bool,
    on_support: bool,
    lift_clearance: float,
    approach_distance: float = 0.3,
    lift_clearance_target: float = 0.04,
) -> dict[str, float]:
    """Return a bounded phase-aware pickup potential and its components.

    The phases are deliberately separated: approach and contact matter only
    while the target remains on its original support, a registered grasp moves
    to a higher potential band, and an unheld dropped target receives no
    progress credit.
    """
    approach_quality = (
        max(0.0, 1.0 - min(float(eef_distance) / approach_distance, 1.0))
        if math.isfinite(eef_distance)
        else 0.0
    )
    contact_quality = float(target_contact)
    lift_quality = min(max(float(lift_clearance) / lift_clearance_target, 0.0), 1.0)

    if in_hand:
        potential = 1.0 if not on_support else 0.65 + 0.25 * lift_quality
    elif on_support:
        potential = 0.35 * approach_quality + 0.20 * contact_quality
    else:
        potential = 0.0

    return {
        "potential": potential,
        "approach_quality": approach_quality,
        "contact_quality": contact_quality,
        "lift_quality": lift_quality,
    }


def apply_reward_overrides(
    reward_spec: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a validated runtime reward spec without mutating the manifest."""
    resolved = dict(reward_spec)
    if overrides:
        resolved.update(overrides)
    SubtaskRewardSpec.from_mapping(resolved)
    return resolved


def resolve_demo_reward_spec(
    value: Mapping[str, Any],
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """Resolve an export-time reward template for one audited demo suffix.

    ``max_steps_multiplier`` derives the concrete timeout from the suffix
    duration. ``step_penalty_budget`` keeps the cumulative time penalty fixed
    even when snapshots have different horizons. The returned mapping contains
    only runtime fields accepted by :class:`SubtaskRewardSpec`.
    """
    resolved = dict(value)
    max_steps_multiplier = resolved.pop("max_steps_multiplier", None)
    step_penalty_budget = resolved.pop("step_penalty_budget", None)

    primitive_steps = end_frame - start_frame
    if primitive_steps <= 0:
        raise ValueError(
            f"Demo horizon must be positive, got [{start_frame}, {end_frame})."
        )
    if max_steps_multiplier is not None:
        if "max_steps" in resolved:
            raise ValueError(
                "Reward template must define either max_steps or "
                "max_steps_multiplier, not both."
            )
        multiplier = float(max_steps_multiplier)
        if multiplier <= 0:
            raise ValueError("max_steps_multiplier must be positive.")
        resolved["max_steps"] = int(primitive_steps * multiplier)

    if step_penalty_budget is not None:
        if "step_penalty" in resolved:
            raise ValueError(
                "Reward template must define either step_penalty or "
                "step_penalty_budget, not both."
            )
        if "max_steps" not in resolved:
            raise ValueError(
                "step_penalty_budget requires max_steps or max_steps_multiplier."
            )
        resolved["step_penalty"] = float(step_penalty_budget) / int(
            resolved["max_steps"]
        )

    spec = SubtaskRewardSpec.from_mapping(resolved)
    validate_demo_horizon(
        spec,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return resolved


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
    "compute_pickup_potential_v2",
    "SubtaskRewardOutcome",
    "SubtaskRewardSpec",
    "SubtaskRewardTracker",
    "get_stage_info",
    "resolve_demo_reward_spec",
    "validate_demo_horizon",
]

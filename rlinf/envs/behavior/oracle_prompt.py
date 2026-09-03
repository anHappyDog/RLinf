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

"""Online prompt selection for paired BEHAVIOR hierarchical evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def extract_sequential_reward_info(info: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the task-specific sequential-reward payload from an env info."""
    if "current_stage_name" in info:
        return info

    reward_info = info.get("reward")
    if not isinstance(reward_info, Mapping):
        return {}

    task_specific = reward_info.get("task_specific")
    if isinstance(task_specific, Mapping):
        return task_specific

    for payload in reward_info.values():
        if isinstance(payload, Mapping) and "current_stage_name" in payload:
            return payload
    return {}


@dataclass
class StagePromptController:
    """Select either the full-task prompt or the simulator's active stage.

    The active stage comes from OmniGibson's sequential task reward. This keeps
    Oracle-HL switching tied to task predicates (distance, grasp, toggle, and
    placement state), rather than replaying demonstration timestamps after the
    policy has diverged from a demonstration.
    """

    task_prompt: str
    num_envs: int
    mode: str = "task"
    stage_prompts: Mapping[str, str] = field(default_factory=dict)
    initial_stage: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"task", "oracle_stage"}:
            raise ValueError(
                f"prompt mode must be 'task' or 'oracle_stage', got {self.mode!r}."
            )
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.mode == "oracle_stage":
            if not self.stage_prompts:
                raise ValueError("oracle_stage mode requires stage_prompts.")
            if self.initial_stage is None:
                self.initial_stage = next(iter(self.stage_prompts))
            if self.initial_stage not in self.stage_prompts:
                raise ValueError(
                    f"initial stage {self.initial_stage!r} has no configured prompt."
                )
        self.reset()

    def reset(self) -> None:
        """Reset every environment to the first oracle stage."""
        self._stages = [self.initial_stage] * self.num_envs

    def prompts(self) -> list[str]:
        """Return the prompt currently assigned to each vector environment."""
        if self.mode == "task":
            return [self.task_prompt] * self.num_envs
        return [self.stage_prompts[stage] for stage in self._stages]

    def update(self, infos: Sequence[Mapping[str, Any]]) -> list[str]:
        """Advance prompts from the latest sequential-reward stage reports."""
        if len(infos) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} info records, got {len(infos)}."
            )
        if self.mode == "task":
            return self.prompts()

        for index, info in enumerate(infos):
            reward_info = extract_sequential_reward_info(info)
            stage = reward_info.get("current_stage_name")
            if stage in {None, "done"}:
                continue
            if stage not in self.stage_prompts:
                raise KeyError(
                    f"Sequential reward reported stage {stage!r}, but no oracle "
                    "prompt is configured for it."
                )
            self._stages[index] = stage
        return self.prompts()

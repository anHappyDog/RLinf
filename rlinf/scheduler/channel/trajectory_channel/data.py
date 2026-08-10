# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from rlinf.data.schema.embodied_types import (
    EmbodiedRolloutResult,
    EnvResult,
    TrajectorySource,
    put_tensor_device,
)


@dataclass(kw_only=True)
class PolicyStep:
    """Policy inference associated with one or more environment sources."""

    sources: list[TrajectorySource]
    obs: dict[str, Any]
    rollout_result: EmbodiedRolloutResult

    def __post_init__(self) -> None:
        """Move inference payloads to CPU for transport."""
        self.obs = put_tensor_device(self.obs, "cpu")


@dataclass(kw_only=True)
class EnvStepResult:
    """Environment outcome for policy actions identified by ``sources``."""

    sources: list[TrajectorySource]
    result: EnvResult
    needs_terminal: bool = False


@dataclass(kw_only=True)
class TrajectoryStart:
    """Initial environment state for one trajectory source and epoch."""

    source: TrajectorySource
    result: EnvResult


@dataclass(kw_only=True)
class TerminalResult:
    """Policy-side processing result for terminal or final observations."""

    sources: list[TrajectorySource]
    obs: dict[str, Any]
    bootstrap_values: torch.Tensor | None
    forward_inputs: dict[str, Any] | None

    def __post_init__(self) -> None:
        """Move terminal payloads to CPU for transport."""
        self.obs = put_tensor_device(self.obs, "cpu")
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.forward_inputs is not None:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")


@dataclass(kw_only=True, frozen=True)
class TrajectoryEnd:
    """Signal that one producer has finished a training step."""

    step_id: int
    source: tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class TrajectoryEpochEnd:
    """Signal that one environment source has finished a pipeline epoch."""

    step_id: int
    epoch_id: int
    source: tuple[int, int]


TrajectoryData: TypeAlias = (
    PolicyStep
    | TrajectoryStart
    | EnvStepResult
    | TerminalResult
    | TrajectoryEnd
    | TrajectoryEpochEnd
)

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
    put_tensor_device,
)


@dataclass(kw_only=True)
class TrajectorySegment:
    """One append operation for a set of logical trajectory sources."""

    step_id: int
    epoch_id: int
    sources: list[tuple[int, int, int]]
    obs: dict[str, Any]
    next_obs: dict[str, Any]
    env_result: EnvResult
    rollout_result: EmbodiedRolloutResult
    initial_env_result: EnvResult | None = None
    forward_inputs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Move segment payloads to CPU for transport."""
        self.obs = put_tensor_device(self.obs, "cpu")
        self.next_obs = put_tensor_device(self.next_obs, "cpu")
        if self.forward_inputs is not None:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")


@dataclass(kw_only=True, frozen=True)
class TrajectoryEnd:
    """Signal that one producer has finished a training step."""

    step_id: int
    source: tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class TrajectoryEpochEnd:
    """Signal that one producer has finished a pipeline epoch."""

    step_id: int
    epoch_id: int
    source: tuple[int, int]
    sources: list[tuple[int, int, int]]
    final_prev_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Move final values to contiguous CPU storage."""
        if self.final_prev_values is not None:
            object.__setattr__(
                self,
                "final_prev_values",
                self.final_prev_values.cpu().contiguous(),
            )


TrajectoryData: TypeAlias = TrajectorySegment | TrajectoryEnd | TrajectoryEpochEnd

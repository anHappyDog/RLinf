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

"""Embodied rollout schemas and trajectory collection APIs."""

from rlinf.data.schema.embodied.trajectory import (
    TrajectoryCollector,
    TrajectoryMode,
    TrajectoryPlan,
    select_trajectory_collector,
    select_trajectory_dispatcher,
)
from rlinf.data.schema.embodied.types import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    LeRobotChunk,
    LeRobotFrame,
    LeRobotStep,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    RTCActionResponse,
    RTCRequest,
    Trajectory,
    TrajectoryKey,
    TrajectoryPart,
    TrajectorySource,
    TrajectoryStep,
    get_model_weights_id,
)

__all__ = [
    "EnvOutput",
    "EnvPart",
    "EnvTransition",
    "LeRobotChunk",
    "LeRobotFrame",
    "LeRobotStep",
    "PolicyInput",
    "PolicyOutput",
    "PolicyPart",
    "RTCActionResponse",
    "RTCRequest",
    "Trajectory",
    "TrajectoryCollector",
    "TrajectoryKey",
    "TrajectoryMode",
    "TrajectoryPart",
    "TrajectoryPlan",
    "TrajectorySource",
    "TrajectoryStep",
    "get_model_weights_id",
    "select_trajectory_collector",
    "select_trajectory_dispatcher",
]

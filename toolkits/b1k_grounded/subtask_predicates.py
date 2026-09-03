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

"""Pure helpers for calibrated B1K subtask success predicates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def base_planar_pose_from_behavior_state(state: np.ndarray) -> np.ndarray:
    """Extract ``(x, y, yaw)`` from one 256-dimensional B1K state."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (256,):
        raise ValueError(f"Expected a 256-dimensional B1K state, got {state.shape}.")
    yaw = math.atan2(state[148], state[145])
    return np.asarray([state[140], state[141], yaw], dtype=np.float64)


def planar_pose_error(
    current_pose: np.ndarray, target_pose: np.ndarray
) -> tuple[float, float]:
    """Return planar position distance and wrapped absolute yaw error."""
    current_pose = np.asarray(current_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    if current_pose.shape != (3,) or target_pose.shape != (3,):
        raise ValueError("Planar poses must have shape (3,).")
    position_error = float(np.linalg.norm(current_pose[:2] - target_pose[:2]))
    yaw_delta = current_pose[2] - target_pose[2]
    yaw_error = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
    return position_error, yaw_error


def demo_terminal_pose_result(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    position_threshold: float,
    yaw_threshold: float,
) -> dict[str, Any]:
    """Evaluate navigation against the annotated demo's terminal base region."""
    if position_threshold <= 0 or yaw_threshold <= 0:
        raise ValueError("Pose thresholds must be positive.")
    position_error, yaw_error = planar_pose_error(current_pose, target_pose)
    return {
        "completed": (
            position_error <= position_threshold and yaw_error <= yaw_threshold
        ),
        "base_position_error": position_error,
        "base_yaw_error": yaw_error,
        "base_position_threshold": position_threshold,
        "base_yaw_threshold": yaw_threshold,
    }


def is_successful_predicate_termination(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> bool:
    """Return whether B1K stopped only because its task predicate succeeded."""
    if not terminated or truncated:
        return False
    conditions = info.get("done", {}).get("termination_conditions", {})
    timeout = conditions.get("timeout", {})
    predicate = conditions.get("predicate", {})
    return bool(
        predicate.get("done", False)
        and predicate.get("success", False)
        and not timeout.get("done", False)
    )

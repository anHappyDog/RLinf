# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Controller-URDF limits for the deployment's FR3 joint tracker."""

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rlinf.robotics.parts.arms.franky import FrankyArm


@dataclass(frozen=True)
class FR3Model:
    """Limits in controller raw-q order, joint1 through joint7, in radians."""

    name: str
    urdf_sha256: str
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_urdf(cls, path: str) -> "FR3Model":
        """Parse the saved controller model; reject a different joint convention."""
        content = Path(path).read_bytes()
        root = ET.fromstring(content)
        if root.attrib.get("name") not in ("fr3v2", "fr3v2.1"):
            raise ValueError("Expected the controller-exported fr3v2 or fr3v2.1 URDF.")
        movable = [
            joint for joint in root.findall("joint") if joint.attrib["type"] != "fixed"
        ]
        names = [f"joint{i}" for i in range(1, 8)]
        if [joint.attrib["name"] for joint in movable] != names:
            raise ValueError(
                "URDF must contain joint1..joint7 in controller raw-q order."
            )
        limits = []
        for index, joint in enumerate(movable):
            if (
                joint.attrib["type"] != "revolute"
                or joint.find("parent").attrib["link"] != f"link{index}"
                or joint.find("child").attrib["link"] != f"link{index + 1}"
                or joint.find("mimic") is not None
                or not np.array_equal(
                    np.fromstring(joint.find("axis").attrib["xyz"], sep=" "), [0, 0, 1]
                )
            ):
                raise ValueError(f"Unexpected joint convention for {names[index]}.")
            limit = joint.find("limit").attrib
            limits.append([float(limit[key]) for key in ("lower", "upper", "velocity")])
        limits = np.asarray(limits)
        if (
            not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
            or np.any(limits[:, 2] <= 0)
        ):
            raise ValueError(
                "URDF must provide finite ordered position limits and positive velocities."
            )
        return cls(
            root.attrib["name"], hashlib.sha256(content).hexdigest(), *limits.T.copy()
        )

    def validate_targets(self, targets: np.ndarray) -> None:
        """Reject invalid targets rather than silently clipping a policy action."""
        targets = np.asarray(targets)
        if targets.shape[-1:] != (7,) or not np.isfinite(targets).all():
            raise ValueError(
                "Expected finite absolute targets in raw joint1..joint7 order."
            )
        if np.any(targets < self.lower) or np.any(targets > self.upper):
            raise ValueError(
                "Joint targets exceed the controller-exported FR3 position limits."
            )

    def contract(self) -> dict:
        """Return the exact model limits recorded by this deployment."""
        return {
            "model": self.name,
            "urdf_sha256": self.urdf_sha256,
            "joint_order": [f"joint{i}" for i in range(1, 8)],
            "joint_zero_offset_rad": [0.0] * 7,
            "lower_rad": self.lower.tolist(),
            "upper_rad": self.upper.tolist(),
            "velocity_rad_s": self.velocity.tolist(),
        }


class FR3Arm(FrankyArm):
    """Use Franky's native reactive joint trajectory generator with FR3 limits."""

    # Franky multiplies this by the upstream robot-wide 0.2 factor.
    MOTION_DYNAMICS_FACTOR = 0.5

    def __init__(self, address: str, *, urdf_path: str, **settings) -> None:
        self.model_contract = FR3Model.from_urdf(urdf_path)
        super().__init__(address, **settings)
        self._native_motion_started = False

    def move_joints(self, joint_positions: np.ndarray) -> None:
        """Replan a native Ruckig trajectory to an absolute, zero-velocity target."""
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError("FR3 joint target must have shape (7,).")
        self.model_contract.validate_targets(q)
        self._check_tracking_motion()
        franky = self._franky
        factor = self.MOTION_DYNAMICS_FACTOR
        motion = franky.JointMotion(
            franky.JointState(position=q, velocity=np.zeros(7)),
            reference_type=franky.ReferenceType.Absolute,
            relative_dynamics_factor=franky.RelativeDynamicsFactor(
                factor, factor, factor
            ),
            return_when_finished=False,
        )
        self._native_motion_started = True
        self._robot.move(motion, asynchronous=True)
        self._check_tracking_motion()

    def _check_tracking_motion(self) -> None:
        """Surface native async exceptions during commands and inherited state reads."""
        super()._check_tracking_motion()
        if not self._native_motion_started:
            return
        try:
            self._robot.poll_motion()
        except Exception as error:
            self._tracking_error = RuntimeError(
                f"FR3 native JointMotion failed: {error}"
            )
            raise self._tracking_error from error

    def cleanup(self) -> None:
        """Stop and join the native async motion, retaining controller errors."""
        if self._robot is None or not self._native_motion_started:
            return
        errors = []
        try:
            self._check_tracking_motion()
        except Exception as error:
            errors.append(error)
        try:
            self._robot.stop()
        except Exception as error:
            errors.append(error)
        try:
            if not self._robot.join_motion(timeout=5.0):
                raise TimeoutError(
                    "FR3 native JointMotion did not stop within 5 seconds."
                )
        except Exception as error:
            errors.append(error)
        self._native_motion_started = False
        for error in errors:
            self._logger.error("FR3 native motion cleanup: %s", error)
        if errors:
            raise errors[0]

    def reset_joint(self, positions: list[float]) -> None:
        """Exclude the upstream Panda-specific reset path from this deployment."""
        raise NotImplementedError("FR3 deployment has no automatic joint reset.")

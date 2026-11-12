# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Deferred Polymetis arm connection with an explicit controller-exported FR3 model."""

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from fr3_arm import FR3Model

from rlinf.robotics.parts.arms.base import BaseArm


@dataclass
class PolymetisArmState:
    """Measured joint state from one Polymetis RobotState message."""

    arm_joint_position: np.ndarray
    arm_joint_velocity: np.ndarray
    timestamp_s: float
    controller_latency_ms: float
    previous_command_successful: bool

    def to_dict(self) -> dict:
        """Expose only the fields provided by this backend."""
        return vars(self).copy()


class PolymetisArm(BaseArm):
    """Use the official adaptive joint impedance policy without homing or recovery."""

    STATE_FIELDS = ("arm_joint_position", "arm_joint_velocity")

    def __init__(self, server_host: str, server_port: int, *, urdf_path: str) -> None:
        if (
            not server_host
            or not isinstance(server_port, int)
            or not 0 < server_port < 65536
        ):
            raise ValueError(
                "An explicit Polymetis server_host and valid server_port are required."
            )
        self.server_host = server_host
        self.server_port = server_port
        self.model_contract = FR3Model.from_urdf(urdf_path)
        self._owns_policy = False
        self.server_metadata = None

    @property
    def action_features(self) -> dict:
        """Accept absolute joint targets in controller raw-q order."""
        return {"joint_position": {"shape": (7,)}}

    def _open(self):
        """Connect to the isolated server and validate its model without sending a policy."""
        from polymetis import RobotInterface

        interface = RobotInterface(ip_address=self.server_host, port=self.server_port)
        try:
            metadata = interface.metadata
            actual_sha = hashlib.sha256(metadata.urdf_file.encode()).hexdigest()
            if actual_sha != self.model_contract.urdf_sha256 or metadata.dof != 7:
                raise ValueError(
                    "Polymetis server must publish the exact controller-exported FR3 URDF and seven DOFs."
                )
            if interface.is_running_policy():
                raise RuntimeError(
                    "The isolated Polymetis server already has an active policy."
                )
            self.server_metadata = {
                "host": self.server_host,
                "port": self.server_port,
                "urdf_sha256": actual_sha,
                "ee_link_name": metadata.ee_link_name,
                "hz": metadata.hz,
                "polymetis_version": metadata.polymetis_version,
                "default_Kq": list(metadata.default_Kq),
                "default_Kqd": list(metadata.default_Kqd),
                "default_Kx": list(metadata.default_Kx),
                "default_Kxd": list(metadata.default_Kxd),
            }
            return interface
        except BaseException:
            interface.channel.close()
            raise

    def start_control(self) -> None:
        """Start this episode's default adaptive controller at the measured configuration."""
        if not self.is_connected:
            raise RuntimeError("Connect to Polymetis before starting joint impedance.")
        if self._owns_policy:
            return
        if self._device.is_running_policy():
            raise RuntimeError("Refusing to replace an existing Polymetis policy.")
        self._owns_policy = True
        self._device.start_joint_impedance()

    def move_to_joint_positions(
        self, positions: np.ndarray, *, time_to_go: float
    ) -> None:
        """Run the official blocking home trajectory before the policy starts."""
        import torch

        q = np.asarray(positions, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError("Home joint positions must have shape (7,).")
        self.model_contract.validate_targets(q)
        if not self.is_connected or self._owns_policy:
            raise RuntimeError("Connect to an idle Polymetis server before homing.")
        if self._device.is_running_policy():
            raise RuntimeError(
                "Refusing to replace an existing Polymetis policy for homing."
            )
        self._owns_policy = True
        self._device.move_to_joint_positions(
            torch.tensor(q, dtype=torch.float32),
            time_to_go=time_to_go,
            blocking=False,
        )
        # Wait without downloading the 1kHz state log through the control service.
        while self._device.is_running_policy():
            time.sleep(0.02)
        self._owns_policy = False

    def stop_control(self) -> None:
        """Terminate only the controller started by this arm instance."""
        if self._owns_policy and self.is_connected:
            if self._device.is_running_policy():
                self._device.terminate_current_policy(return_log=False)
            self._owns_policy = False

    def _release(self, device) -> None:
        """End the owned policy and close the gRPC channel even when termination fails."""
        try:
            self.stop_control()
        finally:
            device.channel.close()

    def get_state(self) -> PolymetisArmState:
        """Read measured q/dq together without deriving a desired or cached state."""
        if not self.is_connected:
            raise RuntimeError("Connect to Polymetis before reading joint state.")
        state = self._device.get_robot_state()
        if state.error_code:
            raise RuntimeError(
                f"Polymetis robot error_code={state.error_code}; no automatic recovery."
            )
        q = np.asarray(state.joint_positions, dtype=np.float64)
        dq = np.asarray(state.joint_velocities, dtype=np.float64)
        if (
            q.shape != (7,)
            or dq.shape != (7,)
            or not np.isfinite(q).all()
            or not np.isfinite(dq).all()
        ):
            raise RuntimeError(
                "Expected seven finite measured joint positions and velocities."
            )
        return PolymetisArmState(
            q,
            dq,
            state.timestamp.seconds + state.timestamp.nanos * 1e-9,
            state.prev_controller_latency_ms,
            state.prev_command_successful,
        )

    def send_action(self, action: dict) -> dict:
        """Update the impedance setpoint without scaling or generating a trajectory."""
        import torch

        q = np.asarray(action["joint_position"], dtype=np.float64)
        if q.shape != (7,):
            raise ValueError("Expected one absolute joint_position with shape (7,).")
        self.model_contract.validate_targets(q)
        self.start_control()
        if not self._device.is_running_policy():
            raise RuntimeError("The owned Polymetis joint impedance policy stopped.")
        self._device.update_desired_joint_positions(
            torch.tensor(q, dtype=torch.float32)
        )
        return {"joint_position": q.copy()}

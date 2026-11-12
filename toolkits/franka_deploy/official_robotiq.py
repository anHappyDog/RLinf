# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Use the published Polymetis Robotiq register mappings on the existing link.

Mappings and completion flags follow FacebookResearch/fairo (MIT), commit
0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2, robotiq_2f_gripper.py:102,130-163.
The source's force argument is percent, not Newtons.
"""

import time
from dataclasses import dataclass

import numpy as np
from robotiq_link import KeepaliveRobotiqGripper

SOURCE_URL = (
    "https://github.com/facebookresearch/fairo/blob/"
    "0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2/polymetis/polymetis/python/"
    "polymetis/robot_client/robotiq_gripper/third_party/"
    "robotiq_2finger_grippers/robotiq_2f_gripper.py"
)


def command_bytes(
    width: float, speed: float, force: float, stroke: float = 0.085
) -> tuple[int, int, int]:
    """Return the exact upstream position/speed/force bytes; force is percent."""
    if not np.isfinite([width, speed, force, stroke]).all() or stroke <= 0:
        raise ValueError(
            "Gripper width, speed, force and positive stroke must be finite."
        )
    position = int(np.clip((3.0 - 230.0) / stroke * width + 230.0, 0, 255))
    velocity = int(np.clip(255.0 / (0.1 - 0.013) * speed - 0.013, 0, 255))
    effort = int(np.clip(255.0 / 100.0 * force, 0, 255))
    return position, velocity, effort


def width_from_position(position: int, stroke: float = 0.085) -> float:
    """Apply the upstream get_pos feedback mapping with nominal endpoints 3/230."""
    return float(np.clip(stroke / (3.0 - 230.0) * (float(position) - 230.0), 0, stroke))


@dataclass(frozen=True)
class OfficialGripperState:
    """Width and flags derived from one live Modbus status sample."""

    width: float
    is_grasped: bool
    is_moving: bool
    raw_status: dict

    @property
    def motion_done(self) -> bool:
        """Use the upstream is_stopped condition, including object contact."""
        return self.raw_status["gOBJ"] != 0


class OfficialRobotiqGripper(KeepaliveRobotiqGripper):
    """Keep serial locking/polling while exposing the official goto/state API."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        slave_id: int = 9,
        max_width: float = 0.085,
        motion_timeout_s: float = 5.0,
    ) -> None:
        if not np.isfinite(motion_timeout_s) or motion_timeout_s <= 0:
            raise ValueError("motion_timeout_s must be positive and finite.")
        if not np.isfinite(max_width) or max_width <= 0:
            raise ValueError("max_width must be positive and finite.")
        super().__init__(
            port=port, baudrate=baudrate, slave_id=slave_id, max_width=max_width
        )
        self.motion_timeout_s = float(motion_timeout_s)

    def _activate(self) -> None:
        """Allow the known communication timeout to clear through read-only polls."""
        with self._io_lock:
            status = self._read_status()
            if status is not None and status["fault"] == 9 and status["gACT"] == 1:
                for _ in range(10):
                    time.sleep(0.1)
                    status = self._read_status()
                    if status is None or status["fault"] != 9 or status["gACT"] != 1:
                        break
            if self._ready(status):
                self._activated = True
                return
            if status is None or status["fault"]:
                raise RuntimeError(
                    f"Robotiq activation refused; no automatic fault recovery: {status}"
                )
            super()._activate()

    def get_state(self) -> OfficialGripperState:
        """Read width and object/motion flags from the same uncached status."""
        status = self._read_status()
        if not self._ready(status):
            raise RuntimeError(f"Robotiq is not ready: {status}")
        return OfficialGripperState(
            width=width_from_position(status["position"], self.max_width),
            is_grasped=status["gOBJ"] in (1, 2),
            is_moving=status["gGTO"] == 1 and status["gOBJ"] == 0,
            raw_status=status,
        )

    @property
    def position(self) -> float:
        """Return the upstream width estimate in metres."""
        return self.get_state().width

    def goto(
        self, width: float, speed: float, force: float, blocking: bool = False
    ) -> None:
        """Send upstream register values and optionally wait for this command to stop.

        Speed uses the published driver's velocity argument; force uses percent.
        The speed formula is preserved exactly, including its subtraction order.
        """
        if not self.is_connected:
            raise RuntimeError("Connect the Robotiq before sending a goto command.")
        position, velocity, effort = command_bytes(width, speed, force, self.max_width)
        # Same six Modbus bytes as upstream; transport health and locking stay owned by the parent.
        self._write_output_regs(0x0900, position, (velocity << 8) | effort)
        if not blocking:
            return
        deadline = time.monotonic() + self.motion_timeout_s
        while True:
            state = self.get_state()
            status = state.raw_status
            if (
                status["gGTO"] == 1
                and status["position_echo"] == position
                and state.motion_done
            ):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Robotiq goto did not complete within {self.motion_timeout_s}s: {status}"
                )
            time.sleep(min(0.02, remaining))

    def move(self, width: float, speed: float = 0.085) -> None:
        """Use official defaults when the existing width-only API is called."""
        self.goto(width, speed, 10.0)

    def open(self, speed: float = 0.085) -> None:
        """Request the official nominal fully-open width."""
        self.goto(self.max_width, speed, 10.0)
        self._is_open_flag = True

    def close(self, speed: float = 0.085, force: float = 10.0) -> None:
        """Request the official nominal closed width with force in percent."""
        self.goto(0.0, speed, force)
        self._is_open_flag = False

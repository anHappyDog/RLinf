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

"""Maintain the Robotiq Modbus link while policy inference is in progress."""

import threading
from typing import Any

from rlinf.robotics.parts.end_effectors.base import EndEffector
from rlinf.robotics.parts.end_effectors.grippers.robotiq import (
    _OUTPUT_REG_ADDR,
    RobotiqGripper,
)


@EndEffector.register("robotiq_keepalive")
class KeepaliveRobotiqGripper(RobotiqGripper):
    """Poll status at 10 Hz and latch communication or readiness failures."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._io_lock = threading.RLock()
        self._stop_polling = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._link_error: Exception | None = None
        self._monitor_ready = False

    @staticmethod
    def _ready(status: dict | None) -> bool:
        return status is not None and (
            status["gACT"] == 1 and status["gSTA"] == 3 and status["fault"] == 0
        )

    def _check_link(self) -> None:
        if self._link_error is not None:
            raise RuntimeError(
                f"Robotiq link failed on {self._port}"
            ) from self._link_error

    def _activate(self) -> None:
        with self._io_lock:
            status = super()._read_status()
            if self._ready(status):
                self._activated = True
                return
            super()._activate()
            status = super()._read_status()
            if not self._ready(status):
                raise RuntimeError(
                    f"Robotiq activation did not reach ready state: {status}"
                )

    def _open(self) -> Any:
        self._stop_polling.clear()
        self._link_error = None
        device = super()._open()
        self._monitor_ready = True
        try:
            self._read_status()
            self._poll_thread = threading.Thread(
                target=self._poll_status, name="robotiq-status", daemon=True
            )
            self._poll_thread.start()
        except Exception:
            self._monitor_ready = False
            super()._release(device)
            raise
        return device

    def _read_status(self) -> dict | None:
        with self._io_lock:
            self._check_link()
            try:
                status = super()._read_status()
                if self._monitor_ready and not self._ready(status):
                    raise RuntimeError(f"Robotiq lost readiness: {status}")
                return status
            except Exception as error:
                self._link_error = error
                self._activated = False
                self._stop_polling.set()
                raise

    def _write_output_regs(self, reg0: int, reg1: int, reg2: int) -> None:
        with self._io_lock:
            self._check_link()
            try:
                # The parent discards the response; check it before returning control.
                response = self._client.write_registers(
                    address=_OUTPUT_REG_ADDR,
                    values=[reg0, reg1, reg2],
                    **{self._slave_kwarg: self._slave_id},
                )
                if response.isError():
                    raise RuntimeError(f"Robotiq Modbus write failed: {response}")
            except Exception as error:
                self._link_error = error
                self._activated = False
                self._stop_polling.set()
                raise

    def _poll_status(self) -> None:
        while not self._stop_polling.wait(0.1):
            try:
                self._read_status()
            except Exception:
                # _read_status retains the error for the caller's next read or action.
                return

    def _release(self, device: Any) -> None:
        self._stop_polling.set()
        if self._poll_thread is not None:
            self._poll_thread.join()
            self._poll_thread = None
        with self._io_lock:
            self._monitor_ready = False
            super()._release(device)

# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Record camera caches independently of policy and hardware control calls."""

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("franka_deploy.video_recorder")


class CameraVideoRecorder:
    """Sample immutable camera cache snapshots without consuming capture queues."""

    def __init__(self, cameras: dict, output: Path, wrist_rotation: int = 0) -> None:
        import cv2

        self._stop = threading.Event()
        self._threads = []
        self._writers = {}
        self._errors = []
        output.mkdir(parents=True, exist_ok=True)
        self._output = output
        self._wrist_rotation = wrist_rotation
        try:
            for role, camera in cameras.items():
                info = camera.camera_info
                writer = cv2.VideoWriter(
                    str(output / f"{role}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(info.fps),
                    tuple(info.resolution),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open {role}.mp4 for recording.")
                self._writers[role] = writer
                self._threads.append(
                    threading.Thread(
                        target=self._record,
                        args=(role, camera, writer),
                        name=f"video-{role}",
                        daemon=True,
                    )
                )
        except BaseException:
            for writer in self._writers.values():
                writer.release()
            raise
        for thread in self._threads:
            thread.start()

    def _record(self, role, camera, writer) -> None:
        frames = 0
        unique = 0
        last_source = None
        period = 1.0 / camera.camera_info.fps
        deadline = time.monotonic()
        try:
            with (self._output / f"{role}_timestamps.jsonl").open("x") as trace:
                while not self._stop.is_set():
                    # Capture replaces this tuple; it never mutates the cached image.
                    # Reading it leaves the policy's frame queue and SDK untouched.
                    latest = camera._latest_frame
                    if latest is not None:
                        captured_at, bgr = latest
                        if role == "wrist" and self._wrist_rotation == 180:
                            bgr = np.ascontiguousarray(np.rot90(bgr, 2))
                        writer.write(bgr)
                        frames += 1
                        unique += captured_at != last_source
                        last_source = captured_at
                        trace.write(
                            json.dumps(
                                {
                                    "frame": frames,
                                    "written_monotonic_s": time.monotonic(),
                                    "captured_monotonic_s": captured_at,
                                }
                            )
                            + "\n"
                        )
                    deadline += period
                    self._stop.wait(max(0.0, deadline - time.monotonic()))
        except Exception as error:
            self._errors.append(error)
            self._stop.set()
            LOGGER.exception("%s video recording failed", role)
        finally:
            writer.release()
            LOGGER.info(
                "%s video saved: %d frames, %d distinct captured frames, %s",
                role,
                frames,
                unique,
                self._output / f"{role}.mp4",
            )

    def check(self) -> None:
        """Surface recording failures to the client's existing cleanup path."""
        if self._errors:
            raise RuntimeError("Camera video recording failed") from self._errors[0]

    def close(self) -> None:
        """Finalize both MP4s before the client disconnects cameras."""
        self._stop.set()
        for thread in self._threads:
            thread.join()

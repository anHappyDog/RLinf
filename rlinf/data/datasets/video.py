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

"""Process-local video decoding with bounded reader reuse."""

from collections import OrderedDict
from typing import Any

import numpy as np


class VideoFrameCache:
    """Bounded process-local PyAV readers with LeRobot frame selection."""

    def __init__(self, max_files: int = 4) -> None:
        self._readers: OrderedDict[str, Any] = OrderedDict()
        if max_files < 1:
            raise ValueError("Video reader cache must retain at least one file")
        self.max_files = max_files

    def read(self, path: str, timestamp: float, tolerance_s: float) -> np.ndarray:
        """Return the closest RGB frame using the existing PyAV seek policy."""
        import av

        path = str(path)
        if path not in self._readers:
            self._readers[path] = av.open(path, metadata_errors="ignore")
            self._readers[path].streams.video[0].codec_context.thread_count = 1
            if len(self._readers) > self.max_files:
                _, reader = self._readers.popitem(last=False)
                reader.close()
        self._readers.move_to_end(path)
        reader = self._readers[path]
        stream = reader.streams.video[0]
        offset = int(round(max(timestamp, 0.0) / stream.time_base))
        reader.seek(offset, backward=True, any_frame=False, stream=stream)
        target = np.float32(timestamp)
        selected = None
        distance = np.float32(np.inf)
        for frame in reader.decode(video=0):
            frame_time = float(frame.pts * frame.time_base)
            candidate = np.abs(target - np.float32(frame_time))
            # Float32 distances and strict updates preserve the first-nearest tie.
            if selected is None or candidate < distance:
                selected, distance = frame, candidate
            if frame_time >= timestamp:
                break
        if selected is None:
            raise ValueError(
                f"No video frames decoded after seeking to {timestamp}s in {path}"
            )
        if not distance < np.float32(tolerance_s):
            raise ValueError(
                f"Closest video frame is {float(distance)}s from {timestamp}s; "
                f"expected less than {tolerance_s}s in {path}"
            )
        return selected.to_rgb().to_ndarray()

    def close(self) -> None:
        """Close all cached video containers; the cache remains reusable."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self) -> dict[str, int]:
        return {"max_files": self.max_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(state["max_files"])

    def __del__(self) -> None:
        self.close()

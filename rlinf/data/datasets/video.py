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
        self.max_files = max_files
        self._readers: OrderedDict[str, Any] = OrderedDict()

    def read(self, path: str, timestamp: float, tolerance_s: float) -> np.ndarray:
        """Return the closest RGB frame using the existing PyAV seek policy."""
        import torch
        import torchvision

        path = str(path)
        if path not in self._readers:
            torchvision.set_video_backend("pyav")
            self._readers[path] = torchvision.io.VideoReader(path, "video")
            if len(self._readers) > self.max_files:
                _, reader = self._readers.popitem(last=False)
                reader.container.close()
        self._readers.move_to_end(path)
        reader = self._readers[path]
        reader.seek(timestamp, keyframes_only=True)
        frames, timestamps = [], []
        for frame in reader:
            frames.append(frame["data"])
            timestamps.append(frame["pts"])
            if frame["pts"] >= timestamp:
                break
        # Match LeRobot's float32 timestamp comparison and first-nearest tie rule.
        distances = torch.cdist(
            torch.tensor([timestamp], dtype=torch.float32)[:, None],
            torch.tensor(timestamps, dtype=torch.float32)[:, None],
            p=1,
        )[0]
        distance, index = distances.min(0)
        if not distance < tolerance_s:
            raise ValueError(
                f"Closest video frame is {float(distance)}s from {timestamp}s; "
                f"expected less than {tolerance_s}s in {path}"
            )
        return frames[index].permute(1, 2, 0).numpy()

    def close(self) -> None:
        """Close all cached video containers; the cache remains reusable."""
        for reader in self._readers.values():
            reader.container.close()
        self._readers.clear()

    def __getstate__(self) -> dict[str, int]:
        return {"max_files": self.max_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(state["max_files"])

    def __del__(self) -> None:
        self.close()

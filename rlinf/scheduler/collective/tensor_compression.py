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

"""Reusable CPU workspaces for collective tensor compression."""

from dataclasses import dataclass
from queue import Empty, LifoQueue
from typing import Literal, Optional

import torch

from rlinf.utils.tensor_codec import TensorCodec, create_tensor_codec


@dataclass(frozen=True)
class TensorCompressionOptions:
    """Configure optional CPU tensor compression for a collective send."""

    codec: Literal["lz4", "zstd"] = "lz4"
    level: int = 1
    min_bytes: int = 64 * 1024
    max_inflight: int = 1

    def __post_init__(self) -> None:
        """Validate compression settings."""
        if self.codec not in ("lz4", "zstd"):
            raise ValueError(f"Unsupported tensor codec: {self.codec!r}.")
        if self.level < 1:
            raise ValueError(f"Compression level must be >= 1, got {self.level}.")
        if self.min_bytes < 1:
            raise ValueError(
                f"Minimum compression size must be >= 1, got {self.min_bytes}."
            )
        if self.max_inflight < 1:
            raise ValueError(
                f"Maximum inflight compression tasks must be >= 1, got {self.max_inflight}."
            )


@dataclass(frozen=True)
class TensorCompressionWireMetadata:
    """Describe compressed CPU tensor payloads in one tensor-list transfer."""

    codec: Literal["lz4", "zstd"]
    level: int
    compressed_numel: tuple[Optional[int], ...]
    version: int = 1


class TensorWorkspaceSlot:
    """Own one codec and reusable uint8 buffers for one in-flight send."""

    def __init__(self, options: TensorCompressionOptions) -> None:
        """Create the slot's non-thread-safe codec and empty buffer cache."""
        self.codec: TensorCodec = create_tensor_codec(
            name=options.codec, level=options.level
        )
        self._buffers: dict[int, torch.Tensor] = {}

    def get_buffer(self, index: int, capacity: int) -> torch.Tensor:
        """Return a reusable CPU uint8 buffer with at least ``capacity`` bytes."""
        buffer = self._buffers.get(index)
        if buffer is None or buffer.numel() < capacity:
            buffer = torch.empty(capacity, dtype=torch.uint8, device="cpu")
            self._buffers[index] = buffer
        return buffer[:capacity]


class CompressionLease:
    """Keep one workspace slot exclusively owned until transfer completion."""

    def __init__(self, pool: "TensorWorkspacePool", slot: TensorWorkspaceSlot) -> None:
        """Bind the leased slot to its pool."""
        self._pool = pool
        self._slot: Optional[TensorWorkspaceSlot] = slot

    @property
    def slot(self) -> TensorWorkspaceSlot:
        """Return the owned workspace slot."""
        if self._slot is None:
            raise RuntimeError("CompressionLease has already been released.")
        return self._slot

    def release(self) -> None:
        """Return the slot to the reusable pool exactly once."""
        if self._slot is None:
            return
        slot = self._slot
        self._slot = None
        self._pool.release(slot)


class TensorWorkspacePool:
    """Bounded, non-blocking pool that prioritizes previously used slots."""

    def __init__(self, options: TensorCompressionOptions) -> None:
        """Create fresh slots without allocating their tensor buffers."""
        self._fresh_slots: LifoQueue[TensorWorkspaceSlot] = LifoQueue(
            maxsize=options.max_inflight
        )
        self._reused_slots: LifoQueue[TensorWorkspaceSlot] = LifoQueue(
            maxsize=options.max_inflight
        )
        for _ in range(options.max_inflight):
            self._fresh_slots.put_nowait(TensorWorkspaceSlot(options))

    def try_acquire(self) -> Optional[CompressionLease]:
        """Return a reusable slot without waiting, or ``None`` when saturated."""
        try:
            return CompressionLease(self, self._reused_slots.get_nowait())
        except Empty:
            try:
                return CompressionLease(self, self._fresh_slots.get_nowait())
            except Empty:
                return None

    def release(self, slot: TensorWorkspaceSlot) -> None:
        """Return a slot so it is preferred over never-used slots next time."""
        self._reused_slots.put_nowait(slot)

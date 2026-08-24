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

"""Bounded reusable CPU tensor buffers for collective payload processing."""

import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class TensorBufferPoolOptions:
    """Configure the Worker-wide collective tensor buffer pool."""

    max_bytes: int = 1024**3

    def __post_init__(self) -> None:
        """Validate the buffer budget."""
        if self.max_bytes < 1:
            raise ValueError(
                f"Maximum tensor buffer pool size must be >= 1, got {self.max_bytes}."
            )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TensorBufferPoolOptions":
        """Build validated options from the public YAML mapping."""
        unknown_keys = set(config) - {"max_bytes"}
        if unknown_keys:
            raise ValueError(
                "Unsupported collective tensor buffer pool options: "
                + ", ".join(sorted(unknown_keys))
            )
        return cls(**config)


class TensorBufferPool:
    """Share reusable CPU byte buffers within a fixed Worker memory budget."""

    def __init__(self, options: TensorBufferPoolOptions) -> None:
        """Create an empty buffer cache bounded by ``options.max_bytes``."""
        self.options = options
        self._allocated_bytes = 0
        self._available_buffers: list[torch.Tensor] = []
        self._lock = threading.Lock()

    @property
    def allocated_bytes(self) -> int:
        """Return bytes owned by active and cached buffers."""
        with self._lock:
            return self._allocated_bytes

    @property
    def cached_bytes(self) -> int:
        """Return bytes currently available for reuse."""
        with self._lock:
            return sum(buffer.numel() for buffer in self._available_buffers)

    def try_acquire(self, capacity: int) -> Optional["BufferLease"]:
        """Acquire a best-fit buffer without exceeding the memory budget."""
        if capacity > self.options.max_bytes:
            return None

        with self._lock:
            best_index = min(
                (
                    index
                    for index, buffer in enumerate(self._available_buffers)
                    if buffer.numel() >= capacity
                ),
                key=lambda index: self._available_buffers[index].numel(),
                default=None,
            )
            if best_index is not None and (
                self._available_buffers[best_index].numel() <= capacity * 2
                or self._allocated_bytes + capacity > self.options.max_bytes
            ):
                buffer = self._available_buffers.pop(best_index)
                return BufferLease(self, buffer)

            while (
                self._allocated_bytes + capacity > self.options.max_bytes
                and self._available_buffers
            ):
                evicted = self._available_buffers.pop(0)
                self._allocated_bytes -= evicted.numel()
                del evicted

            if self._allocated_bytes + capacity > self.options.max_bytes:
                return None

            buffer = torch.empty(capacity, dtype=torch.uint8, device="cpu")
            self._allocated_bytes += buffer.numel()
            return BufferLease(self, buffer)

    def release(self, buffer: torch.Tensor, *, cache: bool) -> None:
        """Return a buffer to the cache or remove it from the budget."""
        with self._lock:
            if cache:
                self._available_buffers.append(buffer)
            else:
                self._allocated_bytes -= buffer.numel()


class BufferLease:
    """Own one tensor buffer until no payload references it."""

    def __init__(self, pool: TensorBufferPool, tensor: torch.Tensor) -> None:
        """Bind the tensor buffer to its pool."""
        self._pool = pool
        self._tensor: Optional[torch.Tensor] = tensor

    @property
    def tensor(self) -> torch.Tensor:
        """Return the owned tensor buffer."""
        if self._tensor is None:
            raise RuntimeError("BufferLease has already been released.")
        return self._tensor

    def release(self, *, cache: bool = True) -> None:
        """Release the buffer exactly once."""
        if self._tensor is None:
            return
        tensor = self._tensor
        self._tensor = None
        self._pool.release(tensor, cache=cache)

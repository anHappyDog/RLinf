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

"""Bounded codec and workspace pools for collective tensor compression."""

from dataclasses import dataclass
from queue import Empty, LifoQueue
from typing import Any, Literal, Optional

import torch

from rlinf.utils.tensor_codec import TensorCodec, create_tensor_codec


@dataclass(frozen=True)
class TensorCompressionOptions:
    """Configure optional CPU tensor compression for a collective send."""

    enabled: bool = True
    codec: Literal["lz4", "zstd"] = "lz4"
    level: int = 1
    min_bytes: int = 64 * 1024
    max_inflight: int = 1

    def __post_init__(self) -> None:
        """Validate compression settings."""
        if not isinstance(self.enabled, bool):
            raise ValueError("Compression enabled must be a boolean.")
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

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TensorCompressionOptions":
        """Build validated options from the ``cluster.collective`` YAML mapping."""
        valid_keys = {"enabled", "codec", "level", "min_bytes", "max_inflight"}
        unknown_keys = set(config) - valid_keys
        if unknown_keys:
            raise ValueError(
                "Unsupported collective tensor compression options: "
                + ", ".join(sorted(unknown_keys))
            )
        return cls(**config)


@dataclass(frozen=True)
class TensorCompressionWireMetadata:
    """Describe compressed CPU tensor payloads in one tensor-list transfer."""

    codec: Literal["lz4", "zstd"]
    level: int
    compressed_numel: tuple[Optional[int], ...]
    version: int = 1


class TensorCompressionSlot:
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

    def __init__(self, pool: "TensorCodecPool", slot: TensorCompressionSlot) -> None:
        """Bind the leased slot to its pool."""
        self._pool = pool
        self._slot: Optional[TensorCompressionSlot] = slot

    @property
    def slot(self) -> TensorCompressionSlot:
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
        self._pool.release_compressor(slot)


class DecompressionLease:
    """Keep one decoder exclusively owned while restoring a received payload."""

    def __init__(self, pool: "TensorCodecPool", codec: TensorCodec) -> None:
        """Bind the borrowed decoder to its pool."""
        self._pool = pool
        self._codec: Optional[TensorCodec] = codec

    @property
    def codec(self) -> TensorCodec:
        """Return the exclusively owned decoder."""
        if self._codec is None:
            raise RuntimeError("DecompressionLease has already been released.")
        return self._codec

    def release(self) -> None:
        """Return the decoder to the pool exactly once."""
        if self._codec is None:
            return
        codec = self._codec
        self._codec = None
        self._pool.release_decompressor(codec)


class TensorCodecPool:
    """Own bounded compression workspaces and independent decoder codecs."""

    def __init__(self, options: TensorCompressionOptions) -> None:
        """Create one codec per compression and decompression slot."""
        self.options = options
        self._fresh_compressors: LifoQueue[TensorCompressionSlot] = LifoQueue(
            maxsize=options.max_inflight
        )
        self._reused_compressors: LifoQueue[TensorCompressionSlot] = LifoQueue(
            maxsize=options.max_inflight
        )
        self._decompressors: LifoQueue[TensorCodec] = LifoQueue(
            maxsize=options.max_inflight
        )
        for _ in range(options.max_inflight):
            self._fresh_compressors.put_nowait(TensorCompressionSlot(options))
            self._decompressors.put_nowait(
                create_tensor_codec(name=options.codec, level=options.level)
            )

    def try_acquire_compressor(self) -> Optional[CompressionLease]:
        """Return a preferred reusable compressor without waiting."""
        try:
            return CompressionLease(self, self._reused_compressors.get_nowait())
        except Empty:
            try:
                return CompressionLease(self, self._fresh_compressors.get_nowait())
            except Empty:
                return None

    def acquire_decompressor(
        self, metadata: TensorCompressionWireMetadata
    ) -> DecompressionLease:
        """Borrow a decoder after checking the wire codec matches this pool."""
        if (metadata.codec, metadata.level) != (
            self.options.codec,
            self.options.level,
        ):
            raise ValueError(
                "Compression metadata does not match this collective's codec settings."
            )
        return DecompressionLease(self, self._decompressors.get())

    def release_compressor(self, slot: TensorCompressionSlot) -> None:
        """Return a compressor so it is preferred over unused slots next time."""
        self._reused_compressors.put_nowait(slot)

    def release_decompressor(self, codec: TensorCodec) -> None:
        """Return a decoder after the received payload has been restored."""
        self._decompressors.put_nowait(codec)

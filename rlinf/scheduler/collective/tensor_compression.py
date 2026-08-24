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

"""Bounded codec pools for collective tensor compression."""

from dataclasses import dataclass
from queue import Empty, LifoQueue
from typing import Any, Literal, Optional

from rlinf.utils.tensor_codec import TensorCodec, create_tensor_codec

from .tensor_buffer_pool import BufferLease


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
        valid_keys = {
            "enabled",
            "codec",
            "level",
            "min_bytes",
            "max_inflight",
        }
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


class CompressionLease:
    """Own one encoder and the buffers backing its wire payloads."""

    def __init__(self, pool: "TensorCodecPool", codec: TensorCodec) -> None:
        """Bind the encoder and future payload buffers to their pools."""
        self._pool = pool
        self._codec: Optional[TensorCodec] = codec
        self._buffers: list[BufferLease] = []

    @property
    def codec(self) -> TensorCodec:
        """Return the exclusively owned encoder."""
        if self._codec is None:
            raise RuntimeError("CompressionLease has already been released.")
        return self._codec

    def retain_buffer(self, buffer: BufferLease) -> None:
        """Keep a tensor buffer alive until the compressed transfer completes."""
        if self._codec is None:
            raise RuntimeError("CompressionLease has already been released.")
        self._buffers.append(buffer)

    def release(self) -> None:
        """Return the encoder and retained buffers exactly once."""
        if self._codec is None:
            return
        codec = self._codec
        self._codec = None
        for buffer in self._buffers:
            buffer.release()
        self._buffers.clear()
        self._pool.release_compressor(codec)


class DecompressionLease:
    """Keep one decompressor exclusively owned while restoring a payload."""

    def __init__(self, pool: "TensorCodecPool", codec: TensorCodec) -> None:
        """Bind the borrowed decompressor to its pool."""
        self._pool = pool
        self._codec: Optional[TensorCodec] = codec

    @property
    def codec(self) -> TensorCodec:
        """Return the exclusively owned decompressor."""
        if self._codec is None:
            raise RuntimeError("DecompressionLease has already been released.")
        return self._codec

    def release(self) -> None:
        """Return the decompressor to the pool exactly once."""
        if self._codec is None:
            return
        codec = self._codec
        self._codec = None
        self._pool.release_decompressor(codec)


class TensorCodecPool:
    """Own the Worker-wide compression and decompression codecs."""

    def __init__(self, options: TensorCompressionOptions) -> None:
        """Create one codec per compression and decompression slot."""
        self.options = options
        self._fresh_compressors: LifoQueue[TensorCodec] = LifoQueue(
            maxsize=options.max_inflight
        )
        self._reused_compressors: LifoQueue[TensorCodec] = LifoQueue(
            maxsize=options.max_inflight
        )
        self._decompressors: LifoQueue[TensorCodec] = LifoQueue(
            maxsize=options.max_inflight
        )
        for _ in range(options.max_inflight):
            self._fresh_compressors.put_nowait(
                create_tensor_codec(name=options.codec, level=options.level)
            )
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
        """Borrow a decompressor after checking the wire codec matches this pool."""
        if (metadata.codec, metadata.level) != (
            self.options.codec,
            self.options.level,
        ):
            raise ValueError(
                "Compression metadata does not match this collective's codec settings."
            )
        return DecompressionLease(self, self._decompressors.get())

    def release_compressor(self, codec: TensorCodec) -> None:
        """Return a compressor so it is preferred over unused slots next time."""
        self._reused_compressors.put_nowait(codec)

    def release_decompressor(self, codec: TensorCodec) -> None:
        """Return a decompressor after restoring the received payload."""
        self._decompressors.put_nowait(codec)

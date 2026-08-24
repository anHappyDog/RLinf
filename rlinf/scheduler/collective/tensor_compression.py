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

"""Codec-specific acquisition policies for collective tensor compression."""

from dataclasses import dataclass, field
from queue import Empty, LifoQueue
from typing import Any, Literal, Optional

import torch

from rlinf.utils.tensor_codec import TensorCodec, create_tensor_codec


@dataclass(frozen=True)
class LZ4CodecProviderOptions:
    """Configure the stateless Worker-wide LZ4 provider."""

    acceleration: int = 1

    def __post_init__(self) -> None:
        """Validate LZ4 provider parameters."""
        if self.acceleration < 1:
            raise ValueError(f"LZ4 acceleration must be >= 1, got {self.acceleration}.")

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "LZ4CodecProviderOptions":
        """Build LZ4 options from its provider-specific parameters."""
        unknown_keys = set(params) - {"acceleration"}
        if unknown_keys:
            raise ValueError(
                "Unsupported LZ4 provider parameters: "
                + ", ".join(sorted(unknown_keys))
            )
        return cls(**params)

    def to_dict(self) -> dict[str, int]:
        """Serialize LZ4 provider parameters."""
        return {"acceleration": self.acceleration}


@dataclass(frozen=True)
class ZstdCodecProviderOptions:
    """Configure bounded Worker-wide Zstd context pools."""

    level: int = 1
    max_inflight: int = 4

    def __post_init__(self) -> None:
        """Validate Zstd provider parameters."""
        if self.level < 1:
            raise ValueError(f"Zstd level must be >= 1, got {self.level}.")
        if self.max_inflight < 1:
            raise ValueError(
                f"Zstd maximum inflight contexts must be >= 1, got {self.max_inflight}."
            )

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ZstdCodecProviderOptions":
        """Build Zstd options from its provider-specific parameters."""
        unknown_keys = set(params) - {"level", "max_inflight"}
        if unknown_keys:
            raise ValueError(
                "Unsupported Zstd provider parameters: "
                + ", ".join(sorted(unknown_keys))
            )
        return cls(**params)

    def to_dict(self) -> dict[str, int]:
        """Serialize Zstd provider parameters."""
        return {"level": self.level, "max_inflight": self.max_inflight}


CodecProviderOptions = LZ4CodecProviderOptions | ZstdCodecProviderOptions


@dataclass(frozen=True)
class TensorCompressionOptions:
    """Configure optional CPU tensor compression for a collective send."""

    enabled: bool = True
    min_bytes: int = 16 * 1024
    excluded_dtypes: tuple[str, ...] = ("float32",)
    provider: CodecProviderOptions = field(default_factory=LZ4CodecProviderOptions)
    _excluded_dtype_values: frozenset[torch.dtype] = field(
        init=False, repr=False, compare=False
    )

    @property
    def codec(self) -> Literal["lz4", "zstd"]:
        """Return the selected provider name."""
        return "lz4" if isinstance(self.provider, LZ4CodecProviderOptions) else "zstd"

    def __post_init__(self) -> None:
        """Validate compression settings."""
        if not isinstance(self.enabled, bool):
            raise ValueError("Compression enabled must be a boolean.")
        if self.min_bytes < 1:
            raise ValueError(
                f"Minimum compression size must be >= 1, got {self.min_bytes}."
            )
        if not isinstance(self.excluded_dtypes, (list, tuple)):
            raise ValueError("Excluded tensor dtypes must be a list.")

        dtype_names = tuple(self.excluded_dtypes)
        if len(set(dtype_names)) != len(dtype_names):
            raise ValueError("Excluded tensor dtypes must not contain duplicates.")
        dtype_values = []
        for name in dtype_names:
            dtype = getattr(torch, name, None) if isinstance(name, str) else None
            if not isinstance(dtype, torch.dtype) or str(dtype) != f"torch.{name}":
                raise ValueError(f"Unsupported excluded tensor dtype: {name!r}.")
            dtype_values.append(dtype)
        object.__setattr__(self, "excluded_dtypes", dtype_names)
        object.__setattr__(self, "_excluded_dtype_values", frozenset(dtype_values))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TensorCompressionOptions":
        """Build validated options from the ``cluster.collective`` YAML mapping."""
        valid_keys = {
            "enabled",
            "codec",
            "min_bytes",
            "excluded_dtypes",
            "params",
        }
        unknown_keys = set(config) - valid_keys
        if unknown_keys:
            raise ValueError(
                "Unsupported collective tensor compression options: "
                + ", ".join(sorted(unknown_keys))
            )
        codec = config.get("codec", "lz4")
        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Tensor codec provider params must be a mapping.")
        if codec == "lz4":
            provider = LZ4CodecProviderOptions.from_dict(params)
        elif codec == "zstd":
            provider = ZstdCodecProviderOptions.from_dict(params)
        else:
            raise ValueError(f"Unsupported tensor codec: {codec!r}.")
        return cls(
            enabled=config.get("enabled", True),
            min_bytes=config.get("min_bytes", 16 * 1024),
            excluded_dtypes=config.get("excluded_dtypes", ["float32"]),
            provider=provider,
        )

    def should_compress(self, tensor: torch.Tensor) -> bool:
        """Return whether a CPU tensor is eligible for a codec attempt."""
        return (
            tensor.is_cpu
            and tensor.dtype not in self._excluded_dtype_values
            and tensor.numel() * tensor.element_size() >= self.min_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validated configuration for Worker propagation."""
        return {
            "enabled": self.enabled,
            "codec": self.codec,
            "min_bytes": self.min_bytes,
            "excluded_dtypes": list(self.excluded_dtypes),
            "params": self.provider.to_dict(),
        }


@dataclass(frozen=True)
class TensorCompressionWireMetadata:
    """Describe compressed CPU tensor payloads in one tensor-list transfer."""

    codec: Literal["lz4", "zstd"]
    compressed_numel: tuple[Optional[int], ...]


class _SharedCodecLease:
    """Expose a thread-safe codec through the common lease interface."""

    def __init__(self, codec: TensorCodec) -> None:
        self.codec = codec

    def release(self) -> None:
        """Keep the shared codec available to all callers."""


class _PooledCodecLease:
    """Keep one stateful codec context exclusively owned during a call."""

    def __init__(
        self, codec: TensorCodec, release_queue: LifoQueue[TensorCodec]
    ) -> None:
        self._release_queue = release_queue
        self._codec: Optional[TensorCodec] = codec

    @property
    def codec(self) -> TensorCodec:
        """Return the exclusively owned codec context."""
        if self._codec is None:
            raise RuntimeError("Codec lease has already been released.")
        return self._codec

    def release(self) -> None:
        """Return the codec context to its pool exactly once."""
        if self._codec is None:
            return
        codec = self._codec
        self._codec = None
        self._release_queue.put_nowait(codec)


CodecLease = _SharedCodecLease | _PooledCodecLease


class _LZ4CodecAcquisition:
    """Share one stateless LZ4 codec without acquisition locks or limits."""

    def __init__(self, options: LZ4CodecProviderOptions) -> None:
        self._lease = _SharedCodecLease(
            create_tensor_codec(name="lz4", level=options.acceleration)
        )

    def try_acquire_compressor(self) -> _SharedCodecLease:
        return self._lease

    def acquire_decompressor(self) -> _SharedCodecLease:
        return self._lease


class _ZstdCodecAcquisition:
    """Bound concurrent access to reusable Zstd compression contexts."""

    def __init__(self, options: ZstdCodecProviderOptions) -> None:
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
                create_tensor_codec(name="zstd", level=options.level)
            )
            self._decompressors.put_nowait(
                create_tensor_codec(name="zstd", level=options.level)
            )

    def try_acquire_compressor(self) -> Optional[_PooledCodecLease]:
        try:
            codec = self._reused_compressors.get_nowait()
        except Empty:
            try:
                codec = self._fresh_compressors.get_nowait()
            except Empty:
                return None
        return _PooledCodecLease(codec, self._reused_compressors)

    def acquire_decompressor(self) -> _PooledCodecLease:
        return _PooledCodecLease(self._decompressors.get(), self._decompressors)


class TensorCodecPool:
    """Own the Worker-wide codec-specific acquisition policy."""

    def __init__(self, options: TensorCompressionOptions) -> None:
        """Create shared LZ4 access or bounded Zstd context pools."""
        self.options = options
        provider = options.provider
        self._acquisition = (
            _LZ4CodecAcquisition(provider)
            if isinstance(provider, LZ4CodecProviderOptions)
            else _ZstdCodecAcquisition(provider)
        )

    def try_acquire_compressor(self) -> Optional[CodecLease]:
        """Return a codec according to its compression acquisition policy."""
        return self._acquisition.try_acquire_compressor()

    def acquire_decompressor(
        self, metadata: TensorCompressionWireMetadata
    ) -> CodecLease:
        """Return a codec according to its decompression acquisition policy."""
        if metadata.codec != self.options.codec:
            raise ValueError(
                "Compression metadata does not match this collective's codec settings."
            )
        return self._acquisition.acquire_decompressor()

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

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import LifoQueue
from typing import Literal, TypeAlias

import torch

from rlinf.data.embodied_io_struct import (
    LeRobotStepResult,
    PolicyInput,
    RewardRequest,
    TensorPath,
    TrajectoryData,
    ValueRequest,
)
from rlinf.scheduler.channel.trajectory_channel.storage import (
    LeRobotEpisodeBatch,
    TrajectoryBatch,
)
from rlinf.utils.tensor_codec import TensorCodec, create_tensor_codec

CodecType: TypeAlias = Literal["lz4", "zstd"]
# (raw_bytes, wire_byted, compressed)
BlockMetaData: TypeAlias = tuple[int, int, bool]


@dataclass(frozen=True)
class TensorCompressionMetadata:
    """Describe the original tensor required to restore compressed blocks."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    block_metadata: tuple[BlockMetaData, ...]


CompressionMetadata: TypeAlias = dict[TensorPath, TensorCompressionMetadata]


@dataclass(frozen=True)
class CompressionConfig:
    """Configure tensor compression and its bounded workspace pool."""

    codec: CodecType = "lz4"
    level: int = 1
    min_bytes: int = 1024 * 64
    block_bytes: int = 1024 * 1024
    num_threads: int = 1
    max_inflight: int = 5
    pin_memory: bool = True

    def __post_init__(self):
        """Validate compression limits at construction time."""
        if self.level < 1:
            raise ValueError(f"Compression level must be >= 1, got {self.level}")
        if self.min_bytes < 1:
            raise ValueError(
                f"Minimum bytes for compression must be >= 1, got {self.min_bytes}"
            )
        if self.block_bytes < 1:
            raise ValueError(
                f"Block size for compression must be >= 1, got {self.block_bytes}"
            )
        if self.num_threads < 1:
            raise ValueError(
                f"Number of threads for compression must be >= 1, got {self.num_threads}"
            )
        if self.max_inflight < 1:
            raise ValueError(
                f"Maximum inflight compression tasks must be >= 1, got {self.max_inflight}"
            )


CompressionConfigDict: TypeAlias = dict[type[TrajectoryData], CompressionConfig]

_COMPRESSION_CONFIGS: CompressionConfigDict = {
    LeRobotStepResult: CompressionConfig(
        codec="lz4",
        level=1,
        min_bytes=1024 * 64,
        block_bytes=1024 * 1024,
        num_threads=4,
    ),
    LeRobotEpisodeBatch: CompressionConfig(
        codec="lz4",
        level=1,
        min_bytes=1024 * 64,
        block_bytes=1024 * 1024,
        num_threads=4,
    ),
    TrajectoryBatch: CompressionConfig(
        codec="lz4",
        level=1,
        min_bytes=1024 * 64,
        block_bytes=1024 * 1024,
        num_threads=4,
    ),
    PolicyInput: CompressionConfig(
        codec="lz4",
        level=1,
        min_bytes=1024 * 64,
        block_bytes=1024 * 1024,
        num_threads=4,
    ),
    ValueRequest: CompressionConfig(
        codec="lz4",
        level=1,
        min_bytes=1024 * 64,
        block_bytes=1024 * 1024,
        num_threads=4,
    ),
    RewardRequest: CompressionConfig(
        codec="lz4",
        level=1,
        min_bytes=1024 * 64,
        block_bytes=1024 * 1024,
        num_threads=4,
    ),
}


def get_compression_configs() -> CompressionConfigDict:
    """Return the default per-record compression settings."""
    return _COMPRESSION_CONFIGS


@dataclass
class _CompressionSlot:
    codecs: tuple[TensorCodec, ...]
    workspaces: tuple[dict[TensorPath, torch.Tensor], ...]


class CompressionLease:
    """Own one compression workspace slot until its transfer completes."""

    def __init__(self, pool: "_CompressionPool", slot: "_CompressionSlot") -> None:
        """Bind a workspace slot to its owning pool."""
        self._pool = pool
        self._slot: "_CompressionSlot" | None = slot
        self._lock = threading.Lock()

    @property
    def slot(self) -> _CompressionSlot:
        """The leased workspace slot."""
        with self._lock:
            if self._slot is None:
                raise RuntimeError("CompressionLease has already been released.")
            return self._slot

    def release(self) -> None:
        """Return the workspace slot to its pool once."""
        with self._lock:
            slot = self._slot
            if slot is None:
                return
            self._slot = None
        self._pool._release(slot)


@dataclass(frozen=True)
class CompressionStats:
    """Summarize the compressed and raw payload sizes."""

    raw_bytes: int
    wire_bytes: int
    compressed_blocks: int
    raw_blocks: int
    workspace_allocations: int
    workspace_bytes: int


@dataclass(frozen=True)
class CompressionOutput:
    """Carry compressed tensors, restoration metadata, and an optional lease."""

    tensors: dict[TensorPath, torch.Tensor]
    metadata: CompressionMetadata
    stats: CompressionStats
    lease: CompressionLease | None


class _CompressionPool:
    def __init__(self, config: CompressionConfig):
        self._config = config
        self._available_slots: LifoQueue[_CompressionSlot] = LifoQueue(
            maxsize=config.max_inflight
        )
        for _ in range(config.max_inflight):
            codecs = tuple(
                create_tensor_codec(name=config.codec, level=config.level)
                for _ in range(config.num_threads)
            )
            workspaces = tuple({} for _ in range(config.num_threads))
            self._available_slots.put(_CompressionSlot(codecs, workspaces))

    def acquire(self) -> CompressionLease:
        slot = self._available_slots.get()
        return CompressionLease(pool=self, slot=slot)

    async def acquire_async(self) -> CompressionLease:
        slot = await asyncio.to_thread(self._available_slots.get)
        return CompressionLease(pool=self, slot=slot)

    def _release(self, slot: _CompressionSlot) -> None:
        self._available_slots.put_nowait(slot)


@dataclass(frozen=True)
class _CompressedTensor:
    wire_key: TensorPath
    wire: torch.Tensor
    metadata: TensorCompressionMetadata | None
    raw_bytes: int
    compressed_blocks: int
    raw_blocks: int
    workspace_allocations: int
    workspace_bytes: int


class TensorCompressor:
    """Compress and restore record tensor payloads with reusable workspaces."""

    def __init__(self, config: CompressionConfig):
        """Initialize the codec pool from a compression configuration."""
        self._config = config
        self._pool = _CompressionPool(config)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=config.num_threads,
                thread_name_prefix=f"TrajectoryChannel-Compression-{config.codec}",
            )
            if config.num_threads > 1
            else None
        )

    def compress(self, tensors: dict[TensorPath, torch.Tensor]) -> CompressionOutput:
        """Compress tensors synchronously and retain a lease when needed."""
        buffer_lease = self._pool.acquire()
        try:
            wire, metadata, stats = self._compress_with_slot(buffer_lease.slot, tensors)
        except Exception:
            buffer_lease.release()
            raise

        if not metadata:
            buffer_lease.release()
            buffer_lease = None
        return CompressionOutput(
            tensors=wire, metadata=metadata, stats=stats, lease=buffer_lease
        )

    async def compress_async(
        self, tensors: dict[TensorPath, torch.Tensor]
    ) -> CompressionOutput:
        """Compress tensors without blocking the worker event loop."""
        lease = await self._pool.acquire_async()
        task = asyncio.create_task(
            asyncio.to_thread(self._compress_with_slot, lease.slot, tensors)
        )
        try:
            tensors, metadata, stats = await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            finally:
                lease.release()
            raise
        except BaseException:
            lease.release()
            raise
        if not metadata:
            lease.release()
            lease = None
        return CompressionOutput(
            tensors=tensors, metadata=metadata, stats=stats, lease=lease
        )

    def decompress(
        self, tensors: dict[TensorPath, torch.Tensor], metadata: CompressionMetadata
    ) -> dict[TensorPath, torch.Tensor]:
        """Restore tensors synchronously from compressed blocks."""
        lease = self._pool.acquire()
        try:
            tensors = self._decompress_with_slot(lease.slot, tensors, metadata)
        finally:
            lease.release()
            lease = None
        return tensors

    async def decompress_async(
        self, tensors: dict[TensorPath, torch.Tensor], metadata: CompressionMetadata
    ) -> dict[TensorPath, torch.Tensor]:
        """Restore tensors without blocking the worker event loop."""
        lease = await self._pool.acquire_async()
        try:
            tensors = await asyncio.to_thread(
                self._decompress_with_slot, lease.slot, tensors, metadata
            )
        finally:
            lease.release()
            lease = None
        return tensors

    def _decompress_with_slot(
        self,
        slot: _CompressionSlot,
        tensors: dict[TensorPath, torch.Tensor],
        metadata: CompressionMetadata,
    ) -> dict[TensorPath, torch.Tensor]:
        codecs = slot.codecs
        items = tuple(metadata.items())
        lanes = tuple(items[i :: len(codecs)] for i in range(len(codecs)))

        if self._executor is None:
            lane_results = (_decompress_lane(lanes[0], tensors, codecs[0]),)
        else:
            lane_results = self._executor.map(
                _decompress_lane, lanes, [tensors] * len(lanes), codecs
            )
        return _merge_decompressed(tensors, lane_results)

    def _compress_with_slot(
        self, slot: _CompressionSlot, tensors: dict[TensorPath, torch.Tensor]
    ) -> tuple[dict[TensorPath, torch.Tensor], CompressionMetadata, CompressionStats]:
        codecs = slot.codecs
        workspaces = slot.workspaces
        items = tuple(tensors.items())
        lanes = tuple(items[i :: len(codecs)] for i in range(len(codecs)))

        if self._executor is None:
            lane_results = (
                _compress_lane(lanes[0], self._config, workspaces[0], codecs[0]),
            )
        else:
            lane_results = self._executor.map(
                _compress_lane, lanes, [self._config] * len(lanes), workspaces, codecs
            )

        results = {
            key: result for lane_result in lane_results for key, result in lane_result
        }
        return _merge_compressed(items, results)


def _merge_compressed(
    items: tuple[tuple[TensorPath, torch.Tensor], ...],
    results: dict[TensorPath, _CompressedTensor],
) -> tuple[dict[TensorPath, torch.Tensor], CompressionMetadata, CompressionStats]:
    wire: dict[TensorPath, torch.Tensor] = {}
    metadata: CompressionMetadata = {}
    raw_bytes: int = 0
    wire_bytes: int = 0
    compressed_blocks: int = 0
    raw_blocks: int = 0
    workspace_allocations: int = 0
    workspace_bytes: int = 0
    for key, _ in items:
        result = results[key]
        wire[result.wire_key] = result.wire
        if result.metadata is not None:
            metadata[key] = result.metadata
        raw_bytes += result.raw_bytes
        wire_bytes += result.wire.numel() * result.wire.element_size()
        compressed_blocks += result.compressed_blocks
        raw_blocks += result.raw_blocks
        workspace_allocations += result.workspace_allocations
        workspace_bytes += result.workspace_bytes
    if wire_bytes > raw_bytes:
        raise RuntimeError("raw fallback failed to bound compressed trajectory bytes")
    stats: CompressionStats = CompressionStats(
        raw_bytes=raw_bytes,
        wire_bytes=wire_bytes,
        compressed_blocks=compressed_blocks,
        raw_blocks=raw_blocks,
        workspace_allocations=workspace_allocations,
        workspace_bytes=workspace_bytes,
    )
    return wire, metadata, stats


def _merge_decompressed(
    wire: dict[TensorPath, torch.Tensor],
    lane_results: tuple[list[tuple[TensorPath, torch.Tensor]], ...],
) -> dict[TensorPath, torch.Tensor]:
    tensors = dict(wire)

    for lane_result in lane_results:
        for key, tensor in lane_result:
            tensors[key] = tensor

    return tensors


def _compress_lane(
    lane: tuple[tuple[TensorPath, torch.Tensor], ...],
    config: CompressionConfig,
    workspace: dict[TensorPath, torch.Tensor],
    codec: TensorCodec,
) -> list[tuple[TensorPath, _CompressedTensor]]:
    return [
        (key, _compress_tensor(key, tensor, config, workspace, codec))
        for key, tensor in lane
    ]


def _decompress_lane(
    metadata_items: tuple[tuple[TensorPath, TensorCompressionMetadata], ...],
    wire: dict[TensorPath, torch.Tensor],
    codec: TensorCodec,
) -> list[tuple[TensorPath, torch.Tensor]]:
    return [
        (key, _decompress_tensor(key, metadata, wire, codec))
        for key, metadata in metadata_items
    ]


def _compress_tensor(
    key: TensorPath,
    tensor: torch.Tensor,
    config: CompressionConfig,
    workspace: dict[TensorPath, torch.Tensor],
    codec: TensorCodec,
) -> _CompressedTensor:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError(
            f"Tensor {key} must be a contiguous CPU tensor for compression, but got {tensor.device.type} and {tensor.is_contiguous()}."
        )

    tensor_bytes = tensor.numel() * tensor.element_size()
    if tensor_bytes < config.min_bytes:
        return _CompressedTensor(
            wire_key=key,
            wire=tensor,
            metadata=None,
            raw_bytes=tensor_bytes,
            compressed_blocks=0,
            raw_blocks=1,
            workspace_allocations=0,
            workspace_bytes=0,
        )

    source = tensor.view(torch.uint8).reshape(-1)
    block_sizes = tuple(
        min(config.block_bytes, tensor_bytes - offset)
        for offset in range(0, tensor_bytes, config.block_bytes)
    )
    capacity = sum(codec.compress_bound(size) for size in block_sizes)
    workspace_allocations = 0
    workspace_bytes = 0
    if key not in workspace or workspace[key].numel() < capacity:
        workspace[key] = torch.empty(
            capacity, dtype=torch.uint8, device="cpu", pin_memory=config.pin_memory
        )
        workspace_allocations = 1
        workspace_bytes = capacity
    destination = workspace[key][:capacity]
    blocks: list[tuple[int, int, bool]] = []
    any_compressed = False
    compressed_blocks = 0
    raw_blocks = 0
    raw_offset = 0
    wire_offset = 0
    for block_bytes in block_sizes:
        block = source[raw_offset : raw_offset + block_bytes]
        bound = codec.compress_bound(block_bytes)
        encoded_bytes = codec.compress_into(
            block, workspace[key][wire_offset : wire_offset + bound]
        )
        compressed = encoded_bytes < block_bytes
        if compressed:
            any_compressed = True
            compressed_blocks += 1
        else:
            encoded_bytes = block_bytes
            destination[wire_offset : wire_offset + block_bytes].copy_(block)
            raw_blocks += 1
        blocks.append((block_bytes, encoded_bytes, compressed))
        raw_offset += block_bytes
        wire_offset += encoded_bytes
    if not any_compressed:
        return _CompressedTensor(
            wire_key=key,
            wire=tensor,
            metadata=None,
            raw_bytes=tensor_bytes,
            compressed_blocks=0,
            raw_blocks=1,
            workspace_allocations=workspace_allocations,
            workspace_bytes=workspace_bytes,
        )
    else:
        metadata = TensorCompressionMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            block_metadata=tuple(blocks),
        )
        return _CompressedTensor(
            wire_key=key,
            wire=destination[:wire_offset],
            metadata=metadata,
            raw_bytes=tensor_bytes,
            compressed_blocks=compressed_blocks,
            raw_blocks=raw_blocks,
            workspace_allocations=workspace_allocations,
            workspace_bytes=workspace_bytes,
        )


def _decompress_tensor(
    key: TensorPath,
    metadata: TensorCompressionMetadata,
    wire: dict[TensorPath, torch.Tensor],
    codec: TensorCodec,
) -> torch.Tensor:
    destination = torch.empty(metadata.shape, dtype=metadata.dtype, device="cpu")
    dst_uint8_view = destination.view(torch.uint8).reshape(-1)
    try:
        packed = wire[key]
    except KeyError as error:
        raise ValueError(
            f"missing trajectory blocks {key!r} in wire data for decompression"
        ) from error
    raw_offset = 0
    wire_offset = 0

    for raw_bytes, wire_bytes, compressed in metadata.block_metadata:
        source = packed[wire_offset : wire_offset + wire_bytes]
        target = dst_uint8_view[raw_offset : raw_offset + raw_bytes]
        if compressed:
            codec.decompress_into(source, wire_bytes, target)
        else:
            if source.numel() != raw_bytes:
                raise ValueError(f"raw trajectory block {key!r}")
            target.copy_(source)
        raw_offset += raw_bytes
        wire_offset += wire_bytes

    if raw_offset != dst_uint8_view.numel() or wire_offset != packed.numel():
        raise ValueError(
            f"trajectory block {key!r} size mismatch, expected {dst_uint8_view.numel()} raw bytes and {packed.numel()} wire bytes, but got {raw_offset} raw bytes and {wire_offset} wire bytes"
        )
    return destination

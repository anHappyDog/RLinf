# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block-wise lossless compression for Actor-bound trajectory tensors."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from rlinf.workers.trajectory.tensor_codec import TensorCodec, create_tensor_codec


@dataclass(frozen=True)
class CompressionConfig:
    """Static Actor-transfer compression policy."""

    enabled: bool = False
    codec: str = "lz4"
    level: int = 1
    min_bytes: int = 64 * 1024
    block_bytes: int = 1024 * 1024
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.codec not in ("lz4", "zstd"):
            raise ValueError("compression codec must be 'lz4' or 'zstd'")
        if self.level < 1:
            raise ValueError("compression level must be positive")
        if self.min_bytes < 1:
            raise ValueError("compression min_bytes must be positive")
        if self.block_bytes < 1:
            raise ValueError("compression block_bytes must be positive")
        if self.num_threads < 1:
            raise ValueError("compression num_threads must be positive")


class CompressionPipeline:
    """Own reusable buffers and independent codec contexts for one endpoint.

    Compressed views remain valid until the next :meth:`compress` call. This
    matches StorageWorker's synchronous send: the collective finishes before
    the workspace can be reused.
    """

    def __init__(self, config: CompressionConfig) -> None:
        self._config = config
        self._codecs = tuple(
            create_tensor_codec(config.codec, level=config.level)
            for _ in range(config.num_threads)
        )
        self._workspaces: tuple[dict[str, torch.Tensor], ...] = tuple(
            {} for _ in self._codecs
        )
        self._executor = (
            ThreadPoolExecutor(
                max_workers=config.num_threads,
                thread_name_prefix="trajectory-codec",
            )
            if config.num_threads > 1
            else None
        )

    def compress(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, int]]:
        """Compress tensors directly into reusable packed wire buffers."""
        items = tuple(tensors.items())
        lanes = tuple(
            items[index :: len(self._codecs)] for index in range(len(self._codecs))
        )

        def run_lane(lane: int) -> list[tuple[str, _CompressedTensor]]:
            return [
                (
                    key,
                    _compress_tensor(
                        key,
                        tensor,
                        self._config,
                        self._codecs[lane],
                        self._workspaces[lane],
                    ),
                )
                for key, tensor in lanes[lane]
            ]

        if self._executor is None:
            lane_results = (run_lane(0),)
        else:
            lane_results = tuple(self._executor.map(run_lane, range(len(lanes))))
        results = {key: result for lane in lane_results for key, result in lane}
        return _merge_compressed(items, results)

    def decompress(
        self,
        wire: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Decompress tensors in parallel into their final owned outputs."""
        metadata_items = tuple(metadata.items())
        lanes = tuple(
            metadata_items[index :: len(self._codecs)]
            for index in range(len(self._codecs))
        )

        def run_lane(lane: int) -> list[tuple[str, torch.Tensor]]:
            return [
                (
                    key,
                    _decompress_tensor(key, tensor_metadata, wire, self._codecs[lane]),
                )
                for key, tensor_metadata in lanes[lane]
            ]

        if self._executor is None:
            lane_results = (run_lane(0),)
        else:
            lane_results = tuple(self._executor.map(run_lane, range(len(lanes))))
        return _merge_decompressed(wire, metadata, lane_results)

    def close(self) -> None:
        """Release codec threads after outstanding work has completed."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None


@dataclass(frozen=True)
class _CompressedTensor:
    wire_key: str
    wire: torch.Tensor
    metadata: dict[str, Any] | None
    raw_bytes: int
    compressed_blocks: int
    raw_blocks: int
    workspace_allocations: int
    workspace_bytes: int


def compress_tensors(
    tensors: dict[str, torch.Tensor],
    config: CompressionConfig,
    codec: TensorCodec,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, int]]:
    """Compress eligible tensors with per-block raw fallback.

    Metadata contains only tensors with at least one compressed block. A tensor
    for which compression saves no bytes remains one raw wire tensor.
    """
    items = tuple(tensors.items())
    results = {
        key: _compress_tensor(key, tensor, config, codec, {}) for key, tensor in items
    }
    return _merge_compressed(items, results)


def _compress_tensor(
    key: str,
    tensor: torch.Tensor,
    config: CompressionConfig,
    codec: TensorCodec,
    workspace: dict[str, torch.Tensor],
) -> _CompressedTensor:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError(f"trajectory tensor {key!r} must be contiguous on CPU")
    tensor_bytes = tensor.numel() * tensor.element_size()
    if tensor_bytes < config.min_bytes:
        return _CompressedTensor(key, tensor, None, tensor_bytes, 0, 1, 0, 0)

    source = tensor.view(torch.uint8).reshape(-1)
    block_sizes = tuple(
        min(config.block_bytes, tensor_bytes - offset)
        for offset in range(0, tensor_bytes, config.block_bytes)
    )
    capacity = sum(codec.compress_bound(size) for size in block_sizes)
    destination = workspace.get(key)
    workspace_allocations = 0
    workspace_bytes = 0
    if destination is None or destination.numel() < capacity:
        destination = torch.empty(capacity, dtype=torch.uint8)
        workspace[key] = destination
        workspace_allocations = 1
        workspace_bytes = capacity

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
            block, destination[wire_offset : wire_offset + bound]
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
            key,
            tensor,
            None,
            tensor_bytes,
            0,
            1,
            workspace_allocations,
            workspace_bytes,
        )
    wire_key = f"{key}:blocks"
    return _CompressedTensor(
        wire_key,
        destination[:wire_offset],
        {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "wire_key": wire_key,
            "blocks": tuple(blocks),
        },
        tensor_bytes,
        compressed_blocks,
        raw_blocks,
        workspace_allocations,
        workspace_bytes,
    )


def _merge_compressed(
    items: tuple[tuple[str, torch.Tensor], ...],
    results: dict[str, _CompressedTensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, int]]:
    wire: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    raw_bytes = 0
    wire_bytes = 0
    compressed_blocks = 0
    raw_blocks = 0
    workspace_allocations = 0
    workspace_bytes = 0
    for key, _tensor in items:
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
    return (
        wire,
        metadata,
        {
            "raw_bytes": raw_bytes,
            "wire_bytes": wire_bytes,
            "compressed_blocks": compressed_blocks,
            "raw_blocks": raw_blocks,
            "workspace_allocations": workspace_allocations,
            "workspace_bytes": workspace_bytes,
        },
    )


def decompress_tensors(
    wire: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    codec: TensorCodec,
) -> dict[str, torch.Tensor]:
    """Restore a tensor dictionary produced by :func:`compress_tensors`."""
    lane = tuple(
        (key, _decompress_tensor(key, tensor_metadata, wire, codec))
        for key, tensor_metadata in metadata.items()
    )
    return _merge_decompressed(wire, metadata, (lane,))


def _decompress_tensor(
    key: str,
    tensor_metadata: dict[str, Any],
    wire: dict[str, torch.Tensor],
    codec: TensorCodec,
) -> torch.Tensor:
    try:
        dtype = getattr(torch, tensor_metadata["dtype"])
    except AttributeError as error:
        raise ValueError(
            f"unsupported trajectory tensor dtype {tensor_metadata['dtype']!r}"
        ) from error
    destination = torch.empty(tensor_metadata["shape"], dtype=dtype)
    destination_bytes = destination.view(torch.uint8).reshape(-1)
    wire_key = tensor_metadata["wire_key"]
    try:
        packed = wire[wire_key]
    except KeyError as error:
        raise ValueError(f"missing trajectory blocks {wire_key!r}") from error
    raw_offset = 0
    wire_offset = 0
    for raw_bytes, encoded_bytes, compressed in tensor_metadata["blocks"]:
        source = packed[wire_offset : wire_offset + encoded_bytes]
        target = destination_bytes[raw_offset : raw_offset + raw_bytes]
        if compressed:
            codec.decompress_into(source, source.numel(), target)
        else:
            if source.numel() != raw_bytes:
                raise ValueError(f"raw trajectory block {wire_key!r} has wrong size")
            target.copy_(source)
        raw_offset += raw_bytes
        wire_offset += encoded_bytes
    if raw_offset != destination_bytes.numel() or wire_offset != packed.numel():
        raise ValueError(f"trajectory tensor {key!r} block sizes do not match shape")
    return destination


def _merge_decompressed(
    wire: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    lane_results: tuple[list[tuple[str, torch.Tensor]], ...],
) -> dict[str, torch.Tensor]:
    compressed_keys = set(metadata)
    block_keys = {tensor_metadata["wire_key"] for tensor_metadata in metadata.values()}
    raw_keys = set(wire) - block_keys
    if raw_keys.intersection(compressed_keys):
        raise ValueError("compressed tensor also appeared as a raw wire tensor")
    if set(wire) != raw_keys | block_keys:
        raise ValueError("trajectory compression metadata does not match wire tensors")
    tensors = {key: wire[key] for key in raw_keys}
    tensors.update(result for lane in lane_results for result in lane)
    return tensors

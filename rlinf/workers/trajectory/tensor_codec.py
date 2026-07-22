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

"""Lossless codecs operating directly on contiguous tensor memory."""

import ctypes
import ctypes.util
import importlib.util
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch


class TensorCodec(ABC):
    """Compress tensors into preallocated uint8 tensors without Python bytes."""

    @abstractmethod
    def compress_bound(self, source_bytes: int) -> int:
        """Return the required worst-case destination capacity."""

    @abstractmethod
    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        """Compress source into destination and return the encoded byte count."""

    @abstractmethod
    def decompress_into(
        self,
        source: torch.Tensor,
        compressed_bytes: int,
        destination: torch.Tensor,
    ) -> None:
        """Decompress source directly into destination."""


class LZ4TensorCodec(TensorCodec):
    """LZ4-fast tensor codec backed by the system liblz4 library."""

    _MAX_INPUT_SIZE = 0x7E000000

    def __init__(self, acceleration: int = 1) -> None:
        if acceleration <= 0:
            raise ValueError("LZ4 acceleration must be positive.")
        self.acceleration = acceleration
        self._library = _load_library("lz4")
        self._library.LZ4_compressBound.argtypes = [ctypes.c_int]
        self._library.LZ4_compressBound.restype = ctypes.c_int
        self._library.LZ4_compress_fast.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.LZ4_compress_fast.restype = ctypes.c_int
        self._library.LZ4_decompress_safe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.LZ4_decompress_safe.restype = ctypes.c_int

    def compress_bound(self, source_bytes: int) -> int:
        if not 0 <= source_bytes <= self._MAX_INPUT_SIZE:
            raise ValueError(f"LZ4 source size must be in [0, {self._MAX_INPUT_SIZE}].")
        return int(self._library.LZ4_compressBound(source_bytes))

    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        source_bytes = _tensor_bytes(source, "source")
        destination_bytes = _byte_tensor(destination, "destination")
        required_bytes = self.compress_bound(source_bytes)
        if destination_bytes < required_bytes:
            raise ValueError(
                f"LZ4 destination requires {required_bytes} bytes, got "
                f"{destination_bytes}."
            )
        if source_bytes == 0:
            return 0
        compressed_bytes = self._library.LZ4_compress_fast(
            source.data_ptr(),
            destination.data_ptr(),
            source_bytes,
            destination_bytes,
            self.acceleration,
        )
        if compressed_bytes <= 0:
            raise RuntimeError("LZ4 compression failed.")
        return int(compressed_bytes)

    def decompress_into(
        self,
        source: torch.Tensor,
        compressed_bytes: int,
        destination: torch.Tensor,
    ) -> None:
        source_capacity = _byte_tensor(source, "source")
        destination_bytes = _tensor_bytes(destination, "destination")
        _validate_compressed_size(compressed_bytes, source_capacity)
        if compressed_bytes == 0 and destination_bytes == 0:
            return
        restored_bytes = self._library.LZ4_decompress_safe(
            source.data_ptr(),
            destination.data_ptr(),
            compressed_bytes,
            destination_bytes,
        )
        if restored_bytes < 0:
            raise RuntimeError("LZ4 decompression detected invalid compressed data.")
        if restored_bytes != destination_bytes:
            raise ValueError(
                f"LZ4 restored {restored_bytes} bytes, expected {destination_bytes}."
            )


class ZstdTensorCodec(TensorCodec):
    """Zstandard tensor codec with reusable native contexts.

    One instance must not be used concurrently by multiple threads.
    """

    def __init__(self, level: int = 1) -> None:
        self.level = level
        self._library = _load_library("zstd")
        self._library.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self._library.ZSTD_compressBound.restype = ctypes.c_size_t
        self._library.ZSTD_createCCtx.restype = ctypes.c_void_p
        self._library.ZSTD_freeCCtx.argtypes = [ctypes.c_void_p]
        self._library.ZSTD_compressCCtx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._library.ZSTD_compressCCtx.restype = ctypes.c_size_t
        self._library.ZSTD_createDCtx.restype = ctypes.c_void_p
        self._library.ZSTD_freeDCtx.argtypes = [ctypes.c_void_p]
        self._library.ZSTD_decompressDCtx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._library.ZSTD_decompressDCtx.restype = ctypes.c_size_t
        self._library.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self._library.ZSTD_isError.restype = ctypes.c_uint
        self._library.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self._library.ZSTD_getErrorName.restype = ctypes.c_char_p
        self._compression_context = self._library.ZSTD_createCCtx()
        self._decompression_context = self._library.ZSTD_createDCtx()
        if not self._compression_context or not self._decompression_context:
            raise RuntimeError("Zstd failed to allocate codec contexts.")
        self._compression_finalizer = weakref.finalize(
            self, self._library.ZSTD_freeCCtx, self._compression_context
        )
        self._decompression_finalizer = weakref.finalize(
            self, self._library.ZSTD_freeDCtx, self._decompression_context
        )

    def compress_bound(self, source_bytes: int) -> int:
        if source_bytes < 0:
            raise ValueError("Zstd source size must be non-negative.")
        return int(self._library.ZSTD_compressBound(source_bytes))

    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        source_bytes = _tensor_bytes(source, "source")
        destination_bytes = _byte_tensor(destination, "destination")
        required_bytes = self.compress_bound(source_bytes)
        if destination_bytes < required_bytes:
            raise ValueError(
                f"Zstd destination requires {required_bytes} bytes, got "
                f"{destination_bytes}."
            )
        compressed_bytes = self._library.ZSTD_compressCCtx(
            self._compression_context,
            destination.data_ptr(),
            destination_bytes,
            source.data_ptr(),
            source_bytes,
            self.level,
        )
        self._check_result(compressed_bytes, "compression")
        return int(compressed_bytes)

    def decompress_into(
        self,
        source: torch.Tensor,
        compressed_bytes: int,
        destination: torch.Tensor,
    ) -> None:
        source_capacity = _byte_tensor(source, "source")
        destination_bytes = _tensor_bytes(destination, "destination")
        _validate_compressed_size(compressed_bytes, source_capacity)
        restored_bytes = self._library.ZSTD_decompressDCtx(
            self._decompression_context,
            destination.data_ptr(),
            destination_bytes,
            source.data_ptr(),
            compressed_bytes,
        )
        self._check_result(restored_bytes, "decompression")
        if restored_bytes != destination_bytes:
            raise ValueError(
                f"Zstd restored {restored_bytes} bytes, expected {destination_bytes}."
            )

    def _check_result(self, result: int, operation: str) -> None:
        if self._library.ZSTD_isError(result):
            error = self._library.ZSTD_getErrorName(result).decode()
            raise RuntimeError(f"Zstd {operation} failed: {error}.")


class _NvCompLZ4CompressOptions(ctypes.Structure):
    _fields_ = [
        ("data_type", ctypes.c_int),
        ("bitshuffle_mode", ctypes.c_int),
        ("reserved", ctypes.c_char * 56),
    ]


class _NvCompLZ4DecompressOptions(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("sort_before_hw_decompress", ctypes.c_int),
        ("data_type", ctypes.c_int),
        ("bitshuffle_mode", ctypes.c_int),
        ("reserved", ctypes.c_char * 48),
    ]


@dataclass
class NvCompLZ4Workspace:
    """Reusable CUDA buffers for one nvCOMP LZ4 operation at a time."""

    source_bytes: int
    chunk_bytes: int
    raw_chunk_sizes: tuple[int, ...]
    max_compressed_chunk_bytes: int
    output: torch.Tensor
    temp: torch.Tensor
    input_ptrs: torch.Tensor
    input_sizes: torch.Tensor
    output_ptrs: torch.Tensor
    destination_ptrs: torch.Tensor
    compressed_sizes: torch.Tensor
    restored_sizes: torch.Tensor
    statuses: torch.Tensor

    @property
    def num_chunks(self) -> int:
        return len(self.raw_chunk_sizes)


@dataclass(frozen=True)
class NvCompLZ4Compressed:
    """Fixed-slot nvCOMP output whose sizes and statuses remain on the GPU."""

    storage: torch.Tensor
    compressed_sizes: torch.Tensor
    statuses: torch.Tensor
    raw_chunk_sizes: tuple[int, ...]
    max_compressed_chunk_bytes: int


class NvCompLZ4TensorCodec:
    """Batched LZ4 codec backed by the low-level libnvcomp C API.

    Compression and decompression are enqueued on PyTorch's current CUDA stream.
    The caller must keep the source, destination, compressed result, and workspace
    alive until that stream completes. One workspace must not be used by
    overlapping operations.
    """

    _NVCOMP_SUCCESS = 0
    _NVCOMP_TYPE_CHAR = 0
    _NVCOMP_BITSHUFFLE_NONE = 0
    _NVCOMP_DECOMPRESS_BACKEND_CUDA = 2
    _MAX_CHUNK_BYTES = 1 << 24
    _REQUIRED_COMPRESSION_ALIGNMENT = 4

    def __init__(self, chunk_bytes: int = 64 * 1024) -> None:
        if not 0 < chunk_bytes <= self._MAX_CHUNK_BYTES:
            raise ValueError(
                f"nvCOMP LZ4 chunk size must be in [1, {self._MAX_CHUNK_BYTES}]."
            )
        if chunk_bytes % self._REQUIRED_COMPRESSION_ALIGNMENT != 0:
            raise ValueError("nvCOMP LZ4 chunk size must be a multiple of 4 bytes.")
        self.chunk_bytes = chunk_bytes
        self._compress_options = _NvCompLZ4CompressOptions(
            self._NVCOMP_TYPE_CHAR,
            self._NVCOMP_BITSHUFFLE_NONE,
            bytes(56),
        )
        self._decompress_options = _NvCompLZ4DecompressOptions(
            self._NVCOMP_DECOMPRESS_BACKEND_CUDA,
            0,
            self._NVCOMP_TYPE_CHAR,
            self._NVCOMP_BITSHUFFLE_NONE,
            bytes(48),
        )
        self._library = _load_nvcomp_library()
        self._configure_library()

    def allocate_workspace(
        self, source_bytes: int, device: torch.device | str
    ) -> NvCompLZ4Workspace:
        """Allocate reusable device buffers for a fixed maximum source size."""
        if source_bytes < 0:
            raise ValueError("nvCOMP source size must be non-negative.")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("nvCOMP workspace must be allocated on CUDA.")

        raw_chunk_sizes = tuple(
            min(self.chunk_bytes, source_bytes - offset)
            for offset in range(0, source_bytes, self.chunk_bytes)
        )
        num_chunks = len(raw_chunk_sizes)
        max_compressed_chunk_bytes = self._max_output_chunk_size()
        temp_bytes = max(
            self._compression_temp_size(num_chunks, source_bytes),
            self._decompression_temp_size(num_chunks, source_bytes),
        )
        output = torch.empty(
            num_chunks * max_compressed_chunk_bytes,
            dtype=torch.uint8,
            device=device,
        )
        output_ptrs = torch.arange(num_chunks, dtype=torch.int64, device=device)
        output_ptrs.mul_(max_compressed_chunk_bytes).add_(output.data_ptr())
        return NvCompLZ4Workspace(
            source_bytes=source_bytes,
            chunk_bytes=self.chunk_bytes,
            raw_chunk_sizes=raw_chunk_sizes,
            max_compressed_chunk_bytes=max_compressed_chunk_bytes,
            output=output,
            temp=torch.empty(temp_bytes, dtype=torch.uint8, device=device),
            input_ptrs=torch.empty(num_chunks, dtype=torch.int64, device=device),
            input_sizes=torch.tensor(raw_chunk_sizes, dtype=torch.int64, device=device),
            output_ptrs=output_ptrs,
            destination_ptrs=torch.empty(num_chunks, dtype=torch.int64, device=device),
            compressed_sizes=torch.empty(num_chunks, dtype=torch.int64, device=device),
            restored_sizes=torch.empty(num_chunks, dtype=torch.int64, device=device),
            statuses=torch.empty(num_chunks, dtype=torch.int32, device=device),
        )

    def compress_into(
        self, source: torch.Tensor, workspace: NvCompLZ4Workspace
    ) -> NvCompLZ4Compressed:
        """Enqueue batched compression into fixed-size workspace slots."""
        source_bytes = _cuda_tensor_bytes(source, "source")
        if source.data_ptr() % self._REQUIRED_COMPRESSION_ALIGNMENT != 0:
            raise ValueError("nvCOMP LZ4 source must be aligned to 4 bytes.")
        self._validate_workspace(workspace, source_bytes, source.device)
        if workspace.num_chunks == 0:
            return self._compressed_result(workspace)

        torch.arange(
            workspace.num_chunks,
            dtype=torch.int64,
            device=source.device,
            out=workspace.input_ptrs,
        )
        workspace.input_ptrs.mul_(workspace.chunk_bytes).add_(source.data_ptr())
        status = self._library.nvcompBatchedLZ4CompressAsync(
            workspace.input_ptrs.data_ptr(),
            workspace.input_sizes.data_ptr(),
            workspace.chunk_bytes,
            workspace.num_chunks,
            workspace.temp.data_ptr(),
            workspace.temp.numel(),
            workspace.output_ptrs.data_ptr(),
            workspace.compressed_sizes.data_ptr(),
            self._compress_options,
            workspace.statuses.data_ptr(),
            torch.cuda.current_stream(source.device).cuda_stream,
        )
        self._check_status(status, "compression launch")
        return self._compressed_result(workspace)

    def decompress_into(
        self,
        source: NvCompLZ4Compressed,
        destination: torch.Tensor,
        workspace: NvCompLZ4Workspace,
    ) -> None:
        """Enqueue batched decompression directly into destination storage."""
        destination_bytes = _cuda_tensor_bytes(destination, "destination")
        self._validate_workspace(workspace, destination_bytes, destination.device)
        if source.storage.data_ptr() != workspace.output.data_ptr():
            raise ValueError("compressed source does not belong to this workspace")
        if source.raw_chunk_sizes != workspace.raw_chunk_sizes:
            raise ValueError("compressed source layout does not match workspace")
        if workspace.num_chunks == 0:
            return

        torch.arange(
            workspace.num_chunks,
            dtype=torch.int64,
            device=destination.device,
            out=workspace.destination_ptrs,
        )
        workspace.destination_ptrs.mul_(workspace.chunk_bytes).add_(
            destination.data_ptr()
        )
        status = self._library.nvcompBatchedLZ4DecompressAsync(
            workspace.output_ptrs.data_ptr(),
            workspace.compressed_sizes.data_ptr(),
            workspace.input_sizes.data_ptr(),
            workspace.restored_sizes.data_ptr(),
            workspace.num_chunks,
            workspace.temp.data_ptr(),
            workspace.temp.numel(),
            workspace.destination_ptrs.data_ptr(),
            self._decompress_options,
            workspace.statuses.data_ptr(),
            torch.cuda.current_stream(destination.device).cuda_stream,
        )
        self._check_status(status, "decompression launch")

    def check_statuses(self, workspace: NvCompLZ4Workspace, operation: str) -> None:
        """Synchronize the workspace device and raise for per-chunk failures."""
        torch.cuda.synchronize(workspace.output.device)
        failures = [
            (index, int(status))
            for index, status in enumerate(workspace.statuses.cpu().tolist())
            if status != self._NVCOMP_SUCCESS
        ]
        if failures:
            index, status = failures[0]
            raise RuntimeError(
                f"nvCOMP LZ4 {operation} failed for chunk {index}: "
                f"{self._status_name(status)}."
            )

    def _configure_library(self) -> None:
        self._library.nvcompGetStatusString.argtypes = [ctypes.c_int]
        self._library.nvcompGetStatusString.restype = ctypes.c_char_p
        self._library.nvcompBatchedLZ4CompressGetMaxOutputChunkSize.argtypes = [
            ctypes.c_size_t,
            _NvCompLZ4CompressOptions,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.nvcompBatchedLZ4CompressGetMaxOutputChunkSize.restype = (
            ctypes.c_int
        )
        self._library.nvcompBatchedLZ4CompressGetTempSizeAsync.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            _NvCompLZ4CompressOptions,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        self._library.nvcompBatchedLZ4CompressGetTempSizeAsync.restype = ctypes.c_int
        self._library.nvcompBatchedLZ4DecompressGetTempSizeAsync.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            _NvCompLZ4DecompressOptions,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        self._library.nvcompBatchedLZ4DecompressGetTempSizeAsync.restype = ctypes.c_int
        self._library.nvcompBatchedLZ4CompressAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            _NvCompLZ4CompressOptions,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._library.nvcompBatchedLZ4CompressAsync.restype = ctypes.c_int
        self._library.nvcompBatchedLZ4DecompressAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            _NvCompLZ4DecompressOptions,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._library.nvcompBatchedLZ4DecompressAsync.restype = ctypes.c_int

    def _max_output_chunk_size(self) -> int:
        result = ctypes.c_size_t()
        status = self._library.nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
            self.chunk_bytes, self._compress_options, ctypes.byref(result)
        )
        self._check_status(status, "output size query")
        return result.value

    def _compression_temp_size(self, num_chunks: int, source_bytes: int) -> int:
        if num_chunks == 0:
            return 0
        result = ctypes.c_size_t()
        status = self._library.nvcompBatchedLZ4CompressGetTempSizeAsync(
            num_chunks,
            self.chunk_bytes,
            self._compress_options,
            ctypes.byref(result),
            source_bytes,
        )
        self._check_status(status, "compression workspace query")
        return result.value

    def _decompression_temp_size(self, num_chunks: int, source_bytes: int) -> int:
        if num_chunks == 0:
            return 0
        result = ctypes.c_size_t()
        status = self._library.nvcompBatchedLZ4DecompressGetTempSizeAsync(
            num_chunks,
            self.chunk_bytes,
            self._decompress_options,
            ctypes.byref(result),
            source_bytes,
        )
        self._check_status(status, "decompression workspace query")
        return result.value

    def _validate_workspace(
        self,
        workspace: NvCompLZ4Workspace,
        source_bytes: int,
        device: torch.device,
    ) -> None:
        if workspace.source_bytes != source_bytes:
            raise ValueError(
                f"nvCOMP workspace is for {workspace.source_bytes} bytes, got "
                f"{source_bytes}."
            )
        if workspace.output.device != device:
            raise ValueError("nvCOMP workspace and tensor must be on the same device.")
        if workspace.chunk_bytes != self.chunk_bytes:
            raise ValueError("nvCOMP workspace was allocated by a different codec.")

    def _compressed_result(self, workspace: NvCompLZ4Workspace) -> NvCompLZ4Compressed:
        return NvCompLZ4Compressed(
            storage=workspace.output,
            compressed_sizes=workspace.compressed_sizes,
            statuses=workspace.statuses,
            raw_chunk_sizes=workspace.raw_chunk_sizes,
            max_compressed_chunk_bytes=workspace.max_compressed_chunk_bytes,
        )

    def _check_status(self, status: int, operation: str) -> None:
        if status != self._NVCOMP_SUCCESS:
            raise RuntimeError(
                f"nvCOMP LZ4 {operation} failed: {self._status_name(status)}."
            )

    def _status_name(self, status: int) -> str:
        return self._library.nvcompGetStatusString(status).decode()


def create_tensor_codec(name: str, *, level: int = 1) -> TensorCodec:
    """Construct a direct tensor codec by transport name."""
    if name == "lz4":
        return LZ4TensorCodec(acceleration=level)
    if name == "zstd":
        return ZstdTensorCodec(level=level)
    raise ValueError(f"Unsupported tensor codec: {name!r}.")


def _load_library(name: str) -> ctypes.CDLL:
    library_path = ctypes.util.find_library(name)
    if library_path is None:
        raise RuntimeError(f"System lib{name} is not available.")
    return ctypes.CDLL(library_path)


def _load_nvcomp_library() -> ctypes.CDLL:
    package = importlib.util.find_spec("nvidia.libnvcomp")
    if package is None or package.origin is None:
        raise RuntimeError("nvidia-libnvcomp-cu12 is not available.")
    library_directory = Path(package.origin).parent / "lib64"
    libraries = sorted(library_directory.glob("libnvcomp.so.*"))
    if not libraries:
        raise RuntimeError("libnvcomp shared library is not available.")
    return ctypes.CDLL(str(libraries[-1]))


def _tensor_bytes(tensor: torch.Tensor, name: str) -> int:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    return tensor.numel() * tensor.element_size()


def _byte_tensor(tensor: torch.Tensor, name: str) -> int:
    size = _tensor_bytes(tensor, name)
    if tensor.dtype != torch.uint8:
        raise ValueError(f"{name} must have dtype torch.uint8.")
    return size


def _cuda_tensor_bytes(tensor: torch.Tensor, name: str) -> int:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    return tensor.numel() * tensor.element_size()


def _validate_compressed_size(compressed_bytes: int, capacity: int) -> None:
    if not 0 <= compressed_bytes <= capacity:
        raise ValueError(
            f"Compressed size must be in [0, {capacity}], got {compressed_bytes}."
        )

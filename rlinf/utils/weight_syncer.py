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
import io
import zlib
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from omegaconf import DictConfig, OmegaConf
import torch
from torch.distributed.tensor import DTensor


def _bytes_to_uint8_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(memoryview(data), dtype=torch.uint8).clone()


def _uint8_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def downscale_nonnegative_indices(tensor: torch.Tensor) -> torch.Tensor:
    assert torch.all(tensor >= 0), "Delta encoded indices must be non-negative"
    if tensor.numel() == 0:
        return tensor.to(torch.uint8)
    max_value = int(tensor.max().item())
    if max_value <= torch.iinfo(torch.uint8).max:
        return tensor.to(torch.uint8)
    elif max_value <= torch.iinfo(torch.uint16).max:
        return tensor.to(torch.uint16)
    elif max_value <= torch.iinfo(torch.uint32).max:
        return tensor.to(torch.uint32)
    else:
        return tensor.to(torch.int64)


def as_coo_2d_view(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = tensor.shape

    if tensor.ndim == 0:
        view = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        view = tensor.reshape(1, tensor.shape[0])
    else:
        view = tensor.reshape(tensor.shape[0], -1)
    return view, original_shape


def materialize_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    assert isinstance(tensor, torch.Tensor), "Expected a torch.Tensor or DTensor"
    return tensor


class WeightSyncer(ABC):
    @abstractmethod
    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: Callable[[Any], Awaitable[None]],
        version: int | torch.Tensor,
    ) -> None: ...

    @abstractmethod
    async def apply(
        self, model: torch.nn.Module, recv: Callable[[], Awaitable[Any]]
    ) -> None: ...

    def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    ):
        pass

    def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    ) -> None:
        pass

    @classmethod
    def create(cls, config: DictConfig) -> "WeightSyncer":
        assert config is not None, "Weight syncer config must be provided"
        syncer_type = OmegaConf.select(config, "type")
        if syncer_type == "bucket":
            bucket_config = OmegaConf.select(config, "bucket")
            assert bucket_config is not None, (
                "Bucket config must be provided for bucket weight syncer"
            )
            return BucketWeightSyncer(
                bucket_size=OmegaConf.select(bucket_config, "bucket_size"),
                bucket_dtype=OmegaConf.select(bucket_config, "bucket_dtype"),
                bucket_device=OmegaConf.select(bucket_config, "bucket_device"),
                is_agent=OmegaConf.select(bucket_config, "is_agent", default=False),
                load_instant=OmegaConf.select(
                    bucket_config, "load_instant", default=True
                ),
            )
        elif syncer_type == "patch":
            patch_config = OmegaConf.select(config, "patch")
            assert patch_config is not None, (
                "Patch config must be provided for patch weight syncer"
            )
            return PatchWeightSyncer(
                snapshot_dtype=OmegaConf.select(
                    patch_config, "snapshot_dtype", default=torch.bfloat16
                ),
                snapshot_device=OmegaConf.select(
                    patch_config, "snapshot_device", default="cpu"
                ),
                delta_encoding=OmegaConf.select(
                    patch_config, "delta_encoding", default=True
                ),
                compression_algorithm=OmegaConf.select(
                    patch_config, "compression_algorithm", default="none"
                ),
                transport_device=OmegaConf.select(
                    patch_config, "transport_device", default="cuda"
                ),
            )
        else:
            raise ValueError(f"Unsupported weight syncer type: {syncer_type}")


class BucketWeightSyncer(WeightSyncer):
    def __init__(
        self,
        bucket_size: int,
        bucket_dtype: torch.dtype,
        bucket_device: str | torch.device,
        is_agent: bool = False,
        load_instant: bool = True,
    ):
        self.bucket_size = bucket_size
        self.bucket_dtype = bucket_dtype
        self.bucket_device = bucket_device
        self.is_agent = is_agent
        self.load_instant = load_instant

    def divide_into_buckets(
        self, state_dict: dict[str, torch.Tensor | DTensor]
    ) -> list[dict[str, torch.Tensor]]:
        buckets = []
        currently_hold = 0
        bucket = {}
        has_visual = any("visual." in k for k in state_dict.keys())
        for key, value in state_dict.items():
            name = key
            if "_extra_state" in name:
                continue
            if (
                has_visual
                and self.is_agent
                and name.startswith("model.language_model.")
            ):
                name = "model." + name[len("model.language_model.") :]

            bucket[name] = materialize_tensor(value).to(
                device=self.bucket_device, dtype=self.bucket_dtype, non_blocking=True
            )
            currently_hold += bucket[name].numel() * bucket[name].element_size()

            if currently_hold >= self.bucket_size:
                buckets.append(bucket)
                bucket = {}
                currently_hold = 0

        if len(bucket) > 0:
            buckets.append(bucket)

        assert len(buckets) > 0, "No parameters to sync"
        buckets[0]["total_buckets"] = torch.tensor(
            len(buckets), dtype=torch.uint16, device=self.bucket_device
        )
        return buckets

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: Callable[[Any], Awaitable[None]],
        version: int | torch.Tensor,
    ) -> None:
        buckets = self.divide_into_buckets(state_dict)

        for bucket in buckets:
            await send(bucket)

    async def apply(
        self,
        model: torch.nn.Module,
        recv: Callable[[], Awaitable[Any]],
    ) -> None:
        bucket: dict[str, torch.Tensor] = await recv()
        total_buckets = int(bucket.pop("total_buckets").item())

        if self.load_instant:
            model.load_state_dict(bucket, strict=False)
        else:
            cpu_buffer: dict[str, torch.Tensor] = {}
            for k, v in bucket.items():
                cpu_buffer[k] = v.to("cpu", non_blocking=True)
        del bucket

        for _ in range(total_buckets - 1):
            bucket: dict[str, torch.Tensor] = await recv()
            if self.load_instant:
                model.load_state_dict(bucket, strict=False)
            else:
                for k, v in bucket.items():
                    cpu_buffer[k] = v.to("cpu", non_blocking=True)
            del bucket

        if not self.load_instant:
            model.load_state_dict(cpu_buffer, strict=True)
            del cpu_buffer


@dataclass
class TensorPatch:
    ordinal: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    values: torch.Tensor


@dataclass
class WeightPatch:
    version: torch.Tensor
    tensors: list[TensorPatch]


class PatchWeightSyncer(WeightSyncer):
    def __init__(
        self,
        snapshot_dtype: torch.dtype | str = torch.bfloat16,
        snapshot_device: torch.device | str = "cpu",
        transport_device: torch.device | str = "cuda",
        delta_encoding: bool = True,
        compression_algorithm: str = "none",
    ):
        self.snapshot: None | dict[str, torch.Tensor] = None
        self.original_shapes: None | dict[str, torch.Size] = None
        self.ordered_keys: None | list[str] = None
        self.delta_encoding = delta_encoding
        self.compression_algorithm = compression_algorithm
        self.transport_device = transport_device
        self.snapshot_dtype = snapshot_dtype
        self.snapshot_device = snapshot_device

    def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    ):
        assert (
            self.snapshot is None
            and self.original_shapes is None
            and self.ordered_keys is None
        ), "Sender already initialized"
        snapshot: dict[str, torch.Tensor] = {}
        self.original_shapes = {}
        self.ordered_keys = []
        for key, value in state_dict.items():
            value, original_shape = as_coo_2d_view(materialize_tensor(value))
            snapshot[key] = (
                value.detach()
                .to(
                    device=self.snapshot_device,
                    dtype=self.snapshot_dtype,
                    non_blocking=True,
                    copy=True,
                )
                .pin_memory()
                if torch.device(self.snapshot_device) == torch.device("cpu")
                else value.detach().to(
                    device=self.snapshot_device,
                    dtype=self.snapshot_dtype,
                    non_blocking=True,
                    copy=True,
                )
            )
            self.original_shapes[key] = original_shape
            self.ordered_keys.append(key)
        self.snapshot = snapshot

    def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    ) -> None:
        assert self.original_shapes is None and self.ordered_keys is None, (
            "Receiver already initialized"
        )
        self.original_shapes = {
            key: materialize_tensor(value).shape for key, value in state_dict.items()
        }
        self.ordered_keys = list(state_dict.keys())

    def _serialize_patch(self, patch: WeightPatch) -> bytes:
        buffer = io.BytesIO()
        torch.save(patch, buffer)
        return buffer.getvalue()

    def _deserialize_patch(self, data: bytes) -> WeightPatch:
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def _compress_bytes(self, data: bytes) -> bytes:

        if self.compression_algorithm == "none":
            return data
        elif self.compression_algorithm == "zstd":
            import zstandard as zstd

            return zstd.ZstdCompressor(level=1).compress(data)
        elif self.compression_algorithm == "zlib":
            return zlib.compress(data)
        elif self.compression_algorithm == "lz4":
            import lz4.frame

            return lz4.frame.compress(data, compression_level=0)

        raise ValueError(
            f"Unsupported compression algorithm: {self.compression_algorithm}"
        )

    def _decompress_bytes(self, data: bytes) -> bytes:
        if self.compression_algorithm == "none":
            return data
        elif self.compression_algorithm == "zstd":
            import zstandard as zstd

            return zstd.ZstdDecompressor().decompress(data)
        elif self.compression_algorithm == "zlib":
            return zlib.decompress(data)
        elif self.compression_algorithm == "lz4":
            import lz4.frame

            return lz4.frame.decompress(data)

        raise ValueError(
            f"Unsupported compression algorithm: {self.compression_algorithm}"
        )

    def delta_encode(
        self, rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows.numel() > 0, "No indices to encode"
        assert rows.numel() == cols.numel(), (
            "Rows and columns must have the same number of elements"
        )
        if rows.numel() == 1:
            return downscale_nonnegative_indices(rows), downscale_nonnegative_indices(
                cols
            )
        row_deltas = torch.empty_like(rows)
        col_deltas = torch.empty_like(cols)
        row_deltas[0] = rows[0]
        col_deltas[0] = cols[0]
        row_deltas[1:] = rows[1:] - rows[:-1]

        same_row = rows[1:] == rows[:-1]
        col_deltas[1:] = torch.where(same_row, cols[1:] - cols[:-1], cols[1:])

        return downscale_nonnegative_indices(row_deltas), downscale_nonnegative_indices(
            col_deltas
        )

    def delta_decode(
        self, rows_delta: torch.Tensor, cols_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows_delta.numel() > 0, "No indices to decode"
        assert rows_delta.numel() == cols_delta.numel(), (
            "Rows and columns must have the same number of elements"
        )
        rows_delta, cols_delta = rows_delta.to(torch.int64), cols_delta.to(torch.int64)
        rows = torch.empty_like(rows_delta)
        cols = torch.empty_like(cols_delta)
        rows[0] = rows_delta[0]
        cols[0] = cols_delta[0]
        for i in range(1, rows_delta.numel()):
            rows[i] = rows[i - 1] + rows_delta[i]
            if rows_delta[i] == 0:
                cols[i] = cols[i - 1] + cols_delta[i]
            else:
                cols[i] = cols_delta[i]
        return rows, cols

    def compress(self, patch: WeightPatch) -> torch.Tensor:
        data = self._serialize_patch(patch)
        compressed_data = self._compress_bytes(data)
        return _bytes_to_uint8_tensor(compressed_data)

    def decompress(self, payload: torch.Tensor) -> WeightPatch:
        compressed_data = _uint8_tensor_to_bytes(payload)
        data = self._decompress_bytes(compressed_data)
        patch = self._deserialize_patch(data)
        return patch

    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> WeightPatch:
        assert self.snapshot is not None and self.ordered_keys is not None, (
            "Snapshot not initialized"
        )
        assert list(state_dict.keys()) == self.ordered_keys, (
            "State dict keys do not match snapshot keys"
        )

        tensor_patches = []
        for ordinal, key in enumerate(self.ordered_keys):
            value, _ = as_coo_2d_view(materialize_tensor(state_dict[key]))
            value = value.to(
                device=self.snapshot[key].device,
                dtype=self.snapshot[key].dtype,
                non_blocking=True,
            )

            changed = value.ne(self.snapshot[key])
            if not torch.any(changed):
                continue

            rows, cols = changed.nonzero(as_tuple=True)

            order = torch.argsort(rows * value.shape[1] + cols)
            rows = rows[order].to(torch.int64)
            cols = cols[order].to(torch.int64)
            values = value[rows, cols]

            self.snapshot[key][rows, cols] = values

            if self.delta_encoding:
                rows, cols = self.delta_encode(rows, cols)

            tensor_patch = TensorPatch(
                ordinal=torch.tensor(ordinal, dtype=torch.uint16),
                row=rows,
                col=cols,
                values=values,
            )
            tensor_patches.append(tensor_patch)

        patch = WeightPatch(
            version=torch.tensor(version, dtype=torch.uint64), tensors=tensor_patches
        )
        return patch

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: Callable[[Any], Awaitable[None]],
        version: int | torch.Tensor,
    ) -> None:
        patch = self.create_patch(state_dict, version)
        compressed_patch = self.compress(patch).to(
            device=self.transport_device, non_blocking=True
        )
        await send(compressed_patch)

    @torch.no_grad()
    async def apply(
        self, model: torch.nn.Module, recv: Callable[[], Awaitable[Any]]
    ) -> None:
        assert self.ordered_keys is not None and self.original_shapes is not None, (
            "Snapshot Info not initialized"
        )

        compressed_patch = await recv()
        patch = self.decompress(compressed_patch)

        state_dict = model.state_dict()

        for tensor_patch in patch.tensors:
            key = self.ordered_keys[tensor_patch.ordinal.item()]
            original_shape = self.original_shapes[key]
            value = state_dict[key]
            value_2dview, _ = as_coo_2d_view(value)
            assert value.shape == original_shape, (
                f"Shape mismatch for key {key}: expected {original_shape}, got {value.shape}"
            )

            if self.delta_encoding:
                rows, cols = self.delta_decode(tensor_patch.row, tensor_patch.col)
            else:
                rows, cols = tensor_patch.row, tensor_patch.col
            rows, cols = (
                rows.to(
                    device=value_2dview.device, dtype=torch.int64, non_blocking=True
                ),
                cols.to(
                    device=value_2dview.device, dtype=torch.int64, non_blocking=True
                ),
            )

            value_2dview[rows, cols] = tensor_patch.values.to(
                device=value_2dview.device, dtype=value_2dview.dtype, non_blocking=True
            )

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from omegaconf import DictConfig, OmegaConf
import torch
from rlinf.utils.logging import get_logger
from torch.distributed.tensor import DTensor

logger = get_logger()

def downscale_nonnegative_indices(tensor: torch.Tensor) -> torch.Tensor:
    assert torch.all(tensor >= 0), "Delta encoded indices must be non-negative"
    if tensor.numel() == 0:
        return tensor.to(torch.uint8)
    max_value = int(tensor.max().item())
    if max_value <= torch.iinfo(torch.uint8).max:
        return tensor.to(torch.uint8)
    elif max_value <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
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

def normalize_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = dtype.lower()
        if key in mapping:
            return mapping[key]
    raise TypeError(f"Unsupported dtype: {dtype}")

def normalize_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


class WeightSyncer(ABC):
    
    def __init__(self):
        self._sender_initialized: bool = False
        self._receiver_initialized: bool = False

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

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: Callable[[Any], Awaitable[None]],
    ) -> None:
        self._sender_initialized = True

    async def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    recv: Callable[[], Awaitable[Any]]
    ) -> None:
        self._receiver_initialized = True

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

    def sender_initialized(self) -> bool:
        return self._sender_initialized

    def receiver_initialized(self) -> bool:
        return self._receiver_initialized



class BucketWeightSyncer(WeightSyncer):
    def __init__(
        self,
        bucket_size: int,
        bucket_dtype: torch.dtype,
        bucket_device: str | torch.device,
        is_agent: bool = False,
        load_instant: bool = True,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.bucket_dtype = normalize_dtype(bucket_dtype)
        self.bucket_device = normalize_device(bucket_device)
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
class WeightPatch:
    version: torch.Tensor
    ordinals: torch.Tensor
    nnz_per_tensor: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "WeightPatch":
        device = normalize_device(device)
        return WeightPatch(
            version=self.version.to(device=device, non_blocking=non_blocking),
            ordinals=self.ordinals.to(device=device, non_blocking=non_blocking),
            nnz_per_tensor=self.nnz_per_tensor.to(
                device=device, non_blocking=non_blocking
            ),
            rows=self.rows.to(device=device, non_blocking=non_blocking),
            cols=self.cols.to(device=device, non_blocking=non_blocking),
            values=self.values.to(device=device, non_blocking=non_blocking),
        )


class PatchWeightSyncer(WeightSyncer):
    def __init__(
        self,
        snapshot_dtype: torch.dtype | str = torch.bfloat16,
        snapshot_device: torch.device | str = "cpu",
        transport_device: torch.device | str = "cuda",
        delta_encoding: bool = True,
        compression_algorithm: str = "none",
    ):
        super().__init__()
        self.snapshot: None | dict[str, torch.Tensor] = None
        self.original_shapes: None | dict[str, torch.Size] = None
        self.ordered_keys: None | list[str] = None
        self.delta_encoding = delta_encoding
        self.compression_algorithm = compression_algorithm
        self.transport_device = normalize_device(transport_device)
        self.snapshot_dtype = normalize_dtype(snapshot_dtype)
        self.snapshot_device = normalize_device(snapshot_device)
        if self.compression_algorithm != "none":
            logger.warning(
                "PatchWeightSyncer uses flat tensor transport; "
                f"compression_algorithm={self.compression_algorithm} is ignored for now."
            )

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: Callable[[Any], Awaitable[None]],
    ):
        assert self.sender_initialized() is False, "Sender already initialized"

        snapshot: dict[str, torch.Tensor] = {}
        self.original_shapes = {}
        self.ordered_keys = []
        for key, value in state_dict.items():
            value, original_shape = as_coo_2d_view(materialize_tensor(value))
            copy_non_blocking = self.snapshot_device.type != "cpu"
            snapshot[key] = (
                value.detach()
                .to(
                    device=self.snapshot_device,
                    dtype=self.snapshot_dtype,
                    non_blocking=copy_non_blocking,
                    copy=True,
                )
                .pin_memory()
                if torch.device(self.snapshot_device) == torch.device("cpu")
                else value.detach().to(
                    device=self.snapshot_device,
                    dtype=self.snapshot_dtype,
                    non_blocking=copy_non_blocking,
                    copy=True,
                )
            )
            self.original_shapes[key] = original_shape
            self.ordered_keys.append(key)
        self.snapshot = snapshot
        
        metadata = {
            "ordered_keys": self.ordered_keys,
            "original_shapes": self.original_shapes,
        }
        await send(metadata)
        self._sender_initialized = True
        
        
    async def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        recv: Callable[[], Awaitable[Any]]
    ) -> None:
        assert self.receiver_initialized() is False, "Receiver already initialized"
        metadata = await recv()
        self.ordered_keys = metadata["ordered_keys"]
        self.original_shapes = metadata["original_shapes"]
        self._receiver_initialized = True

    def delta_encode(
        self, rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows.numel() > 0, "No indices to encode"
        assert rows.numel() == cols.numel(), (
            "Rows and columns must have the same number of elements"
        )
        if rows.numel() == 1:
            return rows, cols
        row_deltas = torch.empty_like(rows)
        col_deltas = torch.empty_like(cols)
        row_deltas[0] = rows[0]
        col_deltas[0] = cols[0]
        row_deltas[1:] = rows[1:] - rows[:-1]

        same_row = rows[1:] == rows[:-1]
        col_deltas[1:] = torch.where(same_row, cols[1:] - cols[:-1], cols[1:])

        return row_deltas, col_deltas

    def delta_decode(
        self, rows_delta: torch.Tensor, cols_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows_delta.numel() > 0
        assert rows_delta.numel() == cols_delta.numel()

        rows_delta = rows_delta.to(torch.int64)
        cols_delta = cols_delta.to(torch.int64)

        rows = torch.cumsum(rows_delta, dim=0)

        start_mask = torch.zeros_like(rows_delta, dtype=torch.bool)
        start_mask[0] = True
        start_mask[1:] = rows_delta[1:] != 0

        idx = torch.arange(rows_delta.numel(), device=rows_delta.device, dtype=torch.int64)
        start_idx = torch.where(start_mask, idx, torch.zeros_like(idx))
        start_idx = torch.cummax(start_idx, dim=0).values

        cum_cols = torch.cumsum(cols_delta, dim=0)
        base = (cum_cols - cols_delta)[start_idx]
        cols = cum_cols - base
        return rows, cols

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

        ordinals: list[torch.Tensor] = []
        nnz_per_tensor: list[torch.Tensor] = []
        row_chunks: list[torch.Tensor] = []
        col_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        total_bytes = 0
        for ordinal, key in enumerate(self.ordered_keys):
            value, _ = as_coo_2d_view(materialize_tensor(state_dict[key]))
            value = value.to(
                device=self.snapshot[key].device,
                dtype=self.snapshot[key].dtype,
                non_blocking=self.snapshot[key].device.type != "cpu",
                copy=True,
            )

            changed = value.ne(self.snapshot[key])
            if not torch.any(changed):
                continue

            # `nonzero` already returns indices in row-major lexicographic order,
            # which is exactly what delta encoding expects here.
            rows, cols = changed.nonzero(as_tuple=True)
            rows = rows.to(torch.int64)
            cols = cols.to(torch.int64)
            values = value[rows, cols]

            self.snapshot[key][rows, cols] = values

            if self.delta_encoding:
                rows, cols = self.delta_encode(rows, cols)

            ordinals.append(
                torch.tensor(ordinal, dtype=torch.int32, device=rows.device)
            )
            nnz_per_tensor.append(
                torch.tensor(values.numel(), dtype=torch.int32, device=rows.device)
            )
            row_chunks.append(rows.contiguous())
            col_chunks.append(cols.contiguous())
            value_chunks.append(values.contiguous())
            # logger.info(f"Sender: Key {key} has {values.numel()} changed elements")
            total_bytes += (
                rows.element_size() * rows.numel()
                + cols.element_size() * cols.numel()
                + values.element_size() * values.numel()
            )
        patch_device = self.snapshot_device
        if row_chunks:
            rows_tensor = torch.cat(row_chunks, dim=0)
            cols_tensor = torch.cat(col_chunks, dim=0)
            rows_tensor = downscale_nonnegative_indices(rows_tensor)
            cols_tensor = downscale_nonnegative_indices(cols_tensor)
            patch = WeightPatch(
                version=torch.tensor(version, dtype=torch.int64, device=patch_device),
                ordinals=torch.stack(ordinals),
                nnz_per_tensor=torch.stack(nnz_per_tensor),
                rows=rows_tensor,
                cols=cols_tensor,
                values=torch.cat(value_chunks, dim=0),
            )
        else:
            patch = WeightPatch(
                version=torch.tensor(version, dtype=torch.int64, device=patch_device),
                ordinals=torch.empty(0, dtype=torch.int32, device=patch_device),
                nnz_per_tensor=torch.empty(0, dtype=torch.int32, device=patch_device),
                rows=torch.empty(0, dtype=torch.uint8, device=patch_device),
                cols=torch.empty(0, dtype=torch.uint8, device=patch_device),
                values=torch.empty(0, dtype=self.snapshot_dtype, device=patch_device),
            )
        # logger.info(
        #     f"Total patch tensor size before transport: {total_bytes / (1024 ** 2):.2f} MB"
        # )
        return patch

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: Callable[[Any], Awaitable[None]],
        version: int | torch.Tensor,
    ) -> None:
        patch = self.create_patch(state_dict, version)
        transport_patch = patch.to(
            device=self.transport_device,
            non_blocking=self.transport_device.type != "cpu",
        )
        await send(transport_patch)

    @torch.no_grad()
    async def apply(
        self, model: torch.nn.Module, recv: Callable[[], Awaitable[Any]]
    ) -> None:
        assert self.ordered_keys is not None and self.original_shapes is not None, (
            "Snapshot Info not initialized"
        )

        patch: WeightPatch = await recv()

        state_dict = model.state_dict()
        # total_changed_elements = int(patch.values.numel())
        # logger.info(
        #     f"Applying patch version {patch.version.item()} "
        #     f"with {total_changed_elements} changed elements"
        # )
        offset = 0
        for patch_idx in range(patch.ordinals.numel()):
            key = self.ordered_keys[patch.ordinals[patch_idx].item()]
            original_shape = self.original_shapes[key]
            value = state_dict[key]
            value_2dview, _ = as_coo_2d_view(value)
            assert value.shape == original_shape, (
                f"Shape mismatch for key {key}: expected {original_shape}, got {value.shape}"
            )
            nnz = int(patch.nnz_per_tensor[patch_idx].item())
            next_offset = offset + nnz
            row_slice = patch.rows[offset:next_offset]
            col_slice = patch.cols[offset:next_offset]
            value_slice = patch.values[offset:next_offset]
            offset = next_offset

            if self.delta_encoding:
                row_delta = row_slice.to(
                    value_2dview.device, dtype=torch.int64, non_blocking=True
                )
                col_delta = col_slice.to(
                    value_2dview.device, dtype=torch.int64, non_blocking=True
                )
                rows, cols = self.delta_decode(row_delta, col_delta)
            else:
                rows = row_slice.to(
                    device=value_2dview.device, dtype=torch.int64, non_blocking=True
                )
                cols = col_slice.to(
                    device=value_2dview.device, dtype=torch.int64, non_blocking=True
                )

            value_2dview[rows, cols] = value_slice.to(
                device=value_2dview.device, dtype=value_2dview.dtype, non_blocking=True
            )
        assert offset == patch.rows.numel(), "Patch offsets do not match payload size"

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

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

from .base import RecvFn, SendFn, WeightSyncer, materialize_tensor, normalize_device, normalize_dtype
from .compressor import PatchCompressor


def downscale_nonnegative_indices(tensor: torch.Tensor) -> torch.Tensor:
    assert torch.all(tensor >= 0), "Delta encoded indices must be non-negative"
    if tensor.numel() == 0:
        return tensor.to(torch.uint8)
    max_value = int(tensor.max().item())
    if max_value <= torch.iinfo(torch.uint8).max:
        return tensor.to(torch.uint8)
    if max_value <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
    return tensor.to(torch.int64)


def as_coo_2d_view(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = tensor.shape
    if tensor.ndim == 0:
        view = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 1:
        view = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        view = tensor
    else:
        try:
            view = tensor.view(tensor.shape[0], -1)
        except RuntimeError as error:
            raise ValueError(
                "PatchWeightSyncer only supports ndim>=3 tensors whose trailing "
                "dimensions can be flattened as a view. "
                f"Got shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}."
            ) from error
    return view, original_shape


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


@dataclass
class CompressedWeightPatch:
    version: torch.Tensor
    ordinals: torch.Tensor
    nnz_per_tensor: torch.Tensor
    rows_compressed: torch.Tensor
    cols_compressed: torch.Tensor
    values_compressed: torch.Tensor
    rows_dtype_code: torch.Tensor
    cols_dtype_code: torch.Tensor
    values_dtype_code: torch.Tensor


WeightPatchTransport = WeightPatch | CompressedWeightPatch


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
        self.snapshot: dict[str, torch.Tensor] | None = None
        self.original_shapes: dict[str, torch.Size] | None = None
        self.ordered_keys: list[str] | None = None
        self.delta_encoding = delta_encoding
        self.transport_device = normalize_device(transport_device)
        self.snapshot_dtype = normalize_dtype(snapshot_dtype)
        self.snapshot_device = normalize_device(snapshot_device)
        self.compressor = PatchCompressor.create(
            compression_algorithm=compression_algorithm,
            transport_device=self.transport_device,
        )

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
    ) -> None:
        assert not self.sender_initialized(), "Sender already initialized"

        snapshot: dict[str, torch.Tensor] = {}
        self.original_shapes = {}
        self.ordered_keys = []
        for key, value in state_dict.items():
            value_2dview, original_shape = as_coo_2d_view(materialize_tensor(value))
            copy_non_blocking = self.snapshot_device.type != "cpu"
            snapshot[key] = (
                value_2dview.detach()
                .to(
                    device=self.snapshot_device,
                    dtype=self.snapshot_dtype,
                    non_blocking=copy_non_blocking,
                    copy=True,
                )
                .pin_memory()
                if self.snapshot_device == torch.device("cpu")
                else value_2dview.detach().to(
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
        state_dict: dict[str, torch.Tensor | DTensor] | None,
        recv: RecvFn,
    ) -> None:
        del state_dict
        assert not self.receiver_initialized(), "Receiver already initialized"
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

        idx = torch.arange(
            rows_delta.numel(), device=rows_delta.device, dtype=torch.int64
        )
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
        patch_device = self.snapshot_device

        for ordinal, key in enumerate(self.ordered_keys):
            value_2dview, _ = as_coo_2d_view(materialize_tensor(state_dict[key]))
            snapshot_value = self.snapshot[key]
            value_2dview = value_2dview.to(
                device=snapshot_value.device,
                dtype=snapshot_value.dtype,
                non_blocking=snapshot_value.device.type != "cpu",
                copy=True,
            )

            changed = value_2dview.ne(snapshot_value)
            if not torch.any(changed):
                continue

            rows, cols = changed.nonzero(as_tuple=True)
            rows = rows.to(torch.int64)
            cols = cols.to(torch.int64)
            values = value_2dview[rows, cols]

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

        if row_chunks:
            rows_tensor = downscale_nonnegative_indices(torch.cat(row_chunks, dim=0))
            cols_tensor = downscale_nonnegative_indices(torch.cat(col_chunks, dim=0))
            return WeightPatch(
                version=torch.tensor(version, dtype=torch.int64, device=patch_device),
                ordinals=torch.stack(ordinals),
                nnz_per_tensor=torch.stack(nnz_per_tensor),
                rows=rows_tensor,
                cols=cols_tensor,
                values=torch.cat(value_chunks, dim=0),
            )

        return WeightPatch(
            version=torch.tensor(version, dtype=torch.int64, device=patch_device),
            ordinals=torch.empty(0, dtype=torch.int32, device=patch_device),
            nnz_per_tensor=torch.empty(0, dtype=torch.int32, device=patch_device),
            rows=torch.empty(0, dtype=torch.uint8, device=patch_device),
            cols=torch.empty(0, dtype=torch.uint8, device=patch_device),
            values=torch.empty(0, dtype=self.snapshot_dtype, device=patch_device),
        )

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        patch = self.create_patch(state_dict, version)
        transport_patch = patch.to(
            device=self.transport_device,
            non_blocking=self.transport_device.type != "cpu",
        )
        await send(self.compressor.compress(transport_patch))

    @torch.no_grad()
    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int:
        assert self.ordered_keys is not None and self.original_shapes is not None, (
            "Snapshot info not initialized"
        )

        patch = self.compressor.decompress(await recv())
        applied_version = int(patch.version.item())
        state_dict = model.state_dict()

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
                    device=value_2dview.device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
                col_delta = col_slice.to(
                    device=value_2dview.device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
                rows, cols = self.delta_decode(row_delta, col_delta)
            else:
                rows = row_slice.to(
                    device=value_2dview.device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
                cols = col_slice.to(
                    device=value_2dview.device,
                    dtype=torch.int64,
                    non_blocking=True,
                )

            value_2dview[rows, cols] = value_slice.to(
                device=value_2dview.device,
                dtype=value_2dview.dtype,
                non_blocking=True,
            )

        assert offset == patch.rows.numel(), "Patch offsets do not match payload size"
        return applied_version

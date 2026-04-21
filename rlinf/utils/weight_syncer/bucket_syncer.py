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

import torch
from torch.distributed.tensor import DTensor

from .base import RecvFn, SendFn, WeightSyncer, materialize_tensor, normalize_device, normalize_dtype


class BucketWeightSyncer(WeightSyncer):
    def __init__(
        self,
        bucket_size: int,
        bucket_dtype: torch.dtype | str,
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
        buckets: list[dict[str, torch.Tensor]] = []
        currently_hold = 0
        bucket: dict[str, torch.Tensor] = {}
        has_visual = any("visual." in key for key in state_dict.keys())
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
                device=self.bucket_device,
                dtype=self.bucket_dtype,
                non_blocking=True,
            )
            currently_hold += bucket[name].numel() * bucket[name].element_size()

            if currently_hold >= self.bucket_size:
                buckets.append(bucket)
                bucket = {}
                currently_hold = 0

        if bucket:
            buckets.append(bucket)

        assert buckets, "No parameters to sync"
        buckets[0]["total_buckets"] = torch.tensor(
            len(buckets), dtype=torch.int32, device=self.bucket_device
        )
        return buckets

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        del version
        buckets = self.divide_into_buckets(state_dict)
        for bucket in buckets:
            await send(bucket)

    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> None:
        bucket: dict[str, torch.Tensor] = await recv()
        total_buckets = int(bucket.pop("total_buckets").item())

        if self.load_instant:
            model.load_state_dict(bucket, strict=False)
        else:
            cpu_buffer: dict[str, torch.Tensor] = {}
            for key, value in bucket.items():
                cpu_buffer[key] = value.to("cpu", non_blocking=True)
        del bucket

        for _ in range(total_buckets - 1):
            bucket = await recv()
            if self.load_instant:
                model.load_state_dict(bucket, strict=False)
            else:
                for key, value in bucket.items():
                    cpu_buffer[key] = value.to("cpu", non_blocking=True)
            del bucket

        if not self.load_instant:
            model.load_state_dict(cpu_buffer, strict=False)
            del cpu_buffer

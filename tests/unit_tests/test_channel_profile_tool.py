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

import torch

from toolkits.channel_profile.profile_two_node_channel import (
    build_payload,
    parse_size,
    payload_num_bytes,
    resolve_global_gpu_rank,
)


def test_parse_size_units() -> None:
    assert parse_size("1KB") == 1024
    assert parse_size("2MB") == 2 * 1024 * 1024
    assert parse_size("1.5GB") == int(1.5 * 1024**3)


def test_resolve_global_gpu_rank() -> None:
    accelerator_ranks = [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert resolve_global_gpu_rank(accelerator_ranks, node_rank=0, local_gpu=2) == 2
    assert resolve_global_gpu_rank(accelerator_ranks, node_rank=1, local_gpu=1) == 5


def test_build_bytes_payload_uses_exact_size() -> None:
    payload = build_payload("bytes", 4097)
    assert isinstance(payload, bytes)
    assert payload_num_bytes(payload) == 4097


def test_build_cpu_tensor_payload_uses_exact_size() -> None:
    payload = build_payload("cpu_tensor", 8193)
    assert isinstance(payload, torch.Tensor)
    assert payload.device.type == "cpu"
    assert payload.dtype == torch.uint8
    assert payload_num_bytes(payload) == 8193

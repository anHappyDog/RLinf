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

import pytest
import torch

from toolkits.benchmark_trajectory_transfer import (
    distribution,
    make_workload,
    tensor_schema,
    total_bytes,
    workload_config,
)


def test_target_workload_matches_openpi_libero_shard_dimensions() -> None:
    config = workload_config("target")

    assert config.slots == 8
    assert config.chunk_steps == 48
    assert config.image_height == config.image_width == 256
    assert config.image_fields == 4


def test_smoke_workload_is_deterministic_contiguous_and_manifested() -> None:
    config = workload_config("smoke")
    first = make_workload(config, seed=4)
    second = make_workload(config, seed=4)
    schema = tensor_schema(first)

    assert first.keys() == second.keys()
    assert all(tensor.device.type == "cpu" for tensor in first.values())
    assert all(tensor.is_contiguous() for tensor in first.values())
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert total_bytes(first) == sum(item["bytes"] for item in schema.values())
    assert schema["image_0"]["shape"] == [3, 2, 64, 64, 3]


def test_distribution_reports_required_percentiles() -> None:
    result = distribution([0.001, 0.002, 0.003, 0.004, 0.005])

    assert result.count == 5
    assert result.p50_ms == 3.0
    assert result.p95_ms == pytest.approx(4.8)
    assert result.p99_ms == pytest.approx(4.96)
    assert result.p999_ms == pytest.approx(4.996)

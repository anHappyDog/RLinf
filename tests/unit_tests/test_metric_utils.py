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


import math

import pytest
import torch

from rlinf.utils.metric_utils import compute_evaluate_metrics


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


def test_compute_evaluate_metrics_reports_per_subtask_outcomes():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/subtask_id": torch.tensor([0, 1]),
                "env/subpool_id": torch.tensor([0, 1]),
                "env/success": torch.tensor([1.0, 0.0]),
                "env/subtask_timeout": torch.tensor([0.0, 1.0]),
            },
            {
                "env/subtask_id": torch.tensor([0, 1]),
                "env/subpool_id": torch.tensor([2, 1]),
                "env/success": torch.tensor([0.0, 1.0]),
                "env/subtask_timeout": torch.tensor([1.0, 0.0]),
            },
        ]
    )

    assert float(metrics["env/subtask/0/success"]) == pytest.approx(0.5)
    assert float(metrics["env/subtask/0/timeout"]) == pytest.approx(0.5)
    assert metrics["env/subtask/0/attempts"] == 2
    assert float(metrics["env/subtask/1/success"]) == pytest.approx(0.5)
    assert float(metrics["env/subtask/1/timeout"]) == pytest.approx(0.5)
    assert metrics["env/subtask/1/attempts"] == 2
    assert float(metrics["env/subtask/0/pool/0/success"]) == pytest.approx(1.0)
    assert float(metrics["env/subtask/0/pool/2/timeout"]) == pytest.approx(1.0)
    assert metrics["env/subtask/1/pool/1/attempts"] == 2


def test_compute_evaluate_metrics_rejects_misaligned_subtask_metrics():
    with pytest.raises(ValueError, match="one subtask id"):
        compute_evaluate_metrics(
            [
                {
                    "subtask_id": torch.tensor([0, 1]),
                    "success": torch.tensor([1.0]),
                    "subtask_timeout": torch.tensor([0.0, 1.0]),
                }
            ]
        )

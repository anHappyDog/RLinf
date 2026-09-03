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

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import torch

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import R1ManifestEntry
from toolkits.mem.action_preservation import action_drift_metrics
from toolkits.mem.r0_overfit import R0Sample
from toolkits.mem.r1_counterfactual import mismatch_inputs
from toolkits.mem.r1_train import (
    TaskBalancedSampler,
    _balanced_subset,
    _generation_metrics,
)


def _entry(task_index: int, episode_index: int) -> R1ManifestEntry:
    return R1ManifestEntry(
        split="train",
        task_index=task_index,
        task_name=f"task_{task_index}",
        task=f"do task {task_index}",
        episode_index=episode_index,
        frame_index=0,
        primitive_index=0,
        subtask="press button",
        target_source="primitive",
    )


def test_balanced_subset_and_sampler_do_not_follow_task_frequency():
    entries = [*(_entry(0, i) for i in range(20)), *(_entry(1, 100 + i) for i in range(4))]

    subset = _balanced_subset(entries, sample_count=8, seed=7)
    sampled_indices = list(TaskBalancedSampler(entries, seed=7))

    assert Counter(entry.task_index for entry in subset) == {0: 4, 1: 4}
    sampled_tasks = Counter(entries[index].task_index for index in sampled_indices)
    assert abs(sampled_tasks[0] - sampled_tasks[1]) <= 1


def test_generation_metrics_report_task_and_semantic_components():
    predictions = [
        {
            "task_index": 0,
            "target": "pick up radio from coffee table",
            "prediction": "pick up radio from coffee table",
            "exact_match": True,
            "ended_with_eos": True,
        },
        {
            "task_index": 1,
            "target": "place radio on coffee table",
            "prediction": "place radio on floor",
            "exact_match": False,
            "ended_with_eos": True,
        },
    ]

    metrics = _generation_metrics(predictions)

    assert metrics["exact_match"] == 0.5
    assert metrics["macro_task_exact_match"] == 0.5
    assert metrics["verb_accuracy"] == 1.0
    assert metrics["object_accuracy"] == 1.0
    assert metrics["destination_accuracy"] == 0.5
    assert metrics["step_count_accuracy"] == 1.0


def test_action_drift_metrics_are_zero_for_identical_outputs():
    actions = torch.tensor([[[1.0, -2.0], [0.5, 0.25]]])

    metrics = action_drift_metrics(actions, actions.clone())

    assert metrics["rmse"] == 0.0
    assert metrics["relative_rmse"] == 0.0
    assert metrics["first_action_relative_rmse"] == 0.0
    assert abs(metrics["mean_cosine_similarity"] - 1.0) < 1e-6


def test_counterfactual_inputs_come_from_another_target():
    entries = [
        dataclasses.replace(_entry(0, 0), subtask="press radio"),
        dataclasses.replace(_entry(0, 1), subtask="pick up radio"),
    ]
    samples = [
        R0Sample(
            manifest=entry,
            images={"image": np.asarray([index])},
            train_tokens=np.asarray([index]),
            train_input_mask=np.asarray([True]),
            train_ar_mask=np.asarray([False]),
            train_loss_mask=np.asarray([False]),
            prefix_tokens=np.asarray([index]),
            prefix_input_mask=np.asarray([True]),
            prefix_ar_mask=np.asarray([False]),
        )
        for index, entry in enumerate(entries)
    ]

    mismatched_images = mismatch_inputs(samples, modality="images")
    mismatched_states = mismatch_inputs(samples, modality="state")

    assert mismatched_images[0].images["image"].item() == 1
    assert mismatched_states[0].prefix_tokens.item() == 1
    assert mismatched_images[0].manifest.subtask == "press radio"

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

from types import SimpleNamespace

import pytest
import torch

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    PrimitivePromptInterval,
)


def test_behavior_sft_sets_oracle_primitive_prompt():
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.prompt_source = "primitive"
    dataset.meta = SimpleNamespace(fps=30)
    dataset.prompt_action_horizon = 4
    dataset.primitive_prompt_intervals = {
        10: [
            PrimitivePromptInterval(
                start_frame=100,
                end_frame=200,
                primitive_index=2,
                subtask="press radio",
                target_source="primitive",
            )
        ]
    }
    dataset._get_fine_grained_task = lambda item: "Turn on the radio."
    item = {
        "episode_index": torch.tensor(10),
        "task_index": torch.tensor(0),
        "timestamp": torch.tensor(4.0),
    }

    dataset._set_prompt(item)

    assert item["task"] == "Turn on the radio."
    assert item["prompt"] == "press radio"
    assert item["primitive_label"] == "press radio"
    assert item["primitive_index"] == 2


def test_behavior_sft_rejects_cross_boundary_primitive_prompt():
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.prompt_source = "primitive"
    dataset.meta = SimpleNamespace(fps=30)
    dataset.prompt_action_horizon = 32
    dataset.primitive_prompt_intervals = {
        10: [
            PrimitivePromptInterval(
                start_frame=100,
                end_frame=200,
                primitive_index=2,
                subtask="press radio",
                target_source="primitive",
            )
        ]
    }
    dataset._get_fine_grained_task = lambda item: "Turn on the radio."
    item = {
        "episode_index": torch.tensor(10),
        "task_index": torch.tensor(0),
        "timestamp": torch.tensor(190 / 30),
    }

    with pytest.raises(ValueError, match="crosses a boundary"):
        dataset._set_prompt(item)


def test_behavior_sft_mixes_task_and_primitive_prompts_reproducibly():
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.prompt_source = "mixed"
    dataset.primitive_prompt_probability = 0.5
    dataset.seed = 42
    dataset.meta = SimpleNamespace(fps=30)
    dataset.prompt_action_horizon = 4
    dataset.primitive_prompt_intervals = {
        10: [
            PrimitivePromptInterval(
                start_frame=100,
                end_frame=300,
                primitive_index=2,
                subtask="press radio",
                target_source="primitive",
            )
        ]
    }
    dataset._get_fine_grained_task = lambda item: "Turn on the radio."

    def prompt_source(frame_index: int) -> str:
        item = {
            "episode_index": torch.tensor(10),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(frame_index / 30),
        }
        dataset._set_prompt(item)
        expected_prompt = (
            "press radio"
            if item["prompt_source"] == "primitive"
            else "Turn on the radio."
        )
        assert item["prompt"] == expected_prompt
        return item["prompt_source"]

    first_pass = [prompt_source(frame_index) for frame_index in range(100, 180)]
    second_pass = [prompt_source(frame_index) for frame_index in range(100, 180)]

    assert first_pass == second_pass
    assert set(first_pass) == {"task", "primitive"}


def test_behavior_sft_mixed_mode_keeps_task_prompts_boundary_safe():
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.prompt_source = "mixed"
    dataset.primitive_prompt_probability = 0.0
    dataset.mixed_boundary_fallback_to_task = False
    dataset.seed = 42
    dataset.meta = SimpleNamespace(fps=30)
    dataset.prompt_action_horizon = 32
    dataset.primitive_prompt_intervals = {
        10: [
            PrimitivePromptInterval(
                start_frame=100,
                end_frame=200,
                primitive_index=2,
                subtask="press radio",
                target_source="primitive",
            )
        ]
    }
    dataset._get_fine_grained_task = lambda item: "Turn on the radio."
    item = {
        "episode_index": torch.tensor(10),
        "task_index": torch.tensor(0),
        "timestamp": torch.tensor(190 / 30),
    }

    with pytest.raises(ValueError, match="crosses a boundary"):
        dataset._set_prompt(item)


def test_behavior_sft_mixed_mode_can_keep_boundary_transition_as_task():
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.prompt_source = "mixed"
    dataset.primitive_prompt_probability = 1.0
    dataset.mixed_boundary_fallback_to_task = True
    dataset.seed = 42
    dataset.meta = SimpleNamespace(fps=30)
    dataset.prompt_action_horizon = 32
    dataset.primitive_prompt_intervals = {
        10: [
            PrimitivePromptInterval(
                start_frame=100,
                end_frame=200,
                primitive_index=2,
                subtask="press radio",
                target_source="primitive",
            )
        ]
    }
    dataset._get_fine_grained_task = lambda item: "Turn on the radio."
    item = {
        "episode_index": torch.tensor(10),
        "task_index": torch.tensor(0),
        "timestamp": torch.tensor(190 / 30),
    }

    dataset._set_prompt(item)

    assert item["prompt_source"] == "task"
    assert item["prompt"] == "Turn on the radio."
    assert "primitive_label" not in item


def test_behavior_sft_non_shuffled_stream_starts_first_worker_chunk():
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset._dist_rank = 0
    dataset._dist_world_size = 1
    dataset.chunks = [(10, 20, 10), (30, 40, 30)]
    dataset.seed = 42
    dataset.shuffle = False

    dataset._select_streaming_chunk()

    assert dataset._active_chunks == dataset.chunks
    assert dataset.current_streaming_chunk_idx == 0
    assert dataset.current_streaming_frame_idx == 10

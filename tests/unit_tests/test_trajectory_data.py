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

from dataclasses import fields

import pytest
import torch

from rlinf.data.trajectory import (
    EnvResult,
    PolicyInput,
    PolicyOutput,
    RewardResult,
    RolloutResult,
    TrajectoryData,
    ValueRequest,
    ValueResult,
)
from rlinf.models.embodiment.openpi.forward_inputs import (
    OpenPILiberoForwardInputs,
)


def coordinates(batch_size: int = 2) -> dict:
    return {
        "global_step": 3,
        "rollout_epoch": 1,
        "chunk_step": 7,
        "slot_ids": tuple(range(10, 10 + batch_size)),
    }


def observations(batch_size: int = 2) -> dict:
    return {
        "main_images": torch.zeros(batch_size, 256, 256, 3, dtype=torch.uint8),
        "states": torch.zeros(batch_size, 8),
        "task_descriptions": [f"task {index}" for index in range(batch_size)],
        "wrist_images": None,
    }


def openpi_forward_inputs(batch_size: int = 2) -> OpenPILiberoForwardInputs:
    return OpenPILiberoForwardInputs(
        chains=torch.zeros(batch_size, 5, 5, 32),
        denoise_inds=torch.zeros(batch_size, 4, dtype=torch.int64),
        tokenized_prompt=torch.zeros(batch_size, 16, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(batch_size, 16, dtype=torch.bool),
        action=torch.zeros(batch_size, 35),
        model_action=torch.zeros(batch_size, 50 * 32),
        image=torch.zeros(batch_size, 8, 8, 3, dtype=torch.uint8),
        wrist_image=torch.zeros(batch_size, 8, 8, 3, dtype=torch.uint8),
        state=torch.zeros(batch_size, 8),
    )


def test_trajectory_data_keeps_slot_order() -> None:
    data = TrajectoryData(
        global_step=0,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(8, 3),
    )

    assert data.batch_size == 2
    assert data.slot_ids == (8, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_step", -1),
        ("rollout_epoch", True),
        ("chunk_step", 1.5),
    ],
)
def test_trajectory_data_rejects_invalid_coordinates(field: str, value) -> None:
    kwargs = coordinates()
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        TrajectoryData(**kwargs)


@pytest.mark.parametrize("slot_ids", [(), (1, 1), (-1, 2), [1, 2]])
def test_trajectory_data_rejects_invalid_slot_ids(slot_ids) -> None:
    with pytest.raises(ValueError, match="slot_ids"):
        TrajectoryData(**{**coordinates(), "slot_ids": slot_ids})


def test_policy_input_contains_only_live_inference_data() -> None:
    policy_input = PolicyInput(
        **coordinates(),
        observations=observations(),
        rlt_switch_flags=torch.zeros(2, 5, dtype=torch.bool),
        intervene_requested=torch.zeros(2, 5, dtype=torch.bool),
    )

    assert policy_input.batch_size == 2
    assert {field.name for field in fields(PolicyInput)} == {
        "global_step",
        "rollout_epoch",
        "chunk_step",
        "slot_ids",
        "observations",
        "rlt_switch_flags",
        "intervene_requested",
    }


def test_policy_input_requires_boolean_controls() -> None:
    with pytest.raises(TypeError, match="intervene_requested"):
        PolicyInput(
            **coordinates(),
            observations=observations(),
            intervene_requested=torch.zeros(2, 5),
        )


def test_policy_output_contains_only_executable_actions() -> None:
    output = PolicyOutput(**coordinates(), actions=torch.zeros(2, 35))

    assert output.actions.shape == (2, 35)
    assert {field.name for field in fields(PolicyOutput)} == {
        "global_step",
        "rollout_epoch",
        "chunk_step",
        "slot_ids",
        "actions",
    }


def test_env_result_accepts_macro_transition_data() -> None:
    result = EnvResult(
        **coordinates(),
        rewards=torch.zeros(2, 5),
        dones=torch.zeros(2, 5, dtype=torch.bool),
        terminations=torch.zeros(2, 5, dtype=torch.bool),
        truncations=torch.zeros(2, 5, dtype=torch.bool),
        observations=observations(),
        next_observations=observations(),
    )

    assert result.batch_size == 2


def test_env_result_requires_aligned_transition_shapes() -> None:
    with pytest.raises(ValueError, match="rewards must have shape"):
        EnvResult(
            **coordinates(),
            rewards=torch.zeros(2, 1),
            dones=torch.zeros(2, 5, dtype=torch.bool),
            terminations=torch.zeros(2, 5, dtype=torch.bool),
            truncations=torch.zeros(2, 5, dtype=torch.bool),
        )


def test_env_result_requires_boolean_masks() -> None:
    with pytest.raises(TypeError, match="dones must have dtype torch.bool"):
        EnvResult(
            **coordinates(),
            rewards=torch.zeros(2, 5),
            dones=torch.zeros(2, 5),
            terminations=torch.zeros(2, 5, dtype=torch.bool),
            truncations=torch.zeros(2, 5, dtype=torch.bool),
        )


def test_rollout_result_accepts_typed_forward_inputs() -> None:
    result = RolloutResult(
        **coordinates(),
        actions=torch.zeros(2, 35),
        forward_inputs=openpi_forward_inputs(),
        prev_logprobs=torch.zeros(2, 5, 7),
        state_values=torch.zeros(2, 1),
    )

    assert result.state_values.shape == (2, 1)


def test_rollout_result_rejects_untyped_forward_inputs() -> None:
    with pytest.raises(TypeError, match="ForwardInputs"):
        RolloutResult(
            **coordinates(),
            actions=torch.zeros(2, 35),
            forward_inputs={"chains": torch.zeros(1, 4, 5, 32)},
        )


def test_rollout_result_rejects_misaligned_forward_inputs() -> None:
    with pytest.raises(ValueError, match="forward_inputs"):
        RolloutResult(
            **coordinates(),
            actions=torch.zeros(2, 35),
            forward_inputs=openpi_forward_inputs(batch_size=1),
        )


def test_history_reward_requires_lengths() -> None:
    with pytest.raises(ValueError, match="history_lengths is required"):
        RewardResult(
            **coordinates(),
            rewards=torch.zeros(2, 1),
            mode="history_buffer",
        )


def test_history_reward_accepts_per_slot_lengths() -> None:
    result = RewardResult(
        **coordinates(),
        rewards=torch.zeros(2, 1),
        mode="history_buffer",
        history_lengths=torch.tensor([3, 1]),
    )

    assert result.history_lengths.tolist() == [3, 1]


def test_non_history_reward_rejects_history_lengths() -> None:
    with pytest.raises(ValueError, match="only valid for history_buffer"):
        RewardResult(
            **coordinates(),
            rewards=torch.zeros(2, 1),
            history_lengths=torch.tensor([1, 1]),
        )


@pytest.mark.parametrize("kind", ["timeout", "tail"])
def test_value_request_accepts_each_boundary_kind(kind: str) -> None:
    request = ValueRequest(
        **coordinates(),
        kind=kind,
        observations=observations(),
    )

    assert request.kind == kind


def test_value_request_validates_observation_batch() -> None:
    with pytest.raises(ValueError, match="observations.states"):
        ValueRequest(
            **coordinates(),
            kind="timeout",
            observations={"states": torch.zeros(1, 8)},
        )


def test_value_result_requires_scalar_value_per_slot() -> None:
    with pytest.raises(ValueError, match=r"shape \[batch_size, 1\]"):
        ValueResult(
            **coordinates(),
            kind="tail",
            values=torch.zeros(2, 5),
        )


def test_value_result_accepts_model_versions() -> None:
    result = ValueResult(
        **coordinates(),
        kind="timeout",
        values=torch.zeros(2, 1),
        versions=torch.full((2, 1), 4),
    )

    assert result.versions.tolist() == [[4], [4]]

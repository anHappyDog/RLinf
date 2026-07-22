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

from rlinf.algorithms.embodied_gae import (
    compose_embodied_rewards,
    compute_embodied_gae,
)
from rlinf.data.trajectory import EnvResult, RewardResult, RolloutResult, ValueResult
from rlinf.workers.trajectory.storage import StorageConfig, TrajectoryStorage


def _gae_inputs(
    rewards: list[list[float]],
    values: list[list[float]],
    *,
    terminations: list[list[bool]],
    truncations: list[list[bool]],
    timeout_values: list[list[float]],
    tail_values: list[float],
) -> dict[str, torch.Tensor]:
    reward_tensor = torch.tensor([rewards], dtype=torch.float32)
    value_tensor = torch.tensor([values], dtype=torch.float32).unsqueeze(-1)
    termination_tensor = torch.tensor([terminations], dtype=torch.bool).unsqueeze(-1)
    truncation_tensor = torch.tensor([truncations], dtype=torch.bool).unsqueeze(-1)
    done_tensor = termination_tensor | truncation_tensor
    timeout_mask = truncation_tensor.squeeze(-1) & ~termination_tensor.squeeze(-1)
    tail_mask = ~done_tensor[:, -1, :, 0]
    return {
        "rewards": reward_tensor,
        "state_values": value_tensor,
        "dones": done_tensor,
        "terminations": termination_tensor,
        "truncations": truncation_tensor,
        "timeout_values": torch.tensor([timeout_values], dtype=torch.float32).unsqueeze(
            -1
        ),
        "timeout_mask": timeout_mask,
        "tail_values": torch.tensor([tail_values], dtype=torch.float32).unsqueeze(-1),
        "tail_mask": tail_mask,
    }


def test_true_termination_has_no_boundary_bootstrap() -> None:
    inputs = _gae_inputs(
        [[1.0], [2.0]],
        [[0.5], [0.7]],
        terminations=[[False], [True]],
        truncations=[[False], [False]],
        timeout_values=[[99.0], [99.0]],
        tail_values=[99.0],
    )

    advantages, returns = compute_embodied_gae(**inputs, gamma=0.9, gae_lambda=0.8)

    assert advantages[0, :, 0, 0].tolist() == pytest.approx([2.066, 1.3])
    assert returns[0, :, 0, 0].tolist() == pytest.approx([2.566, 2.0])


def test_timeout_uses_terminal_value_and_cuts_reset_episode_recursion() -> None:
    inputs = _gae_inputs(
        [[1.0], [2.0], [3.0]],
        [[0.5], [0.6], [100.0]],
        terminations=[[False], [False], [False]],
        truncations=[[False], [True], [False]],
        timeout_values=[[0.0], [1.5], [0.0]],
        tail_values=[100.0],
    )

    advantages, returns = compute_embodied_gae(**inputs, gamma=0.9, gae_lambda=0.8)

    assert advantages[0, 1, 0, 0].item() == pytest.approx(2.75)
    assert advantages[0, 0, 0, 0].item() == pytest.approx(3.02)
    assert returns[0, 1, 0, 0].item() == pytest.approx(3.35)
    # V(reset_obs)=100 is state_values[t+1], but must not enter the timeout delta.
    assert advantages[0, 1, 0, 0].item() != pytest.approx(0.9 * 100 + 2 - 0.6)


def test_alive_segment_cutoff_uses_tail_value() -> None:
    inputs = _gae_inputs(
        [[0.0], [0.0]],
        [[1.0], [2.0]],
        terminations=[[False], [False]],
        truncations=[[False], [False]],
        timeout_values=[[0.0], [0.0]],
        tail_values=[3.0],
    )

    advantages, returns = compute_embodied_gae(**inputs, gamma=0.9, gae_lambda=1.0)

    assert advantages[0, :, 0, 0].tolist() == pytest.approx([1.43, 0.7])
    assert returns[0, :, 0, 0].tolist() == pytest.approx([2.43, 2.7])


def test_mixed_batch_distinguishes_termination_timeout_and_alive_tail() -> None:
    inputs = _gae_inputs(
        [[0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0]],
        terminations=[[True, False, False]],
        truncations=[[False, True, False]],
        timeout_values=[[0.0, 4.0, 0.0]],
        tail_values=[0.0, 0.0, 5.0],
    )

    advantages, returns = compute_embodied_gae(**inputs, gamma=0.5, gae_lambda=0.9)

    assert advantages[0, 0, :, 0] == pytest.approx([-1.0, 1.0, 1.5])
    assert returns[0, 0, :, 0] == pytest.approx([0.0, 2.0, 2.5])


def test_rollout_epochs_are_independent_segments() -> None:
    rewards = torch.zeros(2, 1, 1)
    state_values = torch.ones(2, 1, 1, 1)
    masks = torch.zeros(2, 1, 1, 1, dtype=torch.bool)
    timeout_values = torch.zeros_like(state_values)
    timeout_mask = torch.zeros(2, 1, 1, dtype=torch.bool)
    tail_values = torch.tensor([[[2.0]], [[10.0]]])
    tail_mask = torch.ones(2, 1, dtype=torch.bool)

    advantages, _ = compute_embodied_gae(
        rewards,
        state_values,
        masks,
        masks,
        masks,
        timeout_values,
        timeout_mask,
        tail_values,
        tail_mask,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages[:, 0, 0, 0] == pytest.approx([1.0, 9.0])


def test_normalization_uses_macro_loss_mask_and_does_not_change_returns() -> None:
    inputs = _gae_inputs(
        [[0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0]],
        terminations=[[True, False, False]],
        truncations=[[False, True, False]],
        timeout_values=[[0.0, 4.0, 0.0]],
        tail_values=[0.0, 0.0, 5.0],
    )
    loss_mask = torch.tensor([[[[True, False], [True, True], [False, False]]]])
    _, raw_returns = compute_embodied_gae(**inputs, gamma=0.5, gae_lambda=0.9)

    normalized, returns = compute_embodied_gae(
        **inputs,
        gamma=0.5,
        gae_lambda=0.9,
        normalize_advantages=True,
        loss_mask=loss_mask,
    )

    valid = normalized.squeeze(-1)[loss_mask.any(dim=-1)]
    assert valid.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert valid.std().item() == pytest.approx(1.0, abs=1e-4)
    assert torch.equal(returns, raw_returns)


def test_reward_sources_are_reduced_before_weighting_and_inputs_stay_raw() -> None:
    env_rewards = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    external_rewards = torch.tensor([[[[10.0]], [[float("nan")]]]])
    external_mask = torch.tensor([[[True], [False]]])
    original_env = env_rewards.clone()
    original_external = external_rewards.clone()

    effective = compose_embodied_rewards(
        env_rewards,
        external_rewards=external_rewards,
        external_reward_mask=external_mask,
        env_weight=2.0,
        external_weight=3.0,
    )

    assert effective[0, :, 0].tolist() == pytest.approx([36.0, 14.0])
    assert torch.equal(env_rewards, original_env)
    torch.testing.assert_close(external_rewards, original_external, equal_nan=True)


def test_storage_rewards_remain_raw_and_bootstrap_is_applied_once() -> None:
    config = StorageConfig(
        global_step=2,
        rollout_epochs=1,
        chunk_steps=1,
        slot_ids=(3,),
        rollout_fields=frozenset({"state_values"}),
        reward_mode="per_step",
        reward_steps=(0,),
        boundary_values=True,
    )
    storage = TrajectoryStorage(config)
    storage.write(
        EnvResult(
            global_step=2,
            rollout_epoch=0,
            chunk_step=0,
            slot_ids=(3,),
            rewards=torch.tensor([[1.0, 2.0]]),
            dones=torch.tensor([[False, False]]),
            terminations=torch.tensor([[False, False]]),
            truncations=torch.tensor([[False, False]]),
        )
    )
    storage.write(
        RolloutResult(
            global_step=2,
            rollout_epoch=0,
            chunk_step=0,
            slot_ids=(3,),
            actions=torch.zeros(1, 7),
            state_values=torch.tensor([[1.0]]),
        )
    )
    storage.write(
        RewardResult(
            global_step=2,
            rollout_epoch=0,
            chunk_step=0,
            slot_ids=(3,),
            rewards=torch.tensor([[3.0]]),
        )
    )
    storage.write(
        ValueResult(
            global_step=2,
            rollout_epoch=0,
            chunk_step=1,
            slot_ids=(3,),
            kind="tail",
            values=torch.tensor([[2.0]]),
        )
    )
    batch = storage.export()
    raw_env_rewards = batch.env_rewards.clone()
    raw_external_rewards = batch.external_rewards.clone()
    effective_rewards = compose_embodied_rewards(
        batch.env_rewards,
        external_rewards=batch.external_rewards,
        external_reward_mask=batch.reward_mask,
    )

    advantages, returns = compute_embodied_gae(
        effective_rewards,
        batch.state_values,
        batch.dones,
        batch.terminations,
        batch.truncations,
        batch.timeout_values,
        batch.timeout_mask,
        batch.tail_values,
        batch.tail_mask,
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert effective_rewards.item() == 6.0
    assert advantages.item() == 6.0
    assert returns.item() == 7.0
    assert torch.equal(batch.env_rewards, raw_env_rewards)
    assert torch.equal(batch.external_rewards, raw_external_rewards)
    assert torch.equal(storage.export().env_rewards, raw_env_rewards)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("timeout_mask", "timeout_mask must exactly match"),
        ("tail_mask", "tail_mask must exactly match"),
        ("dones", "dones must equal"),
    ],
)
def test_inconsistent_boundary_masks_fail(field: str, message: str) -> None:
    inputs = _gae_inputs(
        [[0.0]],
        [[1.0]],
        terminations=[[False]],
        truncations=[[True]],
        timeout_values=[[2.0]],
        tail_values=[0.0],
    )
    inputs[field] = ~inputs[field]

    with pytest.raises(ValueError, match=message):
        compute_embodied_gae(**inputs, gamma=0.9, gae_lambda=0.95)


def test_termination_wins_when_transition_also_reaches_time_limit() -> None:
    inputs = _gae_inputs(
        [[0.0]],
        [[1.0]],
        terminations=[[True]],
        truncations=[[False]],
        timeout_values=[[2.0]],
        tail_values=[0.0],
    )
    inputs["truncations"][:] = True

    advantages, returns = compute_embodied_gae(**inputs, gamma=0.9, gae_lambda=0.95)

    assert advantages.item() == pytest.approx(-1.0)
    assert returns.item() == pytest.approx(0.0)

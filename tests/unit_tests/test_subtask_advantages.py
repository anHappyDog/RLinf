import pytest
import torch

import rlinf.algorithms  # noqa: F401 - populate algorithm registries
from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.algorithms.subtask import (
    align_subtask_ids,
    balanced_subtask_weights,
    compute_subtask_gae,
    discounted_chunk_rewards,
    taskwise_normalize,
)
from rlinf.algorithms.utils import preprocess_loss_inputs


def test_discounted_chunk_rewards_use_only_executed_prefix():
    rewards = torch.tensor([[[1.0, 2.0, 100.0, 100.0]]])
    mask = torch.tensor([[[True, True, False, False]]])
    macro_rewards, discounts = discounted_chunk_rewards(rewards, mask, gamma=0.5)
    torch.testing.assert_close(macro_rewards, torch.tensor([[2.0]]))
    torch.testing.assert_close(discounts, torch.tensor([[0.25]]))


def test_discounted_chunk_rewards_reject_non_prefix_mask():
    rewards = torch.ones(1, 1, 3)
    mask = torch.tensor([[[True, False, True]]])
    with pytest.raises(ValueError, match="prefix-contiguous"):
        discounted_chunk_rewards(rewards, mask, gamma=0.99)


def test_subtask_gae_stops_at_done_and_subtask_boundary():
    rewards = torch.tensor([[1.0], [10.0], [100.0]])
    discounts = torch.ones_like(rewards)
    dones = torch.tensor([[True], [False], [False]])
    values = torch.zeros(4, 1)
    subtask_ids = torch.tensor([[0], [1], [2]])
    valid = torch.ones_like(dones)
    advantages, returns = compute_subtask_gae(
        rewards,
        discounts,
        dones,
        values,
        subtask_ids,
        valid,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_std_floor=0.1,
    )
    torch.testing.assert_close(advantages, rewards)
    torch.testing.assert_close(returns, rewards)


def test_taskwise_normalization_prevents_reward_scale_dominance():
    advantages = torch.tensor([[1.0, 100.0], [3.0, 300.0]])
    task_ids = torch.tensor([[0, 1], [0, 1]])
    valid = torch.ones_like(task_ids, dtype=torch.bool)
    normalized = taskwise_normalize(advantages, task_ids, valid, std_floor=1e-4)
    torch.testing.assert_close(normalized[:, 0], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(normalized[:, 1], torch.tensor([-1.0, 1.0]))


def test_taskwise_normalization_preserves_single_transition_signal():
    advantages = torch.tensor([[5.0, -2.0]])
    task_ids = torch.tensor([[0, 1]])
    valid = torch.ones_like(task_ids, dtype=torch.bool)
    normalized = taskwise_normalize(advantages, task_ids, valid, std_floor=0.1)
    torch.testing.assert_close(normalized, torch.tensor([[1.0, -1.0]]))


def test_balanced_weights_give_each_subtask_equal_total_weight():
    task_ids = torch.tensor([[0, 0, 1], [0, 0, 1]])
    valid = torch.tensor([[True, True, True], [True, False, True]])
    weights = balanced_subtask_weights(task_ids, valid)
    torch.testing.assert_close(
        weights[valid & (task_ids == 0)].sum(), torch.tensor(3.0)
    )
    torch.testing.assert_close(
        weights[valid & (task_ids == 1)].sum(), torch.tensor(3.0)
    )
    assert weights[~valid].eq(0).all()


@pytest.mark.parametrize("with_trailing_dim", [False, True])
def test_align_subtask_ids_preserves_singleton_batch(with_trailing_dim):
    reference = torch.ones(3, 1, dtype=torch.bool)
    task_ids = torch.tensor([[0], [0], [1]])
    if with_trailing_dim:
        task_ids = task_ids.unsqueeze(-1)

    aligned = align_subtask_ids(task_ids, reference)

    assert aligned.shape == (3, 1)
    torch.testing.assert_close(aligned, torch.tensor([[0], [0], [1]]))


def test_align_subtask_ids_rejects_missing_batch_axis():
    with pytest.raises(ValueError, match="transition reference shape"):
        align_subtask_ids(torch.tensor([0, 1, 2]), torch.ones(3, 1))


def test_embodied_registry_preserves_macro_transition_shape():
    rewards = torch.ones(2, 2, 4)
    executed = torch.tensor(
        [
            [[True, True, False, False], [True, True, True, True]],
            [[True, False, False, False], [True, True, False, False]],
        ]
    )
    dones = torch.zeros(3, 2, 4, dtype=torch.bool)
    dones[1, 0, 1] = True
    values = torch.zeros(3, 2, 1)
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="subtask_gae",
        reward_type="subtask_chunk_level",
        rewards=rewards,
        dones=dones,
        values=values,
        subtask_ids=torch.tensor([[0, 1], [0, 1]]),
        executed_action_mask=executed,
        loss_mask=executed,
        gamma=0.5,
        gae_lambda=1.0,
        normalize_advantages=False,
    )
    assert result["advantages"].shape == (2, 2, 1)
    assert result["returns"].shape == (2, 2, 1)
    torch.testing.assert_close(result["returns"][0, 0, 0], torch.tensor(1.5))


def test_embodied_registry_preserves_singleton_batch_axis():
    rewards = torch.ones(2, 1, 4)
    executed = torch.ones(2, 1, 4, dtype=torch.bool)
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="subtask_gae",
        reward_type="subtask_chunk_level",
        rewards=rewards,
        dones=torch.zeros(3, 1, 4, dtype=torch.bool),
        values=torch.zeros(3, 1, 1),
        subtask_ids=torch.zeros(2, 1, dtype=torch.long),
        executed_action_mask=executed,
        loss_mask=executed,
        gamma=0.5,
        gae_lambda=1.0,
        normalize_advantages=False,
    )

    assert result["advantages"].shape == (2, 1, 1)
    assert result["returns"].shape == (2, 1, 1)


def test_chunk_logprob_excludes_unexecuted_actions():
    inputs = preprocess_loss_inputs(
        logprobs=torch.ones(1, 4, 2),
        old_logprobs=torch.zeros(1, 4, 2),
        advantages=torch.ones(1),
        logprob_type="chunk_level",
        reward_type="subtask_chunk_level",
        single_action_dim=2,
        executed_action_mask=torch.tensor([[True, True, False, False]]),
        loss_mask=torch.ones(1, dtype=torch.bool),
    )
    torch.testing.assert_close(inputs["logprobs"], torch.tensor([4.0]))
    torch.testing.assert_close(inputs["old_logprobs"], torch.tensor([0.0]))

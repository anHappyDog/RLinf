import pytest
import torch

from rlinf.algorithms.subtask import (
    align_subtask_ids,
    compute_subtask_gae,
    discounted_chunk_rewards,
)
from toolkits.b1k_grounded.analyze_critic_targets import analyze_critic_targets


def test_analyze_critic_targets_matches_monte_carlo_when_lambda_is_one():
    time_size = 3
    batch_size = 2
    action_horizon = 2
    rewards = torch.zeros(time_size, batch_size, action_horizon)
    rewards[-1, 0, -1] = 1.0
    executed_action_mask = torch.ones_like(rewards, dtype=torch.bool)
    dones = torch.zeros(time_size + 1, batch_size, action_horizon, dtype=torch.bool)
    dones[-1] = True
    values = torch.zeros(time_size + 1, batch_size, 1)
    loss_mask = torch.ones(time_size, batch_size, 1, dtype=torch.bool)
    subtask_ids = torch.ones(time_size, batch_size, dtype=torch.int64)

    macro_rewards, discounts = discounted_chunk_rewards(
        rewards, executed_action_mask, gamma=0.9
    )
    aligned_subtask_ids = align_subtask_ids(subtask_ids, macro_rewards)
    raw_advantages, returns = compute_subtask_gae(
        macro_rewards,
        discounts,
        dones[1:].any(dim=-1),
        values.squeeze(-1),
        aligned_subtask_ids,
        loss_mask.squeeze(-1),
        gae_lambda=1.0,
        normalize_advantages=True,
        advantage_std_floor=0.1,
    )
    artifact = {
        "metadata": {
            "global_step": 4,
            "trajectory_count": batch_size,
            "source_shards": [
                {
                    "gamma": 0.9,
                    "gae_lambda": 1.0,
                    "local_trajectory_count": batch_size,
                }
            ],
        },
        "trajectory_outcomes": torch.tensor([True, False]),
        "batch": {
            "returns": returns.unsqueeze(-1),
            "advantages": raw_advantages.unsqueeze(-1),
            "prev_values": values,
            "rewards": rewards,
            "dones": dones,
            "loss_mask": loss_mask,
            "executed_action_mask": executed_action_mask,
            "subtask_ids": subtask_ids,
        },
    }

    result = analyze_critic_targets(artifact)

    assert result["consistency"]["stored_td_return_max_abs_error"] == 0.0
    assert result["consistency"]["stored_normalized_advantage_max_abs_error"] == 0.0
    assert result["metadata"]["initial_transition_count"] == batch_size
    assert result["targets"]["td_minus_monte_carlo"]["max"] == 0.0
    current = result["advantages"]["current_td_vs_monte_carlo"]
    assert current["sign_agreement"] == 1.0
    assert current["cosine_similarity"] == pytest.approx(1.0)

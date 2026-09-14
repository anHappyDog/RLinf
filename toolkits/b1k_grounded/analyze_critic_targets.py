#!/usr/bin/env python3
"""Compare bootstrapped critic targets with Monte-Carlo returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rlinf.algorithms.subtask import (
    align_subtask_ids,
    compute_subtask_gae,
    discounted_chunk_rewards,
    taskwise_normalize,
)


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.float()[mask]


def _distribution(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = _masked_values(values, mask)
    quantiles = torch.quantile(selected, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "min": float(selected.min()),
        "mean": float(selected.mean()),
        "std": float(selected.std(correction=0)),
        "max": float(selected.max()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator <= 1e-12:
        return float("nan")
    return float((left * right).sum() / denominator)


def _explained_variance(target: torch.Tensor, prediction: torch.Tensor) -> float:
    variance = target.float().var(correction=0)
    if variance <= 1e-12:
        return float("nan")
    residual_variance = (target.float() - prediction.float()).var(correction=0)
    return float(1.0 - residual_variance / variance)


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    prediction = _masked_values(prediction, mask)
    target = _masked_values(target, mask)
    error = prediction - target
    return {
        "explained_variance": _explained_variance(target, prediction),
        "correlation": _correlation(target, prediction),
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "mean_error": float(error.mean()),
    }


def _advantage_comparison(
    td_advantages: torch.Tensor,
    mc_advantages: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    td_values = _masked_values(td_advantages, mask)
    mc_values = _masked_values(mc_advantages, mask)
    denominator = td_values.norm() * mc_values.norm()
    cosine = (
        float(torch.dot(td_values, mc_values) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    return {
        "cosine_similarity": cosine,
        "correlation": _correlation(td_values, mc_values),
        "sign_agreement": float((td_values.sign() == mc_values.sign()).float().mean()),
        "td_positive_mc_negative": float(
            ((td_values > 0) & (mc_values < 0)).float().mean()
        ),
        "td_negative_mc_positive": float(
            ((td_values < 0) & (mc_values > 0)).float().mean()
        ),
        "mean_absolute_difference": float((td_values - mc_values).abs().mean()),
    }


def _first_episode_mask(dones: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    ended = dones[1:].any(dim=-1).to(torch.int64)
    ended_before = ended.cumsum(dim=0) - ended
    return valid_mask & (ended_before == 0)


def _initial_transition_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask & (valid_mask.to(torch.int64).cumsum(dim=0) == 1)


def _shardwise_normalize(
    advantages: torch.Tensor,
    subtask_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    source_shards: list[dict[str, Any]],
) -> torch.Tensor:
    normalized = torch.zeros_like(advantages)
    offset = 0
    for shard in source_shards:
        count = int(shard["local_trajectory_count"])
        shard_slice = slice(offset, offset + count)
        normalized[:, shard_slice] = taskwise_normalize(
            advantages[:, shard_slice],
            subtask_ids[:, shard_slice],
            valid_mask[:, shard_slice],
            std_floor=0.1,
        )
        offset += count
    if offset != advantages.shape[1]:
        raise ValueError(
            f"Shard metadata covers {offset} trajectories, expected "
            f"{advantages.shape[1]}."
        )
    return normalized


def _serial_outcome_group_ids(
    artifact: dict[str, Any],
    valid_mask: torch.Tensor,
    outcome_group_size: int,
) -> torch.Tensor:
    world_size = int(artifact["metadata"]["actor_world_size"])
    if outcome_group_size <= 0 or outcome_group_size % world_size != 0:
        raise ValueError(
            "Serial outcome group size must be positive and divisible by the "
            f"actor world size ({world_size}), got {outcome_group_size}."
        )
    trajectories_per_rank_group = outcome_group_size // world_size
    local_indices = artifact["trajectory_local_indices"]
    trajectory_group_ids = local_indices // trajectories_per_rank_group
    counts = torch.bincount(trajectory_group_ids)
    if not torch.all(counts == outcome_group_size):
        raise ValueError(
            "Reconstructed serial outcome groups do not all contain "
            f"{outcome_group_size} trajectories: {counts.tolist()}."
        )
    return trajectory_group_ids.unsqueeze(0).expand_as(valid_mask)


def analyze_critic_targets(
    artifact: dict[str, Any],
    *,
    outcome_group_size: int | None = None,
) -> dict[str, Any]:
    """Return diagnostics for TD(lambda) targets and Monte-Carlo targets."""
    batch = artifact["batch"]
    metadata = artifact["metadata"]
    source_metadata = metadata["source_shards"][0]
    gamma = float(source_metadata["gamma"])
    gae_lambda = float(source_metadata["gae_lambda"])

    valid_mask = batch["loss_mask"].squeeze(-1).to(torch.bool)
    macro_rewards, discounts = discounted_chunk_rewards(
        batch["rewards"],
        batch["executed_action_mask"],
        gamma=gamma,
    )
    macro_dones = batch["dones"][1:].any(dim=-1)
    subtask_ids = align_subtask_ids(batch["subtask_ids"], macro_rewards)
    values = batch["prev_values"].squeeze(-1).float()

    td_raw_advantages, td_returns = compute_subtask_gae(
        macro_rewards,
        discounts,
        macro_dones,
        values,
        subtask_ids,
        valid_mask,
        gae_lambda=gae_lambda,
        normalize_advantages=False,
        advantage_std_floor=0.1,
    )
    mc_raw_advantages, mc_returns = compute_subtask_gae(
        macro_rewards,
        discounts,
        macro_dones,
        values,
        subtask_ids,
        valid_mask,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_std_floor=0.1,
    )
    source_shards = metadata["source_shards"]
    td_advantages = _shardwise_normalize(
        td_raw_advantages, subtask_ids, valid_mask, source_shards
    )
    mc_advantages = _shardwise_normalize(
        mc_raw_advantages, subtask_ids, valid_mask, source_shards
    )
    state_group_ids = None
    if outcome_group_size is not None:
        state_group_ids = _serial_outcome_group_ids(
            artifact, valid_mask, outcome_group_size
        )
        statewise_td_advantages = taskwise_normalize(
            td_raw_advantages,
            state_group_ids,
            valid_mask,
            std_floor=0.1,
        )
        statewise_mc_advantages = taskwise_normalize(
            mc_raw_advantages,
            state_group_ids,
            valid_mask,
            std_floor=0.1,
        )

    stored_returns = batch["returns"].squeeze(-1).float()
    stored_advantages = batch["advantages"].squeeze(-1).float()
    first_episode = _first_episode_mask(batch["dones"], valid_mask)
    initial_transition = _initial_transition_mask(valid_mask)
    prediction = values[:-1]
    target_difference = td_returns - mc_returns

    result: dict[str, Any] = {
        "metadata": {
            "global_step": int(metadata["global_step"]),
            "trajectory_count": int(metadata["trajectory_count"]),
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "valid_transition_count": int(valid_mask.sum()),
            "first_episode_transition_count": int(first_episode.sum()),
            "initial_transition_count": int(initial_transition.sum()),
        },
        "consistency": {
            "stored_td_return_max_abs_error": float(
                (stored_returns - td_returns).abs()[valid_mask].max()
            ),
            "stored_normalized_advantage_max_abs_error": float(
                (stored_advantages - td_advantages).abs()[valid_mask].max()
            ),
        },
        "targets": {
            "td_lambda": _distribution(td_returns, valid_mask),
            "monte_carlo": _distribution(mc_returns, valid_mask),
            "td_minus_monte_carlo": _distribution(target_difference, valid_mask),
            "first_episode_td_minus_monte_carlo": _distribution(
                target_difference, first_episode
            ),
        },
        "critic": {
            "all_valid_td_lambda": _prediction_metrics(
                prediction, td_returns, valid_mask
            ),
            "all_valid_monte_carlo": _prediction_metrics(
                prediction, mc_returns, valid_mask
            ),
            "first_episode_td_lambda": _prediction_metrics(
                prediction, td_returns, first_episode
            ),
            "first_episode_monte_carlo": _prediction_metrics(
                prediction, mc_returns, first_episode
            ),
            "initial_transition_td_lambda": _prediction_metrics(
                prediction, td_returns, initial_transition
            ),
            "initial_transition_monte_carlo": _prediction_metrics(
                prediction, mc_returns, initial_transition
            ),
        },
        "advantages": {
            "current_td_vs_monte_carlo": _advantage_comparison(
                td_advantages, mc_advantages, valid_mask
            ),
            "current_td_vs_monte_carlo_first_episode": _advantage_comparison(
                td_advantages, mc_advantages, first_episode
            ),
        },
    }

    if state_group_ids is not None:
        result["metadata"]["outcome_group_count"] = int(
            torch.unique(state_group_ids[valid_mask]).numel()
        )
        result["advantages"]["statewise_td_vs_monte_carlo"] = _advantage_comparison(
            statewise_td_advantages,
            statewise_mc_advantages,
            valid_mask,
        )
        result["advantages"]["current_td_vs_statewise_monte_carlo"] = (
            _advantage_comparison(
                td_advantages,
                statewise_mc_advantages,
                valid_mask,
            )
        )
        result["advantages"]["current_td_vs_statewise_monte_carlo_first_episode"] = (
            _advantage_comparison(
                td_advantages,
                statewise_mc_advantages,
                first_episode,
            )
        )

    outcomes = artifact.get("trajectory_outcomes")
    if outcomes is not None:
        outcome_mask = outcomes.to(torch.bool).unsqueeze(0).expand_as(valid_mask)
        result["advantages"]["current_first_episode_success"] = _advantage_comparison(
            td_advantages, mc_advantages, first_episode & outcome_mask
        )
        result["advantages"]["current_first_episode_failure"] = _advantage_comparison(
            td_advantages, mc_advantages, first_episode & ~outcome_mask
        )
        result["advantages"]["current_initial_success"] = _advantage_comparison(
            td_advantages, mc_advantages, initial_transition & outcome_mask
        )
        result["advantages"]["current_initial_failure"] = _advantage_comparison(
            td_advantages, mc_advantages, initial_transition & ~outcome_mask
        )
        result["critic"]["first_episode_success_monte_carlo"] = _prediction_metrics(
            prediction,
            mc_returns,
            first_episode & outcome_mask,
        )
        result["critic"]["first_episode_failure_monte_carlo"] = _prediction_metrics(
            prediction,
            mc_returns,
            first_episode & ~outcome_mask,
        )
        result["outcomes"] = {
            "successes": int(outcomes.sum()),
            "failures": int((~outcomes).sum()),
            "initial_value_success": _distribution(
                prediction, initial_transition & outcome_mask
            ),
            "initial_value_failure": _distribution(
                prediction, initial_transition & ~outcome_mask
            ),
            "initial_td_target_success": _distribution(
                td_returns, initial_transition & outcome_mask
            ),
            "initial_td_target_failure": _distribution(
                td_returns, initial_transition & ~outcome_mask
            ),
            "initial_monte_carlo_target_success": _distribution(
                mc_returns, initial_transition & outcome_mask
            ),
            "initial_monte_carlo_target_failure": _distribution(
                mc_returns, initial_transition & ~outcome_mask
            ),
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Merged critic batch artifact.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--outcome-group-size",
        type=int,
        default=None,
        help="Reconstruct serial state groups of this many trajectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    result = analyze_critic_targets(
        artifact,
        outcome_group_size=args.outcome_group_size,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n")
    print(serialized)


if __name__ == "__main__":
    main()

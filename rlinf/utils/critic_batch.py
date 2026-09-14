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

"""Serializable rollout batches for offline critic experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from rlinf.algorithms.subtask import reduce_first_episode_successes
from rlinf.utils.nested_dict_process import clone_nested_to_cpu

_CRITIC_BATCH_KEYS = (
    "returns",
    "advantages",
    "prev_values",
    "rewards",
    "dones",
    "terminations",
    "truncations",
    "loss_mask",
    "loss_mask_sum",
    "sample_weights",
    "executed_action_mask",
    "subtask_ids",
    "successes",
    "outcome_group_ids",
    "outcome_logical_group_ids",
    "outcome_episode_indices",
)

_CRITIC_OBSERVATION_KEYS = (
    "obs_state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
)


def _select_critic_observation(forward_inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep inputs needed to recompute frozen VLM features."""
    selected = {
        key: value
        for key, value in forward_inputs.items()
        if key in _CRITIC_OBSERVATION_KEYS
        or key.startswith("obs_image__")
        or key.startswith("obs_image_mask__")
    }
    missing = [key for key in _CRITIC_OBSERVATION_KEYS if key not in selected]
    if missing:
        raise ValueError(f"Critic batch is missing observation fields: {missing}.")
    if not any(key.startswith("obs_image__") for key in selected):
        raise ValueError("Critic batch must contain at least one observation image.")
    return selected


def build_critic_batch_artifact(
    rollout_batch: dict[str, Any],
    *,
    actor_rank: int,
    actor_world_size: int,
    global_step: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one CPU actor shard before rollout shuffling and optimization."""
    for key in ("returns", "loss_mask", "forward_inputs"):
        if key not in rollout_batch:
            raise ValueError(f"Critic batch is missing required field {key!r}.")

    batch = {
        key: rollout_batch[key] for key in _CRITIC_BATCH_KEYS if key in rollout_batch
    }
    batch["forward_inputs"] = _select_critic_observation(
        rollout_batch["forward_inputs"]
    )

    returns = batch["returns"]
    if not isinstance(returns, torch.Tensor) or returns.ndim < 2:
        raise ValueError("Critic returns must have [time, batch, ...] dimensions.")

    artifact_metadata = {
        "format_version": 1,
        "actor_rank": int(actor_rank),
        "actor_world_size": int(actor_world_size),
        "global_step": int(global_step),
        "time_size": int(returns.shape[0]),
        "local_trajectory_count": int(returns.shape[1]),
    }
    if metadata:
        artifact_metadata.update(metadata)

    return {
        "metadata": artifact_metadata,
        "batch": clone_nested_to_cpu(batch),
    }


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    """Write a torch artifact atomically on the destination filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def export_critic_batch_shard(
    rollout_batch: dict[str, Any],
    export_dir: str | Path,
    *,
    actor_rank: int,
    actor_world_size: int,
    global_step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export one actor's unshuffled critic batch to a rank-specific file."""
    artifact = build_critic_batch_artifact(
        rollout_batch,
        actor_rank=actor_rank,
        actor_world_size=actor_world_size,
        global_step=global_step,
        metadata=metadata,
    )
    path = (
        Path(export_dir)
        / f"global_step_{global_step:06d}"
        / f"actor_rank_{actor_rank:04d}.pt"
    )
    return atomic_torch_save(artifact, path)


def _cat_batch_dimension(values: list[Any], path: tuple[str, ...]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim < 2:
            dotted_path = ".".join(path)
            raise ValueError(
                f"Critic batch tensor {dotted_path!r} has no batch dimension."
            )
        return torch.cat(values, dim=1)
    if isinstance(first, dict):
        keys = list(first)
        expected_keys = set(keys)
        if any(set(value) != expected_keys for value in values[1:]):
            dotted_path = ".".join(path) or "batch"
            raise ValueError(f"Critic shard keys differ under {dotted_path!r}.")
        return {
            key: _cat_batch_dimension([value[key] for value in values], (*path, key))
            for key in keys
        }
    dotted_path = ".".join(path)
    raise TypeError(f"Unsupported critic batch value {dotted_path!r}: {type(first)}")


def merge_critic_batch_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge all actor shards and attach stable trajectory provenance."""
    if not shards:
        raise ValueError("At least one critic batch shard is required.")

    shards = sorted(shards, key=lambda shard: shard["metadata"]["actor_rank"])
    metadata = [shard["metadata"] for shard in shards]
    ranks = [int(item["actor_rank"]) for item in metadata]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"Critic batch contains duplicate actor ranks: {ranks}.")

    expected_world_size = int(metadata[0]["actor_world_size"])
    if any(int(item["actor_world_size"]) != expected_world_size for item in metadata):
        raise ValueError("Critic shards disagree on actor_world_size.")
    if len(shards) != expected_world_size:
        raise ValueError(
            f"Expected {expected_world_size} actor shards, received {len(shards)}."
        )

    global_steps = {int(item["global_step"]) for item in metadata}
    if len(global_steps) != 1:
        raise ValueError(f"Critic shards disagree on global step: {global_steps}.")

    merged_batch = _cat_batch_dimension(
        [shard["batch"] for shard in shards], ("batch",)
    )
    actor_ranks = []
    local_indices = []
    for item in metadata:
        count = int(item["local_trajectory_count"])
        actor_ranks.extend([int(item["actor_rank"])] * count)
        local_indices.extend(range(count))

    outcomes = None
    if "terminations" in merged_batch and "dones" in merged_batch:
        outcomes = reduce_first_episode_successes(
            merged_batch["terminations"], merged_batch["dones"]
        )

    return {
        "metadata": {
            "format_version": 1,
            "global_step": global_steps.pop(),
            "actor_world_size": expected_world_size,
            "trajectory_count": len(actor_ranks),
            "source_shards": metadata,
        },
        "trajectory_actor_ranks": torch.tensor(actor_ranks, dtype=torch.int64),
        "trajectory_local_indices": torch.tensor(local_indices, dtype=torch.int64),
        "trajectory_outcomes": (
            torch.tensor(outcomes, dtype=torch.bool) if outcomes is not None else None
        ),
        "batch": merged_batch,
    }


def summarize_critic_batch(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return compact validation statistics for a merged critic artifact."""
    batch = artifact["batch"]
    mask = batch["loss_mask"].to(torch.bool)
    returns = batch["returns"].float()
    valid_returns = returns.masked_select(mask.expand_as(returns))
    outcomes = artifact.get("trajectory_outcomes")
    quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    return_quantiles = torch.quantile(valid_returns, quantile_levels)
    summary = {
        "trajectory_count": int(artifact["metadata"]["trajectory_count"]),
        "valid_target_count": int(valid_returns.numel()),
        "successes": int(outcomes.sum()) if outcomes is not None else None,
        "failures": int((~outcomes).sum()) if outcomes is not None else None,
        "return_min": float(valid_returns.min()),
        "return_mean": float(valid_returns.mean()),
        "return_std": float(valid_returns.std(correction=0)),
        "return_max": float(valid_returns.max()),
        "return_quantiles": {
            str(float(level)): float(value)
            for level, value in zip(quantile_levels, return_quantiles, strict=True)
        },
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
        "observation_shapes": {
            key: list(value.shape)
            for key, value in batch["forward_inputs"].items()
            if isinstance(value, torch.Tensor)
        },
    }
    prev_values = batch.get("prev_values")
    if isinstance(prev_values, torch.Tensor):
        aligned_prev_values = prev_values[: returns.shape[0]]
        valid_prev_values = aligned_prev_values.float().masked_select(
            mask.expand_as(aligned_prev_values)
        )
        target_variance = valid_returns.var(correction=0)
        explained_variance = 1.0 - (valid_returns - valid_prev_values).var(
            correction=0
        ) / target_variance.clamp(min=1e-12)
        summary.update(
            {
                "prev_value_min": float(valid_prev_values.min()),
                "prev_value_mean": float(valid_prev_values.mean()),
                "prev_value_std": float(valid_prev_values.std(correction=0)),
                "prev_value_max": float(valid_prev_values.max()),
                "initial_explained_variance": float(explained_variance),
            }
        )
    return summary

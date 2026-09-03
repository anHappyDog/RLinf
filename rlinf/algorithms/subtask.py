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

"""Credit-assignment primitives for task-balanced embodied PPO."""

from __future__ import annotations

import torch


def align_subtask_ids(
    subtask_ids: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Align transition subtask IDs with a ``[T, B]`` reference tensor.

    Trajectories store IDs as ``[T, B]``.  This also accepts the legacy
    ``[T, B, 1]`` representation without collapsing a singleton batch axis.
    """
    if subtask_ids.shape == reference.shape:
        return subtask_ids
    if (
        subtask_ids.ndim == reference.ndim + 1
        and subtask_ids.shape[-1] == 1
        and subtask_ids.shape[:-1] == reference.shape
    ):
        return subtask_ids.squeeze(-1)
    raise ValueError(
        "subtask_ids must match the transition reference shape "
        f"{tuple(reference.shape)} or add one trailing singleton dimension; "
        f"got {tuple(subtask_ids.shape)}."
    )


def discounted_chunk_rewards(
    rewards: torch.Tensor,
    executed_action_mask: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse primitive rewards into duration-aware macro transitions.

    Args:
        rewards: Primitive rewards with shape ``[T, B, A]``.
        executed_action_mask: True for actions actually executed, same shape.
        gamma: Primitive-step discount factor.

    Returns:
        A pair containing discounted macro rewards and bootstrap discounts, both
        with shape ``[T, B]``. A transition executing ``m`` actions receives
        discount ``gamma ** m``.
    """
    if rewards.ndim != 3 or executed_action_mask.shape != rewards.shape:
        raise ValueError(
            "rewards and executed_action_mask must have the same [T, B, A] shape."
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")

    mask = executed_action_mask.to(dtype=torch.bool)
    prefix_mask = mask.to(torch.int64).cumprod(dim=-1).to(torch.bool)
    if not torch.equal(mask, prefix_mask):
        raise ValueError("executed_action_mask must be prefix-contiguous per chunk.")

    powers = torch.arange(rewards.shape[-1], device=rewards.device)
    weights = (
        torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device) ** powers
    )
    macro_rewards = (rewards * mask * weights).sum(dim=-1)
    executed_steps = mask.sum(dim=-1)
    macro_discounts = (
        torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device)
        ** executed_steps
    )
    return macro_rewards, macro_discounts


def taskwise_normalize(
    advantages: torch.Tensor,
    subtask_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    std_floor: float,
) -> torch.Tensor:
    """Normalize valid advantages independently for each subtask."""
    if advantages.shape != subtask_ids.shape or advantages.shape != valid_mask.shape:
        raise ValueError("advantages, subtask_ids, and valid_mask must share a shape.")
    if std_floor <= 0:
        raise ValueError("std_floor must be positive.")

    normalized = torch.zeros_like(advantages)
    for subtask_id in torch.unique(subtask_ids[valid_mask]):
        task_mask = valid_mask & (subtask_ids == subtask_id)
        values = advantages[task_mask]
        mean = values.mean()
        std = values.std(unbiased=False)
        if std < std_floor:
            # A near-terminal predecessor state may yield only one valid macro
            # transition. Mean-centering that task would erase its entire policy
            # gradient, so use an RMS-like scale without centering in this
            # degenerate regime. This also preserves a shared success/failure
            # sign when every sampled return is effectively identical.
            scale = values.square().mean().sqrt().clamp_min(std_floor)
            normalized[task_mask] = values / scale
        else:
            normalized[task_mask] = (values - mean) / std
    return normalized


def balanced_subtask_weights(
    subtask_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Give every represented subtask equal total optimization weight.

    The returned valid weights sum to ``subtask_ids.numel()`` so taking a plain
    tensor mean remains exactly composable across gradient-accumulation
    microbatches.
    """
    if subtask_ids.shape != valid_mask.shape:
        raise ValueError("subtask_ids and valid_mask must share a shape.")
    valid_mask = valid_mask.to(torch.bool)
    weights = torch.zeros_like(subtask_ids, dtype=torch.float32)
    represented = torch.unique(subtask_ids[valid_mask])
    if represented.numel() == 0:
        return weights
    target_total = subtask_ids.numel() / represented.numel()
    for subtask_id in represented:
        task_mask = valid_mask & (subtask_ids == subtask_id)
        weights[task_mask] = target_total / task_mask.sum()
    return weights


def compute_subtask_gae(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    subtask_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gae_lambda: float,
    normalize_advantages: bool,
    advantage_std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute duration-aware GAE without leaking across subtask boundaries.

    All tensors except ``values`` have shape ``[T, B]``. ``values`` has shape
    ``[T + 1, B]`` and ``dones[t]`` denotes termination after transition ``t``.
    """
    expected_shape = rewards.shape
    if rewards.ndim != 2:
        raise ValueError("Subtask GAE inputs must have shape [T, B].")
    for name, tensor in (
        ("discounts", discounts),
        ("dones", dones),
        ("subtask_ids", subtask_ids),
        ("valid_mask", valid_mask),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
    if values.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError(
            f"values must have shape {(rewards.shape[0] + 1, rewards.shape[1])}."
        )
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}.")

    valid_mask = valid_mask.to(torch.bool)
    dones = dones.to(torch.bool)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)

    for step in reversed(range(rewards.shape[0])):
        continues = ~dones[step]
        if step + 1 < rewards.shape[0]:
            continues &= subtask_ids[step + 1] == subtask_ids[step]
            continues &= valid_mask[step + 1]
        delta = (
            rewards[step]
            + discounts[step] * continues * values[step + 1]
            - values[step]
        )
        gae = delta + discounts[step] * gae_lambda * continues * gae
        gae = torch.where(valid_mask[step], gae, torch.zeros_like(gae))
        advantages[step] = gae

    returns = advantages + values[:-1]
    if normalize_advantages:
        advantages = taskwise_normalize(
            advantages,
            subtask_ids,
            valid_mask,
            std_floor=advantage_std_floor,
        )
    return advantages, returns


__all__ = [
    "align_subtask_ids",
    "balanced_subtask_weights",
    "compute_subtask_gae",
    "discounted_chunk_rewards",
    "taskwise_normalize",
]

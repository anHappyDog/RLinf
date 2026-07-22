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

from typing import Any

import torch

from rlinf.algorithms.embodied_gae import (
    compose_embodied_rewards,
    compute_embodied_gae,
)
from rlinf.data.forward_inputs import ForwardInputs
from rlinf.workers.trajectory.storage import TrajectoryBatch


def prepare_actor_batch(
    trajectory: TrajectoryBatch,
    *,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
    env_reward_weight: float = 1.0,
    external_reward_weight: float = 1.0,
) -> dict[str, Any]:
    """Convert Storage output into the existing embodied Actor batch order.

    Storage owns coordinates in ``[rollout_epoch, chunk_step, slot]`` order.
    The embodied Actor trains in ``[chunk_step, rollout_epoch * slot]`` order.
    This function is the only compatibility boundary between those layouts.
    """
    required = {
        "state_values": trajectory.state_values,
        "timeout_values": trajectory.timeout_values,
        "timeout_mask": trajectory.timeout_mask,
        "tail_values": trajectory.tail_values,
        "tail_mask": trajectory.tail_mask,
        "prev_logprobs": trajectory.prev_logprobs,
        "forward_inputs": trajectory.forward_inputs,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"trajectory is missing Actor fields: {missing}")

    rewards = compose_embodied_rewards(
        trajectory.env_rewards,
        external_rewards=trajectory.external_rewards,
        external_reward_mask=trajectory.reward_mask,
        env_weight=env_reward_weight,
        external_weight=external_reward_weight,
    )
    loss_mask = _loss_mask(trajectory.dones)
    advantages, returns = compute_embodied_gae(
        rewards,
        trajectory.state_values,
        trajectory.dones,
        trajectory.terminations,
        trajectory.truncations,
        trajectory.timeout_values,
        trajectory.timeout_mask,
        trajectory.tail_values,
        trajectory.tail_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=normalize_advantages,
        loss_mask=loss_mask,
    )

    batch: dict[str, Any] = {
        "rewards": _actor_order(rewards.unsqueeze(-1)),
        "dones": _actor_order(trajectory.dones),
        "terminations": _actor_order(trajectory.terminations),
        "truncations": _actor_order(trajectory.truncations),
        "actions": _actor_order(trajectory.actions),
        "prev_logprobs": _actor_order(trajectory.prev_logprobs),
        "prev_values": _actor_order(trajectory.state_values),
        "advantages": _actor_order(advantages),
        "returns": _actor_order(returns),
        "loss_mask": _actor_order(loss_mask.unsqueeze(-1)),
        "forward_inputs": _actor_forward_inputs(trajectory),
    }
    optional = {
        "versions": trajectory.versions,
        "intervene_flags": trajectory.rollout_intervene_flags,
        "intervene_actions": trajectory.intervene_actions,
        "rlt_switch_flags": trajectory.rlt_switch_flags,
        "curr_obs": trajectory.observations,
        "next_obs": trajectory.next_observations,
    }
    batch.update(
        {
            name: _actor_order(value)
            for name, value in optional.items()
            if value is not None
        }
    )
    return batch


def shuffle_actor_batch(
    batch: dict[str, Any], shuffle_id: torch.Tensor
) -> dict[str, Any]:
    """Flatten transition-aligned ``[S,N]`` axes and apply Actor shuffling."""
    output = {}
    for name, value in batch.items():
        if value is None:
            output[name] = None
        elif isinstance(value, torch.Tensor):
            output[name] = value.flatten(0, 1).index_select(0, shuffle_id)
        elif isinstance(value, dict):
            output[name] = shuffle_actor_batch(value, shuffle_id)
        else:
            raise TypeError(f"unsupported Actor batch value {type(value).__name__}")
    return output


def _actor_order(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim < 3:
            raise ValueError("trajectory grid values must have [E, S, B] dimensions")
        return value.transpose(0, 1).flatten(1, 2).contiguous()
    if isinstance(value, dict):
        return {key: _actor_order(child) for key, child in value.items()}
    raise TypeError(f"unsupported trajectory grid value {type(value).__name__}")


def _actor_forward_inputs(trajectory: TrajectoryBatch) -> dict[str, torch.Tensor]:
    forward_inputs = trajectory.forward_inputs
    assert isinstance(forward_inputs, ForwardInputs)
    epoch_count = trajectory.rollout_epochs
    step_count = trajectory.chunk_steps
    slot_count = len(trajectory.slot_ids)
    expected = epoch_count * step_count * slot_count
    output = {}
    for name, value in forward_inputs.tensor_fields():
        if value.shape[0] != expected:
            raise ValueError(
                f"forward input {name!r} has batch {value.shape[0]}, expected {expected}"
            )
        grid = value.unflatten(0, (epoch_count, step_count, slot_count))
        output[name] = _actor_order(grid)
    return output


def _loss_mask(dones: torch.Tensor) -> torch.Tensor:
    """Mark transitions through and including the first done in each segment."""
    done = dones.any(dim=-1)
    previously_done = done.cumsum(dim=1) - done.to(torch.int64)
    return previously_done == 0

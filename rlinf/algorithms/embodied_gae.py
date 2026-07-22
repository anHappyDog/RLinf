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

import torch


def compose_embodied_rewards(
    env_rewards: torch.Tensor,
    *,
    external_rewards: torch.Tensor | None = None,
    external_reward_mask: torch.Tensor | None = None,
    env_weight: float = 1.0,
    external_weight: float = 1.0,
) -> torch.Tensor:
    """Compose raw reward sources into macro-transition rewards.

    Both reward tensors retain their action-chunk axis. Each source is summed
    independently before weighting, so a scalar external reward is not
    accidentally broadcast across every primitive action in the chunk.

    Args:
        env_rewards: Raw environment rewards shaped ``[E, S, B, A]``.
        external_rewards: Optional aligned rewards shaped ``[E, S, B, R]``.
        external_reward_mask: Valid external coordinates shaped ``[E, S, B]``.
        env_weight: Environment reward coefficient.
        external_weight: External reward coefficient.

    Returns:
        Effective rewards shaped ``[E, S, B]``.
    """
    _validate_reward_tensor(env_rewards, "env_rewards")
    env_macro_rewards = env_rewards.sum(dim=-1)

    if external_rewards is None:
        if external_reward_mask is not None:
            raise ValueError("external_reward_mask requires external_rewards.")
        return env_weight * env_macro_rewards

    _validate_reward_tensor(external_rewards, "external_rewards")
    if external_rewards.shape[:3] != env_rewards.shape[:3]:
        raise ValueError(
            "external_rewards must match env_rewards on [E, S, B], got "
            f"{tuple(external_rewards.shape[:3])} and "
            f"{tuple(env_rewards.shape[:3])}."
        )
    if external_rewards.device != env_rewards.device:
        raise ValueError("env_rewards and external_rewards must share a device.")
    if external_reward_mask is None:
        raise ValueError("external_reward_mask is required with external_rewards.")
    _validate_bool_tensor(
        external_reward_mask,
        "external_reward_mask",
        env_rewards.shape[:3],
        env_rewards.device,
    )

    external_macro_rewards = external_rewards.sum(dim=-1)
    external_macro_rewards = torch.where(
        external_reward_mask,
        external_macro_rewards,
        torch.zeros_like(external_macro_rewards),
    )
    return env_weight * env_macro_rewards + external_weight * external_macro_rewards


def compute_embodied_gae(
    rewards: torch.Tensor,
    state_values: torch.Tensor,
    dones: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    timeout_values: torch.Tensor,
    timeout_mask: torch.Tensor,
    tail_values: torch.Tensor,
    tail_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool = False,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE from transition-aligned embodied trajectory data.

    Args:
        rewards: Macro-transition rewards shaped ``[E, S, B]``.
        state_values: ``V(s_t)`` shaped ``[E, S, B, 1]``.
        dones: Raw action-chunk done masks shaped ``[E, S, B, A]``.
        terminations: Raw true-termination masks with the same prefix.
        truncations: Raw timeout masks with the same prefix.
        timeout_values: Values of terminal observations, shaped ``[E, S, B, 1]``.
        timeout_mask: Valid timeout values shaped ``[E, S, B]``.
        tail_values: Values for alive segment-tail states, shaped ``[E, B, 1]``.
        tail_mask: Valid tail values shaped ``[E, B]``.
        gamma: Discount factor.
        gae_lambda: GAE trace factor.
        normalize_advantages: Whether to normalize advantages globally.
        loss_mask: Optional valid training coordinates, shaped ``[E, S, B]`` or
            ``[E, S, B, A]``.

    Returns:
        Advantages and returns, each shaped ``[E, S, B, 1]``.
    """
    _validate_gae_inputs(
        rewards,
        state_values,
        dones,
        terminations,
        truncations,
        timeout_values,
        timeout_mask,
        tail_values,
        tail_mask,
        gamma,
        gae_lambda,
    )
    done = dones.any(dim=-1)
    terminated = terminations.any(dim=-1)
    truncated = truncations.any(dim=-1)
    if not torch.equal(done, terminated | truncated):
        raise ValueError("dones must equal terminations | truncations.")
    effective_timeout = truncated & ~terminated
    if not torch.equal(timeout_mask, effective_timeout):
        raise ValueError(
            "timeout_mask must exactly match truncations without termination."
        )
    expected_tail_mask = ~done[:, -1]
    if not torch.equal(tail_mask, expected_tail_mask):
        raise ValueError("tail_mask must exactly match alive segment-tail slots.")

    values = state_values.squeeze(-1)
    timeout = timeout_values.squeeze(-1)
    tail = tail_values.squeeze(-1)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[:, 0])

    for step in reversed(range(rewards.shape[1])):
        continuation_values = (
            tail if step == rewards.shape[1] - 1 else values[:, step + 1]
        )
        continuation = (~done[:, step]) * continuation_values
        timeout_bootstrap = timeout_mask[:, step] * timeout[:, step]
        delta = (
            rewards[:, step]
            + gamma * (continuation + timeout_bootstrap)
            - values[:, step]
        )
        gae = delta + gamma * gae_lambda * (~done[:, step]) * gae
        advantages[:, step] = gae

    returns = advantages + values
    if normalize_advantages:
        normalized_mask = _normalize_loss_mask(loss_mask, rewards.shape, rewards.device)
        valid_advantages = (
            advantages.reshape(-1)
            if normalized_mask is None
            else advantages[normalized_mask]
        )
        if valid_advantages.numel() == 1:
            advantages = advantages - valid_advantages[0]
        elif valid_advantages.numel() > 1:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std() + 1e-5
            )

    return advantages.unsqueeze(-1), returns.unsqueeze(-1)


def _validate_reward_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 4 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [E, S, B, A].")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype.")


def _validate_bool_tensor(
    value: torch.Tensor,
    name: str,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}.")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool.")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}.")


def _validate_gae_inputs(
    rewards: torch.Tensor,
    state_values: torch.Tensor,
    dones: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    timeout_values: torch.Tensor,
    timeout_mask: torch.Tensor,
    tail_values: torch.Tensor,
    tail_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> None:
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("rewards must be a torch.Tensor.")
    if rewards.ndim != 3 or min(rewards.shape) < 1:
        raise ValueError("rewards must have non-empty shape [E, S, B].")
    if not torch.is_floating_point(rewards):
        raise TypeError("rewards must have a floating-point dtype.")

    value_shapes = {
        "state_values": (*rewards.shape, 1),
        "timeout_values": (*rewards.shape, 1),
        "tail_values": (rewards.shape[0], rewards.shape[2], 1),
    }
    for name, value in (
        ("state_values", state_values),
        ("timeout_values", timeout_values),
        ("tail_values", tail_values),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.shape != value_shapes[name]:
            raise ValueError(f"{name} must have shape {value_shapes[name]}.")
        if value.dtype != rewards.dtype:
            raise TypeError(f"{name} must have dtype {rewards.dtype}.")
        if value.device != rewards.device:
            raise ValueError(f"{name} must be on device {rewards.device}.")
    transition_prefix = rewards.shape
    for name, value in (
        ("dones", dones),
        ("terminations", terminations),
        ("truncations", truncations),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if (
            value.ndim != 4
            or value.shape[:3] != transition_prefix
            or value.shape[3] < 1
        ):
            raise ValueError(f"{name} must have shape [E, S, B, A].")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool.")
        if value.device != rewards.device:
            raise ValueError(f"{name} must be on device {rewards.device}.")
    if terminations.shape != dones.shape or truncations.shape != dones.shape:
        raise ValueError("dones, terminations, and truncations must have equal shape.")

    _validate_bool_tensor(timeout_mask, "timeout_mask", rewards.shape, rewards.device)
    _validate_bool_tensor(
        tail_mask,
        "tail_mask",
        (rewards.shape[0], rewards.shape[2]),
        rewards.device,
    )
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")


def _normalize_loss_mask(
    loss_mask: torch.Tensor | None,
    transition_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor | None:
    if loss_mask is None:
        return None
    if not isinstance(loss_mask, torch.Tensor):
        raise TypeError("loss_mask must be a torch.Tensor.")
    if loss_mask.dtype != torch.bool:
        raise TypeError("loss_mask must have dtype torch.bool.")
    if loss_mask.device != device:
        raise ValueError(f"loss_mask must be on device {device}.")
    if loss_mask.shape == transition_shape:
        return loss_mask
    if loss_mask.ndim == 4 and loss_mask.shape[:3] == transition_shape:
        return loss_mask.any(dim=-1)
    raise ValueError("loss_mask must have shape [E, S, B] or [E, S, B, A].")

# Copyright 2025 The RLinf Authors.
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

import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def _masked_to_nan(x: torch.Tensor, mask: torch.Tensor):
    x = x.float().detach().cpu()
    m = mask.detach().cpu().bool()
    x[~m] = float("nan")
    return x


def plot_decoupled_ppo_debug(debug_tensors: dict, save_dir: str, tag: str = "step"):
    os.makedirs(save_dir, exist_ok=True)

    adv = _masked_to_nan(debug_tensors["advantages"], debug_tensors["loss_mask"])
    ratio = _masked_to_nan(debug_tensors["proximal_ratio"], debug_tensors["loss_mask"])
    clipped_ratio = _masked_to_nan(
        debug_tensors["clipped_proximal_ratio"], debug_tensors["loss_mask"]
    )
    pg1 = _masked_to_nan(debug_tensors["pg_loss1"], debug_tensors["loss_mask"])
    pg2 = _masked_to_nan(debug_tensors["pg_loss2"], debug_tensors["loss_mask"])
    pg = _masked_to_nan(debug_tensors["pg_loss_mat"], debug_tensors["loss_mask"])
    behav_w = _masked_to_nan(debug_tensors["behav_weight"], debug_tensors["behav_mask"])
    kl = _masked_to_nan(debug_tensors["kl_token"], debug_tensors["loss_mask"])

    # 找到最大的 advantage 位置（忽略 nan）
    adv_abs = torch.nan_to_num(torch.abs(adv), nan=-1.0)
    flat_idx = torch.argmax(adv_abs).item()
    B, T = adv.shape
    i, t = flat_idx // T, flat_idx % T

    def _heatmap(x, title, fname):
        plt.figure()
        plt.imshow(x, aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("step/t")
        plt.ylabel("batch/env")
        plt.scatter([t], [i], s=40)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=160)
        plt.close()

    _heatmap(adv, f"advantages (max @ {i},{t})", f"{tag}_adv.png")
    _heatmap(ratio, f"proximal_ratio (max-adv @ {i},{t})", f"{tag}_ratio.png")
    _heatmap(clipped_ratio, "clipped_ratio", f"{tag}_clipped_ratio.png")
    _heatmap(pg1, "|pg_loss1| = |-adv*ratio|", f"{tag}_pg1.png")
    _heatmap(pg2, "|pg_loss2| = |-adv*clipped|", f"{tag}_pg2.png")
    _heatmap(pg, "pg_loss_mat = max(pg1, pg2)", f"{tag}_pg_max.png")
    _heatmap(behav_w, "behav_weight (masked)", f"{tag}_behav_weight.png")
    _heatmap(kl, "token_KL = -(logp - prox_logp)", f"{tag}_kl.png")

    print(f"[debug] max |adv| at (batch/env={i}, step={t})")
    print(
        f"[debug] adv={adv[i, t].item():.4g}, ratio={ratio[i, t].item():.4g}, clipped={clipped_ratio[i, t].item():.4g}, kl={kl[i, t].item():.4g}"
    )


def compute_decoupled_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    proximal_logprobs: Optional[torch.Tensor] = None,
    versions: Optional[torch.Tensor] = None,
    current_version: int | float | torch.Tensor | None = None,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Callable[..., torch.Tensor]
    | None = masked_mean,  # will default to masked_mean if None
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    behave_weight_threshold: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Decoupled PPO actor loss with optional proximal_logprobs linear interpolation.

    Proximal policy anchor:
      - If `proximal_logprobs` is provided, use it directly (original behavior)
      - Else construct it with interpolation:
            logpi_prox = (1-alpha) * logpi_old + alpha * logpi_new
        where alpha can be:
            - directly passed via `proximal_interp_alpha`
            - derived from versions via (proximal_version/current_version)
    """

    # if logprobs is not None:
    #     print(f"in decoupled ppo actor loss, logprobs shape: {logprobs.shape}",flush=True)
    # if old_logprobs is not None:
    #     print(f"in decoupled ppo actor loss, old_logprobs shape: {old_logprobs.shape}",flush=True)
    # if proximal_logprobs is not None:
    #     print(f"in decoupled ppo actor loss, proximal_logprobs shape: {proximal_logprobs.shape}",flush=True)
    # if versions is not None:
    #     print(f"in decoupled ppo actor loss, versions shape: {versions.shape}",flush=True)
    # if advantages is not None:
    #     print(f"in decoupled ppo actor loss, advantages shape: {advantages.shape}",flush=True)
    # if loss_mask is not None:
    #     print(f"in decoupled ppo actor loss, loss_mask shape: {loss_mask.shape}",flush=True)
    # if current_version is not None:
    #     print(f"in decoupled ppo actor loss, current_version: {current_version}",flush=True)

    assert logprobs.dtype == torch.float32, (
        f"logprobs dtype: {logprobs.dtype}, needed torch.float32"
    )
    assert old_logprobs.dtype == torch.float32, (
        f"old_logprobs dtype: {old_logprobs.dtype}, needed torch.float32"
    )

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs, dtype=torch.bool)

    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    if proximal_logprobs is None:
        alpha = 0
        v_proximal = current_version - 1
        v_behav = versions.float()
        v_theta = float(current_version)

        version_diff = v_theta - v_behav
        version_gap = v_proximal - v_behav

        generated_tokens_mask = versions >= 0

        alpha = torch.where(
            (version_diff > 0) & generated_tokens_mask,
            version_gap / version_diff,
            torch.zeros_like(v_behav),
        ).unsqueeze(-1)

        alpha = torch.clamp(alpha, 0.0, 1.0)
        proximal_logprobs = old_logprobs + alpha * (logprobs - old_logprobs)
    else:
        assert proximal_logprobs.dtype == torch.float32, (
            f"proximal_logprobs dtype: {proximal_logprobs.dtype}, needed torch.float32"
        )
    # print(f"proximal_logprobs shape: {proximal_logprobs.shape},logprobs shape: {logprobs.shape}, old_logprobs shape: {old_logprobs.shape},alpha shape:{alpha.shape}")
    proximal_ratio = torch.where(
        loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0
    )
    clipped_proximal_ratio = torch.clamp(
        proximal_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    loss_mask_count = loss_mask.count_nonzero() or 1

    pg_loss1 = -advantages * proximal_ratio
    pg_loss2 = -advantages * clipped_proximal_ratio

    pg_loss = torch.max(pg_loss1, pg_loss2)

    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, (
            f"clip_ratio_c should be greater than 1.0, got {clip_ratio_c}"
        )
        pg_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(pg_loss, dtype=torch.bool)

    # Behavior weight term
    # NOTE: if you interpolate proximal_logprobs, this becomes a *soft behavior correction*
    behav_weight = torch.exp(proximal_logprobs - old_logprobs)
    behav_mask = (
        (behav_weight <= behave_weight_threshold).logical_and(loss_mask)
        if behave_weight_threshold is not None
        else loss_mask
    )
    behav_mask_count = behav_mask.count_nonzero() or 1
    behav_weight = torch.where(behav_mask, behav_weight, 0.0)

    if loss_mask_ratio is None:
        pg_loss = loss_agg_func(pg_loss * behav_weight, loss_mask)
    else:
        pg_loss = loss_agg_func(pg_loss * behav_weight, loss_mask, loss_mask_ratio)

    if critic_warmup:
        pg_loss = torch.tensor(0.0, device=pg_loss.device)

    with torch.no_grad():
        clip_mask = pg_loss1 < pg_loss2
        clip_fraction = (
            clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
        )
        dual_clip_fraction = (
            dual_clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
        )

        proximal_approx_kl = torch.where(loss_mask, logprobs - proximal_logprobs, 0.0)
        proximal_approx_kl = -proximal_approx_kl.sum() / loss_mask_count

        behav_approx_kl = torch.where(behav_mask, proximal_logprobs - old_logprobs, 0.0)
        behav_approx_kl = -behav_approx_kl.sum() / behav_mask_count

        behav_clip_fraction = 1.0 - (behav_mask_count / loss_mask_count)

    metrics_data = {
        "actor/policy_loss": pg_loss.detach(),
        "actor/proximal_ratio": (
            proximal_ratio.detach()[loss_mask].mean()
            if loss_mask.any()
            else proximal_ratio.detach().mean()
        ),
        "actor/clipped_proximal_ratio": (
            clipped_proximal_ratio.detach()[loss_mask].mean()
            if loss_mask.any()
            else clipped_proximal_ratio.detach().mean()
        ),
        "actor/clip_fraction": clip_fraction,
        "actor/dual_clip_fraction": dual_clip_fraction,
        "actor/behav_clip_fraction": behav_clip_fraction,
        "actor/proximal_approx_kl": proximal_approx_kl,
        "actor/behav_approx_kl": behav_approx_kl,
    }
    return pg_loss, metrics_data


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
    approx_kl = torch.where(loss_mask, (logprobs - old_logprobs).detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio

    clip_mask = policy_loss1.detach() < policy_loss2.detach()

    policy_loss = torch.max(policy_loss1, policy_loss2)
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask.logical_and_(loss_mask)

    clip_fraction = clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
    approx_kl = -approx_kl.sum() / loss_mask_count

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

    # Compile metrics for logging
    ratio_for_metrics = ratio.detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()
    loss_mask_for_metrics = loss_mask

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]

    # compute value clipping
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]

    # here to use value clipping
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    # value_loss = value_loss_original
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (prev_values - values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
        "critic/explained_variance": explained_variance.detach().item(),
    }
    return value_loss, metrics_data


@register_policy_loss("decoupled_actor_critic")
def compute_decoupled_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_decoupled_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data

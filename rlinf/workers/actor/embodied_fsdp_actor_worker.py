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

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.subtask import (
    outcome_actor_channel_key,
    parallel_outcome_sampling_enabled,
    reduce_first_episode_successes,
    reduce_trajectory_group_ids,
    reduce_trajectory_successes,
)
from rlinf.algorithms.utils import huber_loss, kl_penalty
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory, convert_trajectories_to_batch
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.critic_batch import export_critic_batch_shard
from rlinf.utils.distributed import (
    all_reduce_dict,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    flatten_nested_tensor_time_batch,
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
    trim_nested_tensor_time_dim,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    masked_mean,
    preprocess_embodied_batch,
    reshape_entropy,
)


def _select_rollout_trajectories(batch: dict, mask: torch.Tensor) -> dict:
    """Select trajectory batch entries while preserving nested tensor fields."""
    selected = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            selected[key] = value[:, mask].contiguous()
        elif isinstance(value, dict):
            selected[key] = _select_rollout_trajectories(value, mask)
        else:
            raise TypeError(f"Unsupported rollout batch field {key}: {type(value)}")
    return selected


def _masked_reference_kl(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    *,
    penalty_type: str,
    executed_action_mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Return reference-policy KL averaged over executed action coordinates."""
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            "Current and reference log-probabilities must have the same shape; "
            f"got {tuple(logprobs.shape)} and {tuple(ref_logprobs.shape)}."
        )

    mask = torch.ones_like(logprobs, dtype=torch.bool)
    if executed_action_mask is not None:
        action_mask = executed_action_mask.reshape(logprobs.shape[0], -1, 1)
        mask &= action_mask.to(torch.bool)
    if loss_mask is not None:
        macro_mask = loss_mask.reshape(logprobs.shape[0], -1).any(dim=-1, keepdim=True)
        mask &= macro_mask.unsqueeze(-1)

    weights = mask.to(logprobs.dtype)
    if sample_weights is not None:
        trajectory_weights = sample_weights.reshape(logprobs.shape[0], -1).mean(
            dim=-1, keepdim=True
        )
        weights = weights * trajectory_weights.unsqueeze(-1)

    penalties = kl_penalty(logprobs, ref_logprobs, penalty_type)
    return (penalties * weights).sum() / weights.sum().clamp_min(1.0)


def _gradient_cosines_from_gram(
    gram: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pairwise and per-state-to-aggregate gradient cosines."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"Gradient Gram matrix must be square, got {gram.shape}.")
    norms = gram.diag().clamp_min(0).sqrt()
    denominator = norms[:, None] * norms[None, :]
    pairwise = torch.where(
        denominator > 0,
        gram / denominator.clamp_min(torch.finfo(gram.dtype).eps),
        torch.nan,
    )
    aggregate_norm = gram.sum().clamp_min(0).sqrt()
    state_to_aggregate_denominator = norms * aggregate_norm
    state_to_aggregate = torch.where(
        state_to_aggregate_denominator > 0,
        gram.sum(dim=1)
        / state_to_aggregate_denominator.clamp_min(torch.finfo(gram.dtype).eps),
        torch.nan,
    )
    return pairwise, state_to_aggregate


def _gradient_cosines_to_direction(
    dot_products: torch.Tensor,
    gradient_norms: torch.Tensor,
    direction_norm: torch.Tensor,
) -> torch.Tensor:
    """Return gradient cosines to one shared direction."""
    denominator = gradient_norms * direction_norm
    return torch.where(
        denominator > 0,
        dot_products / denominator.clamp_min(torch.finfo(dot_products.dtype).eps),
        torch.nan,
    )


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self.kl_beta = float(self.cfg.algorithm.get("kl_beta", 0.0))
        self.kl_penalty_type = self.cfg.algorithm.get("kl_penalty_type", "low_var_kl")
        self.combine_reference_model = bool(
            self.cfg.actor.get("combine_reference_model", True)
        )
        self.ref_policy_state_dict = None

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        self._resume_checkpoint_loaded = False
        self._resume_warmup_start_step: int | None = None
        self._pending_policy_gradient_diagnostics: dict | None = None
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        critic_global_batch_size = self.cfg.actor.get("critic_global_batch_size", None)
        self.critic_global_batch_size = int(
            critic_global_batch_size
            if critic_global_batch_size is not None
            else self.cfg.actor.global_batch_size
        )
        if self.critic_global_batch_size % (
            self.cfg.actor.micro_batch_size * self._world_size
        ):
            raise ValueError(
                "actor.critic_global_batch_size must be divisible by "
                "actor.micro_batch_size * actor world size; got "
                f"{self.critic_global_batch_size}, "
                f"{self.cfg.actor.micro_batch_size}, and {self._world_size}."
            )
        self.cache_critic_inputs = bool(
            self.cfg.actor.get("cache_critic_inputs", False)
        )
        if self.cache_critic_inputs:
            if (
                SupportedModel(self.cfg.actor.model.model_type)
                != SupportedModel.OPENPI_RLINF
            ):
                raise ValueError(
                    "actor.cache_critic_inputs currently supports openpi_rlinf only."
                )
            if not self.cfg.actor.model.openpi.get("detach_critic_input", False):
                raise ValueError(
                    "actor.cache_critic_inputs requires "
                    "actor.model.openpi.detach_critic_input=true."
                )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        policy_epochs = self.cfg.actor.get("policy_update_epochs", None)
        critic_epochs = self.cfg.actor.get("critic_update_epochs", None)
        self.use_independent_update_epochs = (
            policy_epochs is not None or critic_epochs is not None
        )
        if self.use_independent_update_epochs:
            if self.cfg.algorithm.loss_type != "actor_critic":
                raise ValueError(
                    "Independent policy/critic update epochs require "
                    "algorithm.loss_type='actor_critic'."
                )
            if self.critic_only:
                raise ValueError(
                    "actor.optim.critic_only cannot be combined with independent "
                    "policy/critic update epochs. Set policy_update_epochs=0 instead."
                )
            self.policy_update_epochs = int(policy_epochs or 0)
            self.critic_update_epochs = int(critic_epochs or 0)
            if self.policy_update_epochs < 0 or self.critic_update_epochs < 0:
                raise ValueError(
                    "Policy and critic update epochs must be non-negative."
                )
            if self.policy_update_epochs + self.critic_update_epochs == 0:
                raise ValueError(
                    "At least one policy or critic update epoch is required."
                )
        self.resume_critic_warmup_global_steps = int(
            self.cfg.actor.get("resume_critic_warmup_global_steps", 0)
        )
        self.resume_critic_warmup_update_epochs = int(
            self.cfg.actor.get(
                "resume_critic_warmup_update_epochs",
                self.critic_update_epochs
                if self.use_independent_update_epochs
                else self.update_epoch,
            )
        )
        warmup_value_clip = self.cfg.actor.get("resume_critic_warmup_value_clip", None)
        self.resume_critic_warmup_value_clip = (
            float(warmup_value_clip) if warmup_value_clip is not None else None
        )

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )
        self._candidate_rollout_batch: dict | None = None
        self._accepted_rollout_batches: list[dict] | None = None

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.kl_beta > 0:
            if not self.combine_reference_model:
                raise NotImplementedError(
                    "Embodied reference KL currently requires "
                    "actor.combine_reference_model=true."
                )

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def load_checkpoint(self, load_path: str) -> None:
        """Restore training state and arm optional post-resume critic warmup."""
        super().load_checkpoint(load_path)
        self._resume_checkpoint_loaded = True

    def _resume_critic_warmup_active(self) -> bool:
        """Whether this global step is reserved for post-resume critic updates."""
        if self._resume_warmup_start_step is None:
            return False
        return self.version < (
            self._resume_warmup_start_step + self.resume_critic_warmup_global_steps
        )

    def _training_update_phases(
        self,
    ) -> tuple[bool, tuple[tuple[str, int, bool, bool], ...]]:
        """Build effective policy/critic phases for the current global step."""
        resume_critic_warmup = self._resume_critic_warmup_active()
        if self.use_independent_update_epochs:
            policy_epochs = 0 if resume_critic_warmup else self.policy_update_epochs
            critic_epochs = (
                self.resume_critic_warmup_update_epochs
                if resume_critic_warmup
                else self.critic_update_epochs
            )
            phases = (
                ("policy", policy_epochs, True, False),
                ("critic", critic_epochs, False, True),
            )
        else:
            phases = (
                (
                    "joint",
                    int(self.cfg.algorithm.get("update_epoch", 1)),
                    not self.critic_only,
                    self.cfg.algorithm.adv_type in ("gae", "subtask_gae"),
                ),
            )
        return resume_critic_warmup, phases

    def capture_reference_policy(self) -> None:
        """Capture the active actor weights as the frozen reference policy.

        The runner invokes this after loading an optional resume checkpoint. This
        ordering is important: capturing during ``init_worker`` would anchor a
        resumed run to ``actor.model.model_path`` instead of the resumed policy.
        """
        if self.kl_beta <= 0:
            return
        if not self.combine_reference_model:
            raise NotImplementedError(
                "Embodied reference KL currently requires "
                "actor.combine_reference_model=true."
            )

        restore_weight_offload = self.is_weight_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        self.ref_policy_state_dict = self._strategy.get_weight_swap_state(self.model)
        if restore_weight_offload:
            self.offload_param_and_grad()
        self.log_info(
            "Captured the active actor as the frozen KL reference "
            f"at policy step {self.version} "
            f"(beta={self.kl_beta:g}, penalty={self.kl_penalty_type})."
        )

    def _gradient_clipping_metrics(
        self,
        total_norm: float,
        active_roles: set[str] | None = None,
    ) -> dict[str, float]:
        """Build explicit total and branch-global gradient metrics."""
        metrics = {
            "actor/grad_norm_before_clip": total_norm,
            "actor/grad_norm_after_clip": self.last_grad_norm_after_clip,
            "actor/grad_clip_coef": self.last_grad_clip_coef,
        }
        branch_namespaces = {
            "policy": "actor/policy",
            "value": "critic/value",
        }
        for role, namespace in branch_namespaces.items():
            if active_roles is not None and role not in active_roles:
                continue
            if role not in self.last_grad_norms_before_clip:
                continue
            metrics[f"{namespace}_grad_norm_before_clip"] = (
                self.last_grad_norms_before_clip[role]
            )
            metrics[f"{namespace}_grad_norm_after_clip"] = (
                self.last_grad_norms_after_clip[role]
            )
            metrics[f"{namespace}_grad_clip_coef"] = self.last_grad_clip_coefs[role]
        return metrics

    def _critic_fixed_batch_snapshot(
        self,
        epoch_metrics: dict[str, list],
        update_index: int,
    ) -> dict[str, float]:
        """Summarize critic fit before one repeated fixed-batch update."""
        visible_metrics = dict(epoch_metrics)
        explained_variance_stats = pop_critic_explained_variance_stats(visible_metrics)
        if not explained_variance_stats or "critic/value_loss" not in visible_metrics:
            return {}

        reduced_stats = all_reduce_dict(
            explained_variance_stats,
            op=torch.distributed.ReduceOp.SUM,
        )
        explained_variance = compute_critic_explained_variance_from_stats(
            reduced_stats
        ).item()
        value_loss = (
            torch.stack(
                [
                    torch.as_tensor(value, device=self.device, dtype=torch.float32)
                    for value in visible_metrics["critic/value_loss"]
                ]
            )
            .mean()
            .item()
        )
        value_loss = all_reduce_dict(
            {"value_loss": value_loss},
            op=torch.distributed.ReduceOp.AVG,
        )["value_loss"]

        if self._rank == 0:
            self.log_info(
                f"Fixed-batch critic before update {update_index}: "
                f"value_loss={value_loss:.6f}, "
                f"explained_variance={explained_variance:.6f}"
            )
        suffix = f"before_update_{update_index:04d}"
        return {
            f"critic/fixed_batch/value_loss_{suffix}": value_loss,
            f"critic/fixed_batch/explained_variance_{suffix}": explained_variance,
        }

    def _phase_batch_config(self, phase: str) -> tuple[int, int, int]:
        """Return global, per-rank, and micro-batch accumulation sizes."""
        global_batch_size = (
            self.critic_global_batch_size
            if phase == "critic"
            else int(self.cfg.actor.global_batch_size)
        )
        batch_size_per_rank = global_batch_size // self._world_size
        gradient_accumulation = batch_size_per_rank // self.cfg.actor.micro_batch_size
        return global_batch_size, batch_size_per_rank, gradient_accumulation

    def _build_critic_input_cache(self) -> dict[str, float]:
        """Cache detached VLM prefix features for repeated critic updates."""
        forward_inputs = self.rollout_batch["forward_inputs"]
        if "critic_pooled_prefix" in forward_inputs:
            return {}

        sample_count = int(self.rollout_batch["prev_logprobs"].shape[0])
        num_chunks = (
            sample_count + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        started = time.perf_counter()
        cached_chunks = []
        for micro_batch in split_dict_to_chunk(forward_inputs, num_chunks):
            micro_batch = put_tensor_device(micro_batch, self.device)
            with torch.no_grad(), self.amp_context:
                output = self.model(
                    forward_inputs=micro_batch,
                    compute_logprobs=False,
                    compute_entropy=False,
                    compute_values=False,
                    export_critic_inputs=True,
                    use_cache=False,
                )
            cached_chunks.append(
                {
                    key: value.detach().to("cpu", copy=True)
                    for key, value in output["critic_inputs"].items()
                }
            )
        cache = cat_list_of_dict_tensor(cached_chunks)
        # Policy updates have already completed, so the raw images, prompts,
        # action chains, and denoise metadata are dead for this global step.
        # Retain only the exact inputs consumed by the detached critic. Besides
        # avoiding repeated VLM compute, this prevents every critic micro-batch
        # from copying the much larger policy payload back to the GPU.
        self.rollout_batch["forward_inputs"] = {
            "obs_state": forward_inputs["obs_state"],
            **cache,
        }
        cache_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in cache.values()
        )
        elapsed = time.perf_counter() - started
        self.log_info(
            "Cached detached critic inputs for "
            f"{sample_count} samples ({cache_bytes / 2**30:.2f} GiB) "
            f"in {elapsed:.2f}s."
        )
        return {
            "critic/cache_enabled": 1.0,
            "critic/cache_samples": float(sample_count * self._world_size),
            "critic/cache_gib_per_rank": cache_bytes / 2**30,
            "critic/cache_build_seconds": elapsed,
        }

    def _critic_post_update_metrics(self) -> dict[str, float]:
        """Evaluate the final critic once, without mixing optimizer epochs."""
        forward_inputs = self.rollout_batch["forward_inputs"]
        returns = self.rollout_batch["returns"]
        loss_mask = self.rollout_batch.get("loss_mask")
        sample_count = int(returns.shape[0])
        num_chunks = (
            sample_count + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        forward_chunks = split_dict_to_chunk(forward_inputs, num_chunks)
        return_chunks = torch.chunk(returns, num_chunks, dim=0)
        mask_chunks = (
            torch.chunk(loss_mask, num_chunks, dim=0)
            if loss_mask is not None
            else [None] * num_chunks
        )

        local_values = []
        local_targets = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for inputs, targets, mask in zip(
                    forward_chunks,
                    return_chunks,
                    mask_chunks,
                    strict=True,
                ):
                    inputs = put_tensor_device(inputs, self.device)
                    with self.amp_context:
                        output = self.model(
                            forward_inputs=inputs,
                            compute_logprobs=False,
                            compute_entropy=False,
                            compute_values=True,
                            use_cache=False,
                        )
                    values = output["values"].detach().float().cpu()
                    targets = targets.detach().float().cpu()
                    if values.shape != targets.shape:
                        if values.numel() != targets.numel():
                            raise ValueError(
                                "Post-update critic predictions and targets have "
                                f"incompatible shapes {tuple(values.shape)} and "
                                f"{tuple(targets.shape)}."
                            )
                        values = values.reshape_as(targets)
                    if mask is not None:
                        mask = mask.detach().to(dtype=torch.bool, device="cpu")
                        if mask.shape != targets.shape:
                            mask = torch.broadcast_to(mask, targets.shape)
                        values = values[mask]
                        targets = targets[mask]
                    else:
                        values = values.reshape(-1)
                        targets = targets.reshape(-1)
                    local_values.append(values)
                    local_targets.append(targets)
        finally:
            self.model.train(was_training)

        values = torch.cat(local_values).to(self.device)
        targets = torch.cat(local_targets).to(self.device)
        errors = targets - values
        count = torch.tensor(float(values.numel()), device=self.device)
        sums = torch.stack(
            (
                count,
                values.sum(),
                (values * values).sum(),
                targets.sum(),
                (targets * targets).sum(),
                errors.sum(),
                (errors * errors).sum(),
                errors.abs().sum(),
            )
        ).float()
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        minima = torch.stack((values.min(), targets.min())).float()
        maxima = torch.stack((values.max(), targets.max())).float()
        torch.distributed.all_reduce(minima, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)

        count = sums[0].clamp_min(1.0)
        value_mean = sums[1] / count
        target_mean = sums[3] / count
        error_mean = sums[5] / count
        value_variance = (sums[2] / count - value_mean.square()).clamp_min(0.0)
        target_variance = (sums[4] / count - target_mean.square()).clamp_min(0.0)
        error_variance = (sums[6] / count - error_mean.square()).clamp_min(0.0)
        mse = sums[6] / count
        huber_delta = self.cfg.algorithm.get("huber_delta", None)
        explained_variance = (
            1.0 - error_variance / target_variance
            if target_variance > 0
            else torch.tensor(float("nan"), device=self.device)
        )
        metrics = {
            CRITIC_EXPLAINED_VARIANCE_KEY: explained_variance.item(),
            "critic/post_update/explained_variance": explained_variance.item(),
            "critic/post_update/valid_samples": sums[0].item(),
            "critic/post_update/value_min": minima[0].item(),
            "critic/post_update/value_mean": value_mean.item(),
            "critic/post_update/value_std": value_variance.sqrt().item(),
            "critic/post_update/value_max": maxima[0].item(),
            "critic/post_update/target_min": minima[1].item(),
            "critic/post_update/target_mean": target_mean.item(),
            "critic/post_update/target_std": target_variance.sqrt().item(),
            "critic/post_update/target_max": maxima[1].item(),
            "critic/post_update/error_mean": error_mean.item(),
            "critic/post_update/mae": (sums[7] / count).item(),
            "critic/post_update/mse": mse.item(),
            "critic/post_update/rmse": mse.sqrt().item(),
        }
        if huber_delta is not None:
            local_huber_sum = huber_loss(errors, float(huber_delta)).sum()
            torch.distributed.all_reduce(
                local_huber_sum, op=torch.distributed.ReduceOp.SUM
            )
            metrics["critic/post_update/huber_loss_unclipped"] = (
                local_huber_sum / count
            ).item()
        return metrics

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(
        self, input_channel: Channel
    ) -> list[bool] | dict[int, list[bool]] | None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        sampling_cfg = self.cfg.algorithm.get("outcome_dynamic_sampling", {})
        sampling_enabled = bool(sampling_cfg.get("enabled", False))
        parallel_groups = parallel_outcome_sampling_enabled(sampling_cfg)
        channel_key = outcome_actor_channel_key(self._rank) if parallel_groups else None

        recv_list = []
        for _ in range(split_num):
            if channel_key is None:
                get_work = input_channel.get(async_op=True)
            else:
                get_work = input_channel.get(key=channel_key, async_op=True)
            trajectory: Trajectory = await get_work.async_wait()
            recv_list.append(trajectory)

        rollout_batch = convert_trajectories_to_batch(recv_list)
        if sampling_enabled:
            if self._accepted_rollout_batches is None:
                raise RuntimeError(
                    "begin_rollout_group_collection must be called before receiving "
                    "outcome-dynamic rollout groups."
                )
            self._candidate_rollout_batch = rollout_batch
        else:
            self.rollout_batch = self._process_received_rollout_batch(rollout_batch)

        if self.cfg.env.train.auto_reset:
            terminations = rollout_batch.get("terminations")
            dones = rollout_batch.get("dones")
            if terminations is None or dones is None:
                raise RuntimeError(
                    "Auto-reset outcome sampling requires trajectory termination "
                    "and done flags."
                )
            outcomes = reduce_first_episode_successes(terminations, dones)
        else:
            successes = rollout_batch.get("successes")
            if successes is None:
                return None
            outcomes = reduce_trajectory_successes(successes)
        if not parallel_groups:
            return outcomes

        group_ids = rollout_batch.get("outcome_group_ids")
        if group_ids is None:
            raise RuntimeError("Parallel outcome sampling requires rollout group IDs.")
        trajectory_group_ids = reduce_trajectory_group_ids(group_ids)
        grouped_outcomes: dict[int, list[bool]] = {}
        for group_id, outcome in zip(trajectory_group_ids, outcomes, strict=True):
            grouped_outcomes.setdefault(group_id, []).append(outcome)
        return grouped_outcomes

    def begin_rollout_group_collection(self) -> None:
        """Start one update's collection of independently accepted groups."""
        self._candidate_rollout_batch = None
        self._accepted_rollout_batches = []

    def accept_rollout_group(
        self,
        logical_group_index: int | None = None,
        episode_index: int | None = None,
        policy_trainable: bool | None = None,
    ) -> None:
        """Retain the most recently received candidate as a trainable group."""
        if self._accepted_rollout_batches is None:
            raise RuntimeError("Rollout group collection has not started.")
        if self._candidate_rollout_batch is None:
            raise RuntimeError("No candidate rollout group is available to accept.")
        self._annotate_candidate_state(
            self._candidate_rollout_batch,
            logical_group_index=logical_group_index,
            episode_index=episode_index,
            policy_trainable=policy_trainable,
        )
        self._accepted_rollout_batches.append(self._candidate_rollout_batch)
        self._candidate_rollout_batch = None

    @staticmethod
    def _annotate_candidate_state(
        batch: dict[str, torch.Tensor],
        *,
        logical_group_index: int | None,
        episode_index: int | None,
        policy_trainable: bool | None = None,
    ) -> None:
        """Attach stable logical-state provenance to an actor rollout batch."""
        if (
            logical_group_index is None
            and episode_index is None
            and policy_trainable is None
        ):
            return
        reference = batch.get("outcome_group_ids")
        if reference is None:
            raise RuntimeError(
                "Logical outcome-state provenance requires outcome_group_ids."
            )
        if logical_group_index is not None:
            batch["outcome_logical_group_ids"] = torch.full_like(
                reference,
                int(logical_group_index),
                dtype=torch.long,
            )
        if episode_index is not None:
            batch["outcome_episode_indices"] = torch.full_like(
                reference,
                int(episode_index),
                dtype=torch.long,
            )
        if policy_trainable is not None:
            batch["outcome_policy_trainable"] = torch.full_like(
                reference,
                bool(policy_trainable),
                dtype=torch.bool,
            )

    def accept_rollout_groups(
        self,
        group_ids: list[int],
        state_metadata: dict[int, dict[str, int | bool]] | None = None,
    ) -> None:
        """Retain selected groups from one concurrently sampled candidate batch."""
        if self._accepted_rollout_batches is None:
            raise RuntimeError("Rollout group collection has not started.")
        if self._candidate_rollout_batch is None:
            raise RuntimeError("No candidate rollout groups are available to accept.")
        candidate_group_ids = self._candidate_rollout_batch.get("outcome_group_ids")
        if candidate_group_ids is None:
            raise RuntimeError("Parallel outcome sampling requires rollout group IDs.")

        reduced_group_ids = torch.tensor(
            reduce_trajectory_group_ids(candidate_group_ids),
            device=candidate_group_ids.device,
        )
        for group_id in group_ids:
            group_mask = reduced_group_ids == group_id
            if group_mask.sum().item() != 1:
                raise RuntimeError(
                    "Each actor rank must receive exactly one trajectory per outcome "
                    f"group; group {group_id} has {group_mask.sum().item()}."
                )
            selected_batch = _select_rollout_trajectories(
                self._candidate_rollout_batch,
                group_mask,
            )
            metadata = (state_metadata or {}).get(group_id, {})
            self._annotate_candidate_state(
                selected_batch,
                logical_group_index=metadata.get("logical_group_index"),
                episode_index=metadata.get("episode_index"),
                policy_trainable=metadata.get("policy_trainable"),
            )
            self._accepted_rollout_batches.append(selected_batch)
        self._candidate_rollout_batch = None

    def finalize_rollout_group_collection(self, expected_groups: int) -> None:
        """Merge accepted groups and prepare the complete actor rollout batch."""
        if self._accepted_rollout_batches is None:
            raise RuntimeError("Rollout group collection has not started.")
        if len(self._accepted_rollout_batches) != expected_groups:
            raise RuntimeError(
                f"Expected {expected_groups} accepted rollout groups, got "
                f"{len(self._accepted_rollout_batches)}."
            )
        merged_batch = cat_list_of_dict_tensor(
            self._accepted_rollout_batches,
            dim=1,
        )
        self._accepted_rollout_batches = None
        self.rollout_batch = self._process_received_rollout_batch(merged_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Merge rollout epochs and construct training masks and weights."""
        return preprocess_embodied_batch(
            rollout_batch,
            rollout_epoch=self.cfg.env.train.rollout_epoch,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            self.compute_opd_teacher_logprobs()

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "prev_logprobs": self.rollout_batch.get("prev_logprobs", None),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "executed_action_mask": self.rollout_batch.get(
                "executed_action_mask", None
            ),
            "subtask_ids": self.rollout_batch.get("subtask_ids", None),
            "outcome_logical_group_ids": self.rollout_batch.get(
                "outcome_logical_group_ids", None
            ),
            "advantage_mode": self.cfg.algorithm.get("advantage_mode", None),
            "advantage_std_floor": self.cfg.algorithm.get("advantage_std_floor", 0.1),
            "advantage_clip": self.cfg.algorithm.get("advantage_clip", None),
            "advantage_normalization_scope": self.cfg.algorithm.get(
                "advantage_normalization_scope", "subtask"
            ),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        export_dir = self.cfg.actor.get("critic_batch_export_dir", None)
        if export_dir:
            artifact_path = export_critic_batch_shard(
                self.rollout_batch,
                export_dir,
                actor_rank=self._rank,
                actor_world_size=self._world_size,
                global_step=self.version,
                metadata={
                    "model_path": str(self.cfg.actor.model.model_path),
                    "reward_type": str(self.cfg.algorithm.reward_type),
                    "adv_type": str(self.cfg.algorithm.adv_type),
                    "gamma": float(self.cfg.algorithm.get("gamma", 1.0)),
                    "gae_lambda": float(self.cfg.algorithm.get("gae_lambda", 1.0)),
                    "outcome_group_size": int(
                        self.cfg.algorithm.get("outcome_dynamic_sampling", {}).get(
                            "group_size", 0
                        )
                    ),
                    "outcome_groups_per_update": int(
                        self.cfg.algorithm.get("outcome_dynamic_sampling", {}).get(
                            "groups_per_update", 0
                        )
                    ),
                    "outcome_parallel_groups": bool(
                        self.cfg.algorithm.get("outcome_dynamic_sampling", {}).get(
                            "parallel_groups", False
                        )
                    ),
                },
            )
            self.logger.info(
                "Exported unshuffled critic batch shard to %s", artifact_path
            )
        return rollout_metrics

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    @Worker.timer("actor/recompute_prev_logprobs")
    def recompute_prev_logprobs(self) -> dict[str, float]:
        """Recompute PPO reference log-probs with the trainable actor weights.

        Rollout and FSDP actor copies can produce measurably different flow-SDE
        log-probs even before an optimizer step. PPO must compare the updated
        policy against a reference evaluated by the same model implementation,
        so replace rollout-side values before shuffling the training batch.
        """
        assert "forward_inputs" in self.rollout_batch, (
            "Actor-side log-prob recomputation requires rollout forward_inputs."
        )
        rollout_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = rollout_logprobs.shape[:2]
        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        flat_batch_size = time_dim * batch_dim
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size

        was_training = self.model.training
        self.model.eval()

        def compute_logprobs() -> torch.Tensor:
            recomputed = []
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    output = self.model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                    )
                recomputed.append(output["logprobs"].detach().cpu())
            return torch.cat(recomputed, dim=0).reshape_as(rollout_logprobs)

        recomputed_logprobs = None
        ref_logprobs = None
        try:
            with torch.no_grad():
                recomputed_logprobs = compute_logprobs()
                if self.kl_beta > 0:
                    if self.ref_policy_state_dict is None:
                        raise RuntimeError(
                            "Reference KL is enabled but the frozen reference "
                            "policy has not been initialized."
                        )
                    with self.swap_sharded_model_state_dict(self.ref_policy_state_dict):
                        ref_logprobs = compute_logprobs()
        finally:
            self.model.train(was_training)

        assert recomputed_logprobs is not None
        assert recomputed_logprobs.shape == rollout_logprobs.shape
        drift = recomputed_logprobs - rollout_logprobs
        self.rollout_batch["prev_logprobs"] = recomputed_logprobs
        metrics = {
            "actor/rollout_logprob_abs_diff": drift.abs().mean().item(),
            "actor/rollout_logprob_diff": drift.mean().item(),
            "actor/rollout_logprob_abs_diff_max": drift.abs().max().item(),
        }
        if ref_logprobs is not None:
            self.rollout_batch["ref_logprobs"] = ref_logprobs
            reference_drift = recomputed_logprobs - ref_logprobs
            metrics.update(
                {
                    "actor/reference_logprob_abs_diff": (
                        reference_drift.abs().mean().item()
                    ),
                    "actor/reference_logprob_diff": reference_drift.mean().item(),
                }
            )
        return metrics

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    @staticmethod
    def _mask_batch_to_logical_state(
        batch: dict[str, torch.Tensor], logical_state_id: int
    ) -> dict[str, torch.Tensor]:
        """Keep one logical state's additive contribution to the PPO loss."""
        logical_ids = batch.get("outcome_logical_group_ids")
        if logical_ids is None:
            raise RuntimeError(
                "Per-state policy diagnostics require "
                "outcome_logical_group_ids in the actor batch."
            )
        selected = logical_ids.eq(logical_state_id)
        sample_weights = batch.get("sample_weights")
        if sample_weights is None:
            sample_weights = torch.ones_like(selected, dtype=torch.float32)
        selector = selected.reshape(
            selected.shape[0], *([1] * (sample_weights.ndim - 1))
        )
        masked_batch = dict(batch)
        masked_batch["sample_weights"] = sample_weights * selector.to(
            sample_weights.dtype
        )
        policy_weights = batch.get("policy_sample_weights")
        if policy_weights is not None:
            masked_batch["policy_sample_weights"] = policy_weights * selector.to(
                policy_weights.dtype
            )
        return masked_batch

    def _backward_policy_diagnostic_batch(
        self,
        train_global_batch: dict[str, torch.Tensor],
        *,
        logical_state_id: int | None,
        include_reference_kl: bool,
        seed: int,
    ) -> list[torch.Tensor | None]:
        """Backpropagate one diagnostic objective without taking an optimizer step."""
        train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
        micro_batches = split_dict_to_chunk(
            train_global_batch,
            train_global_batch_size // self.cfg.actor.micro_batch_size,
        )
        if len(micro_batches) != self.gradient_accumulation:
            raise ValueError(
                "Policy-gradient diagnostics require exactly one global batch per "
                "actor rank."
            )

        self.optimizer.zero_grad()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        original_kl_beta = self.kl_beta
        if not include_reference_kl:
            self.kl_beta = 0.0
        try:
            ignored_metrics: dict[str, list[float]] = {}
            for index, micro_batch in enumerate(micro_batches):
                if logical_state_id is not None:
                    micro_batch = self._mask_batch_to_logical_state(
                        micro_batch, logical_state_id
                    )
                self.train_micro_batch(
                    micro_batch=micro_batch,
                    metrics=ignored_metrics,
                    is_last=(index + 1) == len(micro_batches),
                    update_policy=True,
                    update_value=False,
                )
        finally:
            self.kl_beta = original_kl_beta

        gradients = []
        for parameter in self._optimizer_parameters_by_role["policy"]:
            gradient = parameter.grad
            gradients.append(
                None if gradient is None else gradient.detach().float().cpu().clone()
            )
        self.optimizer.zero_grad()
        return gradients

    @staticmethod
    def _logical_state_metadata(
        train_global_batch: dict[str, torch.Tensor],
    ) -> dict[int, int]:
        logical_ids = train_global_batch.get("outcome_logical_group_ids")
        episode_indices = train_global_batch.get("outcome_episode_indices")
        if logical_ids is None or episode_indices is None:
            raise RuntimeError(
                "Per-state policy diagnostics require logical group and episode IDs."
            )
        metadata = {}
        for logical_state_id in torch.unique(logical_ids).tolist():
            episodes = torch.unique(episode_indices[logical_ids == logical_state_id])
            if episodes.numel() != 1:
                raise RuntimeError(
                    f"Logical state {logical_state_id} maps to episodes "
                    f"{episodes.tolist()} in one actor batch."
                )
            metadata[int(logical_state_id)] = int(episodes.item())
        return metadata

    @staticmethod
    def _state_signal_sufficient_statistics(
        train_global_batch: dict[str, torch.Tensor],
        state_ids: list[int],
    ) -> torch.Tensor:
        """Return count/sum/squared-sum statistics for advantages and returns."""
        logical_ids = train_global_batch["outcome_logical_group_ids"].reshape(-1)
        loss_mask = train_global_batch.get("loss_mask")
        if loss_mask is None:
            valid = torch.ones_like(logical_ids, dtype=torch.bool)
        else:
            valid = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=-1)
        advantages = train_global_batch["advantages"].reshape(-1).float()
        returns = train_global_batch["returns"].reshape(-1).float()
        rows = []
        for state_id in state_ids:
            selected = valid & logical_ids.eq(state_id)
            state_advantages = advantages[selected]
            state_returns = returns[selected]
            rows.append(
                torch.tensor(
                    [
                        float(selected.sum()),
                        float(state_advantages.sum()),
                        float(state_advantages.square().sum()),
                        float((state_advantages > 0).sum()),
                        float(state_returns.sum()),
                        float(state_returns.square().sum()),
                    ],
                    dtype=torch.float64,
                )
            )
        return torch.stack(rows)

    def _run_policy_gradient_diagnostics(
        self, train_global_batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Measure exact PPO gradient conflicts among logical initial states."""
        diagnostics_cfg = self.cfg.actor.policy_gradient_diagnostics
        if self.grad_scaler.is_enabled():
            raise ValueError(
                "Policy-gradient diagnostics require actor FSDP grad_scaler=false."
            )
        if self.enable_sft_co_train or self.cfg.algorithm.entropy_bonus > 0:
            raise ValueError(
                "Policy-gradient diagnostics currently require SFT co-training "
                "and entropy bonuses to be disabled."
            )
        if self.critic_only or self.critic_warmup_steps > 0:
            raise ValueError(
                "Policy-gradient diagnostics require an active, non-warmup policy."
            )
        measure_optimizer_update = bool(
            diagnostics_cfg.get("measure_optimizer_update", False)
        )
        if measure_optimizer_update and diagnostics_cfg.get("only", False):
            raise ValueError(
                "Measuring the optimizer update requires "
                "policy_gradient_diagnostics.only=false."
            )

        local_metadata = self._logical_state_metadata(train_global_batch)
        gathered_metadata: list[dict[int, int] | None] = [
            None
        ] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_metadata, local_metadata)
        state_metadata: dict[int, int] = {}
        for metadata in gathered_metadata:
            assert metadata is not None
            for state_id, episode_index in metadata.items():
                previous = state_metadata.setdefault(state_id, episode_index)
                if previous != episode_index:
                    raise RuntimeError(
                        f"Logical state {state_id} maps to both episode {previous} "
                        f"and episode {episode_index} across actor ranks."
                    )
        state_ids = sorted(state_metadata)
        max_states = int(diagnostics_cfg.get("max_states", 0))
        if 0 < max_states < len(state_ids):
            raise ValueError(
                "policy_gradient_diagnostics.max_states cannot exclude states "
                "from the complete actor batch because that would invalidate the "
                "aggregate-gradient and KL decomposition."
            )
        if len(state_ids) < 2:
            raise ValueError(
                "Policy-gradient conflict diagnostics require at least two states."
            )

        seed = int(diagnostics_cfg.get("seed", self.cfg.actor.seed + self.version))
        self.log_info(
            f"Computing exact PPO gradient conflicts for {len(state_ids)} logical "
            "states without updating model weights."
        )
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        state_gradients = [
            self._backward_policy_diagnostic_batch(
                train_global_batch,
                logical_state_id=state_id,
                include_reference_kl=False,
                seed=seed,
            )
            for state_id in state_ids
        ]
        ppo_global_gradients = self._backward_policy_diagnostic_batch(
            train_global_batch,
            logical_state_id=None,
            include_reference_kl=False,
            seed=seed,
        )
        combined_gradients = self._backward_policy_diagnostic_batch(
            train_global_batch,
            logical_state_id=None,
            include_reference_kl=True,
            seed=seed,
        )

        gram = torch.zeros((len(state_ids), len(state_ids)), dtype=torch.float64)
        ppo_norm_squared = 0.0
        combined_norm_squared = 0.0
        kl_norm_squared = 0.0
        ppo_combined_dot = 0.0
        ppo_kl_dot = 0.0
        state_sum_residual_norm_squared = 0.0
        state_ppo_dots = torch.zeros(len(state_ids), dtype=torch.float64)
        state_combined_dots = torch.zeros(len(state_ids), dtype=torch.float64)
        for parameter_index, combined_gradient in enumerate(combined_gradients):
            available = [gradients[parameter_index] for gradients in state_gradients]
            ppo_global_gradient = ppo_global_gradients[parameter_index]
            if (
                combined_gradient is None
                and ppo_global_gradient is None
                and all(gradient is None for gradient in available)
            ):
                continue
            reference = (
                combined_gradient
                if combined_gradient is not None
                else (
                    ppo_global_gradient
                    if ppo_global_gradient is not None
                    else next(
                        gradient for gradient in available if gradient is not None
                    )
                )
            )
            stacked = torch.stack(
                [
                    torch.zeros_like(reference).reshape(-1)
                    if gradient is None
                    else gradient.reshape(-1)
                    for gradient in available
                ]
            )
            gram += (stacked @ stacked.T).double()
            state_sum_gradient = stacked.sum(dim=0)
            ppo_gradient = (
                torch.zeros_like(state_sum_gradient)
                if ppo_global_gradient is None
                else ppo_global_gradient.reshape(-1)
            )
            if combined_gradient is None:
                combined_flat = torch.zeros_like(ppo_gradient)
            else:
                combined_flat = combined_gradient.reshape(-1)
            kl_gradient = combined_flat - ppo_gradient
            ppo_norm_squared += float(torch.dot(ppo_gradient, ppo_gradient))
            combined_norm_squared += float(torch.dot(combined_flat, combined_flat))
            kl_norm_squared += float(torch.dot(kl_gradient, kl_gradient))
            ppo_combined_dot += float(torch.dot(ppo_gradient, combined_flat))
            ppo_kl_dot += float(torch.dot(ppo_gradient, kl_gradient))
            state_ppo_dots += (stacked @ ppo_gradient).double()
            state_combined_dots += (stacked @ combined_flat).double()
            state_sum_residual = state_sum_gradient - ppo_gradient
            state_sum_residual_norm_squared += float(
                torch.dot(state_sum_residual, state_sum_residual)
            )

        signal_stats = self._state_signal_sufficient_statistics(
            train_global_batch, state_ids
        )
        packed = torch.cat(
            [
                gram.flatten(),
                torch.tensor(
                    [
                        ppo_norm_squared,
                        combined_norm_squared,
                        kl_norm_squared,
                        ppo_combined_dot,
                        ppo_kl_dot,
                        state_sum_residual_norm_squared,
                    ],
                    dtype=torch.float64,
                ),
                state_ppo_dots,
                state_combined_dots,
                signal_stats.flatten(),
            ]
        ).to(self.device)
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        packed = packed.cpu()
        gram_size = len(state_ids) ** 2
        gram = packed[:gram_size].reshape(len(state_ids), len(state_ids))
        gradient_scalars = packed[gram_size : gram_size + 6]
        offset = gram_size + 6
        state_ppo_dots = packed[offset : offset + len(state_ids)]
        offset += len(state_ids)
        state_combined_dots = packed[offset : offset + len(state_ids)]
        offset += len(state_ids)
        signal_stats = packed[offset:].reshape(len(state_ids), 6)

        pairwise_cosines, state_to_aggregate = _gradient_cosines_from_gram(gram)
        ppo_norm, combined_norm, kl_norm = gradient_scalars[:3].clamp_min(0).sqrt()
        ppo_combined_cosine = gradient_scalars[3] / (
            ppo_norm * combined_norm
        ).clamp_min(torch.finfo(torch.float64).eps)
        ppo_kl_cosine = gradient_scalars[4] / (ppo_norm * kl_norm).clamp_min(
            torch.finfo(torch.float64).eps
        )
        off_diagonal = ~torch.eye(len(state_ids), dtype=torch.bool)
        finite_pairwise = pairwise_cosines[off_diagonal]
        finite_pairwise = finite_pairwise[torch.isfinite(finite_pairwise)]
        finite_state_to_aggregate = state_to_aggregate[
            torch.isfinite(state_to_aggregate)
        ]
        state_norms = gram.diag().clamp_min(0).sqrt()
        state_to_ppo = _gradient_cosines_to_direction(
            state_ppo_dots, state_norms, ppo_norm
        )
        state_to_combined = _gradient_cosines_to_direction(
            state_combined_dots, state_norms, combined_norm
        )

        def summarize_cosines(prefix: str, values: torch.Tensor) -> dict[str, float]:
            finite = values[torch.isfinite(values)]
            return {
                f"{prefix}_mean": float(finite.mean()),
                f"{prefix}_min": float(finite.min()),
                f"{prefix}_negative_fraction": float((finite < 0).double().mean()),
            }

        metrics = {
            "diagnostics/policy_gradient/state_count": float(len(state_ids)),
            "diagnostics/policy_gradient/ppo_norm": float(ppo_norm),
            "diagnostics/policy_gradient/combined_norm": float(combined_norm),
            "diagnostics/policy_gradient/kl_component_norm": float(kl_norm),
            "diagnostics/policy_gradient/kl_to_ppo_norm_ratio": float(
                kl_norm / ppo_norm.clamp_min(torch.finfo(torch.float64).eps)
            ),
            "diagnostics/policy_gradient/state_sum_additivity_error_ratio": float(
                gradient_scalars[5].clamp_min(0).sqrt()
                / ppo_norm.clamp_min(torch.finfo(torch.float64).eps)
            ),
            "diagnostics/policy_gradient/ppo_combined_cosine": float(
                ppo_combined_cosine
            ),
            "diagnostics/policy_gradient/ppo_kl_cosine": float(ppo_kl_cosine),
            "diagnostics/policy_gradient/pairwise_cosine_mean": float(
                finite_pairwise.mean()
            ),
            "diagnostics/policy_gradient/pairwise_cosine_min": float(
                finite_pairwise.min()
            ),
            "diagnostics/policy_gradient/pairwise_negative_fraction": float(
                (finite_pairwise < 0).double().mean()
            ),
            "diagnostics/policy_gradient/state_to_aggregate_cosine_mean": float(
                finite_state_to_aggregate.mean()
            ),
            "diagnostics/policy_gradient/state_to_aggregate_cosine_min": float(
                finite_state_to_aggregate.min()
            ),
            "diagnostics/policy_gradient/state_to_aggregate_negative_fraction": float(
                (finite_state_to_aggregate < 0).double().mean()
            ),
        }
        metrics.update(
            summarize_cosines(
                "diagnostics/policy_gradient/state_to_ppo_cosine", state_to_ppo
            )
        )
        metrics.update(
            summarize_cosines(
                "diagnostics/policy_gradient/state_to_combined_cosine",
                state_to_combined,
            )
        )
        state_records = []
        for row_index, state_id in enumerate(state_ids):
            count, adv_sum, adv_square_sum, positive_count, return_sum, _ = (
                signal_stats[row_index]
            )
            count = count.clamp_min(1)
            advantage_mean = adv_sum / count
            advantage_std = (
                (adv_square_sum / count - advantage_mean.square()).clamp_min(0).sqrt()
            )
            episode_index = state_metadata[state_id]
            state_prefix = f"diagnostics/state_episode_{episode_index}"
            state_metrics = {
                f"{state_prefix}/gradient_norm": float(state_norms[row_index]),
                f"{state_prefix}/cosine_to_aggregate": float(
                    state_to_aggregate[row_index]
                ),
                f"{state_prefix}/cosine_to_ppo": float(state_to_ppo[row_index]),
                f"{state_prefix}/cosine_to_combined": float(
                    state_to_combined[row_index]
                ),
                f"{state_prefix}/advantage_mean": float(advantage_mean),
                f"{state_prefix}/advantage_std": float(advantage_std),
                f"{state_prefix}/advantage_positive_fraction": float(
                    positive_count / count
                ),
                f"{state_prefix}/return_mean": float(return_sum / count),
            }
            metrics.update(state_metrics)
            state_records.append(
                {
                    "logical_state_id": state_id,
                    "episode_index": episode_index,
                    "gradient_norm": float(state_norms[row_index]),
                    "cosine_to_aggregate": float(state_to_aggregate[row_index]),
                    "cosine_to_ppo": float(state_to_ppo[row_index]),
                    "cosine_to_combined": float(state_to_combined[row_index]),
                    "advantage_mean": float(advantage_mean),
                    "advantage_std": float(advantage_std),
                    "advantage_positive_fraction": float(positive_count / count),
                    "return_mean": float(return_sum / count),
                    "valid_transition_count": int(signal_stats[row_index, 0]),
                }
            )

        report = {
            "global_step": self.version,
            "objective": "PPO state gradients; combined gradient includes KL",
            "metrics": metrics,
            "states": state_records,
            "state_order": state_ids,
            "pairwise_cosine": pairwise_cosines.tolist(),
        }
        output_dir = diagnostics_cfg.get("output_dir", None)
        destination = None
        if output_dir:
            destination = (
                Path(str(output_dir))
                / f"global_step_{self.version:06d}"
                / "policy_gradient_conflicts.json"
            )
        if destination is not None and self._rank == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            self.log_info(f"Saved policy-gradient diagnostic report to {destination}.")

        if measure_optimizer_update:
            self._pending_policy_gradient_diagnostics = {
                "state_gradients": state_gradients,
                "ppo_gradients": ppo_global_gradients,
                "combined_gradients": combined_gradients,
                "parameters_before": [
                    parameter.detach().float().cpu().clone()
                    for parameter in self._optimizer_parameters_by_role["policy"]
                ],
                "state_norms": state_norms,
                "ppo_norm": ppo_norm,
                "combined_norm": combined_norm,
                "report": report,
                "destination": destination,
            }
        else:
            del state_gradients, ppo_global_gradients, combined_gradients
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        self.optimizer.zero_grad()
        return metrics

    def _finalize_policy_gradient_update_diagnostics(self) -> dict[str, float]:
        """Compare diagnostic gradients with the realized optimizer update."""
        pending = self._pending_policy_gradient_diagnostics
        if pending is None:
            return {}

        state_gradients = pending["state_gradients"]
        ppo_gradients = pending["ppo_gradients"]
        combined_gradients = pending["combined_gradients"]
        state_dots = torch.zeros(len(state_gradients), dtype=torch.float64)
        ppo_dot = 0.0
        combined_dot = 0.0
        update_norm_squared = 0.0
        parameters = self._optimizer_parameters_by_role["policy"]
        for parameter_index, (parameter, parameter_before) in enumerate(
            zip(parameters, pending["parameters_before"], strict=True)
        ):
            # Positive alignment means the realized parameter update locally
            # decreases the corresponding loss: descent = -delta_theta.
            descent = -(parameter.detach().float().cpu() - parameter_before).reshape(-1)
            update_norm_squared += float(torch.dot(descent, descent))
            for state_index, gradients in enumerate(state_gradients):
                gradient = gradients[parameter_index]
                if gradient is not None:
                    state_dots[state_index] += float(
                        torch.dot(gradient.reshape(-1), descent)
                    )
            ppo_gradient = ppo_gradients[parameter_index]
            if ppo_gradient is not None:
                ppo_dot += float(torch.dot(ppo_gradient.reshape(-1), descent))
            combined_gradient = combined_gradients[parameter_index]
            if combined_gradient is not None:
                combined_dot += float(torch.dot(combined_gradient.reshape(-1), descent))

        packed = torch.cat(
            [
                state_dots,
                torch.tensor(
                    [update_norm_squared, ppo_dot, combined_dot],
                    dtype=torch.float64,
                ),
            ]
        ).to(self.device)
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        packed = packed.cpu()
        update_norm = packed[-3].clamp_min(0).sqrt()
        state_to_update = _gradient_cosines_to_direction(
            packed[:-3], pending["state_norms"], update_norm
        )
        ppo_to_update = packed[-2] / (pending["ppo_norm"] * update_norm).clamp_min(
            torch.finfo(torch.float64).eps
        )
        combined_to_update = packed[-1] / (
            pending["combined_norm"] * update_norm
        ).clamp_min(torch.finfo(torch.float64).eps)
        finite = state_to_update[torch.isfinite(state_to_update)]
        metrics = {
            "diagnostics/policy_gradient/optimizer_update_norm": float(update_norm),
            "diagnostics/policy_gradient/ppo_to_optimizer_descent_cosine": float(
                ppo_to_update
            ),
            "diagnostics/policy_gradient/combined_to_optimizer_descent_cosine": float(
                combined_to_update
            ),
            "diagnostics/policy_gradient/state_to_optimizer_descent_cosine_mean": float(
                finite.mean()
            ),
            "diagnostics/policy_gradient/state_to_optimizer_descent_cosine_min": float(
                finite.min()
            ),
            "diagnostics/policy_gradient/state_to_optimizer_descent_negative_fraction": float(
                (finite < 0).double().mean()
            ),
        }

        report = pending["report"]
        report["metrics"].update(metrics)
        for state_record, cosine in zip(
            report["states"], state_to_update.tolist(), strict=True
        ):
            state_record["cosine_to_optimizer_descent"] = cosine
        destination = pending["destination"]
        if destination is not None and self._rank == 0:
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            self.log_info(
                "Updated policy-gradient diagnostic report with realized "
                f"optimizer alignment at {destination}."
            )

        self._pending_policy_gradient_diagnostics = None
        return metrics

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.cfg.actor.get("critic_batch_export_only", False):
            if not self.cfg.actor.get("critic_batch_export_dir", None):
                raise ValueError(
                    "actor.critic_batch_export_only requires "
                    "actor.critic_batch_export_dir."
                )
            self.rollout_batch = {}
            clear_memory()
            return {"critic/export_only": 1.0}

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        resume_critic_warmup, update_phases = self._training_update_phases()
        recompute_metrics = {}
        if (
            self.cfg.actor.get("recompute_prev_logprobs", False)
            and not resume_critic_warmup
        ):
            recompute_metrics = self.recompute_prev_logprobs()

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        metrics = {}
        append_to_dict(metrics, recompute_metrics)
        diagnostics_cfg = self.cfg.actor.get("policy_gradient_diagnostics", {})
        diagnostics_start_step = int(diagnostics_cfg.get("start_step", 0))
        diagnostics_interval = int(diagnostics_cfg.get("interval", 1))
        if diagnostics_interval <= 0:
            raise ValueError("policy_gradient_diagnostics.interval must be positive.")
        run_policy_diagnostics = (
            diagnostics_cfg.get("enabled", False)
            and self.version >= diagnostics_start_step
            and (self.version - diagnostics_start_step) % diagnostics_interval == 0
        )
        if run_policy_diagnostics:
            batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
            if rollout_size != batch_size_per_rank:
                raise ValueError(
                    "Policy-gradient diagnostics require the actor global batch "
                    "to contain the complete rollout batch."
                )
            diagnostic_metrics = self._run_policy_gradient_diagnostics(
                self.rollout_batch
            )
            if diagnostics_cfg.get("only", False):
                self.rollout_batch = {}
                clear_memory()
                return diagnostic_metrics
            append_to_dict(metrics, diagnostic_metrics)
        for phase, update_epochs, update_policy, update_value in update_phases:
            if update_epochs == 0:
                continue
            if phase == "critic" and self.cache_critic_inputs:
                append_to_dict(metrics, self._build_critic_input_cache())
            (
                phase_global_batch_size,
                batch_size_per_rank,
                gradient_accumulation,
            ) = self._phase_batch_config(phase)
            if rollout_size % batch_size_per_rank:
                raise ValueError(
                    f"The per-rank rollout size {rollout_size} must be divisible "
                    f"by the {phase} per-rank batch size "
                    f"{batch_size_per_rank} (global batch size "
                    f"{phase_global_batch_size})."
                )
            fixed_batch_interval = max(1, update_epochs // 10)
            for update_index in range(update_epochs):
                collect_epoch_metrics = (
                    phase == "critic" and self.use_independent_update_epochs
                ) or (self.critic_only and update_epochs > 1)
                epoch_metrics = {} if collect_epoch_metrics else metrics
                rollout_dataloader_iter = split_dict_to_chunk(
                    self.rollout_batch,
                    rollout_size // batch_size_per_rank,
                )
                for train_global_batch in rollout_dataloader_iter:
                    train_global_batch_size = train_global_batch["prev_logprobs"].shape[
                        0
                    ]
                    assert (
                        train_global_batch_size
                        == phase_global_batch_size // torch.distributed.get_world_size()
                    )
                    assert (
                        train_global_batch_size % self.cfg.actor.micro_batch_size == 0
                    ), f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"

                    train_micro_batch = split_dict_to_chunk(
                        train_global_batch,
                        train_global_batch_size // self.cfg.actor.micro_batch_size,
                    )

                    self.optimizer.zero_grad()
                    for idx, batch in enumerate(train_micro_batch):
                        self.train_micro_batch(
                            micro_batch=batch,
                            metrics=epoch_metrics,
                            is_last=(idx + 1) == gradient_accumulation,
                            gradient_accumulation=gradient_accumulation,
                            update_policy=update_policy,
                            update_value=update_value,
                        )
                        train_micro_batch[idx] = None
                        del batch

                    self.torch_platform.empty_cache()

                    grad_norm_before_clip, lr_list = self.optimizer_step()
                    data = self._learning_rate_metrics(lr_list)
                    active_roles = {
                        role
                        for role, enabled in (
                            ("policy", update_policy),
                            ("value", update_value),
                        )
                        if enabled
                    }
                    data.update(
                        self._gradient_clipping_metrics(
                            grad_norm_before_clip,
                            active_roles=active_roles,
                        )
                    )
                    if self.critic_only:
                        data["critic/only_mode"] = 1.0
                    append_to_dict(epoch_metrics, data)

                if epoch_metrics is not metrics:
                    if (
                        update_index % fixed_batch_interval == 0
                        or update_index == update_epochs - 1
                    ):
                        append_to_dict(
                            metrics,
                            self._critic_fixed_batch_snapshot(
                                epoch_metrics,
                                update_index,
                            ),
                        )
                    for key, values in epoch_metrics.items():
                        metrics.setdefault(key, []).extend(values)
            if phase == "policy" and self._pending_policy_gradient_diagnostics:
                append_to_dict(
                    metrics,
                    self._finalize_policy_gradient_update_diagnostics(),
                )
        if self.use_independent_update_epochs:
            policy_update_epochs = update_phases[0][1]
            critic_update_epochs = update_phases[1][1]
            append_to_dict(
                metrics,
                {
                    "actor/policy_update_epochs": float(policy_update_epochs),
                    "critic/update_epochs": float(critic_update_epochs),
                    "critic/resume_warmup": float(resume_critic_warmup),
                },
            )
        post_update_critic_metrics = {}
        if any(
            update_value and epochs > 0 for _, epochs, _, update_value in update_phases
        ):
            post_update_critic_metrics = self._critic_post_update_metrics()
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        # Per-microbatch sufficient statistics describe predictions made before
        # many different optimizer steps. Combining them is not one coherent
        # critic and can produce a misleading EV. The explicit post-update pass
        # above owns the public explained-variance metric.
        pop_critic_explained_variance_stats(metrics)
        append_to_dict(metrics, post_update_critic_metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
        gradient_accumulation: int | None = None,
        update_policy: bool = True,
        update_value: bool = True,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        policy_loss_mask = micro_batch.get("policy_loss_mask", loss_mask)
        policy_sample_weights = micro_batch.get(
            "policy_sample_weights", micro_batch.get("sample_weights", None)
        )
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=update_policy,
                compute_entropy=(
                    update_policy and self.cfg.algorithm.entropy_bonus > 0
                ),
                compute_values=update_value,
                use_cache=False,
                **kwargs,
            )

        if update_policy and SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": (
                self.resume_critic_warmup_value_clip
                if self._resume_critic_warmup_active()
                and self.resume_critic_warmup_value_clip is not None
                else self.cfg.algorithm.get("value_clip", None)
            ),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": policy_loss_mask if update_policy else loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "executed_action_mask": micro_batch.get("executed_action_mask", None),
            "sample_weights": (
                policy_sample_weights
                if update_policy
                else micro_batch.get("sample_weights", None)
            ),
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.critic_only
            or self.optimizer_steps < self.critic_warmup_steps,
            "update_policy": update_policy,
            "update_value": update_value,
        }

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
        ]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if update_policy and self.kl_beta > 0:
            ref_logprobs = micro_batch.get("ref_logprobs")
            if ref_logprobs is None:
                raise RuntimeError(
                    "Reference KL is enabled but ref_logprobs are missing from "
                    "the actor batch."
                )
            kl_loss = _masked_reference_kl(
                output_dict["logprobs"],
                ref_logprobs,
                penalty_type=self.kl_penalty_type,
                executed_action_mask=micro_batch.get("executed_action_mask"),
                loss_mask=loss_mask,
                sample_weights=micro_batch.get("sample_weights"),
            )
            loss = loss + self.kl_beta * kl_loss
        if update_policy:
            metrics_data["actor/kl_loss"] = kl_loss.detach().item()
            metrics_data["actor/kl_beta"] = self.kl_beta

        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if (
            update_policy
            and self.cfg.algorithm.entropy_bonus > 0
            and not loss_kwargs["critic_warmup"]
        ):
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=policy_loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        if update_policy:
            metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train and update_policy:
            loss = self._train_sft_epoch(metrics_data, loss)

        gradient_accumulation = (
            self.gradient_accumulation
            if gradient_accumulation is None
            else gradient_accumulation
        )
        loss /= gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        loss_namespace = "actor" if update_policy else "critic"
        metrics_data[f"{loss_namespace}/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if (
            self._resume_checkpoint_loaded
            and self.resume_critic_warmup_global_steps > 0
            and self._resume_warmup_start_step is None
        ):
            self._resume_warmup_start_step = global_step
            self.log_info(
                "Reserved resumed policy steps "
                f"[{global_step}, "
                f"{global_step + self.resume_critic_warmup_global_steps}) for "
                "critic-only warmup."
            )
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm_before_clip, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = self._learning_rate_metrics(lr_list)
        metric_data.update(self._gradient_clipping_metrics(grad_norm_before_clip))
        if self.critic_only:
            metric_data["critic/only_mode"] = 1.0
        append_to_dict(metrics, metric_data)

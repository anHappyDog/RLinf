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

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
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
                raise ValueError("Policy and critic update epochs must be non-negative.")
            if self.policy_update_epochs + self.critic_update_epochs == 0:
                raise ValueError("At least one policy or critic update epoch is required.")

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

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

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

    def accept_rollout_group(self) -> None:
        """Retain the most recently received candidate as a trainable group."""
        if self._accepted_rollout_batches is None:
            raise RuntimeError("Rollout group collection has not started.")
        if self._candidate_rollout_batch is None:
            raise RuntimeError("No candidate rollout group is available to accept.")
        self._accepted_rollout_batches.append(self._candidate_rollout_batch)
        self._candidate_rollout_batch = None

    def accept_rollout_groups(self, group_ids: list[int]) -> None:
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
            self._accepted_rollout_batches.append(
                _select_rollout_trajectories(
                    self._candidate_rollout_batch,
                    group_mask,
                )
            )
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
            "advantage_mode": self.cfg.algorithm.get("advantage_mode", None),
            "advantage_std_floor": self.cfg.algorithm.get("advantage_std_floor", 0.1),
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
        recomputed = []
        try:
            with torch.no_grad():
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
        finally:
            self.model.train(was_training)

        recomputed_logprobs = torch.cat(recomputed, dim=0).reshape_as(rollout_logprobs)
        assert recomputed_logprobs.shape == rollout_logprobs.shape
        drift = recomputed_logprobs - rollout_logprobs
        self.rollout_batch["prev_logprobs"] = recomputed_logprobs
        return {
            "actor/rollout_logprob_abs_diff": drift.abs().mean().item(),
            "actor/rollout_logprob_diff": drift.mean().item(),
            "actor/rollout_logprob_abs_diff_max": drift.abs().max().item(),
        }

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

        recompute_metrics = {}
        if self.cfg.actor.get("recompute_prev_logprobs", False):
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
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        append_to_dict(metrics, recompute_metrics)
        if self.use_independent_update_epochs:
            update_phases = (
                ("policy", self.policy_update_epochs, True, False),
                ("critic", self.critic_update_epochs, False, True),
            )
        else:
            update_phases = (
                (
                    "joint",
                    int(self.cfg.algorithm.get("update_epoch", 1)),
                    not self.critic_only,
                    self.cfg.algorithm.adv_type in ("gae", "subtask_gae"),
                ),
            )

        for phase, update_epochs, update_policy, update_value in update_phases:
            fixed_batch_interval = max(1, update_epochs // 10)
            for update_index in range(update_epochs):
                collect_epoch_metrics = (
                    (phase == "critic" and self.use_independent_update_epochs)
                    or (self.critic_only and update_epochs > 1)
                )
                epoch_metrics = {} if collect_epoch_metrics else metrics
                rollout_dataloader_iter = split_dict_to_chunk(
                    self.rollout_batch,
                    rollout_size // batch_size_per_rank,
                )
                for train_global_batch in rollout_dataloader_iter:
                    train_global_batch_size = train_global_batch[
                        "prev_logprobs"
                    ].shape[0]
                    assert (
                        train_global_batch_size
                        == self.cfg.actor.global_batch_size
                        // torch.distributed.get_world_size()
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
                            is_last=(idx + 1) == self.gradient_accumulation,
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
        if self.use_independent_update_epochs:
            append_to_dict(
                metrics,
                {
                    "actor/policy_update_epochs": float(self.policy_update_epochs),
                    "critic/update_epochs": float(self.critic_update_epochs),
                },
            )
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
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
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "executed_action_mask": micro_batch.get("executed_action_mask", None),
            "sample_weights": micro_batch.get("sample_weights", None),
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
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        if update_policy:
            metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train and update_policy:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
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

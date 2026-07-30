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

import asyncio
import copy
import gc
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.rlt import (
    build_rlt_route,
    predict_rlt_actions,
)
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    PolicyInput,
    PolicyOutput,
    RolloutResult,
    ValueRequest,
    ValueResult,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Cluster, CommMapper, Worker
from rlinf.utils.nested_dict_process import split_dict
from rlinf.utils.placement import HybridComponentPlacement

if TYPE_CHECKING:
    from rlinf.scheduler.channel.trajectory_channel.channel import TrajectoryChannel


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.only_eval = cfg.runner.get("only_eval", False)
        self.algorithm_cfg = cfg.get("algorithm", {})
        self.model_cfg = cfg.rollout.model if self.only_eval else cfg.actor.model
        self.actor_group_name = (
            cfg.actor.get("group_name", None)
            if cfg.get("actor", None) is not None
            else None
        )
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0
        train_env_cfg = cfg.env.get("train", None)
        eval_env_cfg = cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_online_lerobot = self.enable_train and bool(
            OmegaConf.select(
                cfg,
                "algorithm.dagger.online_lerobot.enabled",
                default=False,
            )
        )
        self.enable_eval = (
            cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.enable_dagger = self.algorithm_cfg.get("loss_type") == "embodied_dagger"
        self.enable_opd = self.algorithm_cfg.get("adv_type") == "opd"
        self.expert_model = None
        self.rlt_feature_model = None
        self.rlt_route = None

        self.total_num_train_envs = (
            cfg.env.train.total_num_envs if self.enable_train else 0
        )
        self.total_num_eval_envs = (
            cfg.env.eval.total_num_envs if self.enable_eval else 0
        )
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = self.total_num_train_envs // self.num_pipeline_stages
        self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages

        self.per_node_train_batch_size = (
            self.train_batch_size // self._world_size if self.enable_train else 0
        )
        self.per_node_eval_batch_size = (
            self.eval_batch_size // self._world_size if self.enable_eval else 0
        )

        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // self.model_cfg.num_action_chunks
            if self.enable_train
            else 0
        )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None

        self.weight_syncer = None
        self._sync_weight_comm_options = None
        if not self.only_eval:
            weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
            assert weight_syncer_cfg is not None, (
                "rollout.weight_syncer config must be provided"
            )
            self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
            self._sync_weight_comm_options = self.weight_syncer.comm_options

        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # save the run-time imformation in communicate channel for decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {
                "rollout_results": [],
            }
        self.rollout_queue_size = self.cfg.rollout.get("rollout_queue_size", 0)
        self._value_consumer_task: asyncio.Task[None] | None = None

        self._policy_requests_per_chunk = self._policy_request_count(
            self.total_num_train_envs if self.enable_train else 0,
            rollout_world_size,
        )
        self._eval_policy_requests_per_chunk = self._policy_request_count(
            self.total_num_eval_envs if self.enable_eval else 0,
            rollout_world_size,
        )

    def _policy_request_count(
        self, total_num_envs: int, rollout_world_size: int
    ) -> int:
        """Return the number of routed policy requests served by this rank."""
        request_world_size = self.placement.get_world_size(
            "actor" if "actor" in self.placement.components else "rollout"
        )
        env_world_size = self.placement.get_world_size("env")
        logical_env_world_size = env_world_size * self.num_pipeline_stages
        request_count = sum(
            len(
                CommMapper.get_dst_ranks(
                    batch_size=total_num_envs,
                    src_world_size=logical_env_world_size,
                    dst_world_size=request_world_size,
                    src_rank=logical_env_rank,
                )
            )
            for logical_env_rank in range(logical_env_world_size)
        )
        requests_per_rank, remainder = divmod(request_count, rollout_world_size)
        return requests_per_rank + (self._rank < remainder)

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is not None:
            self.rlt_feature_model = get_model(copy.deepcopy(rlt_feature_model_config))
            self.rlt_feature_model.eval()
            self.rlt_feature_model.requires_grad_(False)
            self.rlt_route = build_rlt_route(self.cfg)

        if self.cfg.rollout.get("expert_model", None) and not self.enable_opd:
            expert_model_config = build_expert_model_config(
                self.cfg,
                self.model_cfg,
                rlt_feature_model_config=rlt_feature_model_config,
            )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # sampling parameters for rollout
        sampling_params = self.cfg.rollout.get("sampling_params", None)
        if sampling_params is not None:
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            self._train_sampling_params = {
                "do_sample": sampling_params["do_sample"],
                "temperature": sampling_params["temperature_train"]
                if sampling_params["do_sample"]
                else 1.0,
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
            self._eval_sampling_params = {
                "do_sample": True
                if sampling_params.get("temperature_eval", -1) > 0
                else False,
                "temperature": sampling_params["temperature_eval"],
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
        else:
            self._train_sampling_params = {}
            self._eval_sampling_params = {}

        if self.expert_model is not None and self.enable_dagger:
            self._dagger_sampling_params = {
                "beta": self.algorithm_cfg.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.algorithm_cfg.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def update_dagger_beta(self):
        if self.expert_model is None or not self.enable_dagger:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.OPENPI_PYTORCH,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
        ]:
            if self.enable_dagger:
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.algorithm_cfg.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None and self.enable_dagger:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and self.enable_dagger  # only re-label in DAgger mode
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs["model_action"]
                expert_action = expert_forward_inputs["action"]
                if expert_target is not None:
                    result["forward_inputs"]["action"] = expert_action
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _predict_rollout_actions(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        final_obs: dict[str, Any] | None = None,
        rlt_switch_flags: torch.Tensor | None = None,
        intervene_requested: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.rlt_feature_model is not None:
            return predict_rlt_actions(
                policy_model=self.hf_model,
                feature_model=self.rlt_feature_model,
                rlt_route=self.rlt_route,
                env_obs=env_obs,
                final_obs=final_obs,
                mode=mode,
                version=self.version,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=self.expert_model,
            )
        return self.predict(env_obs, mode=mode)

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        trajectory_channel: "TrajectoryChannel",
    ):
        if self.enable_offload:
            self.reload_model()

        await self._serve_training_requests(trajectory_channel)

        if self.enable_offload:
            self.offload_model()

    async def _serve_training_requests(
        self, trajectory_channel: "TrajectoryChannel"
    ) -> None:
        self._ensure_value_consumer(trajectory_channel)

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            self.update_dagger_beta()
            for _ in range(self.n_train_chunk_steps):
                for _ in range(self._policy_requests_per_chunk):
                    policy_input = await trajectory_channel.take(
                        PolicyInput, async_op=True
                    ).async_wait()
                    await self._process_policy_input(trajectory_channel, policy_input)

    @Worker.timer("evaluate")
    async def evaluate(self, trajectory_channel: "TrajectoryChannel") -> None:
        """Serve the finite evaluation request stream through TrajectoryChannel."""
        if self.enable_offload:
            self.reload_model()

        for _ in range(self.eval_rollout_epoch):
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self._eval_policy_requests_per_chunk):
                    policy_input = await trajectory_channel.take(
                        PolicyInput, async_op=True
                    ).async_wait()
                    await self._process_policy_input(trajectory_channel, policy_input)

        if self.enable_offload:
            self.offload_model()

    def _ensure_value_consumer(self, trajectory_channel: "TrajectoryChannel") -> None:
        if bool(
            OmegaConf.select(self.cfg, "actor.model.add_value_head", default=False)
        ):
            if self._value_consumer_task is None:
                self._value_consumer_task = asyncio.create_task(
                    self._consume_value_requests(trajectory_channel),
                    name=f"rollout-value-consumer-{self._rank}",
                )
            elif self._value_consumer_task.done():
                self._value_consumer_task.result()
                raise RuntimeError("Value request consumer stopped unexpectedly.")

    async def _process_policy_input(
        self,
        trajectory_channel: "TrajectoryChannel",
        policy_input: PolicyInput,
    ) -> None:
        actions, result = self._predict_rollout_actions(
            policy_input.observations,
            mode=policy_input.mode,
            rlt_switch_flags=policy_input.rlt_switch_flags,
            intervene_requested=policy_input.intervene_flags,
        )
        await self._publish_policy_result(
            trajectory_channel, policy_input, actions, result
        )

    async def _process_policy_inputs(
        self,
        trajectory_channel: "TrajectoryChannel",
        policy_inputs: list[PolicyInput],
    ) -> None:
        """Run one inference batch and return each request independently."""
        if len(policy_inputs) == 1:
            await self._process_policy_input(trajectory_channel, policy_inputs[0])
            return

        merged = self._merge_obs_batches(
            [
                {
                    "obs": request.observations,
                    "rlt_switch_flags": request.rlt_switch_flags,
                    "intervene_flags": request.intervene_flags,
                }
                for request in policy_inputs
            ]
        )
        actions, result = self._predict_rollout_actions(
            merged["obs"],
            mode=policy_inputs[0].mode,
            rlt_switch_flags=merged["rlt_switch_flags"],
            intervene_requested=merged["intervene_flags"],
        )
        split_sizes = [request.batch_size for request in policy_inputs]
        split_actions = torch.split(actions, split_sizes, dim=0)
        split_results = split_dict(result, split_sizes)
        await asyncio.gather(
            *(
                self._publish_policy_result(
                    trajectory_channel,
                    request,
                    request_actions,
                    request_result,
                )
                for request, request_actions, request_result in zip(
                    policy_inputs, split_actions, split_results
                )
            )
        )

    async def _publish_policy_result(
        self,
        trajectory_channel: "TrajectoryChannel",
        policy_input: PolicyInput,
        actions: torch.Tensor,
        result: dict[str, Any],
    ) -> None:
        intervene_flags = result.get("intervene_flags")
        if intervene_flags is None and result.get("expert_label_flag", False):
            intervene_flags = torch.full(
                (actions.shape[0], self.model_cfg.num_action_chunks),
                True,
                dtype=torch.bool,
            )

        prev_logprobs = result.get("prev_logprobs")
        state_values = result.get("prev_values")
        versions = (
            torch.full_like(
                prev_logprobs,
                float(self.version),
                dtype=torch.float32,
            )
            if prev_logprobs is not None
            else None
        )
        forward_inputs = result.get("forward_inputs")
        record_fields = {
            "global_step": policy_input.global_step,
            "rollout_epoch": policy_input.rollout_epoch,
            "chunk_step": policy_input.chunk_step,
            "slot_ids": policy_input.slot_ids,
            "actor_rank": policy_input.actor_rank,
            "pipeline_stage": policy_input.pipeline_stage,
        }
        policy_output = PolicyOutput(
            **record_fields,
            env_rank=policy_input.env_rank,
            actions=actions,
            mode=policy_input.mode,
            expert_actions=(
                forward_inputs.get("action")
                if self.enable_online_lerobot and forward_inputs is not None
                else None
            ),
            intervene_flags=intervene_flags,
        )
        policy_work = trajectory_channel.publish(policy_output, async_op=True)
        await policy_work.async_wait()
        if policy_input.mode == "train" and not self.enable_online_lerobot:
            await trajectory_channel.publish(
                RolloutResult(
                    **record_fields,
                    actions=actions,
                    forward_inputs=forward_inputs,
                    prev_logprobs=(prev_logprobs if self.collect_prev_infos else None),
                    state_values=state_values if self.collect_prev_infos else None,
                    versions=versions,
                    intervene_flags=intervene_flags,
                ),
                async_op=True,
            ).async_wait()

    async def _consume_value_requests(
        self, trajectory_channel: "TrajectoryChannel"
    ) -> None:
        while True:
            request = await trajectory_channel.take(
                ValueRequest, async_op=True
            ).async_wait()
            actions, result = self._predict_rollout_actions(request.observations)
            values = result.get("prev_values")
            if values is None:
                values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
            values = values[:, :1]
            versions = torch.full_like(values, float(self.version), dtype=torch.float32)
            await trajectory_channel.publish(
                ValueResult(
                    global_step=request.global_step,
                    rollout_epoch=request.rollout_epoch,
                    chunk_step=request.chunk_step,
                    slot_ids=request.slot_ids,
                    actor_rank=request.actor_rank,
                    pipeline_stage=request.pipeline_stage,
                    kind=request.value_kind,
                    values=values,
                    versions=versions,
                ),
                async_op=True,
            ).async_wait()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to("cpu")
        if self.expert_model is not None:
            self.expert_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to(self.device)
        if self.expert_model is not None:
            self.expert_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    def _merge_optional_flag_tensors(
        self,
        obs_dicts: list[dict[str, Any]],
        flags_list: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        if not any(flags is not None for flags in flags_list):
            return None
        ref_flags = next(flags for flags in flags_list if flags is not None)
        filled_flags = []
        for obs_dict, flags in zip(obs_dicts, flags_list):
            if flags is None:
                batch_size = self._infer_env_batch_size(obs_dict)
                fill_shape = (batch_size, *ref_flags.shape[1:])
                filled_flags.append(torch.zeros(fill_shape, dtype=ref_flags.dtype))
            else:
                filled_flags.append(flags)
        return torch.cat(filled_flags, dim=0)

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]
        rlt_switch_flags_list = [
            obs_batch.get("rlt_switch_flags", None) for obs_batch in obs_batches
        ]
        intervene_flags_list = [
            obs_batch.get("intervene_flags", None) for obs_batch in obs_batches
        ]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "rlt_switch_flags": self._merge_optional_flag_tensors(
                obs_dicts, rlt_switch_flags_list
            ),
            "intervene_flags": self._merge_optional_flag_tensors(
                obs_dicts, intervene_flags_list
            ),
        }

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)

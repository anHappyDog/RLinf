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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import (
    EnvOutput,
    EnvResult,
    LeRobotStepResult,
    PolicyInput,
    PolicyOutput,
    RewardRequest,
    ValueRequest,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Cluster, CommMapper, Worker
from rlinf.scheduler.channel.trajectory_channel.channel import TrajectoryChannel
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.history_manager import HistoryManager


def _slice_data(data: Any, index: slice | torch.Tensor) -> Any:
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data[index].contiguous()
    if isinstance(data, np.ndarray):
        numpy_index = index.cpu().numpy() if isinstance(index, torch.Tensor) else index
        return np.ascontiguousarray(data[numpy_index])
    if isinstance(data, dict):
        return {key: _slice_data(value, index) for key, value in data.items()}
    if isinstance(data, list):
        if isinstance(index, slice):
            return data[index]
        return [value for value, selected in zip(data, index.tolist()) if selected]
    return data


def _tensor_observations(observations: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in observations.items()
        if isinstance(value, torch.Tensor)
    }


def _trajectory_data(data: Any) -> Any:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(data))
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, dict):
        return {key: _trajectory_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_trajectory_data(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_trajectory_data(value) for value in data)
    return data


@dataclass(frozen=True)
class _Shard:
    actor_rank: int
    slot_ids: tuple[int, ...]
    index: slice


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False
        self.global_step = 0
        self._trajectory_channel: TrajectoryChannel | None = None

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._stage_shards = ()
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        # Env configurations
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.model_cfg = (
            self.cfg.rollout.model if self.only_eval else self.cfg.actor.model
        )
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            self.cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1

        self.train_enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            eval_env_cfg.get("enable_offload", False)
            if eval_env_cfg is not None
            else False
        )
        if self.enable_train:
            self.enable_online_lerobot = bool(
                OmegaConf.select(
                    self.cfg,
                    "algorithm.dagger.online_lerobot.enabled",
                    default=False,
                )
            )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
            self.train_batch_size = self.cfg.env.train.total_num_envs // self.stage_num
        else:
            self.enable_online_lerobot = False
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
            self.eval_batch_size = self.cfg.env.eval.total_num_envs // self.stage_num
        self.n_train_chunk_steps = 0
        if self.enable_train:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        if self.use_training_pipeline and self.enable_train:
            self._init_pipeline_params()

        if self.enable_train:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any]
    ) -> None:
        """Append one environment step's metrics to the current rollout."""
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def store_last_obs_and_intervened_info(self, env_outputs: list[EnvOutput]) -> None:
        """Keep the final observations needed by the next rollout."""
        self.last_obs_list = [env_output.obs for env_output in env_outputs]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_outputs
        ]

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            if self.train_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.env_list
                ), "train envs must have an offload method to enable offload!"

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            if self.eval_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.eval_env_list
                ), "eval envs must have an offload method to enable offload!"

        if self.enable_train:
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
            self._stage_shards = tuple(
                tuple(self._trajectory_shards(stage_id))
                for stage_id in range(self.stage_num)
            )

        self._init_env()

    def update_env_cfg(self):
        if self.enable_train:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(self.cfg.env.train)
        if self.enable_eval:
            eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
            if eval_override_cfgs is not None:
                assert len(eval_override_cfgs) > self._rank, (
                    f"{len(eval_override_cfgs)=} > {self._rank=}"
                )

                general_eval_override_cfg = OmegaConf.to_container(
                    self.cfg.env.eval.get("override_cfg", {}), resolve=True
                )
                eval_override_cfg = OmegaConf.to_container(
                    eval_override_cfgs[self._rank], resolve=True
                ).copy()
                base_eval_cfg = {}
                base_eval_cfg = update_nested_cfg(
                    base_eval_cfg, general_eval_override_cfg
                )
                base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
                setattr(
                    self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg)
                )
            self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _init_pipeline_params(self):
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.shuffle_rollout = self.cfg.algorithm.get("shuffle_rollout", True)
        self.pipeline_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.pipeline_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.pipeline_actor_env_ranks = {
            actor_rank: sorted(
                {
                    logical_src_rank // self.stage_num
                    for logical_src_rank, _ in CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs,
                        src_world_size=logical_env_ws,
                        dst_world_size=actor_ws,
                        dst_rank=actor_rank,
                    )
                }
            )
            for actor_rank in range(actor_ws)
        }
        self.pipeline_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "pipeline_actor"
            )
            for actor_rank in local_actor_ranks
        }
        if self.shuffle_rollout:
            self.shuffle_generators = {
                actor_rank: torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + self._rank * actor_ws
                )
                for actor_rank in local_actor_ranks
            }

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.cfg.env.train.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                if self.train_enable_offload and self.cfg.env.train.get(
                    "enable_init_offload", True
                ):
                    get_env_attr(self.env_list[i], "offload")()
            if self.enable_eval:
                if self.eval_enable_offload:
                    get_env_attr(self.eval_env_list[i], "offload")()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        exec_actions = prepare_actions(
            raw_chunk_actions=chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
            env_cfg=self.cfg.env.train,
        )
        if isinstance(chunk_actions, dict):
            chunk_actions["actions"] = exec_actions
        else:
            chunk_actions = exec_actions
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            rlt_switch_flags=rlt_switch_flags,
        )
        chunk_step_payload = {
            "chunk_actions": exec_actions,
            "obs_list": obs_list,
            "terminations": chunk_terminations,
            "truncations": chunk_truncations,
            "infos_list": infos_list,
        }
        return env_output, env_info, chunk_step_payload

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            env_cfg=self.cfg.env.eval,
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones.any(dim=1)  # [num_envs] bool
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            env_infos=infos if isinstance(infos, dict) else None,
            rlt_switch_flags=rlt_switch_flags,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video:
                    flush_video = get_env_attr(self.env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video:
                    flush_video = get_env_attr(self.eval_env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    env_infos=infos if isinstance(infos, dict) else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def set_global_step(self, global_step: int) -> None:
        self.global_step = global_step

    def _trajectory_shards(self, stage_id: int) -> list[_Shard]:
        total_num_envs = self.cfg.env.train.total_num_envs
        actor_world_size = self._component_placement.get_world_size("actor")
        if total_num_envs % actor_world_size != 0:
            raise ValueError(
                "TrajectoryChannel requires total_num_envs to be divisible by "
                "the actor world size."
            )

        actor_num_slots = total_num_envs // actor_world_size
        stage_start = (
            self._rank * self.stage_num + stage_id
        ) * self.train_num_envs_per_stage
        stage_end = stage_start + self.train_num_envs_per_stage
        shards = []
        for actor_rank in range(actor_world_size):
            actor_start = actor_rank * actor_num_slots
            actor_end = actor_start + actor_num_slots
            shard_start = max(stage_start, actor_start)
            shard_end = min(stage_end, actor_end)
            if shard_start >= shard_end:
                continue
            shards.append(
                _Shard(
                    actor_rank=actor_rank,
                    slot_ids=tuple(
                        range(shard_start - actor_start, shard_end - actor_start)
                    ),
                    index=slice(shard_start - stage_start, shard_end - stage_start),
                )
            )
        return shards

    async def _publish_trajectory_records(
        self, trajectory_channel: TrajectoryChannel, records: list
    ) -> None:
        works = [
            trajectory_channel.publish(record, async_op=True) for record in records
        ]
        await asyncio.gather(*(work.async_wait() for work in works))

    def _build_reward_request_inputs(
        self,
        env_output: EnvOutput,
        stage_id: int,
        *,
        last_run: bool,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None] | None:
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None

        inputs = dict(observations)
        if env_output.env_infos is not None:
            inputs["env_infos"] = self._select_reward_env_infos(env_output.env_infos)
        dones = env_output.dones
        if dones is not None and dones.ndim > 1:
            inputs["dones"] = dones[:, -1]

        history_lengths = None
        if self.reward_mode == "history_buffer":
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=inputs.get("dones")
            )
            inputs["history_input"] = history_input
        if last_run:
            inputs["last_run"] = torch.ones(
                (self.train_num_envs_per_stage, 1), dtype=torch.bool
            )
        return inputs, history_lengths

    async def _take_actions(
        self,
        trajectory_channel: TrajectoryChannel,
        env_outputs: list[EnvOutput],
        rollout_epoch: int,
        chunk_step: int,
    ) -> dict[tuple[int, tuple[int, ...]], PolicyOutput]:
        requests = []
        for stage_id, env_output in enumerate(env_outputs):
            observations = env_output.prepare_observations(env_output.obs)
            for shard in self._stage_shards[stage_id]:
                requests.append(
                    PolicyInput(
                        global_step=self.global_step,
                        rollout_epoch=rollout_epoch,
                        chunk_step=chunk_step,
                        slot_ids=shard.slot_ids,
                        actor_rank=shard.actor_rank,
                        pipeline_stage=stage_id,
                        env_rank=self._rank,
                        observations=_slice_data(observations, shard.index),
                        rlt_switch_flags=_slice_data(
                            env_output.rlt_switch_flags, shard.index
                        ),
                        intervene_flags=_slice_data(
                            env_output.intervene_flags, shard.index
                        ),
                    )
                )
        await self._publish_trajectory_records(trajectory_channel, requests)
        works = [
            trajectory_channel.take(
                PolicyOutput,
                async_op=True,
                partition=(self._rank, "train"),
            )
            for _ in requests
        ]
        outputs = await asyncio.gather(*(work.async_wait() for work in works))
        return {(output.actor_rank, output.slot_ids): output for output in outputs}

    def _lerobot_step(
        self,
        record_kwargs: dict[str, Any],
        policy_output: PolicyOutput,
        chunk_payload: dict[str, Any],
        local_slice: slice,
    ) -> LeRobotStepResult:
        return LeRobotStepResult(
            **record_kwargs,
            env_rank=self._rank,
            chunk_actions=_slice_data(chunk_payload["chunk_actions"], local_slice),
            observations=[
                _slice_data(obs, local_slice) for obs in chunk_payload["obs_list"]
            ],
            terminations=_slice_data(chunk_payload["terminations"], local_slice),
            truncations=_slice_data(chunk_payload["truncations"], local_slice),
            env_infos=[
                _slice_data(info, local_slice) for info in chunk_payload["infos_list"]
            ],
            expert_actions=policy_output.expert_actions,
            intervene_flags=policy_output.intervene_flags,
        )

    def _env_result(
        self,
        record_kwargs: dict[str, Any],
        env_output: EnvOutput,
        observations: dict[str, Any],
        next_observations: dict[str, Any],
        local_slice: slice,
    ) -> EnvResult:
        return EnvResult(
            **record_kwargs,
            rewards=_slice_data(env_output.rewards, local_slice),
            dones=_slice_data(env_output.dones, local_slice),
            terminations=_slice_data(env_output.terminations, local_slice),
            truncations=_slice_data(env_output.truncations, local_slice),
            observations=_slice_data(_tensor_observations(observations), local_slice)
            if self.collect_transitions
            else None,
            next_observations=_slice_data(
                _tensor_observations(next_observations), local_slice
            )
            if self.collect_transitions
            else None,
            intervene_actions=_slice_data(env_output.intervene_actions, local_slice),
            intervene_flags=_slice_data(env_output.intervene_flags, local_slice),
            rlt_switch_flags=_slice_data(env_output.rlt_switch_flags, local_slice),
        )

    def _value_requests(
        self,
        record_kwargs: dict[str, Any],
        env_output: EnvOutput,
        local_slice: slice,
        bootstrap_obs: dict[str, Any],
        chunk_step: int,
    ) -> list[ValueRequest]:
        if not self.cfg.actor.model.get("add_value_head", False):
            return []

        slot_ids = record_kwargs["slot_ids"]
        if self.cfg.algorithm.get("bootstrap_type", "standard") == "standard":
            selected = (
                env_output.truncations[local_slice, -1]
                & ~env_output.terminations[local_slice, -1]
            )
        else:
            selected = env_output.dones[local_slice, -1]

        bootstrap_observations = env_output.prepare_observations(bootstrap_obs)
        requests = []
        if selected.any():
            requests.append(
                ValueRequest(
                    **(
                        record_kwargs
                        | {
                            "slot_ids": tuple(
                                slot_id
                                for slot_id, is_selected in zip(
                                    slot_ids, selected.tolist()
                                )
                                if is_selected
                            )
                        }
                    ),
                    value_kind="truncation",
                    observations=_slice_data(
                        _slice_data(bootstrap_observations, local_slice),
                        selected,
                    ),
                )
            )

        if chunk_step != self.n_train_chunk_steps - 1:
            return requests

        selected = ~env_output.dones[local_slice, -1]
        if selected.any():
            requests.append(
                ValueRequest(
                    **(
                        record_kwargs
                        | {
                            "chunk_step": self.n_train_chunk_steps,
                            "slot_ids": tuple(
                                slot_id
                                for slot_id, is_selected in zip(
                                    slot_ids, selected.tolist()
                                )
                                if is_selected
                            ),
                        }
                    ),
                    value_kind="boundary",
                    observations=_slice_data(
                        _slice_data(
                            env_output.prepare_observations(env_output.obs), local_slice
                        ),
                        selected,
                    ),
                )
            )
        return requests

    def _reward_request(
        self,
        record_fields: dict[str, Any],
        reward_inputs: tuple[dict[str, Any], dict[str, torch.Tensor] | None] | None,
        env_output: EnvOutput,
        shard: _Shard,
    ) -> RewardRequest | None:
        if reward_inputs is None:
            return None

        inputs, history_lengths = reward_inputs
        index: slice | torch.Tensor = shard.index
        slot_ids = shard.slot_ids
        if self.reward_mode == "terminal":
            done = env_output.dones[shard.index].any(dim=1)
            if not done.any():
                return None
            index = torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
            index[shard.index] = done
            slot_ids = tuple(
                slot_id
                for slot_id, is_done in zip(shard.slot_ids, done.tolist())
                if is_done
            )
        return RewardRequest(
            **(record_fields | {"slot_ids": slot_ids}),
            mode=self.reward_mode,
            inputs=_slice_data(inputs, index),
            history_lengths=_slice_data(history_lengths, index),
        )

    def _records(
        self,
        rollout_epoch: int,
        chunk_step: int,
        stage_id: int,
        shard: _Shard,
        policy_output: PolicyOutput,
        observations: dict[str, Any],
        env_output: EnvOutput,
        chunk_payload: dict[str, Any],
        reward_inputs: tuple[dict[str, Any], dict[str, torch.Tensor] | None] | None,
    ) -> list:
        record_fields = {
            "global_step": self.global_step,
            "rollout_epoch": rollout_epoch,
            "chunk_step": chunk_step,
            "slot_ids": shard.slot_ids,
            "actor_rank": shard.actor_rank,
            "pipeline_stage": stage_id,
        }
        if self.enable_online_lerobot:
            return [
                self._lerobot_step(
                    record_fields,
                    policy_output,
                    chunk_payload,
                    shard.index,
                )
            ]

        next_observations = (
            env_output.final_obs
            if env_output.dones.any() and self.cfg.env.train.auto_reset
            else env_output.obs
        )
        records = [
            self._env_result(
                record_fields,
                env_output,
                observations,
                next_observations,
                shard.index,
            )
        ]
        records.extend(
            self._value_requests(
                record_fields,
                env_output,
                shard.index,
                env_output.final_obs or env_output.obs,
                chunk_step,
            )
        )
        reward_request = self._reward_request(
            record_fields, reward_inputs, env_output, shard
        )
        if reward_request is not None:
            records.append(reward_request)
        return records

    async def _run_chunk(
        self,
        trajectory_channel: TrajectoryChannel,
        env_outputs: list[EnvOutput],
        rollout_epoch: int,
        chunk_step: int,
        env_metrics: dict[str, list],
    ) -> list[EnvOutput]:
        outputs = await self._take_actions(
            trajectory_channel, env_outputs, rollout_epoch, chunk_step
        )
        records = []
        next_outputs = []
        for stage_id, env_output in enumerate(env_outputs):
            actions = torch.cat(
                [
                    outputs[(shard.actor_rank, shard.slot_ids)].actions
                    for shard in self._stage_shards[stage_id]
                ],
                dim=0,
            )
            next_output, env_info, chunk_payload = self.env_interact_step(
                actions, stage_id
            )
            if self.enable_online_lerobot:
                chunk_payload = _trajectory_data(chunk_payload)
            reward_inputs = (
                self._build_reward_request_inputs(
                    next_output,
                    stage_id,
                    last_run=(
                        rollout_epoch == self.rollout_epoch - 1
                        and chunk_step == self.n_train_chunk_steps - 1
                    ),
                )
                if self.use_external_reward_model
                else None
            )

            for shard in self._stage_shards[stage_id]:
                records.extend(
                    self._records(
                        rollout_epoch,
                        chunk_step,
                        stage_id,
                        shard,
                        outputs[(shard.actor_rank, shard.slot_ids)],
                        env_output.obs,
                        next_output,
                        chunk_payload,
                        reward_inputs,
                    )
                )
            next_outputs.append(next_output)
            if (
                self.cfg.env.train.auto_reset
                or self.cfg.env.train.ignore_terminations
                or chunk_step == self.n_train_chunk_steps - 1
            ):
                self.record_env_metrics(env_metrics, env_info)

        await self._publish_trajectory_records(trajectory_channel, records)
        return next_outputs

    async def _run_epoch(
        self, trajectory_channel: TrajectoryChannel, rollout_epoch: int
    ) -> dict[str, list]:
        env_metrics = defaultdict(list)
        env_outputs = self.bootstrap_step()
        for chunk_step in range(self.n_train_chunk_steps):
            env_outputs = await self._run_chunk(
                trajectory_channel,
                env_outputs,
                rollout_epoch,
                chunk_step,
                env_metrics,
            )
        self.store_last_obs_and_intervened_info(env_outputs)
        self.finish_rollout()
        return env_metrics

    async def _run_interact_once(
        self, trajectory_channel: TrajectoryChannel
    ) -> dict[str, torch.Tensor]:
        env_metrics = defaultdict(list)
        for rollout_epoch in range(self.rollout_epoch):
            epoch_metrics = await self._run_epoch(trajectory_channel, rollout_epoch)
            for key, values in epoch_metrics.items():
                env_metrics[key].extend(values)

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()
        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        trajectory_channel: TrajectoryChannel,
    ):
        self._trajectory_channel = trajectory_channel
        env_metrics = await self._run_interact_once(trajectory_channel)

        for env in self.env_list:
            if self.train_enable_offload:
                get_env_attr(env, "offload")()

        return env_metrics

    async def evaluate(
        self, trajectory_channel: TrajectoryChannel
    ) -> dict[str, torch.Tensor]:
        """Evaluate through the rank-decoupled policy channel."""
        eval_metrics = defaultdict(list)
        env_outputs: list[EnvOutput | None] = [None] * self.stage_num

        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    observations, infos = self.eval_env_list[stage_id].reset()
                    env_outputs[stage_id] = EnvOutput(
                        obs=observations,
                        final_obs=infos.get("final_observation"),
                        env_infos=infos,
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                requests = [
                    PolicyInput(
                        global_step=self.global_step,
                        rollout_epoch=eval_rollout_epoch,
                        chunk_step=eval_step,
                        slot_ids=tuple(range(self.eval_num_envs_per_stage)),
                        pipeline_stage=stage_id,
                        env_rank=self._rank,
                        observations=env_output.prepare_observations(env_output.obs),
                        mode="eval",
                        rlt_switch_flags=env_output.rlt_switch_flags,
                        intervene_flags=env_output.intervene_flags,
                    )
                    for stage_id, env_output in enumerate(env_outputs)
                ]
                await self._publish_trajectory_records(trajectory_channel, requests)
                output_works = [
                    trajectory_channel.take(
                        PolicyOutput,
                        async_op=True,
                        partition=(self._rank, "eval"),
                    )
                    for _ in requests
                ]
                outputs = {
                    output.pipeline_stage: output
                    for output in await asyncio.gather(
                        *(work.async_wait() for work in output_works)
                    )
                }

                for stage_id in range(self.stage_num):
                    actions = outputs[stage_id].actions.detach().cpu().numpy()
                    env_output, env_info = self.env_evaluate_step(actions, stage_id)
                    env_outputs[stage_id] = env_output
                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

            self.finish_rollout(mode="eval")

        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()
        return {
            key: torch.cat(value, dim=0).contiguous().cpu()
            for key, value in eval_metrics.items()
        }

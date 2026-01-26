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
from typing import Optional, OrderedDict, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from omegaconf import open_dict
from omegaconf.omegaconf import DictConfig, OmegaConf

__all__ = ["ManiskillEnv"]


def extract_termination_from_info(
    info: dict, num_envs: int, device: str | torch.device
) -> torch.Tensor:
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: dict,
        record_metrics: bool = True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

        self.cfg = cfg

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.prev_consecutive_grasp = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        if self.record_metrics:
            self._init_metrics()

    @property
    def total_num_group_envs(self) -> int:
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        return self.env.unwrapped.get_language_instruction()

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _wrap_obs(self, raw_obs: dict) -> dict[str, torch.Tensor]:
        """
        Maniskill env's obs support many modes, including "state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud".
        This function will extract observations based on different `wrap_obs_mode` in cfg.
        If `wrap_obs_mode` is "simple", it will return dict with keys depending on obs_mode:
            - for "state" mode, return {"states": state}
            - for "rgb" mode, return {"main_images": main_camera_rgb, "extra_view_images": extra_view_rgbs, "states": state}
        If `wrap_obs_mode` is not "simple", it will use `_extract_obs_image` function to extract observations,

        Args:
            raw_obs(dict[str,torch.Tensor]): raw observations returned by maniskill env step or reset function.

        Returns:
            wrapped_obs(dict[str,torch.Tensor]): extracted observations after wrapping.
        """
        if getattr(self.cfg, "wrap_obs_mode", "vla") == "simple":
            if self.env.unwrapped.obs_mode == "state":
                wrapped_obs = {
                    "states": raw_obs,
                }
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                state = common.flatten_state_dict(
                    raw_obs, use_torch=True, device=self.device
                )

                main_images = sensor_data["base_camera"]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera")
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )

                wrapped_obs = {
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "states": state,
                }
            else:
                raise NotImplementedError
        else:
            wrapped_obs = self._extract_obs_image(raw_obs)
        return wrapped_obs

    def _extract_obs_image(self, raw_obs: dict) -> dict[str, torch.Tensor]:
        """
        Extract observations from raw observations returned by maniskill env step or reset function.
        It assumes the env obs_mode is "sensor_data" and use "3rd_view_camera" rgb image as main image.
        The raw_obs contains keys:
            - agent: Env agent's proprioception
            - extra: env specific info dict, for sapien env, it's empty dict.
            - sensor_param:
            - sensor_data: a dict contains sensor's data.

        The returned `extracted_obs` contains keys:
            - main_images: rgb images from "3rd_view_camera" sensor.
            - states: agent's proprioception.
            - task_descriptions: language instruction for current task.

        Args:
            raw_obs(dict[str,torch.Tensor]): raw observations returned by maniskill env step or reset function.

        Returns:
            extracted_obs(dict[str,torch.Tensor]): extracted observations after wrapping, including `main_images`,
            `states` and `task_descriptions`.
        """
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        extracted_obs = {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.instruction,
        }
        return extracted_obs

    def _calc_step_reward(self, reward: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate step reward based on raw reward and info dict returned by env step function.
        Maniskill reward calculation supposes `sparse`, `none` mode, using `success` and `fail`.
        This function uses the returned reward, if reward model is raw, just use the given reward
        as base reward, or use `is_src_obj_grasped`*0,1 + `consecutive_grasp`*0.1 + `success`*1.0 as base reward.
        If rel_reward is supported, return the diff to previous step reward as step reward, or return base reward.

        Args:
            reward (torch.Tensor): [num_envs,]: raw reward returned by env step function.
            info (dict): env info returned by Maniskill env.

        Returns:
            step_reward (torch.Tensor): [num_envs,]: calculated step reward.
        """
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            # Time penalty to prevent stalling behavior
            reward += -0.05

            # Grasp reward: small reward for grasping (0.1)
            # reward += info["is_src_obj_grasped"] * 0.1

            # Event-based rewards (only trigger on first occurrence)
            if self.use_rel_reward:
                # Relative reward mode handles state changes naturally via diff
                # But we add extra shaping terms if available in info
                reward += info["consecutive_grasp"] * 0.1
            else:
                # Absolute reward mode needs careful event tracking
                consecutive_grasp = info["consecutive_grasp"]
                # 1. Grasp Event: +0.1 on rising edge
                reward += (consecutive_grasp & ~self.prev_consecutive_grasp) * 0.1

            # State updates for event tracking
            if "consecutive_grasp" in info:
                self.prev_consecutive_grasp = info["consecutive_grasp"].clone()

            # Success reward: strong terminal reward to encourage completion
            reward += (info["success"] & info["is_src_obj_grasped"]) * 2.0

        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self) -> None:
        """
        Init maniskill env's metrics, including `success_once`,
        `fail_once` and `returns`, whose shape are all [num_envs,].
        """
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: Optional[list[int]] = None) -> None:
        """
        Reset metrics for specified envs by `env_idx`. If `env_idx` is None, reset all envs.
        metrics include `success_once`, `fail_once` and `returns`.

        Args:
            env_idx(Optional[list[int]]): list of env indices to reset metrics. If None, reset all envs.
        """
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.prev_consecutive_grasp[mask] = False
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.prev_consecutive_grasp[:] = False
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(
        self, step_reward: torch.Tensor, infos: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Update and return metrics for given step, using step_rewards and env returned infos to record
        `success_once`, `fail_once`, `returns`, `episode_len` and `reward`. It will update episode info
        with average reward, episode_len, returns, success_once and fail_once.

        Args:
            step_reward(torch.Tensor): [num_envs,]: step rewards for current step.
            infos(dict): info dict returned by env step function.

        Return:
            infos(dict): infos with updated `episode` info.
        """
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """
        Call maniskill env's reset function to reset specified envs by `env_idx` value in `options` dict.
        random seed is also supported to be set for the reset function.

        Args:
            seed(Optional[Union[int, list[int]]]): random seed for reset function,can be None,int and list of int,the last
            if for GPU parallelized environments.
            options(Optional[dict]): options for reset function, can be None or dict. If dict, it can contain key `env_idx` to specify which envs to reset,
            and if `self.use_fixed_reset_state_ids` is True, it will also contain key `episode_id` to specify which state ids to reset to.

        Returns:
            extracted_obs(dict[str,torch.Tensor]):
            infos(dict[str,torch.Tensor]):
        """
        if options is None:
            seed = self.seed
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        extracted_obs = self._wrap_obs(raw_obs)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset: bool = True
    ) -> tuple[Array, Array, Array, Array, dict]:
        """
        Use maniskill env's step function to step the environment with given actions.
        and return extracted observations, step rewards, terminations, truncations and infos.


        """
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        extracted_obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward, infos)

        if self.video_cfg.save_video:
            self.add_new_frames(infos=infos, rewards=step_reward)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions: torch.Tensor) -> tuple:
        """
        Do a chunk step with Maniskill Env, return corresponding env outputs, including
        extracted_obs, chunk_rewards,

        Args:
            chunk_actions(torch.Tensor):[num_envs, chunk_steps, action_dim]: batched action tokens.

        Returns:
            extracted_obs(dict): extracted observations after executing chunk actions.
            chunk_rewards(torch.Tensor): [num_envs, chunk_steps]: rewards for each step in the chunk.
            chunk_terminations(torch.Tensor): [num_envs, chunk_steps]: termination flags for each step in the chunk.
            chunk_truncations(torch.Tensor): [num_envs, chunk_steps]: truncation flags for each step in the chunk.
            infos(dict): info dict from the environment after executing the chunk actions.
        """
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        extracted_obs: dict[str, torch.Tensor],
        infos: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Handle auto reset for the environments that are done. It will reset those environments by calling reset function
        and then return extracted observations and infos with final observations and infos included.

        Args:
            dones()
        """

        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos: dict | None = None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info: dict, rew: torch.Tensor | None = None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()

    def add_new_frames(self, infos: dict, rewards: None | torch.Tensor = None):
        image = self.render(infos, rewards)
        self.render_images.append(image)

    def add_new_frames_from_obs(self, raw_obs: dict):
        """For debugging render"""
        raw_imgs = common.to_numpy(raw_obs["main_images"])
        raw_full_img = tile_images(raw_imgs, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(raw_full_img)

    def flush_video(self, video_sub_dir: str | None = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        images_to_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            fps=self.cfg.init_params.sim_config.control_freq,
            verbose=False,
        )
        self.video_cnt += 1
        self.render_images = []

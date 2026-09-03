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

import gc
import inspect
import json
import os
import time
import uuid
from collections import deque
from typing import ClassVar

import gymnasium as gym
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.envs.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.behavior.subpool import (
    SUBPOOL_TYPES,
    SubpoolCatalog,
    SubpoolSnapshot,
    SubpoolStore,
    full_state_sha256,
    validate_round_robin_coverage,
    validate_subpool_env_config,
    validate_subpool_rollout_horizons,
)
from rlinf.envs.behavior.subpool_reward import (
    SubtaskRewardSpec,
    SubtaskRewardTracker,
    get_stage_info,
)
from rlinf.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
    setup_subpool_omni_cfg,
    sync_robot_after_pose_override,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = ["BehaviorEnv", "BehaviorSubpoolEnv"]


def _repeat_terminal_subpool_chunk(last_obs, last_info, chunk_size: int):
    """Return a frozen, non-executed chunk after a subtask has terminated."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    results = []
    for _ in range(chunk_size):
        results.append(
            (
                [last_obs],
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                [last_info],
                torch.zeros(1, dtype=torch.bool),
            )
        )
    return tuple(zip(*results))


def _support_surface_distance(object_aabb, support_aabb) -> float:
    """Distance from an object's bottom center to a support's top footprint."""
    object_lower, object_upper = (torch.as_tensor(value) for value in object_aabb)
    support_lower, support_upper = (torch.as_tensor(value) for value in support_aabb)
    object_center_xy = (object_lower[:2] + object_upper[:2]) / 2
    closest_support_xy = torch.minimum(
        torch.maximum(object_center_xy, support_lower[:2]),
        support_upper[:2],
    )
    horizontal_offset = object_center_xy - closest_support_xy
    vertical_offset = object_lower[2] - support_upper[2]
    return float(
        torch.linalg.vector_norm(
            torch.cat((horizontal_offset, vertical_offset.reshape(1)))
        ).item()
    )


def _move_state_tensors(value, device):
    """Move every tensor in a nested full simulator state to one device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_state_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_tensors(item, device) for item in value)
    return value


def _preload_numba_llvmlite() -> None:
    # Isaac Sim's ``omni.isaac.core_archive`` ships an older numba in its
    # ``pip_prebundle`` and loads a few submodules during Kit startup,
    # which then mix with the venv's newer ``llvmlite`` and fail with
    # ``unknown attr 'nocapture'``. Preload the venv copies of just those
    # submodules so they win the ``sys.modules`` cache.
    import importlib

    for name in (
        "llvmlite",
        "numba",
        "numba.np.arrayobj",
        "numba.core.runtime.context",
    ):
        try:
            importlib.import_module(name)
        except Exception:
            pass


@ray.remote(num_cpus=1)
class BehaviorProcess:
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        pipeline_stage_num: int,
    ):
        _preload_numba_llvmlite()
        from omnigibson.envs import VectorEnvironment

        self.logger = get_logger()
        self.pipeline_stage_num = pipeline_stage_num
        is_subpool = bool(OmegaConf.select(cfg, "subpool.enabled", default=False))
        omni_cfg = setup_subpool_omni_cfg(cfg) if is_subpool else setup_omni_cfg(cfg)
        self.instance_loader = ActivityInstanceLoader.from_omni_cfg(omni_cfg)

        # create env and apply env wrapper if enabled
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        # When pipeline stages > 1, each stage independently advances the
        # global physics per chunk step.  Divide physics_frequency so the
        # total physics rate stays at the configured value.
        if pipeline_stage_num > 1:
            omni_cfg_dict["env"]["physics_frequency"] = (
                omni_cfg_dict["env"]["physics_frequency"] / pipeline_stage_num
            )
        self.env = VectorEnvironment(num_envs, omni_cfg_dict)
        renderer_mode = str(OmegaConf.select(cfg, "renderer_mode", default="rlinf"))
        apply_runtime_renderer_settings(renderer_mode)
        wrapper_name = OmegaConf.select(cfg, "omni_config.env.env_wrapper")
        self.env = apply_env_wrapper(self.env, wrapper_name)

        # Isaac Sim's `omni.kit.app` calls ``gc.disable()`` at startup.
        # OmniGibson has self-referential cycles and leaks memory when
        # cyclic GC is disabled. Since we do not need real-time performance,
        # enable cyclic GC here so that we do not encounter OOMs in long runs.
        gc.enable()

        step_signature = inspect.signature(self.env.step)
        step_params = step_signature.parameters.values()
        step_supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in step_params
        )
        self.step_supports_get_obs = (
            step_supports_kwargs or "get_obs" in step_signature.parameters
        )
        self.step_supports_render = (
            step_supports_kwargs or "render" in step_signature.parameters
        )
        self.step_supports_env_indices = "env_indices" in step_signature.parameters
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )
        self.stop_chunk_on_done = bool(
            OmegaConf.select(cfg, "subpool.enabled", default=False)
        )
        self.subtask_reward_tracker = None
        self.active_task_reward = None
        self.active_subtask_index = None
        self.active_pool_type = None
        self.current_control = None
        self.current_snapshot_metadata = None
        self.control_serializer = None
        self.state_capture_interval = int(
            OmegaConf.select(cfg, "subpool.state_capture_interval", default=8)
        )
        self.recovery_min_lag_states = int(
            OmegaConf.select(cfg, "subpool.recovery_min_lag_states", default=2)
        )
        self.recovery_max_lag_states = int(
            OmegaConf.select(cfg, "subpool.recovery_max_lag_states", default=16)
        )
        self.recovery_rng = __import__("numpy").random.default_rng(int(cfg.seed))
        self.state_ring_size = int(
            OmegaConf.select(cfg, "subpool.state_ring_size", default=32)
        )
        self.state_ring = deque(maxlen=self.state_ring_size)
        self.pending_pool_candidates = None
        self.subpool_episode_done = False
        self.last_subpool_obs = None
        self.last_subpool_info = None
        if self.stop_chunk_on_done:
            if self.state_capture_interval <= 0:
                raise ValueError("subpool.state_capture_interval must be positive.")
            if self.recovery_min_lag_states <= 0:
                raise ValueError("subpool.recovery_min_lag_states must be positive.")
            if self.recovery_max_lag_states < self.recovery_min_lag_states:
                raise ValueError(
                    "subpool.recovery_max_lag_states must be no smaller than "
                    "subpool.recovery_min_lag_states."
                )
            if self.state_ring_size <= self.recovery_min_lag_states:
                raise ValueError(
                    "subpool.state_ring_size must exceed "
                    "subpool.recovery_min_lag_states."
                )

        if self.skip_intermediate_obs_in_chunk and not self.step_supports_get_obs:
            self.logger.warning(
                "skip_intermediate_obs_in_chunk is True but OmniGibson env step does not "
                "support get_obs; this config will be ignored."
            )

        if self.pipeline_stage_num > 1 and not self.step_supports_env_indices:
            self.logger.warning(
                "pipeline_stage_num > 1 but OmniGibson env step does not support env_indices; "
                "this may cause inefficiency since every pipeline step will still "
                "advance every env with zeroed-out actions for inactive envs."
            )
        if self.stop_chunk_on_done:
            self._setup_online_grounding(cfg)

    def _setup_online_grounding(self, cfg: DictConfig) -> None:
        """Enable local masks and the exact P2 serializer used during SFT."""
        from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES

        for wrapped_env in self.env.envs:
            base_env = wrapped_env
            while hasattr(base_env, "env"):
                base_env = base_env.env
            robot = base_env.robots[0]
            for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
                sensor = robot.sensors[camera_name.split("::")[1]]
                for modality in ("seg_semantic", "seg_instance_id"):
                    sensor.add_modality(modality)
            base_env.load_observation_space()

        from rlinf.data.b1k_grounded import (
            ControlSerializer,
            ReservedTokenMapping,
        )

        mapping_path = OmegaConf.select(cfg, "subpool.token_mapping_path")
        if not mapping_path:
            raise ValueError("subpool.token_mapping_path is required for online P2.")
        with open(mapping_path, "r", encoding="utf-8") as mapping_file:
            mapping = ReservedTokenMapping.from_dict(json.load(mapping_file))
        self.control_serializer = ControlSerializer(mapping)

    def _attach_online_grounding(self, raw_obs: dict) -> dict:
        """Recompute the P2 object and part bboxes for one observation."""
        if self.current_control is None or self.control_serializer is None:
            raise RuntimeError("Online grounding was not primed by a subpool reset.")

        import numpy as np
        from omnigibson.sensors.vision_sensor import VisionSensor

        from rlinf.data.b1k_grounded import (
            CameraID,
            ControlProfile,
            EntityResolver,
            ground_control_spec,
        )

        camera_ids = {
            "zed_link:Camera:0": CameraID.HEAD,
            "left_realsense_link:Camera:0": CameraID.LEFT_WRIST,
            "right_realsense_link:Camera:0": CameraID.RIGHT_WRIST,
        }
        segmentations = {}
        for sensor_data in raw_obs.values():
            if not isinstance(sensor_data, dict):
                continue
            for sensor_name, modalities in sensor_data.items():
                if not isinstance(modalities, dict):
                    continue
                for name_fragment, camera_id in camera_ids.items():
                    if name_fragment in sensor_name:
                        segmentation = modalities.get("seg_instance_id")
                        if segmentation is None:
                            raise KeyError(
                                f"Missing seg_instance_id for camera {sensor_name!r}."
                            )
                        if hasattr(segmentation, "cpu"):
                            segmentation = segmentation.cpu()
                        segmentations[camera_id] = np.asarray(segmentation)
        if set(segmentations) != set(camera_ids.values()):
            raise KeyError(
                "Online P2 grounding requires head, left-wrist, and right-wrist "
                "instance masks."
            )
        grounded = ground_control_spec(
            self.current_control,
            segmentations,
            EntityResolver(VisionSensor.INSTANCE_ID_REGISTRY),
            infer_missing_parts=True,
        )
        raw_obs["_subpool"] = {
            "task_description": self.control_serializer.serialize(
                grounded, ControlProfile.P2_GROUND_SG
            )
        }
        return raw_obs

    def get_activity_name(self):
        return self.instance_loader.activity_name

    def _call_step(self, actions, env_indices=None, get_obs=True, render=True):
        """Call ``self.env.step`` forwarding only the kwargs it supports."""
        kwargs = {}
        if self.step_supports_get_obs:
            kwargs["get_obs"] = get_obs
        if self.step_supports_render:
            kwargs["render"] = render
        if env_indices is not None:
            kwargs["env_indices"] = env_indices
        return self.env.step(actions, **kwargs)

    def _call_reset(self, reset_indices=None, get_obs=True):
        """Call ``self.env.reset`` through one normalized code path."""
        kwargs = {"get_obs": get_obs}
        if reset_indices is not None:
            kwargs["env_indices"] = reset_indices
        return self.env.reset(**kwargs)

    def _step_shard(
        self,
        actions: torch.Tensor,
        env_indices: list[int],
        need_obs: bool,
    ):
        """Step one shard for a single chunk timestep.

        ``actions`` is the zero-padded ``[num_shard, action_dim]`` action
        tensor (inactive rows already carry zero actions). ``env_indices``
        is the ascending list of local rows that should advance.

        Returns outputs only for ``env_indices``, in that same order.
        """
        if self.step_supports_env_indices:
            raw_obs, rewards, terminates, truncates, infos = self._call_step(
                [actions[i] for i in env_indices],
                env_indices=env_indices,
                get_obs=need_obs,
                render=need_obs,
            )
        else:
            raw_obs, rewards, terminates, truncates, infos = self._call_step(
                actions,
                get_obs=need_obs,
                render=need_obs,
            )
            if need_obs:
                raw_obs = [raw_obs[i] for i in env_indices]
            rewards = [rewards[i] for i in env_indices]
            terminates = [terminates[i] for i in env_indices]
            truncates = [truncates[i] for i in env_indices]
            infos = [infos[i] for i in env_indices]

        return (
            (
                [self._attach_online_grounding(obs) for obs in raw_obs]
                if need_obs and self.stop_chunk_on_done
                else list(raw_obs)
                if need_obs
                else None
            ),
            to_tensor(rewards),
            to_tensor(terminates),
            to_tensor(truncates),
            list(infos),
        )

    def chunk_step(self, actions, env_indices):
        """Step a full chunk for one shard.

        Args:
            actions: Zero-padded ``[num_shard, chunk, action_dim]`` action
                matrix for this VectorEnvironment.
            env_indices: Ascending local rows that should advance every
                chunk step.
        """
        _, chunk_size, _ = actions.shape

        if self.stop_chunk_on_done:
            return self._chunk_step_until_done(actions, env_indices)

        results: list[tuple] = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            results.append(
                self._step_shard(actions[:, t], env_indices, need_obs=need_obs)
            )
        observations, rewards, terms, truncs, infos = tuple(zip(*results))
        executed = tuple(
            torch.ones(len(env_indices), dtype=torch.bool) for _ in range(chunk_size)
        )
        return observations, rewards, terms, truncs, infos, executed

    @staticmethod
    def _info_done(info: dict) -> bool:
        done = info.get("done", {})
        conditions = done.get("termination_conditions", {})
        return bool(done.get("success", False)) or any(
            bool(value.get("done", False))
            for value in conditions.values()
            if isinstance(value, dict)
        )

    def _chunk_step_until_done(self, actions, env_indices):
        """Execute only the valid action prefix and retain its exact mask."""
        _, chunk_size, _ = actions.shape
        if self.subpool_episode_done:
            if len(env_indices) != 1:
                raise RuntimeError(
                    "A frozen subpool episode requires exactly one environment."
                )
            if self.last_subpool_obs is None or self.last_subpool_info is None:
                raise RuntimeError("Frozen subpool episode has no terminal cache.")
            return _repeat_terminal_subpool_chunk(
                self.last_subpool_obs,
                self.last_subpool_info,
                chunk_size,
            )

        positions = {env_index: pos for pos, env_index in enumerate(env_indices)}
        active_indices = list(env_indices)
        last_obs = [None] * len(env_indices)
        last_infos = [{} for _ in env_indices]
        results = []

        for t in range(chunk_size):
            need_obs = True
            obs_t = list(last_obs)
            rewards_t = torch.zeros(len(env_indices), dtype=torch.float32)
            terms_t = torch.zeros(len(env_indices), dtype=torch.bool)
            truncs_t = torch.zeros(len(env_indices), dtype=torch.bool)
            infos_t = list(last_infos)
            executed_t = torch.zeros(len(env_indices), dtype=torch.bool)

            if active_indices:
                raw_obs, rewards, terms, truncs, infos = self._step_shard(
                    actions[:, t], active_indices, need_obs=need_obs
                )
                next_active = []
                for source_index, env_index in enumerate(active_indices):
                    pos = positions[env_index]
                    obs_t[pos] = raw_obs[source_index]
                    info = infos[source_index]
                    if self.subtask_reward_tracker is None:
                        raise RuntimeError(
                            "Subpool chunk execution started before reward priming."
                        )
                    stage_info = get_stage_info(info, self.active_subtask_index)
                    self._apply_direct_navigation_predicate(stage_info)
                    self._attach_arm_specific_distances(stage_info)
                    outcome = self.subtask_reward_tracker.step(stage_info)
                    info["subpool"] = {
                        "subtask_id": self.active_subtask_index,
                        "pool_type": self.active_pool_type,
                        "success": outcome.success,
                        "timeout": outcome.timeout,
                        "potential": outcome.potential,
                        "progress": outcome.progress,
                    }
                    rewards_t[pos] = outcome.reward
                    terms_t[pos] = outcome.success
                    truncs_t[pos] = outcome.timeout
                    infos_t[pos] = info
                    executed_t[pos] = True
                    is_done = outcome.success or outcome.timeout
                    if is_done:
                        self.subpool_episode_done = True
                        terminal_state = self._dump_subpool_state()
                        recovery_state = None
                        if outcome.timeout:
                            available_max_lag = min(
                                self.recovery_max_lag_states,
                                len(self.state_ring) - 1,
                            )
                            if available_max_lag >= self.recovery_min_lag_states:
                                lag = int(
                                    self.recovery_rng.integers(
                                        self.recovery_min_lag_states,
                                        available_max_lag + 1,
                                    )
                                )
                                recovery_state = list(self.state_ring)[-(lag + 1)]
                        self.pending_pool_candidates = {
                            "success_state": terminal_state
                            if outcome.success
                            else None,
                            "recovery_state": recovery_state,
                        }
                    elif (
                        self.subtask_reward_tracker.steps % self.state_capture_interval
                        == 0
                    ):
                        self.state_ring.append(self._dump_subpool_state())
                    if not is_done:
                        next_active.append(env_index)
                active_indices = next_active

            last_obs = obs_t
            last_infos = infos_t
            results.append((obs_t, rewards_t, terms_t, truncs_t, infos_t, executed_t))

        if self.subpool_episode_done:
            self.last_subpool_obs = last_obs[0]
            self.last_subpool_info = last_infos[0]
        return tuple(zip(*results))

    def _apply_direct_navigation_predicate(self, stage_info) -> None:
        """Use the same demo-terminal base region as grounded evaluation."""
        if self.current_control is None or self.current_control.skill != "move to":
            return
        metadata = self.current_snapshot_metadata or {}
        target_pose = metadata.get("target_base_pose")
        if target_pose is None:
            raise KeyError("Move-to snapshot metadata is missing target_base_pose.")

        import math

        import omnigibson.utils.transform_utils as transform_utils

        wrapped_env = self.env.envs[0]
        base_env = wrapped_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        position, quaternion = base_env.robots[0].get_position_orientation()
        yaw = float(transform_utils.quat2euler(quaternion)[2])
        position_error = math.hypot(
            float(position[0]) - float(target_pose[0]),
            float(position[1]) - float(target_pose[1]),
        )
        yaw_delta = yaw - float(target_pose[2])
        yaw_error = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
        position_threshold = float(metadata.get("move_position_threshold", 0.5))
        yaw_threshold = math.radians(
            float(metadata.get("move_yaw_threshold_deg", 45.0))
        )
        stage_info.update(
            {
                "completed": position_error <= position_threshold
                and yaw_error <= yaw_threshold,
                "base_position_error": position_error,
                "base_yaw_error": yaw_error,
                "base_position_threshold": position_threshold,
                "base_yaw_threshold": yaw_threshold,
                "success_source": "demo_terminal_base_pose",
            }
        )

    def _attach_arm_specific_distances(self, stage_info) -> None:
        """Expose non-minimized arm distances for grounded manipulation rewards."""
        if self.active_task_reward is None or self.active_subtask_index is None:
            return
        stage_defs = getattr(self.active_task_reward, "_stage_defs", ())
        if not 0 <= self.active_subtask_index < len(stage_defs):
            raise IndexError(
                f"Active reward stage {self.active_subtask_index} is unavailable."
            )
        objects = stage_defs[self.active_subtask_index].get("objects", ())
        if not objects:
            return

        import torch as th
        from omnigibson.object_states.toggle import ToggledOn
        from omnigibson.reward_functions.support_utils import get_obj_center

        wrapped_env = self.env.envs[0]
        base_env = wrapped_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        robot = base_env.robots[0]
        target = objects[0]
        target_position = get_obj_center(target)
        if len(objects) >= 2 and objects[1] is not None:
            stage_info["object_to_support_surface_distance"] = (
                _support_surface_distance(target.aabb, objects[1].aabb)
            )

        toggle_state = target.states.get(ToggledOn)
        marker = None if toggle_state is None else toggle_state.visual_marker
        if marker is None:
            toggle_position = target_position
            marker_radius = 0.0
        else:
            toggle_position = marker.get_position_orientation()[0]
            marker_radius = float(th.min(marker.extent * toggle_state.scale).item())

        for arm in robot.arm_names:
            eef_position = robot.get_eef_position(arm)
            stage_info[f"{arm}_eef_to_obj_distance"] = float(
                th.linalg.vector_norm(eef_position - target_position).item()
            )
            stage_info[f"{arm}_eef_to_toggle_distance"] = max(
                float(th.linalg.vector_norm(eef_position - toggle_position).item())
                - marker_radius,
                0.0,
            )

    @staticmethod
    def _dump_subpool_state():
        import omnigibson as og

        return og.sim.dump_state(serialized=False)

    def drain_pool_candidates(self):
        """Return terminal/recovery candidates once, then clear them."""
        candidates = self.pending_pool_candidates
        self.pending_pool_candidates = None
        return candidates

    def load_serialized_state(
        self,
        state,
        *,
        activity_name: str,
        scene_model: str,
        instance_id: int,
        subtask_id: int,
        pool_type: str,
        reward_spec,
        control_json: str,
        snapshot_metadata,
    ):
        """Reset and restore one audited state in a single-env process."""
        if len(self.env) != 1:
            raise RuntimeError("Serialized subpool restore requires exactly one env.")
        if self.instance_loader.activity_name != activity_name:
            raise ValueError(
                f"Snapshot activity {activity_name!r} does not match runtime "
                f"activity {self.instance_loader.activity_name!r}."
            )
        import omnigibson as og

        wrapped_env = self.env.envs[0]
        base_env = wrapped_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        runtime_scene_model = str(getattr(base_env.scene, "scene_model", ""))
        if runtime_scene_model != scene_model:
            raise ValueError(
                f"Snapshot scene {scene_model!r} does not match runtime scene "
                f"{runtime_scene_model!r}."
            )

        self.instance_loader.prepare_reset(self.env)
        self._call_reset(get_obs=False)
        full_state = _move_state_tensors(state, og.sim.device)
        og.sim.load_state(full_state, serialized=False)
        # Simulator state does not include controller goals. Synchronize them
        # before the refresh step so stale reset targets cannot move the robot
        # away from the canonical pose.
        sync_robot_after_pose_override(base_env.robots[0])
        # Official B1K evaluation constructs the scene from the seed template
        # (normally instance 0), then applies the selected challenge instance's
        # TRO state.  Canonical snapshots are dumped after that mutation.  Keep
        # the bootstrap template id separate from the logical task instance and
        # restore the latter after loading the exact serialized simulator state.
        base_env.task.activity_instance_id = instance_id
        # Object-state predicates and camera buffers are stale immediately after
        # load_state. The simulation step refreshes physics and object states;
        # the explicit render then propagates the restored transforms through
        # Fabric before vision sensors read their first frame. Relying on the
        # render embedded in step() is insufficient during concurrent Kit
        # startup and can yield an empty segmentation tensor.
        og.sim.step()
        og.sim.render()
        reward_functions = getattr(base_env.task, "_reward_functions", {})
        task_reward = reward_functions.get("task_specific")
        if task_reward is None or not hasattr(task_reward, "set_active_stage_index"):
            raise TypeError(
                "Subpool RL requires a sequential task_specific reward with "
                "set_active_stage_index()."
            )
        task_reward.set_active_stage_index(subtask_id)
        self.active_task_reward = task_reward
        self.active_subtask_index = subtask_id
        if pool_type not in SUBPOOL_TYPES:
            raise ValueError(f"Unknown subpool type {pool_type!r}.")
        self.active_pool_type = pool_type
        self.subtask_reward_tracker = SubtaskRewardTracker(
            SubtaskRewardSpec.from_mapping(reward_spec)
        )
        obs, info = wrapped_env.get_obs()
        from rlinf.data.b1k_grounded import GroundedControlSpec

        self.current_control = GroundedControlSpec.from_json(control_json)
        self.current_snapshot_metadata = dict(snapshot_metadata)
        self.state_ring.clear()
        self.state_ring.append(self._dump_subpool_state())
        self.pending_pool_candidates = None
        self.subpool_episode_done = False
        self.last_subpool_obs = None
        self.last_subpool_info = None
        return [self._attach_online_grounding(obs)], [info]

    def reset(self, reset_indices=None, get_obs=True):
        self.instance_loader.prepare_reset(self.env)
        result = self._call_reset(
            reset_indices=reset_indices,
            get_obs=get_obs,
        )
        if not get_obs:
            return None, None

        raw_obs, infos = result
        return list(raw_obs), list(infos)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class BehaviorProcessPool:
    """Singleton OmniGibson subprocess pool manager.

    Use :meth:`acquire_shared` to obtain the singleton pool; use :meth:`release_shared` when done.
    """

    _shared_pool: ClassVar["BehaviorProcessPool | None"] = None
    _shared_refcount: ClassVar[int] = 0
    _pipeline_next_idx: ClassVar[int] = 0

    @classmethod
    def acquire_shared(
        cls,
        cfg: DictConfig,
        worker_info,
        pipeline_stage_num: int,
        num_envs: int,
    ) -> tuple["BehaviorProcessPool", int]:
        """Attach to the shared pool and return ``(pool, pool_offset)``."""
        if cls._shared_pool is None:  # pool init
            total_envs = int(OmegaConf.select(cfg, "total_num_envs", default=None))
            total_envs_per_worker = total_envs // worker_info.group_world_size
            num_env_subprocess = int(
                OmegaConf.select(cfg, "num_env_subprocess", default=1)
            )
            cls._shared_pool = cls(
                cfg,
                total_envs_per_worker,
                num_env_subprocess,
                pipeline_stage_num,
            )

        idx = cls._pipeline_next_idx
        global_offset = idx * num_envs
        cls._pipeline_next_idx += 1
        cls._shared_refcount += 1

        pool = cls._shared_pool

        if global_offset + num_envs > pool.total_num_envs:
            raise ValueError(
                f"BehaviorEnv slice [{global_offset}, {global_offset + num_envs}) "
                f"exceeds pool total_num_envs={pool.total_num_envs}."
            )
        return pool, global_offset

    @classmethod
    def release_shared(cls) -> None:
        """Drop refcount; tear down the shared pool when the last env releases."""
        if cls._shared_pool is None:
            return
        cls._shared_refcount -= 1
        if cls._shared_refcount <= 0:
            cls._shared_pool.close()
            cls._shared_pool = None
            cls._pipeline_next_idx = 0

    def __init__(
        self,
        cfg: DictConfig,
        total_num_envs: int,
        num_env_subprocess: int,
        pipeline_stage_num: int,
    ):
        if total_num_envs % num_env_subprocess != 0:
            raise ValueError(
                f"total_num_envs({total_num_envs}) must be divisible by num_env_subprocess({num_env_subprocess})"
            )

        self.logger = get_logger()
        self.cfg = cfg
        self.total_num_envs = total_num_envs
        self.num_env_subprocess = num_env_subprocess
        self.num_env_shard = total_num_envs // num_env_subprocess
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )

        # Create subprocess actors with a retry/backoff loop. Actor startup
        # can fail (e.g. simulator plugin errors); retry a few times to handle
        # transient failures. Configurable via `behavior.init_retry_*` keys.
        max_attempts = int(
            OmegaConf.select(cfg, "behavior.init_retry_count", default=3)
        )
        retry_delay = float(
            OmegaConf.select(cfg, "behavior.init_retry_delay", default=5.0)
        )
        backoff = float(
            OmegaConf.select(cfg, "behavior.init_retry_backoff", default=2.0)
        )

        for attempt in range(1, max_attempts + 1):
            try:
                # A BEHAVIOR process inherits the parent EnvWorker's selected
                # CUDA device and node-specific asset / Python configuration.
                # Nested Ray actors are otherwise free to land on any cluster
                # node, which is incorrect for heterogeneous deployments where
                # only the env node can render OmniGibson.
                node_id = ray.get_runtime_context().get_node_id()
                scheduling_strategy = (
                    ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    )
                )
                child_env_vars = {
                    # BehaviorProcess deliberately does not request another Ray
                    # GPU: its parent EnvWorker already owns the device. Prevent
                    # Ray from replacing the inherited device selection with an
                    # empty CUDA_VISIBLE_DEVICES for this zero-GPU nested actor.
                    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible_devices:
                    child_env_vars["CUDA_VISIBLE_DEVICES"] = visible_devices
                self.env_processes = [
                    BehaviorProcess.options(
                        scheduling_strategy=scheduling_strategy,
                        runtime_env={"env_vars": child_env_vars},
                    ).remote(self.cfg, self.num_env_shard, pipeline_stage_num)
                    for _ in range(self.num_env_subprocess)
                ]

                # Wait for all instances to initialize and fetch their activity name
                activity_names_refs = [
                    proc.get_activity_name.remote() for proc in self.env_processes
                ]
                activity_names = ray.get(activity_names_refs)
                break
            except Exception as e:  # noqa: BLE001 - we want to catch any Ray/OmniGibson init error
                # Best-effort cleanup of any partially-created actors
                for proc in getattr(self, "env_processes", []):
                    try:
                        ray.kill(proc)
                    except Exception:
                        pass
                self.env_processes = []

                if attempt >= max_attempts:
                    self.logger.error(
                        "Failed to start BehaviorProcess actors after %d attempts: %s",
                        attempt,
                        e,
                    )
                    raise

                self.logger.warning(
                    "BehaviorProcess creation failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay *= backoff

        if len(set(activity_names)) != 1:
            raise RuntimeError(
                f"Behavior env subprocesses reported different activity_name: "
                f"{activity_names}"
            )
        self.activity_name = activity_names[0]

    def _slice_plan(
        self, global_start: int, num_envs: int
    ) -> list[tuple[int, list[int], list[int]]]:
        """Build the per-subprocess plan for a contiguous global slice.

        Returns entries of ``(subproc_idx, slice_positions, local_rows)``.
        ``slice_positions`` are indices inside the caller's slice and
        ``local_rows`` are the matching rows owned by that subprocess.
        """
        slice_positions_by_proc = [[] for _ in range(self.num_env_subprocess)]
        local_rows_by_proc = [[] for _ in range(self.num_env_subprocess)]
        for pos in range(num_envs):
            global_idx = global_start + pos
            sp = global_idx % self.num_env_subprocess
            slice_positions_by_proc[sp].append(pos)
            local_rows_by_proc[sp].append(global_idx // self.num_env_subprocess)

        return [
            (sp, slice_positions_by_proc[sp], local_rows_by_proc[sp])
            for sp in range(self.num_env_subprocess)
            if slice_positions_by_proc[sp]
        ]

    def env_reset_slice(self, global_start: int, num_envs: int):
        """Reset envs in ``[global_start, global_start + num_envs)``."""
        if num_envs == 0:
            return [], []
        plan = self._slice_plan(global_start, num_envs)
        refs = [
            self.env_processes[sp].reset.remote(local_rows)
            for sp, _positions, local_rows in plan
        ]

        shard_results = ray.get(refs)
        all_raw_obs: list = [None] * num_envs
        all_infos: list = [None] * num_envs
        for (raw_obs, infos), (_sp, positions, _local_rows) in zip(shard_results, plan):
            for pos, obs, info in zip(positions, raw_obs, infos):
                all_raw_obs[pos] = obs
                all_infos[pos] = info
        return all_raw_obs, all_infos

    def env_chunk_step_slice(
        self,
        global_start: int,
        slice_num_envs: int,
        chunk_actions: torch.Tensor,
    ):
        """Run chunk_step on shards; pool handles all sharding/merging.
        ``chunk_actions`` must be ``[slice_num_envs, chunk, action_dim]``.
        """
        chunk_size = chunk_actions.shape[1]
        action_dim = chunk_actions.shape[-1]
        plan = self._slice_plan(global_start, slice_num_envs)

        refs = []
        for sp, positions, local_rows in plan:
            actions_j = torch.zeros(
                self.num_env_shard,
                chunk_size,
                action_dim,
                dtype=chunk_actions.dtype,
            )
            actions_j[local_rows] = chunk_actions[positions]
            refs.append(self.env_processes[sp].chunk_step.remote(actions_j, local_rows))

        shard_results = ray.get(refs)
        return self._merge_shards(shard_results, plan, slice_num_envs, chunk_size)

    def load_serialized_state(
        self,
        global_start: int,
        num_envs: int,
        state,
        *,
        activity_name: str,
        scene_model: str,
        instance_id: int,
        subtask_id: int,
        pool_type: str,
        reward_spec,
        control_json: str,
        snapshot_metadata,
    ):
        """Restore a state through the only safe single-env subpool layout."""
        if self.num_env_subprocess != 1 or self.total_num_envs != 1:
            raise RuntimeError(
                "Subpool state restore requires one process and one env."
            )
        if global_start != 0 or num_envs != 1:
            raise RuntimeError("Subpool state restore requires the full one-env slice.")
        return ray.get(
            self.env_processes[0].load_serialized_state.remote(
                state,
                activity_name=activity_name,
                scene_model=scene_model,
                instance_id=instance_id,
                subtask_id=subtask_id,
                pool_type=pool_type,
                reward_spec=reward_spec,
                control_json=control_json,
                snapshot_metadata=snapshot_metadata,
            )
        )

    def drain_pool_candidates(self):
        """Drain online state candidates from the one subpool process."""
        if self.num_env_subprocess != 1 or self.total_num_envs != 1:
            raise RuntimeError("Subpool candidates require one process and one env.")
        return ray.get(self.env_processes[0].drain_pool_candidates.remote())

    def _merge_shards(
        self,
        shard_results: list,
        plan: list[tuple[int, list[int], list[int]]],
        slice_num_envs: int,
        chunk_size: int,
    ):
        """Gather per-subprocess shard outputs into ``[chunk][slice]`` order."""
        merged_obs: list = []
        merged_rewards: list = []
        merged_terms: list = []
        merged_trunc: list = []
        merged_infos: list = []
        merged_executed: list = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            obs_t: list | None = [None] * slice_num_envs if need_obs else None
            reward_t = torch.zeros(slice_num_envs, dtype=torch.float32)
            term_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            trunc_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            info_t: list = [{} for _ in range(slice_num_envs)]
            executed_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            for (
                obs_per_t,
                rewards_per_t,
                terms_per_t,
                truncs_per_t,
                infos_per_t,
                executed_per_t,
            ), (
                _sp,
                positions,
                _local_rows,
            ) in zip(shard_results, plan):
                obs_at_t = obs_per_t[t]
                rewards_at_t = rewards_per_t[t]
                terms_at_t = terms_per_t[t]
                truncs_at_t = truncs_per_t[t]
                infos_at_t = infos_per_t[t]
                for i, pos in enumerate(positions):
                    if need_obs:
                        obs_t[pos] = obs_at_t[i]
                    reward_t[pos] = float(rewards_at_t[i])
                    term_t[pos] = bool(terms_at_t[i])
                    trunc_t[pos] = bool(truncs_at_t[i])
                    info_t[pos] = infos_at_t[i]
                    executed_t[pos] = bool(executed_per_t[t][i])
            merged_obs.append(obs_t)
            merged_rewards.append(reward_t)
            merged_terms.append(term_t)
            merged_trunc.append(trunc_t)
            merged_infos.append(info_t)
            merged_executed.append(executed_t)
        return (
            merged_obs,
            merged_rewards,
            merged_terms,
            merged_trunc,
            merged_infos,
            merged_executed,
        )

    def close(self) -> None:
        refs = [proc.close.remote() for proc in self.env_processes]
        ray.get(refs)

        # Kill the procs to free up resources immediately
        for proc in self.env_processes:
            ray.kill(proc)

        self.env_processes = []


class BehaviorEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.logger = get_logger()
        self.reward_coef = cfg.get("reward_coef", 1)

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self._is_start = True
        self.enable_offload = cfg.get("enable_offload", False)
        self.enable_init_offload = cfg.get("enable_init_offload", True)
        self.pool = None
        self.pool_offset = None
        self.task_description = None
        self.last_executed_action_mask = None
        if total_num_processes % worker_info.group_world_size != 0:
            raise ValueError(
                f"total_num_processes ({total_num_processes}) must be divisible by "
                f"worker_info.group_world_size ({worker_info.group_world_size}) to infer pipeline_stage_num."
            )
        self.pipeline_stage_num = total_num_processes // worker_info.group_world_size

        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = torch.tensor(cfg.max_episode_steps)
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        if self.record_metrics:
            self._init_metrics()
        if not (self.enable_offload and not self.enable_init_offload):
            self._ensure_pool()
            self._init_env()

    def _ensure_pool(self):
        if self.pool is None:
            self.pool, self.pool_offset = BehaviorProcessPool.acquire_shared(
                self.cfg,
                self.worker_info,
                self.pipeline_stage_num,
                self.num_envs,
            )

    def _load_tasks_cfg(self, activity_name: str):
        # Read task description

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[activity_name]

    def _init_env(self):
        self._ensure_pool()
        self._load_tasks_cfg(self.pool.activity_name)

    def env_reset(self):
        self._ensure_pool()
        return self.pool.env_reset_slice(self.pool_offset, self.num_envs)

    def env_chunk_step(self, chunk_actions: torch.Tensor):
        self._ensure_pool()
        return self.pool.env_chunk_step_slice(
            self.pool_offset,
            self.num_envs,
            chunk_actions,
        )

    def _extract_obs_image(self, raw_obs):
        state = None
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = convert_uint8_rgb(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = convert_uint8_rgb(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = convert_uint8_rgb(v["rgb"])
                elif "proprio" in k:
                    state = v
        assert state is not None, (
            "state is not found in the observation which is required for the behavior training."
        )

        return {
            "main_images": zed_image,  # [H, W, C]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, H, W, C]
            "state": state,
            "task_description": raw_obs.get("_subpool", {}).get("task_description"),
        }

    def _wrap_obs(self, obs_list):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": [self.task_description for _ in range(self.num_envs)],
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, 32]
        }
        online_descriptions = [
            extracted["task_description"] for extracted in extracted_obs_list
        ]
        if any(description is not None for description in online_descriptions):
            if not all(description is not None for description in online_descriptions):
                raise ValueError(
                    "Online grounding is missing for part of the env batch."
                )
            obs["task_descriptions"] = online_descriptions
        return obs

    def _calc_step_reward(self, reward):
        return self.reward_coef * reward

    def reset(self):
        if self.enable_offload and self.pool is None:
            self._init_env()
        raw_obs, infos = self.env_reset()
        obs = self._wrap_obs(raw_obs)
        rewards = torch.zeros(self.num_envs, dtype=bool)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim].
        chunk_actions = torch.as_tensor(chunk_actions).detach().cpu()
        (
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_truncations_list,
            raw_infos_list,
            raw_executed_list,
        ) = self.env_chunk_step(chunk_actions)
        self.last_executed_action_mask = torch.stack(raw_executed_list, dim=1)

        obs_list = []
        infos_list = []
        scaled_rewards_list = []
        merged_terminations_list = []
        info_done_flags = []
        for raw_obs, raw_rewards, raw_terminations, step_infos in zip(
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_infos_list,
        ):
            if raw_obs is None:
                obs_list.append(None)
            else:
                obs_list.append(self._wrap_obs(raw_obs))
            step_rewards = self._calc_step_reward(raw_rewards)
            infos_list.append(self._record_metrics(step_rewards, step_infos))
            if self.ignore_terminations:
                raw_terminations = torch.zeros_like(raw_terminations)
            merged_terminations_list.append(raw_terminations)
            scaled_rewards_list.append(step_rewards)
            # `raw_infos_list[i]` is a list of per-env info dicts for chunk step i.
            step_done = [
                self._extract_info_done(info) if isinstance(info, dict) else False
                for info in step_infos
            ]
            info_done_flags.append(torch.tensor(step_done, dtype=torch.bool))

        chunk_rewards = torch.stack(
            scaled_rewards_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_terminations = torch.stack(
            merged_terminations_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_truncations = torch.stack(
            raw_truncations_list, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)

        # Some OmniGibson builds may report episode completion primarily via
        # `info["done"]` while leaving `terminations`/`truncations` booleans
        # as all-False for the whole chunk. RLinf's evaluation metrics gate on
        # `terminations|truncations`, so we fall back to info-done here.
        past_info_dones = torch.stack(info_done_flags, dim=1).any(dim=1)
        preserve_primitive_dones = bool(
            OmegaConf.select(self.cfg, "subpool.enabled", default=False)
        )
        if preserve_primitive_dones:
            # Subpool completion is produced by the selected stage predicate in
            # BehaviorProcess. The official whole-task done flag is unrelated.
            past_info_dones = torch.zeros_like(past_info_dones)

        # If the config asks to ignore terminations, map info-done into
        # truncations; otherwise map it into terminations.
        if self.ignore_terminations:
            past_truncations = torch.logical_or(past_truncations, past_info_dones)
        else:
            past_terminations = torch.logical_or(past_terminations, past_info_dones)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if preserve_primitive_dones:
            chunk_terminations = raw_terminations
            chunk_truncations = raw_truncations
        else:
            chunk_terminations = torch.zeros_like(raw_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_truncations)
            chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self):
        return "cuda"

    @property
    def elapsed_steps(self):
        return self.max_episode_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if not self.record_metrics:
            return
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.success_once[mask] = False
        self.returns[mask] = 0

    def _record_metrics(self, rewards, infos):
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            done_dict = info.get("done", {})
            subpool_info = info.get("subpool", {})
            success = bool(subpool_info.get("success", done_dict.get("success", False)))
            episode_info = {
                "success": success,
                "episode_length": info.get("episode_length", 0),
            }
            if subpool_info:
                pool_type = subpool_info.get("pool_type")
                episode_info.update(
                    {
                        "subtask_id": int(subpool_info["subtask_id"]),
                        "subpool_id": SUBPOOL_TYPES.index(pool_type),
                        "subtask_timeout": bool(subpool_info.get("timeout", False)),
                    }
                )
            self.returns[env_idx] += reward
            self.success_once[env_idx] = self.success_once[env_idx] | success
            episode_info["success_once"] = self.success_once[env_idx].clone()

            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )
            if self.ignore_terminations:
                episode_info["success_at_end"] = info["success"]

            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    @staticmethod
    def _extract_info_done(info: dict) -> bool:
        tc = info["done"]["termination_conditions"]
        return any(v["done"] for v in tc.values())

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        # use for multi task training
        pass

    def offload(self):
        self.close()

    def close(self):
        if self.pool:
            BehaviorProcessPool.release_shared()
            self.pool = None
            self.pool_offset = None


class BehaviorSubpoolEnv(BehaviorEnv):
    """Single-env BEHAVIOR adapter with audited simulator-state resets."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        if total_num_processes % worker_info.group_world_size != 0:
            raise ValueError("Cannot infer an integer pipeline_stage_num.")
        pipeline_stage_num = total_num_processes // worker_info.group_world_size
        validate_subpool_env_config(
            cfg,
            num_envs=num_envs,
            pipeline_stage_num=pipeline_stage_num,
        )
        manifest_path = OmegaConf.select(cfg, "subpool.manifest_path")
        if not manifest_path:
            raise ValueError("subpool.manifest_path is required.")
        self.catalog = SubpoolCatalog.from_jsonl(manifest_path)
        validate_subpool_rollout_horizons(
            [
                int(record.metadata["reward"]["max_steps"])
                for record in self.catalog.records
            ],
            episode_horizon=int(cfg.max_episode_steps),
            rollout_horizon=int(cfg.max_steps_per_rollout_epoch),
        )
        self._runtime_signature = self.catalog.runtime_signature
        runtime_activity, runtime_scene, _ = self._runtime_signature
        self._fixed_subtask_id = OmegaConf.select(
            cfg, "subpool.fixed_subtask_id", default=None
        )
        self._subtask_sampling = str(
            OmegaConf.select(cfg, "subpool.subtask_sampling", default="round_robin")
        )
        if self._subtask_sampling != "round_robin":
            raise ValueError(
                "Correctness-first subpool PPO requires round-robin subtask sampling."
            )
        validate_round_robin_coverage(
            self.catalog.subtask_ids,
            env_world_size=total_num_processes,
            fixed_subtask_id=self._fixed_subtask_id,
        )
        # OmniGibson fixes the activity and scene when the persistent simulator
        # process is constructed.  B1K challenge instances are TRO states, not
        # standalone scene templates: construct from the seed template and load
        # the exact challenge-instance state during ``env_reset``.
        bootstrap_instance = int(
            OmegaConf.select(cfg, "subpool.bootstrap_instance_id", default=0)
        )
        if bootstrap_instance < 0:
            raise ValueError("subpool.bootstrap_instance_id must be non-negative.")
        with open_dict(cfg):
            cfg.omni_config.task.activity_name = runtime_activity
            cfg.omni_config.task.activity_instance_id = bootstrap_instance
            cfg.omni_config.task.instance_resample_mode = "disabled"
            cfg.omni_config.task.online_object_sampling = False
            cfg.omni_config.scene.scene_model = runtime_scene
        self._manifest_path = os.path.abspath(manifest_path)
        self._dynamic_updates = bool(
            OmegaConf.select(cfg, "subpool.dynamic_updates", default=True)
        )
        self._max_dynamic_per_subtask_pool = int(
            OmegaConf.select(
                cfg,
                "subpool.max_dynamic_per_subtask_pool",
                default=32,
            )
        )
        if self._max_dynamic_per_subtask_pool <= 0:
            raise ValueError("subpool.max_dynamic_per_subtask_pool must be positive.")
        self._store = SubpoolStore(self._manifest_path)
        self._rng = __import__("numpy").random.default_rng(
            int(cfg.seed) + int(seed_offset)
        )
        self._pool_weights = OmegaConf.to_container(
            OmegaConf.select(cfg, "subpool.pool_weights", default={}), resolve=True
        )
        self._subtask_cursor = int(seed_offset)
        self.current_snapshot = None
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
            record_metrics=record_metrics,
        )

    @property
    def subtask_ids(self) -> torch.Tensor:
        if self.current_snapshot is None:
            raise RuntimeError("Subpool env has not been reset.")
        return torch.tensor([self.current_snapshot.subtask_id], dtype=torch.long)

    @property
    def subpool_ids(self) -> torch.Tensor:
        if self.current_snapshot is None:
            raise RuntimeError("Subpool env has not been reset.")
        return torch.tensor(
            [SUBPOOL_TYPES.index(self.current_snapshot.pool_type)], dtype=torch.long
        )

    def env_reset(self):
        self._ensure_pool()
        if self._dynamic_updates:
            refreshed_catalog = SubpoolCatalog.from_jsonl(self._manifest_path)
            if refreshed_catalog.runtime_signature != self._runtime_signature:
                raise ValueError(
                    "Dynamic subpool update changed the simulator runtime signature."
                )
            self.catalog = refreshed_catalog
        sampled_subtask_id = self._fixed_subtask_id
        if sampled_subtask_id is None:
            subtask_ids = self.catalog.subtask_ids
            sampled_subtask_id = subtask_ids[self._subtask_cursor % len(subtask_ids)]
            self._subtask_cursor += 1
        snapshot = self.catalog.sample(
            self._rng,
            subtask_id=sampled_subtask_id,
            pool_weights=self._pool_weights,
        )
        expected_fingerprint = OmegaConf.select(
            self.cfg, "subpool.asset_fingerprint", default=None
        )
        if expected_fingerprint and snapshot.asset_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Snapshot asset_fingerprint={snapshot.asset_fingerprint!r} does not "
                f"match configured value {expected_fingerprint!r}."
            )
        state = self.catalog.load_state(snapshot)
        raw_obs, infos = self.pool.load_serialized_state(
            self.pool_offset,
            self.num_envs,
            state,
            activity_name=snapshot.activity_name,
            scene_model=snapshot.scene_model,
            instance_id=int(snapshot.metadata["instance_id"]),
            subtask_id=snapshot.subtask_id,
            pool_type=snapshot.pool_type,
            reward_spec=snapshot.metadata["reward"],
            control_json=snapshot.control_json,
            snapshot_metadata=snapshot.metadata,
        )
        self.current_snapshot = snapshot
        return raw_obs, infos

    def chunk_step(self, chunk_actions):
        result = super().chunk_step(chunk_actions)
        _, _, terminations, truncations, _ = result
        if self._dynamic_updates and (terminations.any() or truncations.any()):
            self._append_online_candidates(
                self.pool.drain_pool_candidates(),
                success=bool(terminations.any()),
            )
        return result

    def _append_online_candidates(self, candidates, *, success: bool) -> None:
        if not candidates or self.current_snapshot is None:
            return
        if success:
            state = candidates.get("success_state")
            later_subtasks = [
                subtask_id
                for subtask_id in self.catalog.subtask_ids
                if subtask_id > self.current_snapshot.subtask_id
            ]
            if state is None or not later_subtasks:
                return
            target_subtask_id = min(later_subtasks)
            target_record = next(
                record
                for record in self.catalog.records
                if record.subtask_id == target_subtask_id
                and record.pool_type == "canonical"
            )
            pool_type = "predecessor_success"
        else:
            state = candidates.get("recovery_state")
            if state is None:
                self.logger.warning(
                    "No temporally lagged recovery state was available for %s.",
                    self.current_snapshot.snapshot_id,
                )
                return
            target_record = self.current_snapshot
            target_subtask_id = target_record.subtask_id
            pool_type = "recovery"

        existing_dynamic = sum(
            record.pool_type == pool_type
            and record.subtask_id == target_subtask_id
            and record.snapshot_id.startswith("online-")
            for record in self.catalog.records
        )
        if existing_dynamic >= self._max_dynamic_per_subtask_pool:
            return

        snapshot_id = f"online-{uuid.uuid4().hex}"
        metadata = dict(target_record.metadata)
        metadata["provenance"] = {
            "source_snapshot_id": self.current_snapshot.snapshot_id,
            "source_subtask_id": self.current_snapshot.subtask_id,
            "source_outcome": "success" if success else "timeout",
        }
        record = SubpoolSnapshot(
            snapshot_id=snapshot_id,
            state_path=f"states/{snapshot_id}.pt",
            state_sha256=full_state_sha256(state),
            activity_name=target_record.activity_name,
            scene_model=target_record.scene_model,
            asset_fingerprint=target_record.asset_fingerprint,
            subtask_id=target_subtask_id,
            skill=target_record.skill,
            pool_type=pool_type,
            task_description=target_record.task_description,
            control_json=target_record.control_json,
            episode_index=target_record.episode_index,
            frame_index=target_record.frame_index,
            metadata=metadata,
        )
        self._store.append(record, state)

    def _wrap_obs(self, obs_list):
        obs = super()._wrap_obs(obs_list)
        if self.current_snapshot is not None and not all(obs["task_descriptions"]):
            obs["task_descriptions"] = [self.current_snapshot.task_description]
        return obs

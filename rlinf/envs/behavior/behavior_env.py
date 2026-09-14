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

import copy
import gc
import inspect
import json
import os
import re
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

import gymnasium as gym
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.envs.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.behavior.subpool import (
    SUBPOOL_TYPES,
    FailureStateStore,
    SubpoolCatalog,
    SubpoolSnapshot,
    SubpoolStore,
    analyze_subtask_failure,
    full_state_sha256,
    relative_tilt_angle_deg,
    validate_round_robin_coverage,
    validate_subpool_env_config,
    validate_subpool_rollout_horizons,
)
from rlinf.envs.behavior.subpool_reward import (
    SubtaskRewardSpec,
    SubtaskRewardTracker,
    apply_reward_overrides,
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

_BEHAVIOR_CHILD_ENV_VARS = (
    "TMPDIR",
    "OMNIGIBSON_DATA_PATH",
    "OMNIGIBSON_DATASET_PATH",
    "OMNIGIBSON_KEY_PATH",
    "OMNIGIBSON_ASSET_PATH",
    "OMNIGIBSON_APPDATA_PATH",
    "OMNI_KIT_ACCEPT_EULA",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)


@dataclass
class _SubpoolSlotRuntime:
    """Episode-local state for one scene in a B1K vector environment."""

    reward_tracker: SubtaskRewardTracker | None = None
    task_reward: object | None = None
    subtask_id: int | None = None
    pool_type: str | None = None
    control: object | None = None
    snapshot_metadata: dict | None = None
    snapshot_record: dict | None = None
    sampling_group: int | None = None
    collection_index: int | None = None
    failure_reference_orientation: list[float] | None = None
    failure_tip_stable_count: int = 0
    failure_recovery_event_captured: bool = False
    state_ring: deque = field(default_factory=deque)
    pending_pool_candidates: dict | None = None
    done: bool = False
    last_obs: dict | None = None
    last_info: dict | None = None


def _isolated_appdata_path(
    base_path: str,
    *,
    node_name: str,
    visible_devices: str | None,
    process_index: int,
) -> str:
    """Return an OmniGibson appdata path private to one simulator process.

    OmniGibson explicitly requires its appdata not to be shared by concurrent
    simulator instances. Ray env workers can run on several nodes whose
    ``/mnt/public`` is shared, so the configured base path alone is not a safe
    process boundary. ``node_name`` must be stable across Ray restarts so shader
    and texture caches remain reusable.
    """

    def safe_component(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"

    worker_rank = os.environ.get("RANK", "unknown")
    device = visible_devices or "none"
    return os.path.join(
        base_path,
        f"node_{safe_component(node_name)}",
        f"rank_{safe_component(worker_rank)}_gpu_{safe_component(device)}",
        f"process_{process_index}",
    )


def _repeat_terminal_subpool_chunk(
    last_obs,
    last_info,
    chunk_size: int,
    *,
    skip_intermediate_obs: bool = False,
):
    """Return a frozen, non-executed chunk after a subtask has terminated."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    results = []
    for index in range(chunk_size):
        observation = (
            None if skip_intermediate_obs and index < chunk_size - 1 else [last_obs]
        )
        results.append(
            (
                observation,
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


def _quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[:3] = -result[:3]
    return result


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply two xyzw quaternions."""
    left_xyz, left_w = left[:3], left[3]
    right_xyz, right_w = right[:3], right[3]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz)
    )
    w = left_w * right_w - torch.dot(left_xyz, right_xyz)
    return torch.cat((xyz, w.reshape(1)))


def _rotate_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((vector, torch.zeros(1, dtype=vector.dtype, device=vector.device)))
    return _quaternion_multiply(
        _quaternion_multiply(quaternion, pure),
        _quaternion_conjugate(quaternion),
    )[:3]


def _rebase_scene_state(
    scene_state: dict,
    *,
    target_position: torch.Tensor,
    target_orientation: torch.Tensor,
) -> dict:
    """Move a scene dump between world frames without changing local state.

    Object root poses and world-frame velocities are transformed. Particle
    systems are rejected until their representation receives equivalent audited
    handling; silently copying those states would produce a plausible but wrong
    vector reset.
    """
    result = copy.deepcopy(scene_state)
    source_position = torch.as_tensor(result["pos"])
    source_orientation = torch.as_tensor(result["ori"])
    target_position = torch.as_tensor(
        target_position, dtype=source_position.dtype, device=source_position.device
    )
    target_orientation = torch.as_tensor(
        target_orientation,
        dtype=source_orientation.dtype,
        device=source_orientation.device,
    )
    systems = result["registry"].get("system_registry", {})
    if systems:
        raise NotImplementedError(
            "Vectorized subpool restore does not yet support active particle systems."
        )

    source_inverse = _quaternion_conjugate(source_orientation)
    frame_rotation = _quaternion_multiply(target_orientation, source_inverse)
    for object_state in result["registry"]["object_registry"].values():
        root = object_state.get("root_link")
        if not isinstance(root, dict):
            continue
        local_position = _rotate_vector(
            source_inverse,
            torch.as_tensor(root["pos"]) - source_position,
        )
        local_orientation = _quaternion_multiply(
            source_inverse, torch.as_tensor(root["ori"])
        )
        root["pos"] = target_position + _rotate_vector(
            target_orientation, local_position
        )
        root["ori"] = _quaternion_multiply(target_orientation, local_orientation)
        for velocity_key in ("lin_vel", "ang_vel"):
            if velocity_key in root:
                root[velocity_key] = _rotate_vector(
                    frame_rotation, torch.as_tensor(root[velocity_key])
                )

    result["pos"] = target_position
    result["ori"] = target_orientation
    return result


def _compact_policy_observation(raw_obs: dict) -> dict:
    """Keep only the observation fields consumed outside BehaviorProcess.

    Online grounding needs instance masks inside the simulator process, but the
    policy only consumes the three RGB views, proprioception, and the serialized
    grounded prompt. Compacting here prevents segmentation tensors from crossing
    the Ray actor boundary.
    """
    camera_fragments = {
        "zed_link:Camera:0": "main_images",
        "left_realsense_link:Camera:0": "left_wrist_image",
        "right_realsense_link:Camera:0": "right_wrist_image",
    }
    images = {}
    state = None
    for sensor_data in raw_obs.values():
        if not isinstance(sensor_data, dict):
            continue
        for sensor_name, modalities in sensor_data.items():
            if "proprio" in sensor_name:
                state = modalities
                continue
            if not isinstance(modalities, dict):
                continue
            for fragment, output_key in camera_fragments.items():
                if fragment in sensor_name:
                    images[output_key] = convert_uint8_rgb(modalities["rgb"])
                    break

    missing = [
        output_key
        for output_key in camera_fragments.values()
        if output_key not in images
    ]
    if missing:
        raise KeyError(f"Missing required BEHAVIOR camera observations: {missing}.")
    if state is None:
        raise KeyError("Missing required BEHAVIOR proprio observation.")

    return {
        "main_images": images["main_images"],
        "wrist_images": torch.stack(
            [images["left_wrist_image"], images["right_wrist_image"]], axis=0
        ),
        "state": state,
        "task_description": raw_obs.get("_subpool", {}).get("task_description"),
    }


def _translate_proprio_position_to_scene(
    raw_obs: dict,
    *,
    position_indices,
    scene_position: torch.Tensor,
) -> dict:
    """Remove a vector scene's translation from R1Pro proprioception."""
    translated = False
    for sensor_data in raw_obs.values():
        if not isinstance(sensor_data, dict):
            continue
        for sensor_name, modalities in sensor_data.items():
            if "proprio" not in sensor_name:
                continue
            if not isinstance(modalities, torch.Tensor):
                raise TypeError("BEHAVIOR proprioception must be a torch.Tensor.")
            state = modalities.clone()
            state[position_indices] -= scene_position.to(
                dtype=state.dtype,
                device=state.device,
            )
            sensor_data[sensor_name] = state
            translated = True
    if not translated:
        raise KeyError("Missing required BEHAVIOR proprio observation.")
    return raw_obs


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
        self.num_envs = int(num_envs)
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
        self.step_supports_evaluate_termination = (
            step_supports_kwargs or "evaluate_termination" in step_signature.parameters
        )
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )
        self.stop_chunk_on_done = bool(
            OmegaConf.select(cfg, "subpool.enabled", default=False)
        )
        self.dynamic_pool_updates = bool(
            OmegaConf.select(cfg, "subpool.dynamic_updates", default=True)
        )
        self.skip_official_task_termination = bool(
            OmegaConf.select(
                cfg,
                "subpool.skip_official_task_termination",
                default=False,
            )
        )
        if self.skip_official_task_termination:
            if not self.stop_chunk_on_done:
                raise ValueError(
                    "skip_official_task_termination requires subpool execution."
                )
            if not self.step_supports_evaluate_termination:
                raise ValueError(
                    "skip_official_task_termination requires an OmniGibson "
                    "VectorEnvironment with evaluate_termination support."
                )
        configured_policy_step = OmegaConf.select(
            cfg,
            "subpool.failure_state_capture.policy_global_step",
            default=None,
        )
        self.policy_global_step = (
            None if configured_policy_step is None else int(configured_policy_step)
        )
        if self.policy_global_step is not None and self.policy_global_step < 0:
            raise ValueError(
                "subpool.failure_state_capture.policy_global_step must be non-negative."
            )
        self.control_serializer = None
        capture_enabled = bool(
            OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.enabled",
                default=False,
            )
        )
        self.failure_state_store = None
        if capture_enabled:
            output_dir = OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.output_dir",
                default=None,
            )
            if not output_dir:
                raise ValueError(
                    "subpool.failure_state_capture.output_dir is required when "
                    "failure-state capture is enabled."
                )
            run_id = OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.run_id",
                default=None,
            )
            self.failure_state_store = FailureStateStore(
                output_dir,
                run_id=str(run_id) if run_id is not None else None,
            )
        self.failure_tipped_angle_deg = float(
            OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.tipped_angle_deg",
                default=45.0,
            )
        )
        self.failure_stable_steps = int(
            OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.stable_steps",
                default=8,
            )
        )
        self.failure_max_linear_speed = float(
            OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.max_linear_speed",
                default=0.05,
            )
        )
        self.failure_max_angular_speed = float(
            OmegaConf.select(
                cfg,
                "subpool.failure_state_capture.max_angular_speed",
                default=0.2,
            )
        )
        if self.failure_tipped_angle_deg <= 0:
            raise ValueError("failure_state_capture.tipped_angle_deg must be positive.")
        if self.failure_stable_steps <= 0:
            raise ValueError("failure_state_capture.stable_steps must be positive.")
        if self.failure_max_linear_speed < 0:
            raise ValueError(
                "failure_state_capture.max_linear_speed must be non-negative."
            )
        if self.failure_max_angular_speed < 0:
            raise ValueError(
                "failure_state_capture.max_angular_speed must be non-negative."
            )
        self.state_capture_interval = int(
            OmegaConf.select(cfg, "subpool.state_capture_interval", default=8)
        )
        self.recovery_min_lag_states = int(
            OmegaConf.select(cfg, "subpool.recovery_min_lag_states", default=2)
        )
        self.recovery_max_lag_states = int(
            OmegaConf.select(cfg, "subpool.recovery_max_lag_states", default=16)
        )
        self.state_ring_size = int(
            OmegaConf.select(cfg, "subpool.state_ring_size", default=32)
        )
        self.boundary_render_iterations = int(
            OmegaConf.select(
                cfg,
                "subpool.boundary_render_iterations",
                default=2,
            )
        )
        self.subpool_slots = [
            _SubpoolSlotRuntime(
                state_ring=deque(maxlen=self.state_ring_size),
            )
            for _ in range(self.num_envs)
        ]
        self.recovery_rngs = [
            np.random.default_rng(np.random.SeedSequence([int(cfg.seed), env_index]))
            for env_index in range(self.num_envs)
        ]
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
            if self.boundary_render_iterations <= 0:
                raise ValueError("subpool.boundary_render_iterations must be positive.")

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
                sensor.add_modality("seg_instance_id")
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

    def _attach_online_grounding(self, raw_obs: dict, env_index: int) -> dict:
        """Recompute the P2 object and part bboxes for one observation."""
        slot = self.subpool_slots[env_index]
        if slot.control is None or self.control_serializer is None:
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
            slot.control,
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

    def _prepare_policy_observation(
        self, raw_obs: dict, env_index: int | None = None
    ) -> dict:
        """Ground, then compact an observation before returning it through Ray."""
        if self.stop_chunk_on_done:
            if env_index is None:
                raise ValueError("Subpool observations require an environment index.")
            raw_obs = self._attach_online_grounding(raw_obs, env_index)
        if self.num_envs > 1:
            if env_index is None:
                raise ValueError("Vector observations require an environment index.")
            from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

            scene = self.env.envs[env_index].scene
            scene_position, scene_orientation = scene.get_position_orientation()
            identity = torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                dtype=scene_orientation.dtype,
                device=scene_orientation.device,
            )
            if not torch.allclose(scene_orientation, identity, atol=1e-6, rtol=0.0):
                raise NotImplementedError(
                    "Vector proprio canonicalization requires translated, "
                    "non-rotated B1K scenes."
                )
            raw_obs = _translate_proprio_position_to_scene(
                raw_obs,
                position_indices=PROPRIOCEPTION_INDICES["R1Pro"]["robot_pos"],
                scene_position=scene_position,
            )
        return _compact_policy_observation(raw_obs)

    def _observe_policy(self, env_indices: list[int]) -> list[dict]:
        """Render and observe current states without advancing task or physics time."""
        import omnigibson as og
        from omnigibson.objects.stateful_object import StatefulObject

        selected_scenes = {self.env.envs[index].scene for index in env_indices}
        for scene in selected_scenes:
            for obj in scene.objects:
                if isinstance(obj, StatefulObject) and obj.initialized:
                    obj.update_visuals()
        for _ in range(self.boundary_render_iterations):
            og.sim.render()

        observations = []
        for index in env_indices:
            raw_obs, _obs_info = self.env.envs[index].get_obs()
            observations.append(self._prepare_policy_observation(raw_obs, index))
        return observations

    def get_activity_name(self):
        return self.instance_loader.activity_name

    def _call_step(self, actions, env_indices=None, get_obs=True, render=True):
        """Call ``self.env.step`` forwarding only the kwargs it supports."""
        kwargs = {}
        if self.step_supports_get_obs:
            kwargs["get_obs"] = get_obs
        if self.step_supports_render:
            kwargs["render"] = render
        if self.skip_official_task_termination:
            kwargs["evaluate_termination"] = False
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
                [
                    self._prepare_policy_observation(obs, env_index)
                    for obs, env_index in zip(raw_obs, env_indices, strict=True)
                ]
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
        """Execute valid action prefixes and freeze completed logical slots.

        OmniGibson uses one shared PhysX stage for all scenes. An inactive scene
        still advances physically whenever another slot steps, so completed slots
        are never resumed. Their exact terminal observation, info, and state are
        cached until the next synchronized full-vector restore.
        """
        _, chunk_size, _ = actions.shape
        positions = {env_index: pos for pos, env_index in enumerate(env_indices)}
        active_indices = [
            env_index
            for env_index in env_indices
            if not self.subpool_slots[env_index].done
        ]
        last_obs = [self.subpool_slots[index].last_obs for index in env_indices]
        last_infos = [
            self.subpool_slots[index].last_info or {} for index in env_indices
        ]
        results = []

        for t in range(chunk_size):
            need_obs = not self.skip_intermediate_obs_in_chunk
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
                terminal_indices = []
                for source_index, env_index in enumerate(active_indices):
                    slot = self.subpool_slots[env_index]
                    pos = positions[env_index]
                    if need_obs:
                        obs_t[pos] = raw_obs[source_index]
                    info = infos[source_index]
                    if slot.reward_tracker is None:
                        raise RuntimeError(
                            "Subpool chunk execution started before reward priming."
                        )
                    stage_info = get_stage_info(info, slot.subtask_id)
                    self._apply_direct_navigation_predicate(stage_info, slot, env_index)
                    self._attach_arm_specific_distances(stage_info, slot, env_index)
                    outcome = slot.reward_tracker.step(stage_info)
                    self._maybe_capture_stable_recovery_event(
                        slot, env_index, stage_info, outcome
                    )
                    info["subpool"] = {
                        "subtask_id": slot.subtask_id,
                        "pool_type": slot.pool_type,
                        "success": outcome.success,
                        "timeout": outcome.timeout,
                        "elapsed_steps": slot.reward_tracker.steps,
                        "potential": outcome.potential,
                        "progress": outcome.progress,
                        "reward_progress_return": outcome.cumulative_progress,
                        "reward_step_penalty_return": (outcome.cumulative_step_penalty),
                        "reward_terminal_return": (outcome.cumulative_terminal_reward),
                    }
                    rewards_t[pos] = outcome.reward
                    terms_t[pos] = outcome.success
                    truncs_t[pos] = outcome.timeout
                    infos_t[pos] = info
                    executed_t[pos] = True
                    is_done = outcome.success or outcome.timeout
                    if is_done:
                        terminal_indices.append(env_index)
                        slot.done = True
                        needs_failure_state = (
                            outcome.timeout and self.failure_state_store is not None
                        )
                        terminal_state = (
                            self._dump_subpool_state(env_index)
                            if self.dynamic_pool_updates or needs_failure_state
                            else None
                        )
                        if needs_failure_state:
                            self._capture_failure_terminal_state(
                                slot,
                                terminal_state,
                                stage_info=stage_info,
                                outcome=outcome,
                            )
                        recovery_state = None
                        if self.dynamic_pool_updates and outcome.timeout:
                            available_max_lag = min(
                                self.recovery_max_lag_states,
                                len(slot.state_ring) - 1,
                            )
                            if available_max_lag >= self.recovery_min_lag_states:
                                lag = int(
                                    self.recovery_rngs[env_index].integers(
                                        self.recovery_min_lag_states,
                                        available_max_lag + 1,
                                    )
                                )
                                recovery_state = list(slot.state_ring)[-(lag + 1)]
                        if self.dynamic_pool_updates:
                            slot.pending_pool_candidates = {
                                "success_state": (
                                    terminal_state if outcome.success else None
                                ),
                                "recovery_state": recovery_state,
                            }
                    elif (
                        self.dynamic_pool_updates
                        and slot.reward_tracker.steps % self.state_capture_interval == 0
                    ):
                        slot.state_ring.append(self._dump_subpool_state(env_index))
                    if not is_done:
                        next_active.append(env_index)
                if self.skip_intermediate_obs_in_chunk and (
                    terminal_indices or t == chunk_size - 1
                ):
                    observation_indices = terminal_indices + next_active
                    observations = self._observe_policy(observation_indices)
                    for env_index, observation in zip(
                        observation_indices, observations, strict=True
                    ):
                        obs_t[positions[env_index]] = observation
                for env_index in terminal_indices:
                    slot = self.subpool_slots[env_index]
                    pos = positions[env_index]
                    if obs_t[pos] is None:
                        raise RuntimeError(
                            f"Terminal slot {env_index} has no cached observation."
                        )
                    slot.last_obs = obs_t[pos]
                    slot.last_info = infos_t[pos]
                active_indices = next_active

            last_obs = obs_t
            last_infos = infos_t
            output_obs = (
                obs_t
                if not self.skip_intermediate_obs_in_chunk or t == chunk_size - 1
                else None
            )
            results.append(
                (output_obs, rewards_t, terms_t, truncs_t, infos_t, executed_t)
            )

        return tuple(zip(*results))

    def _apply_direct_navigation_predicate(
        self,
        stage_info,
        slot: _SubpoolSlotRuntime,
        env_index: int,
    ) -> None:
        """Use the same demo-terminal base region as grounded evaluation."""
        if slot.control is None or slot.control.skill != "move to":
            return
        metadata = slot.snapshot_metadata or {}
        target_pose = metadata.get("target_base_pose")
        if target_pose is None:
            raise KeyError("Move-to snapshot metadata is missing target_base_pose.")

        import math

        import omnigibson.utils.transform_utils as transform_utils

        wrapped_env = self.env.envs[env_index]
        base_env = wrapped_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        position, quaternion = base_env.robots[0].get_position_orientation(
            frame="scene"
        )
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

    def _attach_arm_specific_distances(
        self,
        stage_info,
        slot: _SubpoolSlotRuntime,
        env_index: int,
    ) -> None:
        """Expose non-minimized arm distances for grounded manipulation rewards."""
        if (
            slot.task_reward is None
            or slot.subtask_id is None
            or slot.reward_tracker is None
        ):
            return
        required_metrics = {
            term.key for term in slot.reward_tracker.spec.potential_terms
        }
        needs_obj_distance = {
            key for key in required_metrics if key.endswith("_eef_to_obj_distance")
        }
        needs_toggle_distance = {
            key for key in required_metrics if key.endswith("_eef_to_toggle_distance")
        }
        needs_support_distance = (
            "object_to_support_surface_distance" in required_metrics
        )
        if not (needs_obj_distance or needs_toggle_distance or needs_support_distance):
            return
        stage_defs = getattr(slot.task_reward, "_stage_defs", ())
        if not 0 <= slot.subtask_id < len(stage_defs):
            raise IndexError(f"Active reward stage {slot.subtask_id} is unavailable.")
        objects = stage_defs[slot.subtask_id].get("objects", ())
        if not objects:
            return

        import torch as th
        from omnigibson.object_states.toggle import ToggledOn
        from omnigibson.reward_functions.support_utils import get_obj_center

        wrapped_env = self.env.envs[env_index]
        base_env = wrapped_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        robot = base_env.robots[0]
        target = objects[0]
        target_position = get_obj_center(target)
        if needs_support_distance and len(objects) >= 2 and objects[1] is not None:
            stage_info["object_to_support_surface_distance"] = (
                _support_surface_distance(target.aabb, objects[1].aabb)
            )

        if needs_toggle_distance:
            toggle_state = target.states.get(ToggledOn)
            marker = None if toggle_state is None else toggle_state.visual_marker
            if marker is None:
                toggle_position = target_position
                marker_radius = 0.0
            else:
                toggle_position = marker.get_position_orientation()[0]
                marker_radius = float(th.min(marker.extent * toggle_state.scale).item())

        for arm in robot.arm_names:
            obj_key = f"{arm}_eef_to_obj_distance"
            toggle_key = f"{arm}_eef_to_toggle_distance"
            if (
                obj_key not in needs_obj_distance
                and toggle_key not in needs_toggle_distance
            ):
                continue
            eef_position = robot.get_eef_position(arm)
            if obj_key in needs_obj_distance:
                stage_info[obj_key] = float(
                    th.linalg.vector_norm(eef_position - target_position).item()
                )
            if toggle_key in needs_toggle_distance:
                stage_info[toggle_key] = max(
                    float(th.linalg.vector_norm(eef_position - toggle_position).item())
                    - marker_radius,
                    0.0,
                )

    @staticmethod
    def _active_stage_objects(slot: _SubpoolSlotRuntime):
        """Return the simulator objects associated with the active stage."""
        if slot.task_reward is None or slot.subtask_id is None:
            raise RuntimeError("Subtask stage objects requested before reward priming.")
        stage_defs = getattr(slot.task_reward, "_stage_defs", ())
        if not 0 <= slot.subtask_id < len(stage_defs):
            raise IndexError(f"Active reward stage {slot.subtask_id} is unavailable.")
        return tuple(stage_defs[slot.subtask_id].get("objects", ()))

    def _prime_failure_analysis(self, slot: _SubpoolSlotRuntime) -> None:
        """Reset episode-local failure state and record the target reference pose."""
        slot.failure_reference_orientation = None
        slot.failure_tip_stable_count = 0
        slot.failure_recovery_event_captured = False
        if self.failure_state_store is None or slot.control is None:
            return
        if slot.control.skill.strip().lower() != "pick up from":
            return
        objects = self._active_stage_objects(slot)
        if len(objects) < 2 or objects[1] is None:
            raise RuntimeError(
                "Pickup failure analysis requires a target and original support."
            )
        recovery_provenance = (slot.snapshot_metadata or {}).get(
            "recovery_provenance", {}
        )
        reference_orientation = recovery_provenance.get("reference_orientation_xyzw")
        if reference_orientation is None:
            reference_orientation = (
                objects[0]
                .get_position_orientation(frame="scene")[1]
                .detach()
                .cpu()
                .tolist()
            )
        # Validate catalog-provided recovery provenance before it is used by the
        # per-step analyzer. This also returns a plain JSON-safe list.
        relative_tilt_angle_deg(reference_orientation, reference_orientation)
        slot.failure_reference_orientation = [
            float(value) for value in reference_orientation
        ]

    def _failure_facts(self, slot: _SubpoolSlotRuntime, stage_info) -> dict:
        """Extract JSON-safe simulator facts for the active skill."""
        facts = {"completed": bool(stage_info.get("completed", False))}
        if slot.control is None:
            raise RuntimeError("Failure analysis started before control priming.")
        if slot.control.skill.strip().lower() != "pick up from":
            return facts
        if slot.failure_reference_orientation is None:
            raise RuntimeError("Pickup failure analysis has no reference orientation.")

        objects = self._active_stage_objects(slot)
        if len(objects) < 2 or objects[1] is None:
            raise RuntimeError(
                "Pickup failure analysis requires a target and original support."
            )
        target, support = objects[:2]
        position, orientation = target.get_position_orientation(frame="scene")
        linear_speed = float(
            torch.linalg.vector_norm(target.get_linear_velocity()).item()
        )
        angular_speed = float(
            torch.linalg.vector_norm(target.get_angular_velocity()).item()
        )
        orientation_list = orientation.detach().cpu().tolist()
        tilt_angle = relative_tilt_angle_deg(
            slot.failure_reference_orientation,
            orientation_list,
        )
        return {
            "in_hand": bool(stage_info["in_hand"]),
            "on_original_support": bool(stage_info["on_support"]),
            "target_name": str(target.name),
            "support_name": str(support.name),
            "target_position": position.detach().cpu().tolist(),
            "target_orientation_xyzw": orientation_list,
            "reference_orientation_xyzw": list(slot.failure_reference_orientation),
            "tilt_angle_deg": tilt_angle,
            "tipped": tilt_angle >= self.failure_tipped_angle_deg,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "stable": (
                linear_speed <= self.failure_max_linear_speed
                and angular_speed <= self.failure_max_angular_speed
            ),
        }

    @staticmethod
    def _outcome_metadata(
        slot: _SubpoolSlotRuntime, outcome, *, failure_reason: str
    ) -> dict:
        """Build the common reward and termination metadata for a capture."""
        if slot.reward_tracker is None:
            raise RuntimeError("Failure capture started before reward priming.")
        return {
            "failure_reason": failure_reason,
            "success": bool(outcome.success),
            "timeout": bool(outcome.timeout),
            "elapsed_steps": int(slot.reward_tracker.steps),
            "potential": float(outcome.potential),
            "last_progress": float(outcome.progress),
            "progress_return": float(outcome.cumulative_progress),
            "step_penalty_return": float(outcome.cumulative_step_penalty),
            "terminal_return": float(outcome.cumulative_terminal_reward),
            "return": float(
                outcome.cumulative_progress
                + outcome.cumulative_step_penalty
                + outcome.cumulative_terminal_reward
            ),
        }

    def _capture_failure_state(
        self,
        slot: _SubpoolSlotRuntime,
        state,
        *,
        capture_kind: str,
        stage_info,
        outcome,
        failure_reason: str,
    ) -> None:
        """Persist one audited state with a skill-relative interpretation."""
        if self.failure_state_store is None:
            return
        if slot.snapshot_record is None or slot.control is None:
            raise RuntimeError("Failure capture started before a subpool reset.")
        facts = self._failure_facts(slot, stage_info)
        analysis = analyze_subtask_failure(
            slot.control.skill,
            facts,
            termination_reason="timeout" if outcome.timeout else None,
        )
        metadata_path = self.failure_state_store.capture(
            state,
            policy_global_step=self.policy_global_step,
            collection_index=slot.collection_index,
            sampling_group=slot.sampling_group,
            capture_kind=capture_kind,
            source_snapshot=slot.snapshot_record,
            outcome=self._outcome_metadata(
                slot, outcome, failure_reason=failure_reason
            ),
            analysis=analysis.to_dict(),
        )
        self.logger.info(
            "Saved %s failure state (%s, %s) to %s.",
            capture_kind,
            ",".join(analysis.failure_tags),
            analysis.recovery_status,
            metadata_path,
        )

    def _maybe_capture_stable_recovery_event(
        self,
        slot: _SubpoolSlotRuntime,
        env_index: int,
        stage_info,
        outcome,
    ) -> None:
        """Capture the first stable, simulator-certified recovery state."""
        if (
            self.failure_state_store is None
            or slot.failure_recovery_event_captured
            or outcome.success
            or outcome.timeout
            or slot.control is None
            or slot.control.skill.strip().lower() != "pick up from"
        ):
            return
        facts = self._failure_facts(slot, stage_info)
        analysis = analyze_subtask_failure(
            slot.control.skill,
            facts,
            termination_reason=None,
        )
        if analysis.recovery_status == "eligible" and bool(facts["stable"]):
            slot.failure_tip_stable_count += 1
        else:
            slot.failure_tip_stable_count = 0
        if slot.failure_tip_stable_count < self.failure_stable_steps:
            return
        self._capture_failure_state(
            slot,
            self._dump_subpool_state(env_index),
            capture_kind="stable_recovery_event",
            stage_info=stage_info,
            outcome=outcome,
            failure_reason="target_tipped",
        )
        slot.failure_recovery_event_captured = True

    def _dump_subpool_state(self, env_index: int):
        """Dump one scene in the canonical scene-zero coordinate frame."""
        wrapped_env = self.env.envs[env_index]
        base_env = wrapped_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        scene_state = base_env.scene.dump_state(serialized=False)
        return {
            0: _rebase_scene_state(
                scene_state,
                target_position=torch.zeros_like(scene_state["pos"]),
                target_orientation=torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=scene_state["ori"].dtype,
                    device=scene_state["ori"].device,
                ),
            )
        }

    @staticmethod
    def dump_serialized_state():
        """Return the simulator's flat state for deterministic diagnostics."""
        import omnigibson as og

        return og.sim.dump_state(serialized=True)

    def dump_subpool_states(self):
        """Return every vector slot in the canonical scene-zero frame."""
        return [self._dump_subpool_state(index) for index in range(self.num_envs)]

    def _capture_failure_terminal_state(
        self,
        slot: _SubpoolSlotRuntime,
        state,
        *,
        stage_info,
        outcome,
    ) -> None:
        """Persist the exact timeout state without adding it to a recovery pool."""
        self._capture_failure_state(
            slot,
            state,
            capture_kind="terminal",
            stage_info=stage_info,
            outcome=outcome,
            failure_reason="timeout",
        )

    def set_policy_global_step(self, global_step: int) -> None:
        """Associate later terminal-state artifacts with a policy version."""
        global_step = int(global_step)
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        self.policy_global_step = global_step

    def drain_pool_candidates(self):
        """Return terminal/recovery candidates once for every vector slot."""
        candidates = []
        for slot in self.subpool_slots:
            candidates.append(slot.pending_pool_candidates)
            slot.pending_pool_candidates = None
        return candidates

    def load_serialized_states(
        self,
        states,
        *,
        activity_names,
        scene_models,
        instance_ids,
        subtask_ids,
        pool_types,
        reward_specs,
        control_jsons,
        snapshot_metadatas,
        snapshot_records,
        sampling_groups,
        collection_indices,
    ):
        """Synchronously restore one canonical state into every vector scene."""
        fields = {
            "states": states,
            "activity_names": activity_names,
            "scene_models": scene_models,
            "instance_ids": instance_ids,
            "subtask_ids": subtask_ids,
            "pool_types": pool_types,
            "reward_specs": reward_specs,
            "control_jsons": control_jsons,
            "snapshot_metadatas": snapshot_metadatas,
            "snapshot_records": snapshot_records,
            "sampling_groups": sampling_groups,
            "collection_indices": collection_indices,
        }
        wrong_lengths = {
            name: len(value)
            for name, value in fields.items()
            if len(value) != self.num_envs
        }
        if wrong_lengths:
            raise ValueError(
                f"Full-vector restore expects {self.num_envs} values per field; "
                f"got {wrong_lengths}."
            )
        if any(
            activity_name != self.instance_loader.activity_name
            for activity_name in activity_names
        ):
            raise ValueError(
                "Every snapshot activity must match runtime activity "
                f"{self.instance_loader.activity_name!r}."
            )
        import omnigibson as og

        from rlinf.data.b1k_grounded import GroundedControlSpec

        # A partial reset is physically unsafe in OmniGibson's shared stage.
        # Reset all scenes first, then load every state before one common refresh.
        self.instance_loader.prepare_reset(self.env)
        self._call_reset(get_obs=False)
        base_envs = []
        for env_index, wrapped_env in enumerate(self.env.envs):
            base_env = wrapped_env
            while hasattr(base_env, "env"):
                base_env = base_env.env
            base_envs.append(base_env)
            runtime_scene_model = str(getattr(base_env.scene, "scene_model", ""))
            if runtime_scene_model != scene_models[env_index]:
                raise ValueError(
                    f"Snapshot scene {scene_models[env_index]!r} does not match "
                    f"runtime scene {runtime_scene_model!r} for slot {env_index}."
                )
            state = states[env_index]
            if set(state) != {0}:
                raise ValueError(
                    "Canonical subpool states must contain exactly scene key 0, "
                    f"got {sorted(state)} for slot {env_index}."
                )
            target_position, target_orientation = (
                base_env.scene.get_position_orientation()
            )
            scene_state = _rebase_scene_state(
                _move_state_tensors(state[0], og.sim.device),
                target_position=target_position,
                target_orientation=target_orientation,
            )
            base_env.scene.load_state(scene_state, serialized=False)
            # Simulator state excludes controller goals. Synchronize before the
            # common refresh so reset targets cannot move a restored robot.
            sync_robot_after_pose_override(base_env.robots[0])
            base_env.task.activity_instance_id = int(instance_ids[env_index])

            reward_functions = getattr(base_env.task, "_reward_functions", {})
            task_reward = reward_functions.get("task_specific")
            if task_reward is None or not hasattr(
                task_reward, "set_active_stage_index"
            ):
                raise TypeError(
                    "Subpool RL requires a sequential task_specific reward with "
                    "set_active_stage_index()."
                )
            subtask_id = int(subtask_ids[env_index])
            task_reward.set_active_stage_index(subtask_id)
            pool_type = pool_types[env_index]
            if pool_type not in SUBPOOL_TYPES:
                raise ValueError(f"Unknown subpool type {pool_type!r}.")
            self.subpool_slots[env_index] = _SubpoolSlotRuntime(
                reward_tracker=SubtaskRewardTracker(
                    SubtaskRewardSpec.from_mapping(reward_specs[env_index])
                ),
                task_reward=task_reward,
                subtask_id=subtask_id,
                pool_type=pool_type,
                control=GroundedControlSpec.from_json(control_jsons[env_index]),
                snapshot_metadata=dict(snapshot_metadatas[env_index]),
                snapshot_record=dict(snapshot_records[env_index]),
                sampling_group=int(sampling_groups[env_index]),
                collection_index=(
                    None
                    if collection_indices[env_index] is None
                    else int(collection_indices[env_index])
                ),
                state_ring=deque(maxlen=self.state_ring_size),
            )

        # Object-state predicates and camera buffers are stale immediately after
        # scene.load_state. Refresh all scenes together only after every scene is
        # restored, then propagate transforms through Fabric for vision sensors.
        og.sim.step()
        og.sim.render()
        observations = []
        infos = []
        for env_index, (wrapped_env, slot) in enumerate(
            zip(self.env.envs, self.subpool_slots, strict=True)
        ):
            self._prime_failure_analysis(slot)
            if self.dynamic_pool_updates:
                slot.state_ring.append(self._dump_subpool_state(env_index))
            obs, info = wrapped_env.get_obs()
            observations.append(self._prepare_policy_observation(obs, env_index))
            infos.append(info)
        return observations, infos

    def reset(self, reset_indices=None, get_obs=True):
        self.instance_loader.prepare_reset(self.env)
        result = self._call_reset(
            reset_indices=reset_indices,
            get_obs=get_obs,
        )
        if not get_obs:
            return None, None

        raw_obs, infos = result
        return [
            self._prepare_policy_observation(obs, env_index)
            for env_index, obs in enumerate(raw_obs)
        ], list(infos)

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
                node_name = os.environ.get("RLINF_NODE_RANK", socket.gethostname())
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
                # A collector daemon can attach to a Ray cluster that was
                # started before its BEHAVIOR paths were configured. Ray
                # workers inherit the raylet's environment, not the driver's,
                # unless runtime_env explicitly carries these values.
                child_env_vars.update(
                    {
                        name: os.environ[name]
                        for name in _BEHAVIOR_CHILD_ENV_VARS
                        if name in os.environ
                    }
                )
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible_devices:
                    child_env_vars["CUDA_VISIBLE_DEVICES"] = visible_devices
                appdata_base = child_env_vars.get("OMNIGIBSON_APPDATA_PATH")
                self.env_processes = []
                for process_index in range(self.num_env_subprocess):
                    process_env_vars = child_env_vars.copy()
                    if appdata_base:
                        process_env_vars["OMNIGIBSON_APPDATA_PATH"] = (
                            _isolated_appdata_path(
                                appdata_base,
                                node_name=node_name,
                                visible_devices=visible_devices,
                                process_index=process_index,
                            )
                        )
                    process = BehaviorProcess.options(
                        scheduling_strategy=scheduling_strategy,
                        runtime_env={"env_vars": process_env_vars},
                    ).remote(self.cfg, self.num_env_shard, pipeline_stage_num)
                    self.env_processes.append(process)

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

    def set_policy_global_step(
        self,
        global_start: int,
        num_envs: int,
        global_step: int,
    ) -> None:
        """Set the policy version on subprocesses serving one environment slice."""
        plan = self._slice_plan(global_start, num_envs)
        ray.get(
            [
                self.env_processes[sp].set_policy_global_step.remote(global_step)
                for sp, _positions, _local_rows in plan
            ]
        )

    def load_serialized_states(
        self,
        global_start: int,
        num_envs: int,
        states,
        *,
        activity_names,
        scene_models,
        instance_ids,
        subtask_ids,
        pool_types,
        reward_specs,
        control_jsons,
        snapshot_metadatas,
        snapshot_records,
        sampling_groups,
        collection_indices,
    ):
        """Restore a complete synchronized vector through one OG process."""
        if self.num_env_subprocess != 1:
            raise RuntimeError(
                "Vectorized subpool restore requires num_env_subprocess=1."
            )
        if global_start != 0 or num_envs != self.total_num_envs:
            raise RuntimeError(
                "Vectorized subpool restore requires the complete process slice."
            )
        return ray.get(
            self.env_processes[0].load_serialized_states.remote(
                states,
                activity_names=activity_names,
                scene_models=scene_models,
                instance_ids=instance_ids,
                subtask_ids=subtask_ids,
                pool_types=pool_types,
                reward_specs=reward_specs,
                control_jsons=control_jsons,
                snapshot_metadatas=snapshot_metadatas,
                snapshot_records=snapshot_records,
                sampling_groups=sampling_groups,
                collection_indices=collection_indices,
            )
        )

    def drain_pool_candidates(self):
        """Drain online state candidates from every vector slot."""
        if self.num_env_subprocess != 1:
            raise RuntimeError(
                "Vectorized subpool candidates require num_env_subprocess=1."
            )
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
        if {"main_images", "wrist_images", "state"}.issubset(raw_obs):
            return raw_obs
        return _compact_policy_observation(raw_obs)

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

        synchronized_vector_subpool = preserve_primitive_dones and self.num_envs > 1
        if past_dones.any() and self.auto_reset and not synchronized_vector_subpool:
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
            episode_length = subpool_info.get(
                "elapsed_steps", info.get("episode_length", 0)
            )
            episode_info = {
                "success": success,
                "episode_length": episode_length,
            }
            if subpool_info:
                pool_type = subpool_info.get("pool_type")
                episode_info.update(
                    {
                        "subtask_id": int(subpool_info["subtask_id"]),
                        "subpool_id": SUBPOOL_TYPES.index(pool_type),
                        "subtask_timeout": bool(subpool_info.get("timeout", False)),
                        "reward_progress_return": float(
                            subpool_info.get("reward_progress_return", 0.0)
                        ),
                        "reward_step_penalty_return": float(
                            subpool_info.get("reward_step_penalty_return", 0.0)
                        ),
                        "reward_terminal_return": float(
                            subpool_info.get("reward_terminal_return", 0.0)
                        ),
                    }
                )
            self.returns[env_idx] += reward
            self.success_once[env_idx] = self.success_once[env_idx] | success
            episode_info["success_once"] = self.success_once[env_idx].clone()

            episode_info["return"] = self.returns[env_idx].clone()
            if subpool_info:
                episode_info["episode_len"] = torch.as_tensor(
                    episode_length, device=episode_info["return"].device
                )
            else:
                episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / torch.clamp(
                episode_info["episode_len"], min=1
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
    """BEHAVIOR adapter with synchronized, audited vector-state resets."""

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
        self._state_cache_size = int(
            OmegaConf.select(cfg, "subpool.state_cache_size", default=0)
        )
        self.catalog = SubpoolCatalog.from_jsonl(
            manifest_path,
            state_cache_size=self._state_cache_size,
        )
        reward_overrides = OmegaConf.select(cfg, "subpool.reward_overrides", default={})
        self._reward_overrides = dict(
            (
                OmegaConf.to_container(reward_overrides, resolve=True)
                if OmegaConf.is_config(reward_overrides)
                else reward_overrides
            )
            or {}
        )
        validate_subpool_rollout_horizons(
            [
                SubtaskRewardSpec.from_mapping(
                    apply_reward_overrides(
                        record.metadata["reward"], self._reward_overrides
                    )
                ).max_steps
                for record in self.catalog.records
            ],
            episode_horizon=int(cfg.max_episode_steps),
            rollout_horizon=int(cfg.max_steps_per_rollout_epoch),
        )
        self._runtime_signature = self.catalog.runtime_signature
        runtime_activity, runtime_scene = self._runtime_signature
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
            env_world_size=total_num_processes * num_envs,
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
        self._outcome_group_size = int(
            OmegaConf.select(cfg, "subpool.outcome_group_size", default=1)
        )
        if self._outcome_group_size <= 0:
            raise ValueError("subpool.outcome_group_size must be positive.")
        if self._outcome_group_size % num_envs != 0:
            raise ValueError(
                "subpool.outcome_group_size must be divisible by the number of "
                "vector slots per EnvWorker so a worker never straddles two "
                "outcome groups."
            )
        logical_env_world_size = total_num_processes * num_envs
        if logical_env_world_size % self._outcome_group_size != 0:
            raise ValueError(
                "The number of logical BEHAVIOR environments must be divisible by "
                "subpool.outcome_group_size."
            )
        self._sampling_seed = int(cfg.seed)
        first_logical_env = int(seed_offset) * num_envs
        logical_env_ids = range(first_logical_env, first_logical_env + num_envs)
        self._sampling_groups = [
            logical_env_id // self._outcome_group_size
            for logical_env_id in logical_env_ids
        ]
        self._rngs = [
            np.random.default_rng(
                np.random.SeedSequence([self._sampling_seed, logical_env_id])
            )
            for logical_env_id in logical_env_ids
        ]
        self._pool_weights = OmegaConf.to_container(
            OmegaConf.select(cfg, "subpool.pool_weights", default={}), resolve=True
        )
        self._outcome_snapshot_schedule = str(
            OmegaConf.select(
                cfg,
                "subpool.outcome_snapshot_schedule",
                default="random",
            )
        )
        if self._outcome_snapshot_schedule not in {
            "random",
            "shuffled_round_robin",
        }:
            raise ValueError(
                "subpool.outcome_snapshot_schedule must be 'random' or "
                "'shuffled_round_robin'."
            )
        self._sticky_outcome_snapshot = bool(
            OmegaConf.select(
                cfg,
                "subpool.sticky_outcome_snapshot",
                default=False,
            )
        )
        self._fixed_snapshot_per_env = bool(
            OmegaConf.select(
                cfg,
                "subpool.fixed_snapshot_per_env",
                default=False,
            )
        )
        self._subtask_cursors = list(self._sampling_groups)
        self._pending_outcome_collection_index: int | None = None
        self._pending_outcome_logical_group_indices: list[int | None] | None = None
        self._pending_outcome_update_index: int | None = None
        self._current_outcome_logical_group_indices: list[int | None] = [
            None
        ] * num_envs
        self._current_outcome_update_index: int | None = None
        self._active_outcome_snapshots: list[SubpoolSnapshot | None] = [None] * num_envs
        self.current_snapshots: list[SubpoolSnapshot | None] = [None] * num_envs
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
        if any(snapshot is None for snapshot in self.current_snapshots):
            raise RuntimeError("Subpool env has not been reset.")
        return torch.tensor(
            [snapshot.subtask_id for snapshot in self.current_snapshots],
            dtype=torch.long,
        )

    @property
    def subpool_ids(self) -> torch.Tensor:
        if any(snapshot is None for snapshot in self.current_snapshots):
            raise RuntimeError("Subpool env has not been reset.")
        return torch.tensor(
            [
                SUBPOOL_TYPES.index(snapshot.pool_type)
                for snapshot in self.current_snapshots
            ],
            dtype=torch.long,
        )

    def set_policy_global_step(self, global_step: int) -> None:
        """Propagate the active policy version to the simulator process."""
        self._ensure_pool()
        self.pool.set_policy_global_step(
            self.pool_offset,
            self.num_envs,
            global_step,
        )

    def prepare_outcome_group_reset(
        self,
        collection_index: int,
        logical_group_index: int | list[int | None] | None = None,
        update_index: int | None = None,
    ) -> None:
        """Make the next reset deterministic within an outcome-sampling group.

        Auto-reset advances each environment's ordinary RNG independently because
        successful and failed trajectories finish at different times.  DAPO quota
        comparisons require a stronger invariant: every member of an outcome group
        must begin each candidate rollout from exactly the same snapshot.  The
        The runner supplies both the physical collection attempt and the logical
        group identity. A shuffled-round-robin schedule keys the snapshot only by
        update and logical group, so DAPO retries cannot change the initial state.
        """
        collection_index = int(collection_index)
        if collection_index < 0:
            raise ValueError("collection_index must be non-negative.")
        if self._pending_outcome_collection_index is not None:
            raise RuntimeError("An outcome-group reset is already pending.")
        self._pending_outcome_collection_index = collection_index
        if isinstance(logical_group_index, (list, tuple)):
            if len(logical_group_index) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} logical group assignments, got "
                    f"{len(logical_group_index)}."
                )
            logical_group_indices = [
                None if value is None else int(value) for value in logical_group_index
            ]
        else:
            value = None if logical_group_index is None else int(logical_group_index)
            logical_group_indices = [value] * self.num_envs
        self._pending_outcome_logical_group_indices = logical_group_indices
        self._pending_outcome_update_index = (
            None if update_index is None else int(update_index)
        )
        if any(value is not None and value < 0 for value in logical_group_indices):
            raise ValueError("logical_group_index values must be non-negative.")
        if (
            self._pending_outcome_update_index is not None
            and self._pending_outcome_update_index < 0
        ):
            raise ValueError("update_index must be non-negative.")

    @property
    def outcome_group_reset_metadata(self) -> list[dict[str, int | str]]:
        """Describe each snapshot loaded by the most recent vector reset."""
        if any(snapshot is None for snapshot in self.current_snapshots):
            raise RuntimeError("Subpool env has not been reset.")
        return [
            {
                "sampling_group": sampling_group,
                "snapshot_id": snapshot.snapshot_id,
                "episode_index": int(snapshot.episode_index),
                "subtask_id": int(snapshot.subtask_id),
                "pool_type": snapshot.pool_type,
                "logical_group_index": logical_group_index,
                "update_index": self._current_outcome_update_index,
            }
            for sampling_group, snapshot, logical_group_index in zip(
                self._sampling_groups,
                self.current_snapshots,
                self._current_outcome_logical_group_indices,
                strict=True,
            )
        ]

    def _sample_reset_snapshot(self, env_index: int) -> SubpoolSnapshot:
        collection_index = self._pending_outcome_collection_index
        sampling_group = self._sampling_groups[env_index]
        if collection_index is None:
            if (
                self._sticky_outcome_snapshot
                and self._active_outcome_snapshots[env_index] is not None
            ):
                return self._active_outcome_snapshots[env_index]
            if self._fixed_snapshot_per_env:
                sampled_subtask_id = self._fixed_subtask_id
                if sampled_subtask_id is None:
                    subtask_ids = self.catalog.subtask_ids
                    sampled_subtask_id = subtask_ids[sampling_group % len(subtask_ids)]
                return self.catalog.shuffled_round_robin_snapshot(
                    seed=self._sampling_seed,
                    update_index=0,
                    logical_group_index=sampling_group,
                    subtask_id=sampled_subtask_id,
                    pool_weights=self._pool_weights,
                )
            rng = self._rngs[env_index]
            sampled_subtask_id = self._fixed_subtask_id
            if sampled_subtask_id is None:
                subtask_ids = self.catalog.subtask_ids
                sampled_subtask_id = subtask_ids[
                    self._subtask_cursors[env_index] % len(subtask_ids)
                ]
                self._subtask_cursors[env_index] += 1
        else:
            if self._pending_outcome_logical_group_indices is None:
                raise RuntimeError("Outcome logical group assignments are missing.")
            logical_group_index = self._pending_outcome_logical_group_indices[env_index]
            update_index = self._pending_outcome_update_index
            if self._outcome_snapshot_schedule == "shuffled_round_robin":
                if logical_group_index is None or update_index is None:
                    raise RuntimeError(
                        "shuffled_round_robin outcome resets require logical_group_index "
                        "and update_index."
                    )
                sampled_subtask_id = self._fixed_subtask_id
                if sampled_subtask_id is None:
                    subtask_ids = self.catalog.subtask_ids
                    sampled_subtask_id = subtask_ids[
                        logical_group_index % len(subtask_ids)
                    ]
                return self.catalog.shuffled_round_robin_snapshot(
                    seed=self._sampling_seed,
                    update_index=update_index,
                    logical_group_index=logical_group_index,
                    subtask_id=sampled_subtask_id,
                    pool_weights=self._pool_weights,
                )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self._sampling_seed,
                        sampling_group,
                        collection_index,
                        env_index,
                    ]
                )
            )
            sampled_subtask_id = self._fixed_subtask_id
            if sampled_subtask_id is None:
                subtask_ids = self.catalog.subtask_ids
                sampled_subtask_id = subtask_ids[
                    (sampling_group + collection_index) % len(subtask_ids)
                ]

        return self.catalog.sample(
            rng,
            subtask_id=sampled_subtask_id,
            pool_weights=self._pool_weights,
        )

    def env_reset(self):
        self._ensure_pool()
        if self._dynamic_updates:
            refreshed_catalog = SubpoolCatalog.from_jsonl(
                self._manifest_path,
                state_cache_size=self._state_cache_size,
            )
            if refreshed_catalog.runtime_signature != self._runtime_signature:
                raise ValueError(
                    "Dynamic subpool update changed the simulator runtime signature."
                )
            self.catalog = refreshed_catalog
        collection_index = self._pending_outcome_collection_index
        snapshots = [
            self._sample_reset_snapshot(env_index) for env_index in range(self.num_envs)
        ]
        expected_fingerprint = OmegaConf.select(
            self.cfg, "subpool.asset_fingerprint", default=None
        )
        for env_index, snapshot in enumerate(snapshots):
            self.logger.info(
                "Sampled vector slot=%d snapshot=%s episode=%s subtask=%d pool=%s.",
                env_index,
                snapshot.snapshot_id,
                snapshot.episode_index,
                snapshot.subtask_id,
                snapshot.pool_type,
            )
            if (
                expected_fingerprint
                and snapshot.asset_fingerprint != expected_fingerprint
            ):
                raise ValueError(
                    f"Snapshot asset_fingerprint={snapshot.asset_fingerprint!r} "
                    f"does not match configured value {expected_fingerprint!r}."
                )
        raw_obs, infos = self.pool.load_serialized_states(
            self.pool_offset,
            self.num_envs,
            [self.catalog.load_state(snapshot) for snapshot in snapshots],
            activity_names=[snapshot.activity_name for snapshot in snapshots],
            scene_models=[snapshot.scene_model for snapshot in snapshots],
            instance_ids=[
                int(snapshot.metadata["instance_id"]) for snapshot in snapshots
            ],
            subtask_ids=[snapshot.subtask_id for snapshot in snapshots],
            pool_types=[snapshot.pool_type for snapshot in snapshots],
            reward_specs=[
                apply_reward_overrides(
                    snapshot.metadata["reward"], self._reward_overrides
                )
                for snapshot in snapshots
            ],
            control_jsons=[snapshot.control_json for snapshot in snapshots],
            snapshot_metadatas=[snapshot.metadata for snapshot in snapshots],
            snapshot_records=[snapshot.to_dict() for snapshot in snapshots],
            sampling_groups=self._sampling_groups,
            collection_indices=[collection_index] * self.num_envs,
        )
        self.current_snapshots = snapshots
        if collection_index is not None and self._sticky_outcome_snapshot:
            self._active_outcome_snapshots = list(snapshots)
        self._current_outcome_logical_group_indices = list(
            self._pending_outcome_logical_group_indices or [None] * self.num_envs
        )
        self._current_outcome_update_index = self._pending_outcome_update_index
        self._pending_outcome_collection_index = None
        self._pending_outcome_logical_group_indices = None
        self._pending_outcome_update_index = None
        return raw_obs, infos

    def chunk_step(self, chunk_actions):
        result = super().chunk_step(chunk_actions)
        _, _, terminations, truncations, _ = result
        if self._dynamic_updates and (terminations.any() or truncations.any()):
            candidates = self.pool.drain_pool_candidates()
            for env_index, (slot_candidates, snapshot) in enumerate(
                zip(candidates, self.current_snapshots, strict=True)
            ):
                if snapshot is None:
                    raise RuntimeError("Subpool env has not been reset.")
                slot_terminated = bool(terminations[env_index].any())
                slot_truncated = bool(truncations[env_index].any())
                if slot_terminated or slot_truncated:
                    self._append_online_candidates(
                        slot_candidates,
                        snapshot=snapshot,
                        success=slot_terminated,
                    )
        return result

    def _append_online_candidates(
        self,
        candidates,
        *,
        snapshot: SubpoolSnapshot,
        success: bool,
    ) -> None:
        if not candidates:
            return
        if success:
            state = candidates.get("success_state")
            later_subtasks = [
                subtask_id
                for subtask_id in self.catalog.subtask_ids
                if subtask_id > snapshot.subtask_id
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
                    snapshot.snapshot_id,
                )
                return
            target_record = snapshot
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
            "source_snapshot_id": snapshot.snapshot_id,
            "source_subtask_id": snapshot.subtask_id,
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
        if not all(obs["task_descriptions"]):
            if any(snapshot is None for snapshot in self.current_snapshots):
                raise RuntimeError("Subpool env has not been reset.")
            obs["task_descriptions"] = [
                snapshot.task_description for snapshot in self.current_snapshots
            ]
        return obs

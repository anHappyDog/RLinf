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

import inspect

import torch

# Order follows omnigibson.learning.utils.eval_utils.PROPRIOCEPTION_INDICES["R1Pro"].
R1PRO_PROPRIO_KEYS = [
    "joint_qpos",
    "joint_qpos_sin",
    "joint_qpos_cos",
    "joint_qvel",
    "joint_qeffort",
    "robot_pos",
    "robot_ori_cos",
    "robot_ori_sin",
    "robot_2d_ori",
    "robot_2d_ori_cos",
    "robot_2d_ori_sin",
    "robot_lin_vel",
    "robot_ang_vel",
    "arm_left_qpos",
    "arm_left_qpos_sin",
    "arm_left_qpos_cos",
    "arm_left_qvel",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "gripper_left_qvel",
    "arm_right_qpos",
    "arm_right_qpos_sin",
    "arm_right_qpos_cos",
    "arm_right_qvel",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "gripper_right_qvel",
    "trunk_qpos",
    "trunk_qvel",
    "base_qpos",
    "base_qpos_sin",
    "base_qpos_cos",
    "base_qvel",
]

SUPPORTED_ENV_WRAPPERS = ("default", "rgb_lowres", "rich_obs", "comet_rgb")


def set_camera_resolution(camera_cfg: dict | None) -> None:
    """Apply BEHAVIOR camera resolution overrides to OmniGibson eval constants."""
    if camera_cfg is None:
        return

    import omnigibson.learning.utils.eval_utils as eval_utils

    head_resolution = camera_cfg.get("head_resolution")
    wrist_resolution = camera_cfg.get("wrist_resolution")
    if head_resolution is not None:
        eval_utils.HEAD_RESOLUTION = tuple(head_resolution)
    if wrist_resolution is not None:
        eval_utils.WRIST_RESOLUTION = tuple(wrist_resolution)


def get_env_wrapper(wrapper_name: str):
    if wrapper_name == "default":
        from omnigibson.learning.wrappers.default_wrapper import DefaultWrapper

        return DefaultWrapper
    if wrapper_name == "rgb_lowres":
        from omnigibson.learning.wrappers.rgb_low_res_wrapper import RGBLowResWrapper

        return RGBLowResWrapper
    if wrapper_name == "rich_obs":
        from omnigibson.learning.wrappers.rich_obs_wrapper import RichObservationWrapper

        return RichObservationWrapper
    if wrapper_name == "comet_rgb":
        from omnigibson.envs import EnvironmentWrapper

        class CometRGBWrapper(EnvironmentWrapper):
            """Match openpi-comet RGB wrapper behavior for BEHAVIOR eval."""

            def __init__(self, env):
                super().__init__(env=env)
                import omnigibson.learning.utils.eval_utils as eval_utils

                robot = env.robots[0]
                for camera_id, camera_name in eval_utils.ROBOT_CAMERA_NAMES[
                    "R1Pro"
                ].items():
                    sensor_name = camera_name.split("::")[1]
                    if camera_id == "head":
                        robot.sensors[sensor_name].horizontal_aperture = 40.0
                        robot.sensors[
                            sensor_name
                        ].image_height = eval_utils.HEAD_RESOLUTION[0]
                        robot.sensors[
                            sensor_name
                        ].image_width = eval_utils.HEAD_RESOLUTION[1]
                    else:
                        robot.sensors[
                            sensor_name
                        ].image_height = eval_utils.WRIST_RESOLUTION[0]
                        robot.sensors[
                            sensor_name
                        ].image_width = eval_utils.WRIST_RESOLUTION[1]
                env.load_observation_space()

        return CometRGBWrapper
    raise ValueError(
        f"Unsupported wrapper name: {wrapper_name}, expected one of {SUPPORTED_ENV_WRAPPERS}"
    )


def build_eval_utils_omni_cfg(task_idx: int) -> dict:
    """Build task-specific BEHAVIOR env config consistent with omnigibson.learning.eval."""
    from gello.robots.sim_robot.og_teleop_utils import (
        generate_robot_config,
        load_available_tasks,
    )
    from omnigibson.learning.utils.eval_utils import (
        PROPRIOCEPTION_INDICES,
        TASK_INDICES_TO_NAMES,
        generate_basic_environment_config,
    )

    task_name = TASK_INDICES_TO_NAMES[task_idx]
    task_cfg = load_available_tasks()[task_name][0]

    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
    cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
    cfg["task"]["include_obs"] = False
    return cfg


def ensure_uint8_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor to uint8 while preserving float [0,1] / [0,255] semantics."""
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    if image.dtype == torch.uint8:
        return image[..., :3]

    if torch.is_floating_point(image):
        max_val = float(image.detach().max().item()) if image.numel() > 0 else 1.0
        if max_val <= 1.0 + 1e-6:
            image = image * 255.0
        image = image.round().clamp(0, 255).to(torch.uint8)
    else:
        image = image.clamp(0, 255).to(torch.uint8)

    return image[..., :3]


def find_sensor_by_suffix(sensor_names: list[str], suffix: str) -> str:
    if suffix in sensor_names:
        return suffix
    candidates = [name for name in sensor_names if name.endswith(suffix)]
    if not candidates:
        candidates = [name for name in sensor_names if suffix in name]
    if not candidates:
        raise KeyError(
            f"Cannot find sensor suffix '{suffix}' from available sensors: {sensor_names}"
        )
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=len)


def sync_r1pro_camera_names_with_env(env) -> None:
    """Align eval camera-name mapping with actual sensor names from this env."""
    import omnigibson.learning.utils.eval_utils as eval_utils

    robot = env.robots[0]
    sensor_names = list(robot.sensors.keys())
    camera_suffix = {
        "left_wrist": "left_realsense_link:Camera:0",
        "right_wrist": "right_realsense_link:Camera:0",
        "head": "zed_link:Camera:0",
    }
    resolved = {
        camera_id: find_sensor_by_suffix(sensor_names, suffix)
        for camera_id, suffix in camera_suffix.items()
    }
    eval_utils.ROBOT_CAMERA_NAMES["R1Pro"] = {
        camera_id: f"{robot.name}::{sensor_name}"
        for camera_id, sensor_name in resolved.items()
    }


def patch_omnigibson_wrapper_reset_signature() -> None:
    """Patch old OmniGibson wrapper reset to tolerate kwargs like get_obs."""
    from omnigibson.envs.env_wrapper import EnvironmentWrapper

    reset_fn = EnvironmentWrapper.reset
    if getattr(reset_fn, "__rlinf_patched__", False):
        return

    sig = inspect.signature(reset_fn)
    supports_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if supports_kwargs:
        return

    def _reset_with_kwargs(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    _reset_with_kwargs.__rlinf_patched__ = True
    EnvironmentWrapper.reset = _reset_with_kwargs


def apply_env_wrapper(vec_env, wrapper_name: str | None):
    if wrapper_name is None:
        return vec_env
    patch_omnigibson_wrapper_reset_signature()
    wrapper_cls = get_env_wrapper(wrapper_name)
    for i in range(vec_env.num_envs):
        sync_r1pro_camera_names_with_env(vec_env.envs[i])
        vec_env.envs[i] = wrapper_cls(vec_env.envs[i])
    return vec_env

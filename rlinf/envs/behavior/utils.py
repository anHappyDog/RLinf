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
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

SUPPORTED_ENV_WRAPPERS = ("default", "rgb_lowres", "rich_obs")

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

TASK_INSTANCE_FILE_SUFFIX = "_template-tro_state.json"
TASK_INSTANCE_TEMPLATE_FILE_SUFFIX = "_template.json"
SUPPORTED_INSTANCE_RESAMPLE_MODES = ("disabled", "offline", "online")


@dataclass(frozen=True)
class BehaviorActivityInstanceFile:
    instance_id: int
    path: str
    file_format: str


@dataclass(frozen=True)
class BehaviorInstanceResampleConfig:
    activity_name: str
    activity_instance_id: int
    activity_instance_dir: str | None
    instance_resample_mode: str
    cached_activity_instances: tuple[BehaviorActivityInstanceFile, ...] = ()


def set_camera_resolution(camera_cfg: dict | None) -> None:
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
    raise ValueError(
        f"Unsupported wrapper name: {wrapper_name}, expected one of {SUPPORTED_ENV_WRAPPERS}"
    )


def convert_uint8_rgb(image: torch.Tensor) -> torch.Tensor:
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


def patch_omnigibson_wrapper_reset_signature() -> None:
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
        vec_env.envs[i] = wrapper_cls(vec_env.envs[i])
    return vec_env


def override_sub_cfg(omni_cfg: DictConfig, override_cfg: DictConfig, sub_attr: str):
    omni_sub_cfg = OmegaConf.select(omni_cfg, sub_attr)
    override_sub_cfg = OmegaConf.select(override_cfg, sub_attr)
    if override_sub_cfg is not None:
        setattr(
            omni_cfg,
            sub_attr,
            override_sub_cfg
            if omni_sub_cfg is None
            else OmegaConf.merge(omni_sub_cfg, override_sub_cfg),
        )


def setup_omni_cfg(cfg: DictConfig) -> DictConfig:
    """
    Setup OmniGibson's config, overrided by user-set config

    Args:
        cfg(DictConfig): rlinf's env config, must have `omni_config` field

    Returns:
        (DictConfig): overrided OmniGibson config
    """
    import omnigibson as og
    from omnigibson.macros import gm

    override_cfg = OmegaConf.select(cfg, "omni_config")
    cfg_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        omni_cfg = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
    # override env/render/camera/robots/task/scene config
    override_sub_cfg(omni_cfg, override_cfg, "env")
    override_sub_cfg(omni_cfg, override_cfg, "render")
    override_sub_cfg(omni_cfg, override_cfg, "camera")
    override_sub_cfg(omni_cfg, override_cfg, "macro")
    override_sub_cfg(omni_cfg, override_cfg, "task")
    override_sub_cfg(omni_cfg, override_cfg, "scene")
    # here actually we only needs one robot config (and Behavior actually does do that)
    # we must use update rather than merge to keep default robot config fields.
    robot_override = OmegaConf.select(override_cfg, "robots[0]", default=None)
    assert robot_override is not None, (
        "OmniGibson config must contain a non-empty robots list, but robots[0] config is None"
    )
    OmegaConf.update(omni_cfg, "robots[0]", robot_override, merge=True)

    override_proprio_obs = OmegaConf.select(
        override_cfg, "robots[0].proprio_obs", default=None
    )
    if override_proprio_obs is None:
        override_proprio_obs = R1PRO_PROPRIO_KEYS
    OmegaConf.update(
        omni_cfg, "robots[0].proprio_obs", override_proprio_obs, merge=True
    )

    # setup omnigibson macros, according to configuration yaml
    macro_cfg = OmegaConf.select(omni_cfg, "macro")
    gm.HEADLESS = macro_cfg.headless
    gm.ENABLE_FLATCACHE = macro_cfg.enable_flatcache
    gm.ENABLE_OBJECT_STATES = macro_cfg.enable_object_states
    gm.USE_GPU_DYNAMICS = macro_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = macro_cfg.enable_transition_rules
    gm.RENDER_VIEWER_CAMERA = macro_cfg.render_viewer_camera
    gm.USE_NUMPY_CONTROLLER_BACKEND = macro_cfg.use_numpy_controller_backend

    # setup head/wrist camera resolutions
    camera_cfg = OmegaConf.select(omni_cfg, "camera")
    set_camera_resolution(camera_cfg)

    # override behavior's termination config `max_steps` field
    max_episode_steps = OmegaConf.select(cfg, "max_episode_steps")
    assert max_episode_steps is not None, "must set max_episode_steps in config."
    OmegaConf.update(
        omni_cfg,
        "task.termination_config.max_steps",
        max_episode_steps,
    )

    return omni_cfg


def build_instance_resample_config(
    omni_cfg: DictConfig,
) -> BehaviorInstanceResampleConfig:
    activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
    activity_definition_id = OmegaConf.select(
        omni_cfg, "task.activity_definition_id"
    )
    activity_instance_id = OmegaConf.select(omni_cfg, "task.activity_instance_id")
    activity_instance_dir = OmegaConf.select(omni_cfg, "task.activity_instance_dir")
    instance_resample_mode = OmegaConf.select(
        omni_cfg, "task.instance_resample_mode", default="disabled"
    )
    instance_resample_mode = str(instance_resample_mode).lower()
    online_object_sampling = OmegaConf.select(omni_cfg, "task.online_object_sampling")

    assert instance_resample_mode in SUPPORTED_INSTANCE_RESAMPLE_MODES, (
        "task.instance_resample_mode must be one of "
        f"{SUPPORTED_INSTANCE_RESAMPLE_MODES}, got {instance_resample_mode!r}."
    )

    if activity_instance_dir is not None:
        assert not online_object_sampling and instance_resample_mode != "online", (
            "task.activity_instance_dir only supports offline task "
            "instances, please disable online reset-time resampling."
        )

    cached_activity_instances: tuple[BehaviorActivityInstanceFile, ...] = ()
    if activity_instance_dir is not None:
        cached_activity_instances = tuple(
            discover_activity_instance_files(
                activity_instance_dir=activity_instance_dir,
                activity_name=activity_name,
                activity_definition_id=activity_definition_id,
            )
        )

    if instance_resample_mode == "offline":
        assert activity_instance_dir is not None, (
            "task.activity_instance_dir must be set when "
            "task.instance_resample_mode is 'offline'."
        )
    elif instance_resample_mode == "online":
        use_presampled_robot_pose = OmegaConf.select(
            omni_cfg, "task.use_presampled_robot_pose"
        )
        assert online_object_sampling, (
            "task.instance_resample_mode='online' requires "
            "task.online_object_sampling to be True."
        )
        assert not use_presampled_robot_pose, (
            "task.instance_resample_mode='online' requires "
            "task.use_presampled_robot_pose to be False."
        )
    elif activity_instance_dir is not None:
        instance_ids = {entry.instance_id for entry in cached_activity_instances}
        assert activity_instance_id in instance_ids, (
            f"task.activity_instance_id={activity_instance_id} is not present in "
            f"task.activity_instance_dir={activity_instance_dir}."
        )

    return BehaviorInstanceResampleConfig(
        activity_name=activity_name,
        activity_instance_id=activity_instance_id,
        activity_instance_dir=activity_instance_dir,
        instance_resample_mode=instance_resample_mode,
        cached_activity_instances=cached_activity_instances,
    )


def parse_activity_instance_filename(
    filename: str,
    activity_name: str,
) -> tuple[int, int, str] | None:
    file_format = None
    suffix = None
    if filename.endswith(TASK_INSTANCE_FILE_SUFFIX):
        file_format = "tro_state"
        suffix = TASK_INSTANCE_FILE_SUFFIX
    elif filename.endswith(TASK_INSTANCE_TEMPLATE_FILE_SUFFIX):
        file_format = "template"
        suffix = TASK_INSTANCE_TEMPLATE_FILE_SUFFIX
    if suffix is None:
        return None

    infix = f"_task_{activity_name}_"
    if infix not in filename:
        return None

    stem = filename[: -len(suffix)]
    _, suffix_stem = stem.split(infix, 1)
    definition_and_instance = suffix_stem.split("_")
    if len(definition_and_instance) != 2:
        return None

    definition_id, instance_id = definition_and_instance
    if not definition_id.isdigit() or not instance_id.isdigit():
        return None

    return int(definition_id), int(instance_id), file_format


def discover_activity_instance_files(
    activity_instance_dir: str | os.PathLike[str],
    activity_name: str,
    activity_definition_id: int,
) -> list[BehaviorActivityInstanceFile]:
    instance_dir = Path(activity_instance_dir)
    if not instance_dir.is_dir():
        raise ValueError(
            f"activity_instance_dir must be an existing directory, got: {instance_dir}"
        )

    instance_files = {}
    for entry in instance_dir.iterdir():
        if not entry.is_file():
            continue
        parsed = parse_activity_instance_filename(
            entry.name,
            activity_name=activity_name,
        )
        if parsed is None:
            continue

        definition_id, instance_id, file_format = parsed
        if definition_id != activity_definition_id:
            continue

        if instance_id in instance_files:
            raise ValueError(
                f"Duplicate activity instance id {instance_id} found in {instance_dir}."
            )
        instance_files[instance_id] = BehaviorActivityInstanceFile(
            instance_id=instance_id,
            path=str(entry),
            file_format=file_format,
        )

    if not instance_files:
        raise ValueError(
            "No cached BEHAVIOR task instances were found in "
            f"{instance_dir} for activity_name={activity_name}, "
            f"activity_definition_id={activity_definition_id}."
        )

    return [instance_files[k] for k in sorted(instance_files)]


def get_activity_instance_file(
    activity_instances: tuple[BehaviorActivityInstanceFile, ...],
    instance_id: int,
) -> BehaviorActivityInstanceFile:
    for instance_file in activity_instances:
        if instance_file.instance_id == instance_id:
            return instance_file
    raise ValueError(f"Activity instance id {instance_id} was not discovered.")


def get_activity_instance_dir(env, activity_instance_dir: str | None = None) -> str:
    if activity_instance_dir is not None:
        return activity_instance_dir

    from omnigibson.utils.asset_utils import get_task_instance_path

    scene_model = env.task.scene_name
    return os.path.join(
        get_task_instance_path(scene_model),
        f"json/{scene_model}_task_{env.task.activity_name}_instances",
    )


def get_activity_instance_file_path(
    env,
    instance_id: int,
    activity_instance_dir: str | None = None,
) -> str:
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    return os.path.join(
        get_activity_instance_dir(env, activity_instance_dir),
        f"{tro_filename}-tro_state.json",
    )


def load_cached_activity_instance(
    env,
    instance_id: int,
    activity_instance_dir: str | None = None,
    reset_scene: bool = False,
) -> None:
    tro_file_path = get_activity_instance_file_path(
        env,
        instance_id=instance_id,
        activity_instance_dir=activity_instance_dir,
    )
    load_activity_instance_tro_state(
        env,
        instance_id=instance_id,
        tro_file_path=tro_file_path,
        reset_scene=reset_scene,
    )


def load_activity_instance_tro_state(
    env,
    instance_id: int,
    tro_file_path: str,
    reset_scene: bool = False,
) -> None:
    import omnigibson as og
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    env.task.activity_instance_id = instance_id
    with open(tro_file_path, "r", encoding="utf-8") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    robot = env.task.get_agent(env)
    robot_name = getattr(robot, "model_name", getattr(robot, "model", None))
    assert robot_name is not None, (
        "Robot model name is required to load task instances."
    )

    for tro_key, state in tro_state.items():
        if tro_key == "robot_poses":
            assert robot_name in state, (
                f"{robot_name} presampled pose is not found in {tro_file_path}"
            )
            robot_pose = state[robot_name][0]
            robot.set_position_orientation(
                robot_pose["position"],
                robot_pose["orientation"],
                frame="scene",
            )
            env.scene.write_task_metadata(key=tro_key, data=state)
        else:
            env.task.object_scope[tro_key].load_state(state, serialized=False)

    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if entity.exists and not entity.is_system:
                entity.keep_still()

    env.scene.update_initial_file()
    if reset_scene:
        env.scene.reset()


def load_activity_instance_template(
    env,
    instance_id: int,
    template_path: str,
    reset_scene: bool = False,
) -> None:
    env.task.activity_instance_id = instance_id
    env.scene.restore(scene_file=template_path, update_initial_file=True)
    if reset_scene:
        env.scene.reset()


def load_activity_instance_file(
    env,
    instance_file: BehaviorActivityInstanceFile,
    reset_scene: bool = False,
) -> None:
    if instance_file.file_format == "tro_state":
        load_activity_instance_tro_state(
            env,
            instance_id=instance_file.instance_id,
            tro_file_path=instance_file.path,
            reset_scene=reset_scene,
        )
        return
    if instance_file.file_format == "template":
        load_activity_instance_template(
            env,
            instance_id=instance_file.instance_id,
            template_path=instance_file.path,
            reset_scene=reset_scene,
        )
        return

    raise ValueError(f"Unsupported activity instance format: {instance_file.file_format}")


def resample_task(vec_env, omni_task_cfg: DictConfig, num_envs: int):
    online_object_sampling = OmegaConf.select(omni_task_cfg, "online_object_sampling")
    use_presampled_robot_pose = OmegaConf.select(
        omni_task_cfg, "use_presampled_robot_pose"
    )

    assert online_object_sampling and not use_presampled_robot_pose, (
        f"online_object_sampling should be True and use_presampled_robot_pose should be False, but got {online_object_sampling} and  {use_presampled_robot_pose}"
    )

    for i in range(num_envs):
        vec_env.envs[i].update_task(task_config=omni_task_cfg)

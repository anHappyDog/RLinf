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
import hashlib
import inspect
import json
import os
import time
from typing import ClassVar

import gymnasium as gym
import ray
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from rlinf.envs.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = ["BehaviorEnv"]


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
        subproc_idx: int = 0,
        num_env_subprocess: int = 1,
    ):
        _preload_numba_llvmlite()
        from omnigibson.envs import VectorEnvironment

        self.logger = get_logger()
        self.pipeline_stage_num = pipeline_stage_num
        omni_cfg = setup_omni_cfg(cfg)
        self.instance_loader = ActivityInstanceLoader.from_omni_cfg(omni_cfg)

        # create env and apply env wrapper if enabled
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        task_cfg = omni_cfg_dict.get("task", {})
        if task_cfg.get("instance_file_format") == "tro_state" and task_cfg.get(
            "activity_instance_dir"
        ):
            task_cfg["activity_instance_id"] = 0
        elif isinstance(
            task_cfg.get("activity_instance_id"), (list, tuple, ListConfig)
        ):
            activity_instance_ids = task_cfg["activity_instance_id"]
            if len(activity_instance_ids) == 0:
                raise ValueError("task.activity_instance_id is an empty list.")
            task_cfg["activity_instance_id"] = int(activity_instance_ids[0])

        # When pipeline stages > 1, each stage independently advances the
        # global physics per chunk step.  Divide physics_frequency so the
        # total physics rate stays at the configured value.
        if pipeline_stage_num > 1:
            omni_cfg_dict["env"]["physics_frequency"] = (
                omni_cfg_dict["env"]["physics_frequency"] / pipeline_stage_num
            )
        self.env = VectorEnvironment(num_envs, omni_cfg_dict)
        for local_row, env in enumerate(getattr(self.env, "envs", [])):
            env._rlinf_global_env_idx = local_row * num_env_subprocess + subproc_idx
        apply_runtime_renderer_settings()
        wrapper_name = OmegaConf.select(omni_cfg, "env.env_wrapper")
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

    @staticmethod
    def _debug_hash_value(value) -> str:
        """Return a stable short digest for nested simulator state values."""

        hasher = hashlib.sha256()

        def update(item) -> None:
            if item is None or isinstance(item, (bool, int, float, str)):
                hasher.update(repr(item).encode("utf-8"))
                return
            if isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
                hasher.update(str(tensor.dtype).encode("utf-8"))
                hasher.update(tensor.numpy().tobytes())
                return
            if isinstance(item, dict):
                hasher.update(b"{")
                for key in sorted(item, key=lambda x: repr(x)):
                    hasher.update(repr(key).encode("utf-8"))
                    update(item[key])
                hasher.update(b"}")
                return
            if isinstance(item, (list, tuple)):
                hasher.update(b"[")
                for child in item:
                    update(child)
                hasher.update(b"]")
                return
            if hasattr(item, "shape") and hasattr(item, "tobytes"):
                hasher.update(str(getattr(item, "shape", None)).encode("utf-8"))
                hasher.update(str(getattr(item, "dtype", None)).encode("utf-8"))
                hasher.update(item.tobytes())
                return
            hasher.update(repr(item).encode("utf-8", errors="replace"))

        update(value)
        return hasher.hexdigest()[:16]

    @staticmethod
    def _debug_to_plain(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {
                str(k): BehaviorProcess._debug_to_plain(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [BehaviorProcess._debug_to_plain(v) for v in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return repr(value)

    def _debug_vec_envs(self):
        env = self.env
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            envs = getattr(env, "envs", None)
            if envs is not None:
                return list(envs)
            next_env = None
            for attr in ("env", "venv", "_env", "unwrapped"):
                candidate = getattr(env, attr, None)
                if candidate is not None and candidate is not env:
                    next_env = candidate
                    break
            env = next_env
        raise RuntimeError(
            f"Could not find VectorEnvironment.envs through wrapper {type(self.env)!r}."
        )

    def debug_snapshot(self):
        """Capture read-only reset-state fingerprints for each local env row."""
        rows = []
        for local_row, env in enumerate(self._debug_vec_envs()):
            row = {"local_row": local_row}
            try:
                scene = getattr(env, "scene", None)
                task = getattr(env, "task", None)
                row["scene_idx"] = self._debug_to_plain(getattr(scene, "idx", None))
                row["scene_model"] = self._debug_to_plain(
                    getattr(scene, "scene_model", None)
                )
                row["activity_instance_id"] = self._debug_to_plain(
                    getattr(task, "activity_instance_id", None)
                )

                initial_state_parts = {
                    "scene_idx": row["scene_idx"],
                    "scene_model": row["scene_model"],
                    "activity_instance_id": row["activity_instance_id"],
                }
                try:
                    robot = task.get_agent(env) if task is not None else None
                    if robot is not None:
                        pos, orn = robot.get_position_orientation()
                        row["robot_pose"] = {
                            "position": self._debug_to_plain(pos),
                            "orientation": self._debug_to_plain(orn),
                        }
                        row["robot_pose_checksum"] = self._debug_hash_value((pos, orn))
                        initial_state_parts["robot_pose"] = (pos, orn)
                        try:
                            scene_pos, scene_orn = robot.get_position_orientation(
                                frame="scene"
                            )
                        except Exception:
                            scene_pos, scene_orn = (
                                scene.convert_world_pose_to_scene_relative(
                                    pos,
                                    orn,
                                )
                            )
                        row["robot_scene_pose"] = {
                            "position": self._debug_to_plain(scene_pos),
                            "orientation": self._debug_to_plain(scene_orn),
                        }
                        row["robot_scene_pose_checksum"] = self._debug_hash_value(
                            (scene_pos, scene_orn)
                        )
                        initial_state_parts["robot_scene_pose"] = (
                            scene_pos,
                            scene_orn,
                        )
                except Exception as e:  # noqa: BLE001 - debug probe should continue
                    row["robot_pose_error"] = repr(e)

                object_scope = getattr(task, "object_scope", {}) if task else {}
                object_entries = []
                object_scene_pose_entries = []
                object_errors = {}
                for name, obj in sorted(object_scope.items(), key=lambda item: item[0]):
                    try:
                        if hasattr(obj, "dump_state"):
                            state = obj.dump_state(serialized=True)
                        elif hasattr(obj, "get_position_orientation"):
                            state = obj.get_position_orientation()
                        else:
                            state = repr(obj)
                        object_entries.append((name, state))
                    except Exception as e:  # noqa: BLE001 - debug probe should continue
                        object_errors[name] = repr(e)
                    try:
                        if hasattr(obj, "get_position_orientation"):
                            pos, orn = obj.get_position_orientation()
                            scene_pos, scene_orn = (
                                scene.convert_world_pose_to_scene_relative(
                                    pos,
                                    orn,
                                )
                            )
                            object_scene_pose_entries.append(
                                (name, scene_pos, scene_orn)
                            )
                    except Exception:
                        pass
                row["task_relevant_object_count"] = len(object_scope)
                row["task_relevant_object_names"] = sorted(object_scope)[:32]
                row["task_relevant_object_checksum"] = self._debug_hash_value(
                    object_entries
                )
                row["task_relevant_object_scene_pose_checksum"] = (
                    self._debug_hash_value(object_scene_pose_entries)
                )
                initial_state_parts["task_relevant_objects"] = object_entries
                initial_state_parts["task_relevant_object_scene_poses"] = (
                    object_scene_pose_entries
                )
                if object_errors:
                    row["task_relevant_object_errors"] = object_errors
                row["initial_state_checksum"] = self._debug_hash_value(
                    initial_state_parts
                )
            except Exception as e:  # noqa: BLE001 - debug probe should continue
                row["snapshot_error"] = repr(e)
            rows.append(row)
        return rows

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
            list(raw_obs) if need_obs else None,
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

        results: list[tuple] = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            results.append(
                self._step_shard(actions[:, t], env_indices, need_obs=need_obs)
            )
        return tuple(zip(*results))

    def reset(self, reset_indices=None, get_obs=True):
        self.instance_loader.prepare_reset(self.env)
        pre_reset_debug = None
        if os.environ.get("RLINF_BEHAVIOR_CAPTURE_PRE_RESET_SNAPSHOT", "0") == "1":
            snapshot_rows = self.debug_snapshot()
            if reset_indices is None:
                pre_reset_debug = snapshot_rows
            else:
                by_local_row = {row["local_row"]: row for row in snapshot_rows}
                pre_reset_debug = [by_local_row[idx] for idx in reset_indices]
        result = self._call_reset(
            reset_indices=reset_indices,
            get_obs=get_obs,
        )
        if not get_obs:
            return None, None

        raw_obs, infos = result
        if pre_reset_debug is not None:
            for info, row in zip(infos, pre_reset_debug, strict=True):
                info["rlinf_pre_reset_debug"] = row
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
    _RUNTIME_ENV_KEYS: ClassVar[tuple[str, ...]] = (
        "OMNIGIBSON_DATA_PATH",
        "OMNIGIBSON_HEADLESS",
        "OMNIGIBSON_NO_OMNI_LOGS",
        "OMNIGIBSON_DEBUG",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "EXP_PATH",
        "CARB_APP_PATH",
        "RLINF_BEHAVIOR_TRO_STATE_SCOPE_SETTLE",
        "RLINF_BEHAVIOR_TRO_STATE_SETTLE_STEPS",
        "RLINF_BEHAVIOR_TRO_STATE_INSTANCE_BY_GLOBAL_INDEX",
        "RLINF_BEHAVIOR_CAPTURE_PRE_RESET_SNAPSHOT",
    )

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
                self.env_processes = [
                    BehaviorProcess.options(
                        runtime_env=self._subprocess_runtime_env(sp)
                    ).remote(
                        self.cfg,
                        self.num_env_shard,
                        pipeline_stage_num,
                        sp,
                        self.num_env_subprocess,
                    )
                    for sp in range(self.num_env_subprocess)
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

    def _subprocess_runtime_env(self, subproc_idx: int) -> dict:
        env_vars = {
            key: os.environ[key] for key in self._RUNTIME_ENV_KEYS if key in os.environ
        }
        device_override = os.environ.get(
            "RLINF_BEHAVIOR_SUBPROCESS_CUDA_VISIBLE_DEVICES"
        )
        visible_devices = device_override or os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            devices = [
                device.strip()
                for device in visible_devices.split(",")
                if device.strip()
            ]
            if devices:
                selected_device = devices[subproc_idx % len(devices)]
                env_vars["CUDA_VISIBLE_DEVICES"] = selected_device
                self.logger.info(
                    "BehaviorProcess %d using CUDA_VISIBLE_DEVICES=%s "
                    "(source=%s, parent CUDA_VISIBLE_DEVICES=%s)",
                    subproc_idx,
                    selected_device,
                    "RLINF_BEHAVIOR_SUBPROCESS_CUDA_VISIBLE_DEVICES"
                    if device_override
                    else "CUDA_VISIBLE_DEVICES",
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                )
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1"
        )
        return {"env_vars": env_vars}

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

    def debug_snapshot_slice(self, global_start: int, num_envs: int):
        """Capture debug snapshots for ``[global_start, global_start + num_envs)``."""
        if num_envs == 0:
            return []
        plan = self._slice_plan(global_start, num_envs)
        shard_results = ray.get(
            [self.env_processes[sp].debug_snapshot.remote() for sp, _, _ in plan]
        )

        rows: list[dict | None] = [None] * num_envs
        for shard_rows, (sp, positions, local_rows) in zip(shard_results, plan):
            by_local_row = {row["local_row"]: row for row in shard_rows}
            for pos, local_row in zip(positions, local_rows):
                row = dict(by_local_row[local_row])
                row["global_row"] = global_start + pos
                row["slice_position"] = pos
                row["subproc"] = sp
                row["local_row"] = local_row
                rows[pos] = row
        return rows

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
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            obs_t: list | None = [None] * slice_num_envs if need_obs else None
            reward_t = torch.zeros(slice_num_envs, dtype=torch.float32)
            term_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            trunc_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            info_t: list = [{} for _ in range(slice_num_envs)]
            for (obs_per_t, rewards_per_t, terms_per_t, truncs_per_t, infos_per_t), (
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
            merged_obs.append(obs_t)
            merged_rewards.append(reward_t)
            merged_terms.append(term_t)
            merged_trunc.append(trunc_t)
            merged_infos.append(info_t)
        return merged_obs, merged_rewards, merged_terms, merged_trunc, merged_infos

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
        self._last_pre_reset_debug = None
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

    def debug_snapshot(self):
        self._ensure_pool()
        return self.pool.debug_snapshot_slice(self.pool_offset, self.num_envs)

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
        return obs

    def _calc_step_reward(self, reward):
        return self.reward_coef * reward

    def reset(self):
        if self.enable_offload and self.pool is None:
            self._init_env()
        raw_obs, infos = self.env_reset()
        self._last_pre_reset_debug = [
            info.get("rlinf_pre_reset_debug")
            for info in infos
            if isinstance(info, dict) and "rlinf_pre_reset_debug" in info
        ]
        if not self._last_pre_reset_debug:
            self._last_pre_reset_debug = None
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
        ) = self.env_chunk_step(chunk_actions)

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
            episode_info = {
                "success": done_dict.get("success", False),
                "episode_length": info.get("episode_length", 0),
            }
            self.returns[env_idx] += reward
            self.success_once[env_idx] = self.success_once[env_idx] | done_dict.get(
                "success", False
            )
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

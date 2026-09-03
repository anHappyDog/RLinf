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
import pathlib
import time
from typing import ClassVar

import gymnasium as gym
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.demonstration_reset import (
    DemonstrationResetSpec,
    restore_demonstration_observations,
)
from rlinf.envs.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.behavior.oracle_prompt import (
    StagePromptController,
    extract_sequential_reward_info,
)
from rlinf.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = [
    "BehaviorEnv",
    "apply_short_memory_history_ablation",
    "apply_short_memory_history_valid_lengths",
]


_SHORT_MEMORY_HISTORY_ABLATIONS = {
    "none",
    "repeat_current",
    "shuffle_past",
}


def apply_short_memory_history_ablation(
    history: dict[str, torch.Tensor], mode: str
) -> dict[str, torch.Tensor]:
    """Apply a deterministic content control to a short-memory observation.

    The current observation, padding mask, and time offsets remain unchanged.
    ``repeat_current`` replaces every valid past observation with the current
    one. ``shuffle_past`` reverses only the valid past observations, providing
    a reproducible order counterfactual without moving padding into valid slots.
    """
    if mode not in _SHORT_MEMORY_HISTORY_ABLATIONS:
        choices = ", ".join(sorted(_SHORT_MEMORY_HISTORY_ABLATIONS))
        raise ValueError(f"history_ablation must be one of {choices}, got {mode!r}.")
    if mode == "none":
        return history

    frame_mask = history["history_frame_mask"]
    result = dict(history)
    content_keys = (
        "history_main_images",
        "history_wrist_images",
        "history_states",
    )
    for key in content_keys:
        value = history[key]
        controlled = value.clone()
        if mode == "repeat_current":
            content_mask = frame_mask.reshape(
                *frame_mask.shape, *((1,) * (value.ndim - frame_mask.ndim))
            )
            current = value[:, -1:].expand_as(value)
            controlled = torch.where(content_mask, current, controlled)
        else:
            for batch_index, mask in enumerate(frame_mask):
                valid_past = torch.nonzero(mask[:-1], as_tuple=False).flatten()
                controlled[batch_index, valid_past] = value[
                    batch_index, valid_past.flip(0)
                ]
        result[key] = controlled
    return result


def apply_short_memory_history_valid_lengths(
    history: dict[str, torch.Tensor], valid_lengths: list[int]
) -> dict[str, torch.Tensor]:
    """Mask leading history frames to reproduce a loader's padded sample."""
    frame_mask = history["history_frame_mask"]
    batch_size, history_length = frame_mask.shape
    if len(valid_lengths) != batch_size:
        raise ValueError(
            f"Expected {batch_size} valid history lengths, got {len(valid_lengths)}."
        )
    if any(length < 1 or length > history_length for length in valid_lengths):
        raise ValueError(
            f"History valid lengths must be in [1, {history_length}], got "
            f"{valid_lengths}."
        )

    result = dict(history)
    limited_mask = frame_mask.clone()
    for batch_index, valid_length in enumerate(valid_lengths):
        limited_mask[batch_index, : history_length - valid_length] = False
    result["history_frame_mask"] = limited_mask

    for key in (
        "history_main_images",
        "history_wrist_images",
        "history_states",
    ):
        value = history[key]
        content_mask = limited_mask.reshape(
            *limited_mask.shape, *((1,) * (value.ndim - limited_mask.ndim))
        )
        result[key] = torch.where(content_mask, value, torch.zeros_like(value))
    result["history_time_offsets"] = torch.where(
        limited_mask,
        history["history_time_offsets"],
        torch.zeros_like(history["history_time_offsets"]),
    )
    return result


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
        activity_instance_ids: list[int] | None = None,
        demonstration_reset_specs: list[dict] | None = None,
    ):
        _preload_numba_llvmlite()
        from omnigibson.envs import VectorEnvironment

        self.logger = get_logger()
        self.pipeline_stage_num = pipeline_stage_num
        omni_cfg = setup_omni_cfg(cfg)
        self.instance_loader = ActivityInstanceLoader.from_omni_cfg(omni_cfg)
        if activity_instance_ids is not None:
            self.instance_loader.with_fixed_instance_ids(activity_instance_ids)
        self.demonstration_reset_specs = (
            tuple(DemonstrationResetSpec(**spec) for spec in demonstration_reset_specs)
            if demonstration_reset_specs is not None
            else ()
        )
        if self.demonstration_reset_specs and (
            num_envs != 1 or len(self.demonstration_reset_specs) != num_envs
        ):
            raise ValueError(
                "Demonstration reset requires exactly one env and one reset spec "
                "per OmniGibson subprocess."
            )

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
        self.decision_trace_object_name = OmegaConf.select(
            cfg, "decision_trace_object_name", default=None
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

    def _step_shard(
        self,
        actions: torch.Tensor,
        env_indices: list[int],
        need_obs: bool,
        record_trace_state: bool = False,
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

        instance_ids = self.instance_loader.current_instance_ids
        for info, env_index in zip(infos, env_indices, strict=True):
            if instance_ids:
                info["activity_instance_id"] = instance_ids[env_index]
            if record_trace_state and self.decision_trace_object_name is not None:
                env = self.env.envs[env_index]
                target = env.scene.object_registry(
                    "name", self.decision_trace_object_name
                )
                position, orientation = target.get_position_orientation()
                info["decision_trace_state"] = {
                    "object_position": position.detach().cpu().tolist(),
                    "object_orientation": orientation.detach().cpu().tolist(),
                }

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
                self._step_shard(
                    actions[:, t],
                    env_indices,
                    need_obs=need_obs,
                    record_trace_state=is_last,
                )
            )
        return tuple(zip(*results))

    def reset(self, reset_indices=None, get_obs=True):
        instance_ids = self.instance_loader.prepare_reset(self.env)
        if self.demonstration_reset_specs:
            if reset_indices is not None and list(reset_indices) != [0]:
                raise ValueError(
                    "The single-env demonstration reset only accepts reset_indices=[0]."
                )
            self._call_reset(reset_indices=reset_indices, get_obs=False)
            history = restore_demonstration_observations(
                self.env.envs[0], self.demonstration_reset_specs[0]
            )
            result = (
                [history[-1][1]],
                [{"_rlinf_demonstration_history": history}],
            )
        else:
            result = self._call_reset(
                reset_indices=reset_indices,
                get_obs=get_obs,
            )
        if not get_obs:
            return None, None

        raw_obs, infos = result
        if reset_indices is not None:
            instance_ids = tuple(instance_ids[index] for index in reset_indices)
        for info, instance_id in zip(infos, instance_ids, strict=True):
            info["activity_instance_id"] = instance_id
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
        activity_instance_ids: list[int] | None = None,
        demonstration_reset_specs: list[dict] | None = None,
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
                activity_instance_ids=activity_instance_ids,
                demonstration_reset_specs=demonstration_reset_specs,
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
        activity_instance_ids: list[int] | None = None,
        demonstration_reset_specs: list[dict] | None = None,
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
        if (
            activity_instance_ids is not None
            and len(activity_instance_ids) != total_num_envs
        ):
            raise ValueError(
                "activity_instance_ids must contain one id per local BEHAVIOR env, "
                f"got {len(activity_instance_ids)} and {total_num_envs}."
            )
        if (
            demonstration_reset_specs is not None
            and len(demonstration_reset_specs) != total_num_envs
        ):
            raise ValueError(
                "demonstration_reset_specs must contain one spec per local "
                f"BEHAVIOR env, got {len(demonstration_reset_specs)} and "
                f"{total_num_envs}."
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
                    BehaviorProcess.remote(
                        self.cfg,
                        self.num_env_shard,
                        pipeline_stage_num,
                        (
                            activity_instance_ids[process_index::num_env_subprocess]
                            if activity_instance_ids is not None
                            else None
                        ),
                        (
                            demonstration_reset_specs[process_index::num_env_subprocess]
                            if demonstration_reset_specs is not None
                            else None
                        ),
                    )
                    for process_index in range(self.num_env_subprocess)
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
    _decision_trace_dir = None

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
        self.prompt_controller = None
        self.history_length = int(OmegaConf.select(cfg, "history_length", default=1))
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        self.history_decision_stride = int(
            OmegaConf.select(cfg, "history_decision_stride", default=1)
        )
        if self.history_decision_stride <= 0:
            raise ValueError("history_decision_stride must be positive.")
        self.history_ablation = str(
            OmegaConf.select(cfg, "history_ablation", default="none")
        )
        if self.history_ablation not in _SHORT_MEMORY_HISTORY_ABLATIONS:
            choices = ", ".join(sorted(_SHORT_MEMORY_HISTORY_ABLATIONS))
            raise ValueError(
                "history_ablation must be one of "
                f"{choices}, got {self.history_ablation!r}."
            )
        self._observation_history = []
        self._history_step = 0
        self._action_frequency = float(
            OmegaConf.select(cfg, "omni_config.env.action_frequency", default=60.0)
        )
        self._decision_trace_dir = OmegaConf.select(
            cfg, "decision_trace_dir", default=None
        )
        self._decision_trace_object_name = OmegaConf.select(
            cfg, "decision_trace_object_name", default=None
        )
        self._decision_trace_gripper_indices = tuple(
            OmegaConf.select(cfg, "decision_trace_gripper_indices", default=[14, 22])
        )
        if len(self._decision_trace_gripper_indices) != 2:
            raise ValueError("decision_trace_gripper_indices must contain two indices.")
        self._decision_trace_records = []
        self._decision_index = 0
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
            activity_instance_ids = OmegaConf.select(
                self.cfg, "activity_instance_ids", default=None
            )
            local_instance_ids = None
            local_demonstration_specs = None
            start = None
            total_envs_per_worker = None
            if activity_instance_ids is not None:
                activity_instance_ids = list(activity_instance_ids)
                total_num_envs = int(self.cfg.total_num_envs)
                if len(activity_instance_ids) != total_num_envs:
                    raise ValueError(
                        "activity_instance_ids must contain one id per global "
                        f"BEHAVIOR env, got {len(activity_instance_ids)} and "
                        f"total_num_envs={total_num_envs}."
                    )
                if total_num_envs % int(self.worker_info.group_world_size) != 0:
                    raise ValueError(
                        "total_num_envs must be divisible by the env worker world "
                        "size when activity_instance_ids are fixed."
                    )
                total_envs_per_worker = int(self.cfg.total_num_envs) // int(
                    self.worker_info.group_world_size
                )
                start = int(self.worker_info.rank) * total_envs_per_worker
                local_instance_ids = activity_instance_ids[
                    start : start + total_envs_per_worker
                ]
            demonstration_paths = OmegaConf.select(
                self.cfg, "demonstration_reset_paths", default=None
            )
            demonstration_frames = OmegaConf.select(
                self.cfg, "demonstration_reset_frame_indices", default=None
            )
            if demonstration_paths is not None or demonstration_frames is not None:
                if demonstration_paths is None or demonstration_frames is None:
                    raise ValueError(
                        "demonstration_reset_paths and "
                        "demonstration_reset_frame_indices must be configured together."
                    )
                demonstration_paths = list(demonstration_paths)
                demonstration_frames = list(demonstration_frames)
                total_num_envs = int(self.cfg.total_num_envs)
                if (
                    len(demonstration_paths) != total_num_envs
                    or len(demonstration_frames) != total_num_envs
                ):
                    raise ValueError(
                        "Demonstration reset path/frame lists must contain one entry "
                        f"per global env ({total_num_envs})."
                    )
                valid_history_lengths = OmegaConf.select(
                    self.cfg,
                    "demonstration_reset_valid_history_lengths",
                    default=None,
                )
                if valid_history_lengths is not None:
                    valid_history_lengths = list(valid_history_lengths)
                    if len(valid_history_lengths) != total_num_envs:
                        raise ValueError(
                            "demonstration_reset_valid_history_lengths must "
                            f"contain one entry per global env ({total_num_envs})."
                        )
                    if any(
                        int(length) < 1 or int(length) > self.history_length
                        for length in valid_history_lengths
                    ):
                        raise ValueError(
                            "Demonstration valid history lengths must be within "
                            f"[1, {self.history_length}]."
                        )
                if (
                    local_instance_ids is None
                    or start is None
                    or total_envs_per_worker is None
                ):
                    raise ValueError(
                        "Demonstration reset requires matching activity_instance_ids."
                    )
                history_stride = int(
                    OmegaConf.select(
                        self.cfg, "demonstration_reset_history_stride", default=30
                    )
                )
                initial_stage_name = OmegaConf.select(
                    self.cfg, "demonstration_reset_stage", default=None
                )
                local_demonstration_specs = [
                    {
                        "path": demonstration_paths[index],
                        "frame_index": int(demonstration_frames[index]),
                        "history_length": self.history_length,
                        "history_stride": history_stride,
                        "expected_instance_id": int(local_instance_ids[index - start]),
                        "initial_stage_name": initial_stage_name,
                    }
                    for index in range(start, start + total_envs_per_worker)
                ]
            self.pool, self.pool_offset = BehaviorProcessPool.acquire_shared(
                self.cfg,
                self.worker_info,
                self.pipeline_stage_num,
                self.num_envs,
                activity_instance_ids=local_instance_ids,
                demonstration_reset_specs=local_demonstration_specs,
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
        stage_prompts = OmegaConf.select(self.cfg, "oracle_stage_prompts", default={})
        if OmegaConf.is_config(stage_prompts):
            stage_prompts = OmegaConf.to_container(stage_prompts, resolve=True)
        else:
            stage_prompts = dict(stage_prompts)
        self.prompt_controller = StagePromptController(
            task_prompt=self.task_description,
            num_envs=self.num_envs,
            mode=str(OmegaConf.select(self.cfg, "prompt_mode", default="task")),
            stage_prompts=stage_prompts,
            initial_stage=OmegaConf.select(
                self.cfg, "oracle_initial_stage", default=None
            ),
        )

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
        }

    def _wrap_obs(self, obs_list, raw_infos=None, *, record_history=True):
        """Convert raw observations and optionally record a policy memory frame.

        A BEHAVIOR action chunk contains many simulator steps but produces one
        policy decision. Intermediate observations are useful for complete
        videos; they must not change the decision-rate short-memory buffer.
        """
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        if raw_infos is not None:
            self.prompt_controller.update(raw_infos)
        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": self.prompt_controller.prompts(),
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, 32]
        }
        if self.history_length > 1 and record_history:
            self._append_history(obs)
            obs.update(self._build_history_observation())
        return obs

    def _append_history(self, obs):
        entry = {
            "main_images": obs["main_images"],
            "wrist_images": obs["wrist_images"],
            "states": obs["states"],
            "time_seconds": self._history_step / self._action_frequency,
        }
        self._observation_history.append(entry)
        raw_history_length = (
            self.history_length - 1
        ) * self.history_decision_stride + 1
        self._observation_history = self._observation_history[-raw_history_length:]

    def _build_history_observation(self):
        current = self._observation_history[-1]
        selected = self._observation_history[:: -self.history_decision_stride][
            : self.history_length
        ]
        selected.reverse()
        missing = self.history_length - len(selected)

        def _stack_history(key):
            padding = [torch.zeros_like(current[key]) for _ in range(missing)]
            values = [*padding, *(entry[key] for entry in selected)]
            return torch.stack(values, dim=1)

        history_mask = torch.zeros(self.num_envs, self.history_length, dtype=torch.bool)
        history_mask[:, missing:] = True
        current_time = current["time_seconds"]
        offsets = [0.0] * missing + [
            entry["time_seconds"] - current_time for entry in selected
        ]
        time_offsets = torch.tensor(offsets, dtype=torch.float32).expand(
            self.num_envs, -1
        )
        history = {
            "history_main_images": _stack_history("main_images"),
            "history_wrist_images": _stack_history("wrist_images"),
            "history_states": _stack_history("states"),
            "history_frame_mask": history_mask,
            "history_time_offsets": time_offsets,
        }
        return apply_short_memory_history_ablation(history, self.history_ablation)

    def _calc_step_reward(self, reward):
        return self.reward_coef * reward

    def reset(self):
        if self.enable_offload and self.pool is None:
            self._init_env()
        raw_obs, infos = self.env_reset()
        self.prompt_controller.reset()
        self._observation_history = []
        self._history_step = 0
        self._decision_trace_records = []
        self._decision_index = 0
        demonstration_histories = [
            info.pop("_rlinf_demonstration_history", None) for info in infos
        ]
        if any(history is not None for history in demonstration_histories):
            if not all(history is not None for history in demonstration_histories):
                raise ValueError(
                    "Demonstration reset history must be present for every env."
                )
            history_sizes = {len(history) for history in demonstration_histories}
            if history_sizes != {self.history_length}:
                raise ValueError(
                    "Demonstration reset history length does not match the policy: "
                    f"got {sorted(history_sizes)}, expected {self.history_length}."
                )
            for history_position in range(self.history_length):
                self._history_step = history_position * int(
                    OmegaConf.select(
                        self.cfg, "demonstration_reset_history_stride", default=30
                    )
                )
                history_raw_obs = [
                    history[history_position][1] for history in demonstration_histories
                ]
                obs = self._wrap_obs(history_raw_obs)
            valid_history_lengths = OmegaConf.select(
                self.cfg,
                "demonstration_reset_valid_history_lengths",
                default=None,
            )
            if valid_history_lengths is not None:
                start = int(self.worker_info.rank) * self.num_envs
                local_valid_lengths = [
                    int(length)
                    for length in list(valid_history_lengths)[
                        start : start + self.num_envs
                    ]
                ]
                obs.update(
                    apply_short_memory_history_valid_lengths(
                        obs, local_valid_lengths
                    )
                )
        else:
            obs = self._wrap_obs(raw_obs)
        rewards = torch.zeros(self.num_envs, dtype=bool)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim].
        chunk_actions = torch.as_tensor(chunk_actions).detach().cpu()
        decision_prompts = self.prompt_controller.prompts()
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
        chunk_size = len(raw_obs_list)
        for step_index, (
            raw_obs,
            raw_rewards,
            raw_terminations,
            step_infos,
        ) in enumerate(
            zip(
                raw_obs_list,
                raw_rewards_list,
                raw_terminations_list,
                raw_infos_list,
            )
        ):
            self._history_step += 1
            if raw_obs is None:
                obs_list.append(None)
            else:
                obs_list.append(
                    self._wrap_obs(
                        raw_obs,
                        step_infos,
                        record_history=step_index == chunk_size - 1,
                    )
                )
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

        self._record_decision_trace(
            chunk_actions=chunk_actions,
            raw_infos_list=raw_infos_list,
            decision_prompts=decision_prompts,
            final_obs=obs_list[-1],
        )

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

    def _record_decision_trace(
        self,
        chunk_actions: torch.Tensor,
        raw_infos_list: list,
        decision_prompts: list[str],
        final_obs: dict,
    ) -> None:
        """Record one compact trace row per environment and policy decision."""
        if self._decision_trace_dir is None:
            return

        left_index, right_index = self._decision_trace_gripper_indices
        if chunk_actions.shape[-1] <= max(left_index, right_index):
            raise ValueError(
                "Configured gripper action index exceeds action dimension "
                f"{chunk_actions.shape[-1]}."
            )

        final_states = final_obs["states"]
        for env_index in range(self.num_envs):
            sequential_steps = [
                extract_sequential_reward_info(step_infos[env_index])
                for step_infos in raw_infos_list
            ]
            distances = []
            in_hand = []
            on_support = []
            has_left_support = []
            has_picked_up = []
            for sequential_info in sequential_steps:
                stage_name = sequential_info.get("current_stage_name")
                stage_info = sequential_info.get("stage_infos", {}).get(stage_name, {})
                if "eef_to_obj_distance" in stage_info:
                    distances.append(float(stage_info["eef_to_obj_distance"]))
                if "in_hand" in stage_info:
                    in_hand.append(bool(stage_info["in_hand"]))
                if "on_support" in stage_info:
                    on_support.append(bool(stage_info["on_support"]))
                if "has_left_support" in stage_info:
                    has_left_support.append(bool(stage_info["has_left_support"]))
                if "has_picked_up" in stage_info:
                    has_picked_up.append(bool(stage_info["has_picked_up"]))

            final_info = sequential_steps[-1]
            final_raw_info = raw_infos_list[-1][env_index]
            trace_state = final_raw_info.get("decision_trace_state", {})
            policy_state = final_states[env_index]
            self._decision_trace_records.append(
                {
                    "decision_index": self._decision_index,
                    "sim_step_end": self._history_step,
                    "activity_instance_id": final_raw_info.get(
                        "activity_instance_id", -1
                    ),
                    "prompt": decision_prompts[env_index],
                    "stage_name": final_info.get("current_stage_name"),
                    "completed_stage_count": final_info.get("completed_stage_count", 0),
                    "eef_to_obj_distance_min": min(distances) if distances else None,
                    "eef_to_obj_distance_final": distances[-1] if distances else None,
                    "in_hand_any": any(in_hand),
                    "in_hand_final": in_hand[-1] if in_hand else None,
                    "on_support_final": on_support[-1] if on_support else None,
                    "has_left_support_any": any(has_left_support),
                    "has_picked_up_any": any(has_picked_up),
                    "left_gripper_command": chunk_actions[
                        env_index, :, left_index
                    ].tolist(),
                    "right_gripper_command": chunk_actions[
                        env_index, :, right_index
                    ].tolist(),
                    "left_gripper_width": float(policy_state[193:195].sum()),
                    "right_gripper_width": float(policy_state[232:234].sum()),
                    "object_position": trace_state.get("object_position"),
                    "object_orientation": trace_state.get("object_orientation"),
                }
            )
        self._decision_index += 1

    def flush_decision_trace(self) -> None:
        """Write this env worker's decision trace to a rank-specific JSON file."""
        if self._decision_trace_dir is None:
            return

        output_dir = pathlib.Path(self._decision_trace_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"rank_{self.worker_info.rank}_seed_{self.seed}.json"
        )
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "object_name": self._decision_trace_object_name,
                    "gripper_action_indices": self._decision_trace_gripper_indices,
                    "records": self._decision_trace_records,
                },
                file,
                indent=2,
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
                "activity_instance_id": info.get("activity_instance_id", -1),
            }
            sequential_info = extract_sequential_reward_info(info)
            if sequential_info:
                episode_info.update(
                    oracle_stage_index=sequential_info.get("current_stage_idx", -1),
                    oracle_completed_stage_count=sequential_info.get(
                        "completed_stage_count", 0
                    ),
                    oracle_all_stages_completed=sequential_info.get(
                        "all_stages_completed", False
                    ),
                )
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

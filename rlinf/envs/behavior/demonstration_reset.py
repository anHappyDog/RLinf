# Copyright 2026 The RLinf Authors.
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

"""Restore BEHAVIOR environments from official raw demonstration states."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DemonstrationResetSpec:
    """One raw HDF5 state sequence used to initialize an evaluation env."""

    path: str
    frame_index: int
    history_length: int = 1
    history_stride: int = 30
    expected_instance_id: int | None = None
    initial_stage_name: str | None = None

    def frame_indices(self) -> tuple[int, ...]:
        """Return oldest-to-current demonstration frame indices."""
        if self.frame_index < 0:
            raise ValueError("Demonstration reset frame_index must be non-negative.")
        if self.history_length <= 0:
            raise ValueError("Demonstration reset history_length must be positive.")
        if self.history_stride <= 0:
            raise ValueError("Demonstration reset history_stride must be positive.")
        first = self.frame_index - (self.history_length - 1) * self.history_stride
        if first < 0:
            raise ValueError(
                "Demonstration reset does not have enough preceding frames for "
                f"history: first requested index is {first}."
            )
        return tuple(
            first + offset * self.history_stride
            for offset in range(self.history_length)
        )


def read_demonstration_instance_id(path: str | Path) -> int:
    """Read the BEHAVIOR activity instance id recorded in a raw demo."""
    import h5py

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Demonstration reset HDF5 not found: {resolved_path}")
    with h5py.File(resolved_path, "r") as hdf5_file:
        config = json.loads(hdf5_file["data"].attrs["config"])
    return int(config["task"]["activity_instance_id"])


def load_demonstration_states(
    spec: DemonstrationResetSpec,
) -> tuple[tuple[int, torch.Tensor], ...]:
    """Load and validate serialized simulator states for one reset sequence."""
    import h5py

    path = Path(spec.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration reset HDF5 not found: {path}")

    frame_indices = spec.frame_indices()
    with h5py.File(path, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        demo_group = data_group["demo_0"]
        config = json.loads(data_group.attrs["config"])
        instance_id = int(config["task"]["activity_instance_id"])
        if (
            spec.expected_instance_id is not None
            and instance_id != spec.expected_instance_id
        ):
            raise ValueError(
                f"Raw demonstration instance id {instance_id} does not match "
                f"expected id {spec.expected_instance_id}: {path}"
            )

        transitions = json.loads(demo_group.attrs.get("transitions", "{}"))
        relevant_transitions = [
            int(index) for index in transitions if int(index) <= spec.frame_index
        ]
        if relevant_transitions:
            raise ValueError(
                "Mid-stage reset does not yet support demonstrations with dynamic "
                f"scene transitions before frame {spec.frame_index}: "
                f"{sorted(relevant_transitions)}"
            )

        state_dataset = demo_group["state"]
        state_sizes = demo_group["state_size"]
        if frame_indices[-1] >= len(state_dataset):
            raise IndexError(
                f"Frame {frame_indices[-1]} is outside {path}, which contains "
                f"{len(state_dataset)} states."
            )
        states = []
        for frame_index in frame_indices:
            state_size = int(state_sizes[frame_index])
            state = torch.from_numpy(state_dataset[frame_index, :state_size].copy())
            states.append((frame_index, state))
    return tuple(states)


def load_demonstration_scene_file(path: str | Path) -> dict:
    """Load the exact scene registry stored with a raw demonstration."""
    import h5py

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Demonstration reset HDF5 not found: {resolved_path}")
    with h5py.File(resolved_path, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        if "scene_file" not in data_group.attrs:
            raise KeyError(f"Raw demonstration has no scene_file metadata: {resolved_path}")
        scene_file = json.loads(data_group.attrs["scene_file"])
        init_metadata = data_group["demo_0"].get("init_metadata")
        if init_metadata is not None and len(init_metadata):
            raise NotImplementedError(
                "Mid-stage reset does not yet support non-empty init_metadata: "
                f"{resolved_path}"
            )
    return scene_file


def _unwrap_env(env):
    while "env" in vars(env):
        env = vars(env)["env"]
    return env


def _reset_task_bookkeeping_at_current_state(
    env, initial_stage_name: str | None = None
) -> None:
    """Reset task rewards and terminations without resetting the scene."""
    task = env.task
    task._reset_variables(env)  # noqa: SLF001
    for termination_condition in task._termination_conditions.values():  # noqa: SLF001
        termination_condition.reset(task, env)
    for reward_function in task._reward_functions.values():  # noqa: SLF001
        reward_function.reset(task, env)
    if initial_stage_name is not None:
        matching_rewards = []
        for reward_function in task._reward_functions.values():  # noqa: SLF001
            stage_defs = getattr(reward_function, "_stage_defs", ())
            stage_names = [stage.get("name") for stage in stage_defs]
            if initial_stage_name in stage_names and hasattr(
                reward_function, "set_active_stage_index"
            ):
                reward_function.set_active_stage_index(
                    stage_names.index(initial_stage_name)
                )
                matching_rewards.append(reward_function)
        if not matching_rewards:
            raise ValueError(
                f"No sequential task reward contains stage {initial_stage_name!r}."
            )
    env._current_step = 0  # noqa: SLF001


def restore_demonstration_observations(
    env,
    spec: DemonstrationResetSpec,
) -> list[tuple[int, dict]]:
    """Restore a state sequence and render matching raw observations.

    The caller must own exactly one scene in its OmniGibson simulator. The
    returned list is oldest-to-current and contains no executed action.
    """
    import omnigibson as og
    from omnigibson.utils.usd_utils import PoseAPI

    if len(og.sim.scenes) != 1:
        raise ValueError(
            "Raw demonstration reset requires one scene per OmniGibson process, "
            f"found {len(og.sim.scenes)}."
        )

    base_env = _unwrap_env(env)
    observations = []
    states = load_demonstration_states(spec)
    scene_file = load_demonstration_scene_file(spec.path)
    base_env.scene.restore(scene_file, update_initial_file=True)
    base_env.reset()
    for sequence_index, (frame_index, state) in enumerate(states):
        og.sim.load_state(state, serialized=True)
        PoseAPI.invalidate()
        PoseAPI._refresh()  # noqa: SLF001
        if sequence_index == len(states) - 1:
            _reset_task_bookkeeping_at_current_state(
                base_env, spec.initial_stage_name
            )
        for _ in range(3):
            og.sim.render()
        observation, _ = base_env.get_obs()
        observations.append((frame_index, observation))
    return observations

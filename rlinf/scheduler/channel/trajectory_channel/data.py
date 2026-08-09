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

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import torch

from rlinf.data.schema.embodied_types import EnvOutput, put_tensor_device


@dataclass(kw_only=True)
class EnvResult:
    """Environment outcome associated with one policy output."""

    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    terminations: torch.Tensor | None = None
    truncations: torch.Tensor | None = None
    final_obs: dict[str, Any] | None = None
    intervene_actions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None
    reward_model_output: torch.Tensor | None = None
    reward_assign_lengths: list[int] | None = None
    episode_data: dict[str, Any] | None = None

    def __post_init__(self):
        """Move transmitted tensors to contiguous CPU storage."""
        for field_name in (
            "rewards",
            "dones",
            "terminations",
            "truncations",
            "intervene_actions",
            "intervene_flags",
            "rlt_switch_flags",
            "reward_model_output",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.cpu().contiguous())
        if self.final_obs is not None:
            self.final_obs = put_tensor_device(self.final_obs, "cpu")

    @classmethod
    def from_env_output(
        cls, env_output: EnvOutput, reward_model_output: torch.Tensor | None = None
    ) -> "EnvResult":
        """Build a transport result from an environment output."""
        return cls(
            rewards=env_output.rewards,
            dones=env_output.dones,
            terminations=env_output.terminations,
            truncations=env_output.truncations,
            final_obs=(
                env_output.prepare_observations(env_output.final_obs)
                if env_output.final_obs is not None
                else None
            ),
            intervene_actions=env_output.intervene_actions,
            intervene_flags=env_output.intervene_flags,
            rlt_switch_flags=env_output.rlt_switch_flags,
            reward_model_output=reward_model_output,
        )


@dataclass(kw_only=True)
class RolloutResult:
    """Policy inference data retained for trajectory construction."""

    actions: torch.Tensor
    forward_inputs: dict[str, Any]
    bootstrap_values: torch.Tensor | None = None
    prev_logprobs: torch.Tensor | None = None
    prev_values: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    versions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Move inference data to contiguous CPU storage."""
        self.actions = self.actions.cpu().contiguous()
        self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        for field_name in (
            "bootstrap_values",
            "prev_logprobs",
            "prev_values",
            "intervene_flags",
            "versions",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.cpu().contiguous())


@dataclass(kw_only=True)
class PolicyInput:
    """Input to a policy and the result of its preceding action."""

    obs: dict[str, Any]
    env_result: EnvResult
    is_last: bool = False
    sources: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        """Move observations to CPU for transport."""
        self.obs = put_tensor_device(self.obs, "cpu")


@dataclass(kw_only=True)
class PolicyOutput:
    """Action-only response returned to an environment worker."""

    actions: torch.Tensor


@dataclass(kw_only=True)
class TrajectorySegment:
    """One append operation for a set of logical trajectory sources."""

    step_id: int
    epoch_id: int
    sources: list[tuple[int, int, int]]
    obs: dict[str, Any]
    next_obs: dict[str, Any]
    env_result: EnvResult
    rollout_result: RolloutResult
    initial_env_result: EnvResult | None = None
    forward_inputs: dict[str, Any] | None = None

    def __post_init__(self):
        """Move segment payloads to CPU for transport."""
        self.obs = put_tensor_device(self.obs, "cpu")
        self.next_obs = put_tensor_device(self.next_obs, "cpu")
        if self.forward_inputs is not None:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")


@dataclass(kw_only=True, frozen=True)
class TrajectoryEnd:
    """Signal that one producer has finished a training step."""

    step_id: int
    source: tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class TrajectoryEpochEnd:
    """Signal that one producer has finished a pipeline epoch."""

    step_id: int
    epoch_id: int
    source: tuple[int, int]
    sources: list[tuple[int, int, int]]
    final_prev_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Move final values to contiguous CPU storage."""
        if self.final_prev_values is not None:
            object.__setattr__(
                self,
                "final_prev_values",
                self.final_prev_values.cpu().contiguous(),
            )


TrajectoryData: TypeAlias = TrajectorySegment | TrajectoryEnd | TrajectoryEpochEnd


def split_batch_value(value: Any, split_sizes: list[int]) -> list[Any]:
    """Split a recursively nested batch on its leading dimension."""
    if value is None:
        return [None] * len(split_sizes)
    if isinstance(value, torch.Tensor):
        return [chunk.contiguous() for chunk in torch.split(value, split_sizes, dim=0)]
    if isinstance(value, np.ndarray):
        return list(np.split(value, np.cumsum(split_sizes)[:-1], axis=0))
    if isinstance(value, dict):
        chunks = [{} for _ in split_sizes]
        for key, item in value.items():
            for chunk, split_item in zip(chunks, split_batch_value(item, split_sizes)):
                chunk[key] = split_item
        return chunks
    if isinstance(value, list):
        offset = 0
        chunks = []
        for size in split_sizes:
            chunks.append(value[offset : offset + size])
            offset += size
        return chunks
    if isinstance(value, (bool, float, int, str)):
        return [value] * len(split_sizes)
    raise TypeError(f"Unsupported batch value: {type(value)}")


def merge_batch_values(values: list[Any]) -> Any:
    """Merge recursively nested batches on their leading dimension."""
    if not values:
        raise ValueError("Cannot merge an empty list of batch values.")
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Cannot merge present and absent batch values.")

    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(values, axis=0)
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values[1:]):
            raise ValueError("Cannot merge batch dictionaries with different keys.")
        return {
            key: merge_batch_values([value[key] for value in values]) for key in first
        }
    if isinstance(first, list):
        return [item for value in values for item in value]
    if isinstance(first, (bool, float, int, str)):
        return values
    raise TypeError(f"Unsupported batch value: {type(first)}")


def split_episode_data(
    data: dict[str, Any] | None, split_sizes: list[int]
) -> list[dict[str, Any] | None]:
    """Split online LeRobot chunk data without splitting its time dimension."""
    if data is None:
        return [None] * len(split_sizes)

    def split_steps(values: list[Any]) -> list[list[Any]]:
        chunks = [[] for _ in split_sizes]
        for value in values:
            for chunk, split_item in zip(chunks, split_batch_value(value, split_sizes)):
                chunk.append(split_item)
        return chunks

    return [
        {
            "chunk_actions": chunk_actions,
            "obs_list": obs_list,
            "terminations": terminations,
            "truncations": truncations,
            "infos_list": infos_list,
        }
        for chunk_actions, obs_list, terminations, truncations, infos_list in zip(
            split_batch_value(data["chunk_actions"], split_sizes),
            split_steps(data["obs_list"]),
            split_batch_value(data["terminations"], split_sizes),
            split_batch_value(data["truncations"], split_sizes),
            split_steps(data["infos_list"]),
        )
    ]


def merge_episode_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge online LeRobot chunk data without merging its time dimension."""
    if not data:
        raise ValueError("Cannot merge an empty list of episode data.")

    def merge_steps(values: list[list[Any]]) -> list[Any]:
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            raise ValueError("Cannot merge episode data with different chunk lengths.")
        return [merge_batch_values(list(items)) for items in zip(*values)]

    return {
        "chunk_actions": merge_batch_values([value["chunk_actions"] for value in data]),
        "obs_list": merge_steps([value["obs_list"] for value in data]),
        "terminations": merge_batch_values([value["terminations"] for value in data]),
        "truncations": merge_batch_values([value["truncations"] for value in data]),
        "infos_list": merge_steps([value["infos_list"] for value in data]),
    }


def split_policy_input(
    policy_input: PolicyInput, split_sizes: list[int]
) -> list[PolicyInput]:
    """Split a policy input on its batch dimension."""

    def split_sources() -> list[list[tuple[int, int, int]]]:
        if not policy_input.sources:
            return [[] for _ in split_sizes]
        result = [[] for _ in split_sizes]
        source_index = 0
        source_offset = 0
        for result_index, split_size in enumerate(split_sizes):
            remaining = split_size
            while remaining:
                rank, stage, source_size = policy_input.sources[source_index]
                take_size = min(remaining, source_size - source_offset)
                result[result_index].append((rank, stage, take_size))
                source_offset += take_size
                remaining -= take_size
                if source_offset == source_size:
                    source_index += 1
                    source_offset = 0
        return result

    result_fields = {
        field_name: split_batch_value(
            getattr(policy_input.env_result, field_name), split_sizes
        )
        for field_name in policy_input.env_result.__dataclass_fields__
        if field_name != "episode_data"
    }
    source_splits = split_sources()
    episode_splits = split_episode_data(
        policy_input.env_result.episode_data, split_sizes
    )
    return [
        PolicyInput(
            obs=obs,
            env_result=EnvResult(
                **{
                    field_name: field_values[index]
                    for field_name, field_values in result_fields.items()
                }
                | {"episode_data": episode_splits[index]}
            ),
            is_last=policy_input.is_last,
            sources=source_splits[index],
        )
        for index, obs in enumerate(split_batch_value(policy_input.obs, split_sizes))
    ]


def merge_policy_inputs(policy_inputs: list[PolicyInput]) -> PolicyInput:
    """Merge routed policy inputs in source order."""
    if not policy_inputs:
        raise ValueError("Cannot merge an empty list of policy inputs.")

    def get_batch_size(obs: dict[str, Any]) -> int:
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from policy input observations.")

    observations = [policy_input.obs for policy_input in policy_inputs]
    results = [policy_input.env_result for policy_input in policy_inputs]

    def merge_optional_tensor(
        field_name: str,
        *,
        fill_value: float | bool | None = None,
    ) -> torch.Tensor | None:
        values = [getattr(result, field_name) for result in results]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            if fill_value is None:
                raise ValueError(f"Inconsistent policy result field: {field_name}.")
            reference = next(value for value in values if value is not None)
            values = [
                value
                if value is not None
                else torch.full(
                    (get_batch_size(obs), *reference.shape[1:]),
                    fill_value,
                    dtype=reference.dtype,
                )
                for obs, value in zip(observations, values)
            ]
        return merge_batch_values(values)

    final_observations = [result.final_obs for result in results]
    merged_final_obs = None
    if any(obs is not None for obs in final_observations):
        merged_final_obs = merge_batch_values(
            [
                final_obs if final_obs is not None else obs
                for obs, final_obs in zip(observations, final_observations)
            ]
        )

    is_last = policy_inputs[0].is_last
    if any(policy_input.is_last != is_last for policy_input in policy_inputs[1:]):
        raise ValueError("Cannot merge final and non-final policy inputs.")

    sources: list[tuple[int, int, int]] = []
    for policy_input in policy_inputs:
        for rank, stage, size in policy_input.sources:
            if sources and sources[-1][:2] == (rank, stage):
                previous_rank, previous_stage, previous_size = sources[-1]
                sources[-1] = (
                    previous_rank,
                    previous_stage,
                    previous_size + size,
                )
            else:
                sources.append((rank, stage, size))

    episode_data_list = [result.episode_data for result in results]
    if any(data is not None for data in episode_data_list):
        if any(data is None for data in episode_data_list):
            raise ValueError("Inconsistent policy result field: episode_data.")

        episode_data = merge_episode_data(episode_data_list)
    else:
        episode_data = None

    return PolicyInput(
        obs=merge_batch_values(observations),
        env_result=EnvResult(
            rewards=merge_optional_tensor("rewards"),
            dones=merge_optional_tensor("dones"),
            terminations=merge_optional_tensor("terminations"),
            truncations=merge_optional_tensor("truncations"),
            final_obs=merged_final_obs,
            intervene_actions=merge_optional_tensor(
                "intervene_actions", fill_value=0.0
            ),
            intervene_flags=merge_optional_tensor("intervene_flags", fill_value=False),
            rlt_switch_flags=merge_optional_tensor(
                "rlt_switch_flags", fill_value=False
            ),
            reward_model_output=merge_optional_tensor("reward_model_output"),
            reward_assign_lengths=merge_batch_values(
                [result.reward_assign_lengths for result in results]
            )
            if any(result.reward_assign_lengths is not None for result in results)
            else None,
            episode_data=episode_data,
        ),
        is_last=is_last,
        sources=sources,
    )

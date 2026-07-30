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

import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import (
    EmbodiedLerobotRolloutResult,
    EnvResult,
    LeRobotStepResult,
    RewardMode,
    RewardResult,
    RolloutResult,
    Trajectory,
    TrajectoryData,
    TrajectoryRecord,
    ValueResult,
    get_model_weights_id,
)
from rlinf.scheduler.channel.trajectory_channel.owner_key import (
    BatchKey,
    LeRobotOwnerKey,
    OwnerKey,
)

StorageOutput = TypeVar("StorageOutput", bound=TrajectoryData, covariant=True)
LeRobotFrame: TypeAlias = dict[str, Any]
LeRobotEpisode: TypeAlias = list[LeRobotFrame]
TensorTree: TypeAlias = torch.Tensor | dict[str, "TensorTree"]


@dataclass(kw_only=True)
class LeRobotEpisodeBatch(TrajectoryData):
    """Completed actor-local LeRobot episodes ready for ingestion."""

    global_step: int
    actor_rank: int
    episodes: list[LeRobotEpisode]

    @classmethod
    def from_episodes(
        cls,
        *,
        global_step: int,
        actor_rank: int,
        episodes: list[LeRobotEpisode],
    ) -> "LeRobotEpisodeBatch":
        """Convert NumPy episode leaves into transportable tensors."""
        return cls(
            global_step=global_step,
            actor_rank=actor_rank,
            episodes=cast(
                list[LeRobotEpisode], _map_episode_values(episodes, _to_tensor)
            ),
        )

    def to_numpy_episodes(self) -> list[LeRobotEpisode]:
        """Restore the NumPy episode representation consumed by LeRobot."""
        return cast(
            list[LeRobotEpisode],
            _map_episode_values(self.episodes, _to_numpy),
        )


def _map_episode_values(value: Any, convert: Callable[[Any], Any]) -> Any:
    if isinstance(value, dict):
        return {key: _map_episode_values(item, convert) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_episode_values(item, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_episode_values(item, convert) for item in value)
    return convert(value)


def _to_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return value


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


@dataclass(frozen=True, kw_only=True)
class LeRobotStorageContext:
    """Configuration required to collect LeRobot episodes."""

    only_success: bool
    num_action_chunks: int
    action_dim: int
    rollout_epochs: int
    chunk_steps: int
    slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class _LeRobotStreamKey:
    env_rank: int
    actor_rank: int
    pipeline_stage: int


@dataclass(frozen=True)
class _LeRobotRolloutInput:
    forward_inputs: dict[str, torch.Tensor]
    intervene_flags: torch.Tensor | None


@dataclass(frozen=True, kw_only=True)
class TrajectoryBatchContext:
    """Static dimensions and optional streams of a trajectory batch."""

    rollout_epochs: int
    chunk_steps: int
    slot_ids: tuple[int, ...]
    reward_mode: RewardMode | None = None
    reward_steps: tuple[int, ...] = ()
    collect_values: bool = False
    bootstrap_on_termination: bool = False

    def __post_init__(self) -> None:
        """Validate batch dimensions."""
        if self.rollout_epochs < 1 or self.chunk_steps < 1:
            raise ValueError("rollout_epochs and chunk_steps must be positive.")
        if not self.slot_ids or len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("slot_ids must be non-empty and unique.")
        if self.reward_mode is None and self.reward_steps:
            raise ValueError("reward_steps requires reward_mode to be configured.")
        if self.reward_mode in ("per_step", "history_buffer") and not self.reward_steps:
            raise ValueError(f"{self.reward_mode} requires reward_steps.")
        if self.reward_mode == "terminal" and self.reward_steps:
            raise ValueError("terminal reward must use dynamic expectations.")
        if any(not 0 <= step < self.chunk_steps for step in self.reward_steps):
            raise ValueError("reward_steps contains an invalid chunk step.")


@dataclass
class RecordProgress:
    """Track received and expected slot positions for one record type."""

    expected: int = 0
    received: int = 0
    expectation_complete: bool = True

    @property
    def complete(self) -> bool:
        """Whether the record stream is complete."""
        return self.expectation_complete and self.received == self.expected

    def receive(self, count: int) -> None:
        """Account for newly written positions."""
        self.received += count
        if self.received > self.expected and self.expectation_complete:
            raise ValueError("Received more record positions than expected.")


ProgressFactory: TypeAlias = Callable[
    [TrajectoryBatchContext], dict[type[TrajectoryRecord], RecordProgress]
]
_PROGRESS_FACTORIES: dict[str, ProgressFactory] = {}


def register_progress_factory(
    *algorithm_names: str,
) -> Callable[[ProgressFactory], ProgressFactory]:
    """Register one progress definition for one or more algorithms."""

    def register(factory: ProgressFactory) -> ProgressFactory:
        for algorithm_name in algorithm_names:
            key = algorithm_name.lower()
            if key in _PROGRESS_FACTORIES:
                raise ValueError(f"Progress factory {key!r} is already registered.")
            _PROGRESS_FACTORIES[key] = factory
        return factory

    return register


def get_progress_factory(algorithm_name: str) -> ProgressFactory:
    """Return the progress factory registered for an algorithm."""
    try:
        return _PROGRESS_FACTORIES[algorithm_name.lower()]
    except KeyError as error:
        raise ValueError(
            f"No trajectory progress factory registered for {algorithm_name!r}."
        ) from error


@register_progress_factory("ppo", "grpo", "sac", "dagger", "dsrl")
def create_embodied_progress(
    context: TrajectoryBatchContext,
) -> dict[type[TrajectoryRecord], RecordProgress]:
    """Create progress counters for the shared embodied rollout format."""
    slots = len(context.slot_ids)
    step_positions = context.rollout_epochs * context.chunk_steps * slots
    progress = {
        EnvResult: RecordProgress(expected=step_positions),
        RolloutResult: RecordProgress(expected=step_positions),
    }
    if context.reward_mode is not None:
        if context.reward_mode == "terminal":
            progress[RewardResult] = RecordProgress(expectation_complete=False)
        else:
            progress[RewardResult] = RecordProgress(
                expected=context.rollout_epochs * len(context.reward_steps) * slots
            )
    if context.collect_values:
        progress[ValueResult] = RecordProgress(expectation_complete=False)
    return progress


@dataclass(kw_only=True)
class TrajectoryBatch(TrajectoryData):
    """A trajectory batch populated directly from trajectory records."""

    global_step: int
    actor_rank: int
    slot_ids: tuple[int, ...]
    reward_mode: RewardMode | None = None
    env_rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    terminations: torch.Tensor | None = None
    truncations: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    observations: TensorTree | None = None
    next_observations: TensorTree | None = None
    intervene_actions: torch.Tensor | None = None
    env_intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None
    forward_inputs: TensorTree | None = None
    prev_logprobs: torch.Tensor | None = None
    state_values: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    rollout_intervene_flags: torch.Tensor | None = None
    external_rewards: torch.Tensor | None = None
    reward_mask: torch.Tensor | None = None
    reward_history_lengths: torch.Tensor | None = None
    truncation_values: torch.Tensor | None = None
    truncation_mask: torch.Tensor | None = None
    boundary_values: torch.Tensor | None = None
    boundary_mask: torch.Tensor | None = None
    truncation_versions: torch.Tensor | None = None
    boundary_versions: torch.Tensor | None = None
    _context: TrajectoryBatchContext = field(
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    _progress: dict[type[TrajectoryRecord], RecordProgress] = field(
        init=False,
        repr=False,
        metadata={"serialize": False},
    )

    @classmethod
    def create(
        cls,
        global_step: int,
        actor_rank: int,
        context: TrajectoryBatchContext,
        progress_factory: ProgressFactory,
    ) -> "TrajectoryBatch":
        """Create an empty trajectory batch."""
        batch = cls(
            global_step=global_step,
            actor_rank=actor_rank,
            slot_ids=context.slot_ids,
            reward_mode=context.reward_mode,
        )
        batch._context = context
        batch._progress = progress_factory(context)
        return batch

    @property
    def complete(self) -> bool:
        """Whether every enabled record stream is complete."""
        return all(item.complete for item in self._progress.values())

    def record(self, record: TrajectoryRecord) -> None:
        """Write one trajectory record into its batch positions."""
        if record.global_step != self.global_step:
            raise ValueError("Record global_step does not match this batch.")
        if record.actor_rank != self.actor_rank:
            raise ValueError("Record partition does not match this batch.")
        if type(record) not in self._progress:
            raise TypeError(f"Unexpected record type {type(record).__name__}.")

        if isinstance(record, EnvResult):
            self.record_env_result(record)
        elif isinstance(record, RolloutResult):
            self.record_rollout_result(record)
        elif isinstance(record, RewardResult):
            self.record_reward_result(record)
        elif isinstance(record, ValueResult):
            self.record_value_result(record)
        else:
            raise TypeError(f"Unsupported record type {type(record).__name__}.")

        self._progress[type(record)].receive(record.batch_size)
        if isinstance(record, EnvResult):
            self._update_reward_expectation(record)
            self._update_value_expectation(record)

    def record_env_result(self, record: EnvResult) -> None:
        """Write environment outputs."""
        index = self._step_index(record)
        self.env_rewards = self._write(self.env_rewards, record.rewards, index)
        self.dones = self._write(self.dones, record.dones, index)
        self.terminations = self._write(self.terminations, record.terminations, index)
        self.truncations = self._write(self.truncations, record.truncations, index)
        self.observations = self._write(self.observations, record.observations, index)
        self.next_observations = self._write(
            self.next_observations, record.next_observations, index
        )
        self.intervene_actions = self._write(
            self.intervene_actions, record.intervene_actions, index
        )
        self.env_intervene_flags = self._write(
            self.env_intervene_flags, record.intervene_flags, index
        )
        self.rlt_switch_flags = self._write(
            self.rlt_switch_flags, record.rlt_switch_flags, index
        )

    def record_rollout_result(self, record: RolloutResult) -> None:
        """Write rollout outputs."""
        index = self._step_index(record)
        training_actions = (
            record.forward_inputs["action"]
            if record.forward_inputs is not None and "action" in record.forward_inputs
            else record.actions
        )
        self.actions = self._write(self.actions, training_actions, index)
        self.forward_inputs = self._write(
            self.forward_inputs, record.forward_inputs, index
        )
        self.prev_logprobs = self._write(
            self.prev_logprobs, record.prev_logprobs, index
        )
        self.state_values = self._write(self.state_values, record.state_values, index)
        self.versions = self._write(self.versions, record.versions, index)
        self.rollout_intervene_flags = self._write(
            self.rollout_intervene_flags, record.intervene_flags, index
        )

    def record_reward_result(self, record: RewardResult) -> None:
        """Write externally computed rewards."""
        if record.mode != self._context.reward_mode:
            raise ValueError("RewardResult does not match this batch.")
        if (
            record.mode != "terminal"
            and record.chunk_step not in self._context.reward_steps
        ):
            raise ValueError("RewardResult does not match this batch.")
        index = self._step_index(record)
        self.external_rewards = self._write(
            self.external_rewards, record.rewards, index
        )
        self.reward_mask = self._mark(self.reward_mask, index)
        self.reward_history_lengths = self._write(
            self.reward_history_lengths, record.history_lengths, index
        )

    def _update_reward_expectation(self, record: EnvResult) -> None:
        progress = self._progress.get(RewardResult)
        if progress is None or self._context.reward_mode != "terminal":
            return
        progress.expected += int(
            record.dones.reshape(record.batch_size, -1).any(dim=1).sum().item()
        )
        if self._progress[EnvResult].complete:
            progress.expectation_complete = True
            if progress.received > progress.expected:
                raise ValueError("Received more RewardResult positions than expected.")

    def record_value_result(self, record: ValueResult) -> None:
        """Write sparse truncation or rollout-boundary values."""
        if record.kind == "truncation":
            index = self._step_index(record)
            self.truncation_values = self._write(
                self.truncation_values, record.values, index
            )
            self.truncation_mask = self._mark(self.truncation_mask, index)
            self.truncation_versions = self._write(
                self.truncation_versions, record.versions, index
            )
            return
        if record.chunk_step != self._context.chunk_steps:
            raise ValueError("Boundary ValueResult has an invalid chunk_step.")
        index = (record.rollout_epoch, self._slot_indices(record.slot_ids))
        prefix = (self._context.rollout_epochs, len(self.slot_ids))
        self.boundary_values = self._write(
            self.boundary_values, record.values, index, prefix
        )
        self.boundary_mask = self._mark(self.boundary_mask, index, prefix)
        self.boundary_versions = self._write(
            self.boundary_versions, record.versions, index, prefix
        )

    def to_training_batch(self, cfg: DictConfig) -> dict[str, Any]:
        """Convert the storage layout to the legacy embodied training layout."""
        if self.env_rewards is None:
            raise RuntimeError("TrajectoryBatch has no environment rewards.")

        rewards = self.env_rewards.clone()
        if self.external_rewards is not None:
            rewards = self._merge_external_rewards(rewards, cfg)

        if self.truncation_values is not None:
            if self.truncation_mask is None:
                raise RuntimeError("Truncation values require a truncation mask.")
            bonus = self.truncation_values.to(rewards.dtype)
            mask = self.truncation_mask
            while mask.ndim < bonus.ndim:
                mask = mask.unsqueeze(-1)
            bonus = bonus * mask
            bonus = bonus.reshape(*bonus.shape[:3], -1)[..., 0]
            if rewards.ndim == 3:
                rewards += float(cfg.algorithm.gamma) * bonus
            else:
                rewards[..., -1] += float(cfg.algorithm.gamma) * bonus

        batch: dict[str, Any] = {
            "rewards": self._flatten_steps(rewards),
            "dones": self._flatten_boundaries(self.dones),
            "terminations": self._flatten_boundaries(self.terminations),
            "truncations": self._flatten_boundaries(self.truncations),
        }
        fields = {
            "actions": self.actions,
            "curr_obs": self.observations,
            "next_obs": self.next_observations,
            "intervene_flags": self.rollout_intervene_flags,
            "forward_inputs": self.forward_inputs,
            "prev_logprobs": self.prev_logprobs,
            "versions": self.versions,
        }
        for name, value in fields.items():
            if value is not None:
                batch[name] = self._flatten_steps(value)

        if self.state_values is not None:
            batch["prev_values"] = self._flatten_values()
        return batch

    def _merge_external_rewards(
        self, rewards: torch.Tensor, cfg: DictConfig
    ) -> torch.Tensor:
        if self.reward_mask is None:
            raise RuntimeError("External rewards require a reward mask.")

        env_weight = float(cfg.reward.get("env_reward_weight", 0.0))
        reward_weight = float(cfg.reward.get("reward_weight", 1.0))
        external = self.external_rewards.to(rewards.dtype)
        if self.reward_mode == "terminal":
            if self.dones is None:
                raise RuntimeError("Terminal rewards require episode boundaries.")
            terminal = self.dones.bool()
            if rewards.ndim == self.reward_mask.ndim:
                combined = env_weight * rewards + reward_weight * external
                return torch.where(
                    self.reward_mask,
                    combined,
                    rewards,
                )

            terminal = terminal.reshape(*terminal.shape[:3], -1)
            valid = self.reward_mask & terminal.any(dim=-1)
            epoch, step, slot = valid.nonzero(as_tuple=True)
            terminal_step = terminal.to(torch.int64).argmax(dim=-1)[valid]
            external = external.reshape(*external.shape[:3], -1)[..., 0]
            rewards[valid] *= env_weight
            rewards[epoch, step, slot, terminal_step] += reward_weight * external[valid]
            return rewards

        mask = self.reward_mask
        while mask.ndim < rewards.ndim:
            mask = mask.unsqueeze(-1)
        while external.ndim < rewards.ndim:
            external = external.unsqueeze(-1)
        combined = env_weight * rewards + reward_weight * external
        rewards = torch.where(mask, combined, rewards)
        if self.reward_mode == "history_buffer" and cfg.reward.get(
            "history_reward_assign", False
        ):
            self._assign_history_rewards(rewards, reward_weight)
        return rewards

    def _assign_history_rewards(
        self, rewards: torch.Tensor, reward_weight: float
    ) -> None:
        if self.reward_history_lengths is None:
            raise RuntimeError("History reward assignment requires history lengths.")

        num_steps = rewards.shape[0] * rewards.shape[1]
        rewards = rewards.reshape(num_steps, len(self.slot_ids), *rewards.shape[3:])
        external = self.external_rewards.reshape(
            num_steps,
            len(self.slot_ids),
            *self.external_rewards.shape[3:],
        )
        lengths = self.reward_history_lengths.reshape(
            num_steps, len(self.slot_ids), -1
        )[..., 0]
        mask = self.reward_mask.reshape(num_steps, len(self.slot_ids))

        for step, slot in mask.nonzero().tolist():
            history_length = min(int(lengths[step, slot]), step + 1)
            if history_length > 1:
                rewards[step - history_length + 1 : step, slot] += (
                    reward_weight * external[step, slot]
                )

    def to_trajectory(self, cfg: DictConfig) -> Trajectory:
        """Convert the storage layout to the legacy replay-buffer input."""
        batch = self.to_training_batch(cfg)
        trajectory = Trajectory(
            max_episode_length=int(cfg.env.train.max_episode_steps),
            **batch,
        )
        versions = trajectory.versions
        if versions is None:
            versions = torch.zeros(1, dtype=torch.float32)
        trajectory.model_weights_id = get_model_weights_id(versions)
        return trajectory

    def _flatten_steps(self, value: TensorTree) -> TensorTree:
        if isinstance(value, torch.Tensor):
            return value.reshape(-1, *value.shape[2:])
        return {key: self._flatten_steps(item) for key, item in value.items()}

    def _flatten_boundaries(self, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise RuntimeError("TrajectoryBatch is missing episode boundaries.")
        leading = torch.zeros_like(value[:, :1])
        value = torch.cat((leading, value), dim=1)
        return value.reshape(-1, *value.shape[2:])

    def _flatten_values(self) -> torch.Tensor:
        values = cast(torch.Tensor, self.state_values)
        boundary = self.boundary_values
        if boundary is None:
            boundary = torch.zeros_like(values[:, 0])
        elif boundary.shape != values[:, 0, :, :1].shape:
            raise RuntimeError(
                "Boundary value layout does not match state values: "
                f"state_values={tuple(values.shape)}, "
                f"boundary_values={tuple(boundary.shape)}."
            )
        if boundary.shape != values[:, 0].shape:
            boundary_chunk = torch.zeros_like(values[:, 0])
            boundary_chunk[..., :1] = boundary
            boundary = boundary_chunk
        values = torch.cat((values, boundary.unsqueeze(1)), dim=1)
        return values.reshape(-1, *values.shape[2:])

    def _update_value_expectation(self, record: EnvResult) -> None:
        progress = self._progress.get(ValueResult)
        if progress is None:
            return
        truncated = (
            record.dones[..., -1]
            if self._context.bootstrap_on_termination
            else record.truncations[..., -1] & ~record.terminations[..., -1]
        )
        progress.expected += int(
            truncated.reshape(record.batch_size, -1).any(dim=1).sum().item()
        )
        if record.chunk_step == self._context.chunk_steps - 1:
            done_slots = int(
                record.dones[..., -1]
                .reshape(record.batch_size, -1)
                .any(dim=1)
                .sum()
                .item()
            )
            progress.expected += record.batch_size - done_slots
        if self._progress[EnvResult].complete:
            progress.expectation_complete = True
            if progress.received > progress.expected:
                raise ValueError("Received more ValueResult positions than expected.")

    def _step_index(self, record: TrajectoryRecord) -> tuple[int, int, torch.Tensor]:
        if not 0 <= record.rollout_epoch < self._context.rollout_epochs:
            raise ValueError("rollout_epoch is outside this batch.")
        if not 0 <= record.chunk_step < self._context.chunk_steps:
            raise ValueError("chunk_step is outside this batch.")
        return (
            record.rollout_epoch,
            record.chunk_step,
            self._slot_indices(record.slot_ids),
        )

    def _slot_indices(self, slot_ids: tuple[int, ...]) -> torch.Tensor:
        try:
            slot_index = {slot: index for index, slot in enumerate(self.slot_ids)}
            return torch.tensor([slot_index[slot] for slot in slot_ids])
        except KeyError as error:
            raise ValueError(f"Unknown slot {error.args[0]}.") from error

    def _write(
        self,
        target: TensorTree | None,
        value: Any | None,
        index: tuple[Any, ...],
        prefix: tuple[int, ...] | None = None,
    ) -> TensorTree | None:
        if value is None:
            return target
        prefix = prefix or (
            self._context.rollout_epochs,
            self._context.chunk_steps,
            len(self.slot_ids),
        )
        if isinstance(value, torch.Tensor):
            if target is None:
                target = value.new_zeros((*prefix, *value.shape[1:]))
            tensor = cast(torch.Tensor, target)
            tensor[index] = value
            return tensor
        if isinstance(value, Mapping):
            values = cast(Mapping[str, Any], value)
            targets = {} if target is None else cast(dict[str, TensorTree], target)
            if target is not None and targets.keys() != values.keys():
                raise ValueError("Dynamic tensor keys changed within one batch.")
            return {
                key: cast(
                    TensorTree, self._write(targets.get(key), item, index, prefix)
                )
                for key, item in values.items()
            }
        raise TypeError(f"Cannot write value of type {type(value).__name__}.")

    def _mark(
        self,
        target: torch.Tensor | None,
        index: tuple[Any, ...],
        prefix: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if target is None:
            prefix = prefix or (
                self._context.rollout_epochs,
                self._context.chunk_steps,
                len(self.slot_ids),
            )
            target = torch.zeros(prefix, dtype=torch.bool)
        target[index] = True
        return target

    def _seal(self) -> None:
        del self._context
        del self._progress


@dataclass(frozen=True)
class _StorageFailure:
    error: BaseException


class TrajectoryStorage(ABC, Generic[StorageOutput]):
    """Record owner-ordered trajectory data on a background thread pool."""

    def __init__(
        self,
        max_queue_size: int,
        num_record_threads: int,
    ) -> None:
        """Initialize record queues and the background writer pool."""
        if num_record_threads < 1:
            raise ValueError("num_record_threads must be positive.")
        self._max_queue_size = max_queue_size
        self._executor = ThreadPoolExecutor(
            max_workers=num_record_threads, thread_name_prefix="trajectory-record"
        )
        self._record_queues: dict[OwnerKey, asyncio.Queue[TrajectoryRecord]] = {}
        self._record_tasks: dict[OwnerKey, asyncio.Task[None]] = {}
        self._ready: asyncio.Queue[StorageOutput | _StorageFailure] = asyncio.Queue(
            max_queue_size
        )
        self._failure: BaseException | None = None

    async def _submit(self, owner_key: OwnerKey, data: TrajectoryRecord) -> None:
        if self._failure is not None:
            raise RuntimeError("Trajectory storage background writer failed.") from (
                self._failure
            )
        queue = self._record_queues.get(owner_key)
        if queue is None:
            queue = asyncio.Queue(self._max_queue_size)
            self._record_queues[owner_key] = queue
            self._record_tasks[owner_key] = asyncio.create_task(
                self._write_records(owner_key, queue)
            )
        await queue.put(data)

    async def _write_records(
        self,
        owner_key: OwnerKey,
        queue: asyncio.Queue[TrajectoryRecord],
    ) -> None:
        try:
            while True:
                data = await queue.get()
                try:
                    output = await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._record_sync,
                        owner_key,
                        data,
                    )
                    if output is not None:
                        await self._ready.put(output)
                finally:
                    queue.task_done()
                if queue.empty():
                    return
        except Exception as error:
            self._failure = error
            await self._ready.put(_StorageFailure(error))
        finally:
            self._record_tasks.pop(owner_key, None)
            self._record_queues.pop(owner_key, None)

    @abstractmethod
    async def record(self, data: TrajectoryRecord, owner_key: OwnerKey) -> None:
        """Enqueue trajectory data for recording."""

    @abstractmethod
    def _record_sync(
        self, owner_key: OwnerKey, data: TrajectoryRecord
    ) -> StorageOutput | None:
        """Write one record from a record thread."""

    async def take(self) -> StorageOutput:
        """Take the next completed output or raise a writer failure."""
        output = await self._ready.get()
        if isinstance(output, _StorageFailure):
            raise output.error
        return output


class RolloutTrajectoryStorage(TrajectoryStorage[TrajectoryBatch]):
    """Store embodied rollout records directly in trajectory batches."""

    def __init__(
        self,
        algorithm_name: str,
        context: TrajectoryBatchContext,
        max_queue_size: int = 0,
        num_record_threads: int = 4,
    ) -> None:
        """Initialize rollout storage."""
        super().__init__(max_queue_size, num_record_threads)
        self._context = context
        self._progress_factory = get_progress_factory(algorithm_name)
        self._batches: dict[BatchKey, TrajectoryBatch] = {}
        self._batches_lock = threading.Lock()

    async def record(self, data: TrajectoryRecord, owner_key: OwnerKey) -> None:
        """Enqueue a rollout record for background writing."""
        if not isinstance(owner_key, BatchKey):
            raise TypeError("Rollout storage requires a BatchKey.")
        await self._submit(owner_key, data)

    def _record_sync(
        self, owner_key: OwnerKey, data: TrajectoryRecord
    ) -> TrajectoryBatch | None:
        if not isinstance(owner_key, BatchKey):
            raise TypeError("Rollout storage requires a BatchKey.")
        with self._batches_lock:
            batch = self._batches.get(owner_key)
            if batch is None:
                batch = TrajectoryBatch.create(
                    data.global_step,
                    data.actor_rank,
                    self._context,
                    self._progress_factory,
                )
                self._batches[owner_key] = batch
        batch.record(data)
        if batch.complete:
            with self._batches_lock:
                del self._batches[owner_key]
            batch._seal()
            return batch
        return None


class LeRobotTrajectoryStorage(TrajectoryStorage[LeRobotEpisodeBatch]):
    """Collect long-lived LeRobot streams into completed episodes."""

    def __init__(
        self,
        context: LeRobotStorageContext,
        max_queue_size: int = 0,
        num_record_threads: int = 4,
    ) -> None:
        """Initialize LeRobot storage."""
        super().__init__(max_queue_size, num_record_threads)
        self._context = context
        self._collectors: dict[_LeRobotStreamKey, EmbodiedLerobotRolloutResult] = {}
        self._pending_episodes: dict[BatchKey, list[LeRobotEpisode]] = {}
        self._boundary_slots: dict[BatchKey, set[int]] = {}
        self._collectors_lock = threading.Lock()

    async def record(
        self,
        data: TrajectoryRecord,
        owner_key: OwnerKey,
    ) -> None:
        """Enqueue one raw LeRobot chunk for background writing."""
        if not isinstance(data, LeRobotStepResult) or not isinstance(
            owner_key, LeRobotOwnerKey
        ):
            raise TypeError("LeRobot storage requires an actor owner and step result.")
        await self._submit(owner_key, data)

    def _record_sync(
        self, owner_key: OwnerKey, data: TrajectoryRecord
    ) -> LeRobotEpisodeBatch | None:
        if not isinstance(owner_key, LeRobotOwnerKey) or not isinstance(
            data, LeRobotStepResult
        ):
            raise TypeError("LeRobot storage requires an actor owner and step result.")
        stream_key = _LeRobotStreamKey(
            env_rank=data.env_rank,
            actor_rank=data.actor_rank,
            pipeline_stage=data.pipeline_stage,
        )
        with self._collectors_lock:
            collector = self._collectors.get(stream_key)
            if collector is None:
                collector = EmbodiedLerobotRolloutResult(
                    num_envs=data.batch_size,
                    only_success=self._context.only_success,
                    num_action_chunks=self._context.num_action_chunks,
                    action_dim=self._context.action_dim,
                )
                self._collectors[stream_key] = collector
        if collector.num_envs != data.batch_size:
            raise ValueError("LeRobot stream batch size changed.")

        collector.append_chunk_episode_data(
            rollout_result=_LeRobotRolloutInput(
                forward_inputs={"action": data.expert_actions}
                if data.expert_actions is not None
                else {},
                intervene_flags=data.intervene_flags,
            ),
            chunk_actions=data.chunk_actions,
            obs_list=data.observations,
            terminations=data.terminations,
            truncations=data.truncations,
            infos_list=data.env_infos,
        )
        episodes = collector.drain_episodes()
        batch_key = BatchKey(
            global_step=data.global_step,
            actor_rank=data.actor_rank,
        )
        pending = self._pending_episodes.setdefault(batch_key, [])
        pending.extend(episodes)
        rollout_complete = (
            data.rollout_epoch + 1 == self._context.rollout_epochs
            and data.chunk_step + 1 == self._context.chunk_steps
        )
        if not rollout_complete:
            return None
        boundary_slots = self._boundary_slots.setdefault(batch_key, set())
        boundary_slots.update(data.slot_ids)
        if boundary_slots != set(self._context.slot_ids):
            return None
        del self._boundary_slots[batch_key]
        completed = self._pending_episodes.pop(batch_key)
        return LeRobotEpisodeBatch.from_episodes(
            global_step=data.global_step,
            actor_rank=data.actor_rank,
            episodes=completed,
        )


def create_trajectory_storage(
    algorithm_name: str,
    cfg: DictConfig,
    actor_world_size: int,
    max_queue_size: int = 0,
    num_record_threads: int = 4,
) -> TrajectoryStorage[TrajectoryData]:
    """Create the trajectory storage selected by the training mode."""
    total_num_envs = int(cfg.env.train.total_num_envs)
    if total_num_envs % actor_world_size != 0:
        raise ValueError(
            "env.train.total_num_envs must be divisible by actor world size."
        )
    if OmegaConf.select(
        cfg,
        "algorithm.dagger.online_lerobot.enabled",
        default=False,
    ):
        return LeRobotTrajectoryStorage(
            LeRobotStorageContext(
                only_success=bool(
                    OmegaConf.select(
                        cfg,
                        "algorithm.dagger.online_lerobot.only_success",
                        default=False,
                    )
                ),
                num_action_chunks=int(cfg.actor.model.num_action_chunks),
                action_dim=int(cfg.actor.model.action_dim),
                rollout_epochs=int(cfg.env.train.rollout_epoch),
                chunk_steps=(
                    int(cfg.env.train.max_steps_per_rollout_epoch)
                    // int(cfg.actor.model.num_action_chunks)
                ),
                slot_ids=tuple(range(total_num_envs // actor_world_size)),
            ),
            max_queue_size=max_queue_size,
            num_record_threads=num_record_threads,
        )

    chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // int(
        cfg.actor.model.num_action_chunks
    )
    use_external_reward = bool(
        OmegaConf.select(cfg, "reward.use_reward_model", default=False)
    ) and not bool(OmegaConf.select(cfg, "reward.standalone_realworld", default=False))
    reward_mode = (
        OmegaConf.select(cfg, "reward.reward_mode", default="per_step")
        if use_external_reward
        else None
    )
    reward_steps = (
        tuple(range(chunk_steps))
        if reward_mode in ("per_step", "history_buffer")
        else ()
    )
    return RolloutTrajectoryStorage(
        algorithm_name,
        TrajectoryBatchContext(
            rollout_epochs=int(cfg.env.train.rollout_epoch),
            chunk_steps=chunk_steps,
            slot_ids=tuple(range(total_num_envs // actor_world_size)),
            reward_mode=reward_mode,
            reward_steps=reward_steps,
            collect_values=bool(
                OmegaConf.select(cfg, "actor.model.add_value_head", default=False)
            ),
            bootstrap_on_termination=(
                OmegaConf.select(cfg, "algorithm.bootstrap_type", default="standard")
                != "standard"
            ),
        ),
        max_queue_size=max_queue_size,
        num_record_threads=num_record_threads,
    )

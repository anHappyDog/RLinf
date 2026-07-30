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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

from rlinf.data.embodied_io_struct import (
    LeRobotStepResult,
    TrajectoryData,
    TrajectoryRecord,
)
from rlinf.scheduler.manager import WorkerAddress


@dataclass(frozen=True)
class BatchKey:
    """Identify one actor-local rollout batch."""

    global_step: int
    actor_rank: int


@dataclass(frozen=True)
class PipelineBatchKey:
    """Identify one actor-local rollout epoch in the training pipeline."""

    global_step: int
    rollout_epoch: int
    actor_rank: int


@dataclass(frozen=True)
class LeRobotOwnerKey:
    """Identify the StorageWorker that owns one actor's LeRobot streams."""

    actor_rank: int


OwnerKey: TypeAlias = BatchKey | PipelineBatchKey | LeRobotOwnerKey
OwnerKeyBuilder: TypeAlias = Callable[[TrajectoryData, WorkerAddress], OwnerKey]


def trajectory_batch_owner_key(data: TrajectoryData, source: WorkerAddress) -> BatchKey:
    """Build the owner key for an actor-local rollout batch."""
    del source
    record = cast(TrajectoryRecord, data)
    return BatchKey(
        global_step=record.global_step,
        actor_rank=record.actor_rank,
    )


def pipeline_batch_owner_key(
    data: TrajectoryData, source: WorkerAddress
) -> PipelineBatchKey:
    """Build the owner key for one actor-local rollout epoch."""
    del source
    record = cast(TrajectoryRecord, data)
    return PipelineBatchKey(
        global_step=record.global_step,
        rollout_epoch=record.rollout_epoch,
        actor_rank=record.actor_rank,
    )


def lerobot_actor_owner_key(
    data: TrajectoryData, source: WorkerAddress
) -> LeRobotOwnerKey:
    """Keep all LeRobot streams for an actor on one StorageWorker."""
    del source
    return LeRobotOwnerKey(actor_rank=cast(LeRobotStepResult, data).actor_rank)

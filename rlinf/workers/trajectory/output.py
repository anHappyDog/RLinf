# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import ray
import torch

from rlinf.data.forward_inputs import ForwardInputs, get_forward_inputs_type
from rlinf.scheduler import Worker
from rlinf.workers.trajectory.compression import (
    CompressionConfig,
    CompressionPipeline,
)
from rlinf.workers.trajectory.route_plan import RoutePlan
from rlinf.workers.trajectory.storage import (
    TrajectoryBatch,
    merge_trajectory_batches,
)

if TYPE_CHECKING:
    from rlinf.scheduler.worker.worker_group import WorkerGroup
    from rlinf.workers.trajectory.workers import WorkerLayout


def flatten_trajectory(
    batch: TrajectoryBatch,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Separate trajectory tensors from its small serializable skeleton."""
    tensors: dict[str, torch.Tensor] = {}
    paths: list[tuple[str, tuple[str | int, ...]]] = []
    forward_schemas: dict[str, tuple[str, int]] = {}
    values = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, ForwardInputs):
            forward_schemas[field.name] = (
                value.schema_name,
                value.schema_version,
            )
            value = dict(value.tensor_fields())
        values[field.name] = _flatten_value(
            value,
            (field.name,),
            tensors,
            paths,
        )
    skeleton = {
        "values": values,
        "tensor_paths": tuple(paths),
        "forward_schemas": forward_schemas,
    }
    return tensors, skeleton


def restore_trajectory(
    tensors: dict[str, torch.Tensor],
    skeleton: dict[str, Any],
) -> TrajectoryBatch:
    """Restore a trajectory sent as a tensor dict plus tensor-free skeleton."""
    paths = skeleton["tensor_paths"]
    expected_keys = {key for key, _ in paths}
    if tensors.keys() != expected_keys:
        raise ValueError("trajectory tensors do not match skeleton paths")
    by_path = {path: tensors[key] for key, path in paths}
    values = {
        name: _restore_value(value, (name,), by_path)
        for name, value in skeleton["values"].items()
    }
    for field_name, (schema_name, schema_version) in skeleton[
        "forward_schemas"
    ].items():
        forward_type = get_forward_inputs_type(schema_name, schema_version)
        values[field_name] = forward_type.from_model_inputs(values[field_name])
    return TrajectoryBatch(**values)


def _flatten_value(
    value: Any,
    path: tuple[str | int, ...],
    tensors: dict[str, torch.Tensor],
    paths: list[tuple[str, tuple[str | int, ...]]],
) -> Any:
    if isinstance(value, torch.Tensor):
        key = str(len(tensors))
        tensors[key] = value
        paths.append((key, path))
        return None
    if isinstance(value, Mapping):
        return {
            key: _flatten_value(child, (*path, key), tensors, paths)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _flatten_value(child, (*path, index), tensors, paths)
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _flatten_value(child, (*path, index), tensors, paths)
            for index, child in enumerate(value)
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported trajectory value at {path!r}: {type(value).__name__}")


def _restore_value(
    value: Any,
    path: tuple[str | int, ...],
    tensors: dict[tuple[str | int, ...], torch.Tensor],
) -> Any:
    if path in tensors:
        return tensors[path]
    if isinstance(value, dict):
        return {
            key: _restore_value(child, (*path, key), tensors)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _restore_value(child, (*path, index), tensors)
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _restore_value(child, (*path, index), tensors)
            for index, child in enumerate(value)
        )
    return value


class TrajectoryReader:
    """Actor-side façade for pulling and merging ready Storage shards."""

    def __init__(
        self,
        *,
        actors: dict[int, ray.actor.ActorHandle],
        storage_group_name: str,
        route_plan: RoutePlan,
        storage_layout: "WorkerLayout",
        actor_layout: "WorkerLayout",
        storage_participant: str = "storage",
        actor_participant: str = "actor",
        compression: CompressionConfig = CompressionConfig(),
    ) -> None:
        self._actors = actors
        self._storage_group_name = storage_group_name
        self._route_plan = route_plan
        self._storage_layout = storage_layout
        self._actor_layout = actor_layout
        self._storage_participant = storage_participant
        self._actor_participant = actor_participant
        self._compression = compression
        self._compression_pipeline: CompressionPipeline | None = None
        self._current_worker = Worker.current_worker
        expected = {
            storage_participant: len(storage_layout.data_ranks),
            actor_participant: len(actor_layout.data_ranks),
        }
        for participant, world_size in expected.items():
            if route_plan.world_sizes.get(participant) != world_size:
                raise ValueError(
                    f"{participant!r} layout does not match route world size"
                )
        if set(actors) != set(storage_layout.data_ranks):
            raise ValueError("storage actors do not match data-owning ranks")

    @classmethod
    def from_worker_group(
        cls,
        worker_group: "WorkerGroup",
        **kwargs,
    ) -> "TrajectoryReader":
        """Connect an Actor façade to an existing Storage worker group."""
        actors = {
            worker.rank: worker.worker for worker in worker_group.worker_info_list
        }
        return cls(
            actors=actors,
            storage_group_name=worker_group.worker_group_name,
            **kwargs,
        )

    def pull(self) -> TrajectoryBatch:
        """Pull all Storage contributions for the current Actor data rank."""
        worker = self._current_worker
        if worker is None:
            raise RuntimeError("TrajectoryReader.pull must run inside an Actor Worker")
        actor_rank = self._actor_layout.logical_rank(worker._rank)
        return self._pull(actor_rank, worker)

    def pull_via_ray(self, actor_rank: int) -> TrajectoryBatch:
        """Pull one explicit logical Actor rank through Ray object transfer."""
        self._route_plan.slot_range(self._actor_participant, actor_rank)
        return self._pull(actor_rank, worker=None)

    def _pull(self, actor_rank: int, worker: Worker | None) -> TrajectoryBatch:
        start, end = self._route_plan.slot_range(self._actor_participant, actor_rank)
        slot_ids = tuple(range(start, end))
        storage_ranks = tuple(
            sorted(
                {
                    self._route_plan.owner(self._storage_participant, slot_id)
                    for slot_id in slot_ids
                }
            )
        )
        batches = []
        for storage_rank in storage_ranks:
            physical_storage_rank = self._storage_layout.data_ranks[storage_rank]
            actor = self._actors[physical_storage_rank]
            if worker is None:
                batch = ray.get(
                    actor.pull_actor_shard_via_ray.remote(
                        actor_rank, self._actor_participant
                    )
                )
            else:
                pending = actor.pull_actor_shard.remote(
                    worker.worker_address,
                    actor_rank,
                    self._actor_participant,
                )
                tensors, skeleton = worker.recv(
                    self._storage_group_name, physical_storage_rank
                )
                if self._compression.enabled:
                    if self._compression_pipeline is None:
                        self._compression_pipeline = CompressionPipeline(
                            self._compression
                        )
                    tensors = self._compression_pipeline.decompress(
                        tensors, skeleton["compression"]
                    )
                batch = restore_trajectory(tensors, skeleton)
                ray.get(pending)
            batches.append(batch)
        return merge_trajectory_batches(batches, slot_ids)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._current_worker = Worker.current_worker
        self._compression_pipeline = None

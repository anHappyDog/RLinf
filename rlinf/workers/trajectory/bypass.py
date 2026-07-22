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

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import ray
import torch

from rlinf.data.forward_inputs import ForwardInputs
from rlinf.data.trajectory import EnvResult, RewardResult, RolloutResult, ValueResult
from rlinf.scheduler import Worker
from rlinf.workers.trajectory.route_plan import RoutePlan
from rlinf.workers.trajectory.transport import (
    EndpointSchema,
    PreparedSend,
    TransportAck,
    TransportEndpoint,
)

if TYPE_CHECKING:
    from rlinf.scheduler.worker.worker_group import WorkerGroup
    from rlinf.workers.trajectory.storage import TrajectoryStorage
    from rlinf.workers.trajectory.transport import ReceiveResult
    from rlinf.workers.trajectory.workers import WorkerLayout

StorageInput = EnvResult | RolloutResult | RewardResult | ValueResult

_SCHEMA_KINDS = {
    ("EnvResult", ""): 1,
    ("RolloutResult", ""): 2,
    ("RewardResult", "per_step"): 3,
    ("RewardResult", "terminal"): 4,
    ("RewardResult", "history_buffer"): 5,
    ("ValueResult", "timeout"): 6,
    ("ValueResult", "tail"): 7,
}


def endpoint_schema_id(record_type: str, discriminator: str, source_rank: int) -> int:
    """Return the stable schema identifier for one source-owned record lane."""
    try:
        kind = _SCHEMA_KINDS[(record_type, discriminator)]
    except KeyError as error:
        raise ValueError(
            f"unsupported endpoint schema {(record_type, discriminator)!r}"
        ) from error
    if source_rank < 0:
        raise ValueError("source_rank must be non-negative")
    return kind * 1_000_000 + source_rank + 1


class SubmitStatus(str, Enum):
    """Durability reached before a storage submission is acknowledged."""

    RECEIVED = "received"
    INGESTED = "ingested"


@dataclass(frozen=True)
class SubmitAck:
    """Storage acknowledgement for one fixed-frame submission."""

    status: SubmitStatus
    schema_id: int
    sequence_id: int
    inserted: bool | None
    trajectory_ready: bool | None
    backpressure_seconds: float = 0.0

    @property
    def transport_ack(self) -> TransportAck:
        """Return the wire acknowledgement used to release sender ownership."""
        return TransportAck(self.schema_id, self.sequence_id)


@dataclass(frozen=True)
class _PendingSubmission:
    destination_rank: int
    endpoint: TransportEndpoint
    prepared: PreparedSend


class TrajectoryWriter:
    """Producer-side façade that routes records directly to Storage workers."""

    def __init__(
        self,
        *,
        actors: dict[int, ray.actor.ActorHandle],
        storage_group_name: str,
        route_plan: RoutePlan,
        source_participant: str,
        source_layout: "WorkerLayout",
        storage_layout: "WorkerLayout",
        schemas_by_rank: dict[int, tuple[EndpointSchema, ...]],
        storage_participant: str = "storage",
    ) -> None:
        self._actors = actors
        self._storage_group_name = storage_group_name
        self._route_plan = route_plan
        self._source_participant = source_participant
        self._source_layout = source_layout
        self._storage_layout = storage_layout
        self._storage_participant = storage_participant
        if source_participant not in route_plan.world_sizes:
            raise ValueError(f"unknown source participant {source_participant!r}")
        if storage_participant not in route_plan.world_sizes:
            raise ValueError(f"unknown storage participant {storage_participant!r}")
        expected_ranks = set(range(route_plan.world_sizes[source_participant]))
        if set(schemas_by_rank) != expected_ranks:
            raise ValueError("schemas_by_rank must cover every source logical rank")
        self._schemas = {
            (rank, _schema_key_from_schema(schema)): schema
            for rank, schemas in schemas_by_rank.items()
            for schema in schemas
        }
        if len(self._schemas) != sum(map(len, schemas_by_rank.values())):
            raise ValueError("TrajectoryWriter endpoint schemas must be unique")
        if route_plan.world_sizes[source_participant] != len(source_layout.data_ranks):
            raise ValueError("source layout does not match route world size")
        if route_plan.world_sizes[storage_participant] != len(
            storage_layout.data_ranks
        ):
            raise ValueError("storage layout does not match route world size")
        self._send_endpoints: dict[tuple[int, int], TransportEndpoint] = {}
        self._current_worker = Worker.current_worker

    @classmethod
    def from_worker_group(
        cls,
        worker_group: "WorkerGroup",
        **kwargs,
    ) -> "TrajectoryWriter":
        """Connect a producer façade to an existing Storage worker group."""
        actors = {
            worker.rank: worker.worker for worker in worker_group.worker_info_list
        }
        return cls(
            actors=actors,
            storage_group_name=worker_group.worker_group_name,
            **kwargs,
        )

    def submit(
        self,
        data: StorageInput,
        wait_for: SubmitStatus = SubmitStatus.RECEIVED,
    ) -> tuple[SubmitAck, ...]:
        """Route one record batch and wait for the requested durability."""
        if wait_for not in (SubmitStatus.RECEIVED, SubmitStatus.INGESTED):
            raise ValueError("submit can wait for RECEIVED or INGESTED")
        worker = self._current_worker
        if worker is None:
            raise RuntimeError("TrajectoryWriter.submit must run inside a Worker")
        source_rank = self._source_layout.logical_rank(worker._rank)
        pending = self._prepare(data, source_rank)
        acknowledgements: list[SubmitAck] = []
        for submission in pending:
            actor = self._actors[submission.destination_rank]
            reservation_id = ray.get(
                actor.reserve.remote(
                    worker.worker_address.get_name(),
                    submission.endpoint.schema.schema_id,
                )
            )
            result = actor.submit.remote(
                worker.worker_address,
                submission.endpoint.schema.schema_id,
                reservation_id,
                wait_for,
            )
            for tensor in submission.prepared.buffers:
                worker.send_tensor(
                    tensor,
                    self._storage_group_name,
                    submission.destination_rank,
                )
            ack = ray.get(result)
            submission.endpoint.acknowledge(ack.transport_ack)
            acknowledgements.append(ack)
        return tuple(acknowledgements)

    def drain(self) -> None:
        """Verify that every local transport send has been acknowledged."""
        pending = sum(endpoint.in_flight for endpoint in self._send_endpoints.values())
        if pending:
            raise RuntimeError(f"cannot drain writer with {pending} in-flight frames")

    def owned_slots(self) -> tuple[int, ...]:
        """Return global slots owned by the current producer data rank."""
        worker = self._current_worker
        if worker is None:
            raise RuntimeError("owned_slots must run inside a Worker")
        source_rank = self._source_layout.logical_rank(worker._rank)
        start, end = self._route_plan.slot_range(self._source_participant, source_rank)
        return tuple(range(start, end))

    def _prepare(
        self,
        data: StorageInput,
        source_rank: int,
    ) -> tuple[_PendingSubmission, ...]:
        expected_participant = {
            EnvResult: "env",
            RolloutResult: "rollout",
            RewardResult: "reward",
            ValueResult: "rollout",
        }[type(data)]
        if self._source_participant != expected_participant:
            raise TypeError(
                f"{self._source_participant!r} writer cannot submit "
                f"{type(data).__name__}"
            )
        schema_key = _schema_key(data)
        try:
            schema = self._schemas[(source_rank, schema_key)]
        except KeyError as error:
            raise ValueError(
                f"no configured schema for source rank {source_rank} and "
                f"record {schema_key!r}"
            ) from error
        routes = self._route_plan.route_slots(
            self._source_participant,
            source_rank,
            data.slot_ids,
            self._storage_participant,
        )
        submissions = []
        for route in routes:
            destination_rank = self._storage_layout.data_ranks[route.destination_rank]
            key = (schema.schema_id, destination_rank)
            endpoint = self._send_endpoints.setdefault(key, TransportEndpoint(schema))
            shard = select_storage_data(data, route.source_indices)
            submissions.append(
                _PendingSubmission(
                    destination_rank=destination_rank,
                    endpoint=endpoint,
                    prepared=endpoint.encode(shard),
                )
            )
        return tuple(submissions)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._current_worker = Worker.current_worker


def _schema_key(data: StorageInput) -> tuple[str, str]:
    discriminator = ""
    if isinstance(data, RewardResult):
        discriminator = data.mode
    elif isinstance(data, ValueResult):
        discriminator = data.kind
    return type(data).__name__, discriminator


def _schema_key_from_schema(schema: EndpointSchema) -> tuple[str, str]:
    constants = dict(schema.constants)
    discriminator = ""
    if schema.record_type == "RewardResult":
        discriminator = constants[("mode",)]
    elif schema.record_type == "ValueResult":
        discriminator = constants[("kind",)]
    return schema.record_type, discriminator


def select_storage_data(
    data: StorageInput,
    indices: tuple[int, ...],
) -> StorageInput:
    """Select an ordered storage-record subset while preserving its schema."""
    selected: dict[str, Any] = {
        "slot_ids": tuple(data.slot_ids[index] for index in indices)
    }
    for field in fields(data):
        if field.name in {
            "global_step",
            "rollout_epoch",
            "chunk_step",
            "slot_ids",
        }:
            continue
        selected[field.name] = _select(getattr(data, field.name), indices)
    return replace(data, **selected)


def ingest_storage_data(
    storage: "TrajectoryStorage",
    received: "ReceiveResult",
) -> SubmitAck:
    """Ingest one decoded frame and produce its strongest current ack."""
    if not isinstance(
        received.data, (EnvResult, RolloutResult, RewardResult, ValueResult)
    ):
        raise TypeError(f"cannot ingest {type(received.data).__name__}")
    inserted = storage.write(received.data)
    return SubmitAck(
        status=SubmitStatus.INGESTED,
        schema_id=received.ack.schema_id,
        sequence_id=received.ack.sequence_id,
        inserted=inserted,
        trajectory_ready=storage.ready,
    )


def _select(value: Any, indices: tuple[int, ...]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, ForwardInputs):
        index = torch.tensor(indices, dtype=torch.long)
        return value.select(index)
    if isinstance(value, torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, dict):
        return {key: _select(child, indices) for key, child in value.items()}
    if isinstance(value, list):
        return [value[index] for index in indices]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    raise TypeError(f"cannot select storage field of type {type(value).__name__}")

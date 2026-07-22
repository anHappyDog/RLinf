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
from dataclasses import dataclass, fields
from typing import Any

import torch

from rlinf.data.forward_inputs import ForwardInputs, get_forward_inputs_type
from rlinf.data.trajectory import (
    EnvResult,
    RewardResult,
    RolloutResult,
    TrajectoryData,
    ValueResult,
)

PROTOCOL_VERSION = 1
_MAGIC = 0x524C494E
_HEADER_PREFIX_SIZE = 10
_COORDINATE_FIELDS = frozenset(
    {"global_step", "rollout_epoch", "chunk_step", "slot_ids"}
)
_RECORD_TYPES: dict[str, type[TrajectoryData]] = {
    record_type.__name__: record_type
    for record_type in (EnvResult, RolloutResult, RewardResult, ValueResult)
}


@dataclass(frozen=True)
class TensorLayout:
    """One tensor leaf in a registered endpoint schema."""

    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    element_size: int

    def nbytes(self, batch_size: int) -> int:
        """Return the raw byte count for an actual batch."""
        elements = batch_size
        for dimension in self.shape:
            elements *= dimension
        return elements * self.element_size


@dataclass(frozen=True)
class EndpointSchema:
    """Fixed business-data layout shared by one sender and receiver endpoint."""

    schema_id: int
    max_batch_size: int
    record_type: str
    tensors: tuple[TensorLayout, ...]
    constants: tuple[tuple[tuple[str, ...], Any], ...]
    forward_schema: tuple[str, int] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_id, int)
            or isinstance(self.schema_id, bool)
            or self.schema_id < 1
        ):
            raise ValueError("schema_id must be a positive integer.")
        if (
            not isinstance(self.max_batch_size, int)
            or isinstance(self.max_batch_size, bool)
            or self.max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer.")
        if self.record_type not in _RECORD_TYPES:
            raise ValueError(f"Unsupported record type {self.record_type!r}.")
        if not self.tensors:
            raise ValueError("An endpoint schema must contain at least one tensor.")
        paths = [layout.path for layout in self.tensors]
        if len(set(paths)) != len(paths):
            raise ValueError("Tensor layout paths must be unique.")
        for layout in self.tensors:
            if not layout.path or any(
                not isinstance(key, str) or not key for key in layout.path
            ):
                raise ValueError("Tensor layout paths must contain non-empty keys.")
            if any(dimension < 1 for dimension in layout.shape):
                raise ValueError("Tensor layout dimensions must be positive.")
            expected_size = torch.empty((), dtype=layout.dtype).element_size()
            if layout.element_size != expected_size:
                raise ValueError("Tensor layout element_size does not match dtype.")
        constant_paths = [path for path, _ in self.constants]
        if len(set(constant_paths)) != len(constant_paths):
            raise ValueError("Constant layout paths must be unique.")
        if set(paths).intersection(constant_paths):
            raise ValueError("Tensor and constant layout paths must not overlap.")

    @classmethod
    def from_example(
        cls,
        schema_id: int,
        max_batch_size: int,
        data: EnvResult | RolloutResult | RewardResult | ValueResult,
    ) -> "EndpointSchema":
        """Derive a fixed endpoint layout from one validated business object."""
        if not isinstance(data, tuple(_RECORD_TYPES.values())):
            raise TypeError(f"Unsupported endpoint data {type(data).__name__}.")
        layouts, constants, forward_schema = _describe(data)
        schema = cls(
            schema_id=schema_id,
            max_batch_size=max_batch_size,
            record_type=type(data).__name__,
            tensors=layouts,
            constants=constants,
            forward_schema=forward_schema,
        )
        if data.batch_size > schema.max_batch_size:
            raise ValueError("Example batch exceeds max_batch_size.")
        return schema

    @property
    def header_size(self) -> int:
        """Return the fixed number of int64 values in this endpoint header."""
        return _HEADER_PREFIX_SIZE + self.max_batch_size + len(self.tensors)

    def validate(self, data: TrajectoryData) -> tuple[torch.Tensor, ...]:
        """Validate one object and return tensor leaves in wire order."""
        if type(data) is not _RECORD_TYPES[self.record_type]:
            raise TypeError(
                f"Endpoint expects {self.record_type}, got {type(data).__name__}."
            )
        if data.batch_size > self.max_batch_size:
            raise ValueError("Data batch exceeds endpoint max_batch_size.")
        layouts, constants, forward_schema, tensors = _describe(
            data, return_tensors=True
        )
        if layouts != self.tensors:
            expected = {
                layout.path: (layout.shape, layout.dtype) for layout in self.tensors
            }
            actual = {layout.path: (layout.shape, layout.dtype) for layout in layouts}
            raise ValueError(
                "Data tensor layout does not match the endpoint schema: "
                f"expected={expected}, actual={actual}."
            )
        if constants != self.constants or forward_schema != self.forward_schema:
            raise ValueError("Data constants do not match the endpoint schema.")
        for layout, tensor in zip(self.tensors, tensors, strict=True):
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"Tensor {'.'.join(layout.path)!r} must be on CPU before send."
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"Tensor {'.'.join(layout.path)!r} must be contiguous before send."
                )
        return tensors


@dataclass(frozen=True)
class PreparedSend:
    """Header and zero-copy tensor views held until their sequence is acked."""

    header: torch.Tensor
    payloads: tuple[torch.Tensor, ...]

    @property
    def sequence_id(self) -> int:
        return int(self.header[3].item())

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        """Return header followed by payload tensors in send order."""
        return (self.header, *self.payloads)


@dataclass(frozen=True)
class ReceiveBuffers:
    """Reusable maximum-capacity storage owned by one receive operation."""

    header: torch.Tensor
    payloads: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TransportAck:
    """Wire acknowledgement permitting sender-side buffer reuse."""

    schema_id: int
    sequence_id: int


@dataclass(frozen=True)
class ReceiveResult:
    """Decoded data and its retry-aware wire acknowledgement."""

    data: EnvResult | RolloutResult | RewardResult | ValueResult
    ack: TransportAck
    duplicate: bool


@dataclass(frozen=True)
class _Header:
    sequence_id: int
    global_step: int
    rollout_epoch: int
    chunk_step: int
    batch_size: int
    slot_ids: tuple[int, ...]


class TransportEndpoint:
    """Stateful raw fixed-layout codec for one ordered endpoint lane."""

    def __init__(self, schema: EndpointSchema) -> None:
        self.schema = schema
        self._next_send_sequence = 0
        self._next_receive_sequence = 0
        self._in_flight: set[int] = set()

    @property
    def in_flight(self) -> int:
        """Return the number of encoded sends not yet acknowledged."""
        return len(self._in_flight)

    def encode(
        self,
        data: EnvResult | RolloutResult | RewardResult | ValueResult,
    ) -> PreparedSend:
        """Prepare one zero-copy raw send and reserve its sequence number."""
        payloads = self.schema.validate(data)
        sequence_id = self._next_send_sequence
        header = self._encode_header(data, sequence_id)
        self._next_send_sequence += 1
        self._in_flight.add(sequence_id)
        return PreparedSend(header=header, payloads=payloads)

    def allocate_receive_buffers(self) -> ReceiveBuffers:
        """Allocate reusable CPU buffers at the endpoint's maximum batch size."""
        return ReceiveBuffers(
            header=torch.empty(self.schema.header_size, dtype=torch.int64),
            payloads=tuple(
                torch.empty(
                    (self.schema.max_batch_size, *layout.shape),
                    dtype=layout.dtype,
                )
                for layout in self.schema.tensors
            ),
        )

    def payload_views(self, buffers: ReceiveBuffers) -> tuple[torch.Tensor, ...]:
        """Return actual-batch prefixes to pass to in-place tensor receive calls."""
        header = self._parse_header(buffers.header)
        self._validate_receive_buffers(buffers)
        return tuple(payload[: header.batch_size] for payload in buffers.payloads)

    def decode(self, buffers: ReceiveBuffers) -> ReceiveResult:
        """Decode received buffers and validate ordered sequence/retry semantics."""
        header = self._parse_header(buffers.header)
        payloads = self.payload_views(buffers)
        if header.sequence_id == self._next_receive_sequence:
            duplicate = False
        elif 0 <= header.sequence_id < self._next_receive_sequence:
            duplicate = True
        else:
            raise ValueError(
                f"Expected receive sequence {self._next_receive_sequence} or an "
                f"earlier retry, got {header.sequence_id}."
            )

        data = self._restore(header, payloads)
        if not duplicate:
            self._next_receive_sequence += 1
        return ReceiveResult(
            data=data,
            ack=TransportAck(
                schema_id=self.schema.schema_id,
                sequence_id=header.sequence_id,
            ),
            duplicate=duplicate,
        )

    def acknowledge(self, ack: TransportAck) -> bool:
        """Release an in-flight sequence; return false for a duplicate ack."""
        if ack.schema_id != self.schema.schema_id:
            raise ValueError("Ack schema_id does not match this endpoint.")
        if ack.sequence_id in self._in_flight:
            self._in_flight.remove(ack.sequence_id)
            return True
        if 0 <= ack.sequence_id < self._next_send_sequence:
            return False
        raise ValueError(f"Ack references unknown sequence {ack.sequence_id}.")

    def _encode_header(self, data: TrajectoryData, sequence_id: int) -> torch.Tensor:
        slots = (
            *data.slot_ids,
            *([-1] * (self.schema.max_batch_size - data.batch_size)),
        )
        sizes = tuple(layout.nbytes(data.batch_size) for layout in self.schema.tensors)
        return torch.tensor(
            (
                _MAGIC,
                PROTOCOL_VERSION,
                self.schema.schema_id,
                sequence_id,
                data.global_step,
                data.rollout_epoch,
                data.chunk_step,
                data.batch_size,
                len(self.schema.tensors),
                0,
                *slots,
                *sizes,
            ),
            dtype=torch.int64,
        )

    def _parse_header(self, header: torch.Tensor) -> _Header:
        if (
            not isinstance(header, torch.Tensor)
            or header.dtype != torch.int64
            or header.device.type != "cpu"
            or not header.is_contiguous()
            or header.shape != (self.schema.header_size,)
        ):
            raise ValueError(
                "Transport header must be a fixed contiguous CPU int64 tensor."
            )
        values = header.tolist()
        if values[0] != _MAGIC:
            raise ValueError("Invalid transport header magic.")
        if values[1] != PROTOCOL_VERSION:
            raise ValueError("Unsupported transport protocol version.")
        if values[2] != self.schema.schema_id:
            raise ValueError("Transport header schema_id does not match endpoint.")
        if values[3] < 0 or any(value < 0 for value in values[4:7]):
            raise ValueError(
                "Transport sequence and trajectory coordinates must be non-negative."
            )
        batch_size = values[7]
        if not 1 <= batch_size <= self.schema.max_batch_size:
            raise ValueError("Transport header contains an invalid batch size.")
        if values[8] != len(self.schema.tensors):
            raise ValueError("Transport header tensor count does not match endpoint.")
        if values[9] != 0:
            raise ValueError("Transport header contains unsupported flags.")

        slots_start = _HEADER_PREFIX_SIZE
        slots_stop = slots_start + self.schema.max_batch_size
        slots = tuple(values[slots_start : slots_start + batch_size])
        unused_slots = values[slots_start + batch_size : slots_stop]
        if any(slot_id < 0 for slot_id in slots) or len(set(slots)) != len(slots):
            raise ValueError("Transport header contains invalid slot IDs.")
        if any(slot_id != -1 for slot_id in unused_slots):
            raise ValueError("Unused transport slot entries must be -1.")

        sizes = values[slots_stop:]
        expected_sizes = [layout.nbytes(batch_size) for layout in self.schema.tensors]
        if sizes != expected_sizes:
            raise ValueError(
                "Transport header payload sizes do not match endpoint layout."
            )
        return _Header(
            sequence_id=values[3],
            global_step=values[4],
            rollout_epoch=values[5],
            chunk_step=values[6],
            batch_size=batch_size,
            slot_ids=slots,
        )

    def _validate_receive_buffers(self, buffers: ReceiveBuffers) -> None:
        if len(buffers.payloads) != len(self.schema.tensors):
            raise ValueError("Receive buffer count does not match endpoint schema.")
        for layout, payload in zip(self.schema.tensors, buffers.payloads, strict=True):
            expected_shape = (self.schema.max_batch_size, *layout.shape)
            if (
                payload.shape != expected_shape
                or payload.dtype != layout.dtype
                or payload.device.type != "cpu"
                or not payload.is_contiguous()
            ):
                raise ValueError(
                    f"Receive buffer {'.'.join(layout.path)!r} does not match schema."
                )

    def _restore(
        self,
        header: _Header,
        payloads: tuple[torch.Tensor, ...],
    ) -> EnvResult | RolloutResult | RewardResult | ValueResult:
        values: dict[str, Any] = {}
        forward_values: dict[str, torch.Tensor] = {}
        forward_field = "forward_inputs" if self.schema.forward_schema else None
        for path, constant in self.schema.constants:
            _assign(values, path, constant)
        for layout, payload in zip(self.schema.tensors, payloads, strict=True):
            if forward_field is not None and layout.path[0] == forward_field:
                forward_values[layout.path[1]] = payload
            else:
                _assign(values, layout.path, payload)
        if self.schema.forward_schema is not None:
            name, version = self.schema.forward_schema
            forward_type = get_forward_inputs_type(name, version)
            values["forward_inputs"] = forward_type.from_model_inputs(forward_values)

        values.update(
            {
                "global_step": header.global_step,
                "rollout_epoch": header.rollout_epoch,
                "chunk_step": header.chunk_step,
                "slot_ids": header.slot_ids,
            }
        )
        record_type = _RECORD_TYPES[self.schema.record_type]
        return record_type(**values)


def _describe(
    data: TrajectoryData,
    *,
    return_tensors: bool = False,
) -> (
    tuple[
        tuple[TensorLayout, ...],
        tuple[tuple[tuple[str, ...], Any], ...],
        tuple[str, int] | None,
    ]
    | tuple[
        tuple[TensorLayout, ...],
        tuple[tuple[tuple[str, ...], Any], ...],
        tuple[str, int] | None,
        tuple[torch.Tensor, ...],
    ]
):
    layouts: list[TensorLayout] = []
    constants: list[tuple[tuple[str, ...], Any]] = []
    tensors: list[torch.Tensor] = []
    forward_schema: tuple[str, int] | None = None
    for field in fields(data):
        if field.name in _COORDINATE_FIELDS:
            continue
        value = getattr(data, field.name)
        if isinstance(value, ForwardInputs):
            if forward_schema is not None:
                raise ValueError(
                    "Only one ForwardInputs field is supported per endpoint."
                )
            forward_schema = (value.schema_name, value.schema_version)
            for name, tensor in value.tensor_fields():
                _append_tensor((field.name, name), tensor, layouts, tensors)
        else:
            _flatten((field.name,), value, layouts, constants, tensors)
    result = (tuple(layouts), tuple(constants), forward_schema)
    return (*result, tuple(tensors)) if return_tensors else result


def _flatten(
    path: tuple[str, ...],
    value: Any,
    layouts: list[TensorLayout],
    constants: list[tuple[tuple[str, ...], Any]],
    tensors: list[torch.Tensor],
) -> None:
    if isinstance(value, torch.Tensor):
        _append_tensor(path, value, layouts, tensors)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Fixed endpoint mapping keys must be strings.")
        for key in sorted(value):
            _flatten((*path, key), value[key], layouts, constants, tensors)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        constants.append((path, value))
        return
    raise TypeError(
        f"Fixed endpoint field {'.'.join(path)!r} cannot contain "
        f"{type(value).__name__}."
    )


def _append_tensor(
    path: tuple[str, ...],
    tensor: torch.Tensor,
    layouts: list[TensorLayout],
    tensors: list[torch.Tensor],
) -> None:
    if tensor.ndim < 1:
        raise ValueError(f"Tensor {'.'.join(path)!r} must have a batch dimension.")
    layouts.append(
        TensorLayout(
            path=path,
            shape=tuple(tensor.shape[1:]),
            dtype=tensor.dtype,
            element_size=tensor.element_size(),
        )
    )
    tensors.append(tensor)


def _assign(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        nested = current.get(key)
        if nested is None:
            nested = {}
            current[key] = nested
        elif not isinstance(nested, dict):
            raise ValueError(f"Transport schema path collision at {key!r}.")
        current = nested
    current[path[-1]] = value

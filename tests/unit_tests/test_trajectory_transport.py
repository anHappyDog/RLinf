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

import pickle
from dataclasses import fields, is_dataclass, replace
from typing import Any

import pytest
import torch

from rlinf.data.forward_inputs import ForwardInputs
from rlinf.data.trajectory import EnvResult, RewardResult, RolloutResult, ValueResult
from rlinf.models.embodiment.openpi.forward_inputs import OpenPILiberoForwardInputs
from rlinf.workers.trajectory.transport import (
    EndpointSchema,
    TransportAck,
    TransportEndpoint,
)


def _forward_inputs(batch_size: int) -> OpenPILiberoForwardInputs:
    return OpenPILiberoForwardInputs(
        chains=torch.arange(batch_size * 5 * 2 * 7, dtype=torch.float32).reshape(
            batch_size, 5, 2, 7
        ),
        denoise_inds=torch.arange(4).expand(batch_size, 4).clone(),
        tokenized_prompt=torch.arange(3).expand(batch_size, 3).clone(),
        tokenized_prompt_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        action=torch.arange(batch_size * 7, dtype=torch.float32).reshape(batch_size, 7),
        model_action=torch.arange(batch_size * 7, dtype=torch.float32).reshape(
            batch_size, 7
        ),
        image=torch.arange(batch_size * 2 * 2 * 3, dtype=torch.uint8).reshape(
            batch_size, 2, 2, 3
        ),
        wrist_image=torch.arange(batch_size * 2 * 2 * 3, dtype=torch.uint8).reshape(
            batch_size, 2, 2, 3
        ),
        state=torch.arange(batch_size * 8, dtype=torch.float32).reshape(batch_size, 8),
    )


def _env_result(slot_ids: tuple[int, ...] = (2, 5)) -> EnvResult:
    batch_size = len(slot_ids)
    return EnvResult(
        global_step=7,
        rollout_epoch=1,
        chunk_step=3,
        slot_ids=slot_ids,
        rewards=torch.arange(batch_size * 2, dtype=torch.float32).reshape(
            batch_size, 2
        ),
        dones=torch.zeros(batch_size, 2, dtype=torch.bool),
        terminations=torch.zeros(batch_size, 2, dtype=torch.bool),
        truncations=torch.zeros(batch_size, 2, dtype=torch.bool),
        observations={
            "image": torch.arange(batch_size * 2 * 2 * 3, dtype=torch.uint8).reshape(
                batch_size, 2, 2, 3
            ),
            "state": torch.arange(batch_size * 8, dtype=torch.float32).reshape(
                batch_size, 8
            ),
        },
        intervene_flags=torch.zeros(batch_size, 1, dtype=torch.bool),
    )


def _rollout_result(slot_ids: tuple[int, ...] = (2, 5)) -> RolloutResult:
    batch_size = len(slot_ids)
    return RolloutResult(
        global_step=7,
        rollout_epoch=1,
        chunk_step=3,
        slot_ids=slot_ids,
        actions=torch.arange(batch_size * 7, dtype=torch.float32).reshape(
            batch_size, 7
        ),
        forward_inputs=_forward_inputs(batch_size),
        prev_logprobs=torch.arange(batch_size, dtype=torch.float32).reshape(
            batch_size, 1
        ),
        state_values=torch.arange(batch_size, dtype=torch.float32).reshape(
            batch_size, 1
        ),
        versions=torch.arange(batch_size, dtype=torch.int64),
    )


def _reward_result(slot_ids: tuple[int, ...] = (2, 5)) -> RewardResult:
    batch_size = len(slot_ids)
    return RewardResult(
        global_step=7,
        rollout_epoch=1,
        chunk_step=3,
        slot_ids=slot_ids,
        rewards=torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1),
        mode="history_buffer",
        history_lengths=torch.arange(batch_size, dtype=torch.int64) + 1,
    )


def _value_result(slot_ids: tuple[int, ...] = (2, 5)) -> ValueResult:
    batch_size = len(slot_ids)
    return ValueResult(
        global_step=7,
        rollout_epoch=1,
        chunk_step=3,
        slot_ids=slot_ids,
        kind="timeout",
        values=torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1),
        versions=torch.arange(batch_size, dtype=torch.int64),
    )


def _transfer(
    sender: TransportEndpoint,
    receiver: TransportEndpoint,
    data: EnvResult | RolloutResult | RewardResult | ValueResult,
):
    prepared = sender.encode(data)
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)
    views = receiver.payload_views(buffers)
    for source, destination in zip(prepared.payloads, views, strict=True):
        destination.copy_(source)
    return prepared, buffers, receiver.decode(buffers)


def _equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, ForwardInputs):
        return all(
            left_name == right_name and torch.equal(left_value, right_value)
            for (left_name, left_value), (right_name, right_value) in zip(
                left.tensor_fields(), right.tensor_fields(), strict=True
            )
        )
    if is_dataclass(left):
        return all(
            _equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


@pytest.mark.parametrize(
    "data",
    [_env_result(), _rollout_result(), _reward_result(), _value_result()],
)
def test_all_storage_results_round_trip_field_exactly(data: Any) -> None:
    schema = EndpointSchema.from_example(11, 4, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)

    prepared, _, received = _transfer(sender, receiver, data)

    assert all(isinstance(buffer, torch.Tensor) for buffer in prepared.buffers)
    assert _equal(received.data, data)
    assert not received.duplicate
    assert sender.in_flight == 1
    assert sender.acknowledge(received.ack)
    assert sender.in_flight == 0
    assert not sender.acknowledge(received.ack)


def test_receive_views_use_preallocated_max_batch_buffers() -> None:
    data = _env_result((2, 5))
    schema = EndpointSchema.from_example(12, 8, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)
    prepared = sender.encode(data)
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)

    views = receiver.payload_views(buffers)

    assert prepared.payloads[0] is data.rewards
    assert buffers.payloads[0].shape == (8, 2)
    assert views[0].shape == (2, 2)
    assert views[0].data_ptr() == buffers.payloads[0].data_ptr()
    for source, destination in zip(prepared.payloads, views, strict=True):
        destination.copy_(source)
    received = receiver.decode(buffers)
    assert received.data.rewards.data_ptr() == buffers.payloads[0].data_ptr()


def test_schema_is_serializable_as_control_plane_metadata() -> None:
    schema = EndpointSchema.from_example(22, 8, _rollout_result())

    restored = pickle.loads(pickle.dumps(schema))

    assert restored == schema


def test_retry_reuses_sequence_and_returns_the_same_ack() -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(13, 4, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)

    prepared, buffers, first = _transfer(sender, receiver, data)
    second = receiver.decode(buffers)

    assert prepared.sequence_id == 0
    assert not first.duplicate
    assert second.duplicate
    assert second.ack == first.ack
    assert _equal(second.data, data)


def test_older_sequence_can_retry_after_newer_data_arrives() -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(20, 4, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)

    _, first_buffers, first = _transfer(sender, receiver, data)
    _, _, second = _transfer(sender, receiver, data)
    retried = receiver.decode(first_buffers)

    assert first.ack.sequence_id == 0
    assert second.ack.sequence_id == 1
    assert retried.duplicate
    assert retried.ack == first.ack


def test_ack_rejects_wrong_schema_and_unknown_sequence() -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(21, 4, data)
    endpoint = TransportEndpoint(schema)
    endpoint.encode(data)

    with pytest.raises(ValueError, match="schema_id"):
        endpoint.acknowledge(TransportAck(schema_id=99, sequence_id=0))
    with pytest.raises(ValueError, match="unknown sequence"):
        endpoint.acknowledge(TransportAck(schema_id=21, sequence_id=1))


def test_future_sequence_is_rejected_without_consuming_expected_sequence() -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(14, 4, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)
    prepared = sender.encode(data)
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)
    buffers.header[3] = 1
    for source, destination in zip(
        prepared.payloads, receiver.payload_views(buffers), strict=True
    ):
        destination.copy_(source)

    with pytest.raises(ValueError, match="Expected receive sequence 0"):
        receiver.decode(buffers)

    buffers.header[3] = 0
    received = receiver.decode(buffers)
    assert not received.duplicate


@pytest.mark.parametrize(
    ("header_index", "value", "message"),
    [
        (0, 0, "magic"),
        (1, 99, "protocol version"),
        (2, 99, "schema_id"),
        (8, 99, "tensor count"),
        (9, 1, "unsupported flags"),
        (10, -2, "invalid slot IDs"),
        (12, 0, "Unused transport slot"),
    ],
)
def test_corrupt_header_is_rejected(
    header_index: int, value: int, message: str
) -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(15, 4, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)
    prepared = sender.encode(data)
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)
    buffers.header[header_index] = value

    with pytest.raises(ValueError, match=message):
        receiver.payload_views(buffers)


def test_corrupt_payload_size_is_rejected() -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(16, 4, data)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)
    prepared = sender.encode(data)
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)
    payload_sizes_start = 10 + schema.max_batch_size
    buffers.header[payload_sizes_start] += 1

    with pytest.raises(ValueError, match="payload sizes"):
        receiver.payload_views(buffers)


def test_schema_rejects_changed_shape_constants_and_oversized_batch() -> None:
    data = _value_result()
    schema = EndpointSchema.from_example(17, 2, data)
    endpoint = TransportEndpoint(schema)

    changed_shape = replace(data, versions=torch.zeros(2, 1, dtype=torch.int64))
    with pytest.raises(ValueError, match="tensor layout"):
        endpoint.encode(changed_shape)

    changed_kind = replace(data, kind="tail")
    with pytest.raises(ValueError, match="constants"):
        endpoint.encode(changed_kind)

    oversized = _value_result((1, 2, 3))
    with pytest.raises(ValueError, match="max_batch_size"):
        endpoint.encode(oversized)


def test_sender_requires_contiguous_cpu_tensor_lanes() -> None:
    data = _env_result()
    schema = EndpointSchema.from_example(18, 2, data)
    endpoint = TransportEndpoint(schema)
    noncontiguous = torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
    changed = replace(data, rewards=noncontiguous)

    with pytest.raises(ValueError, match="contiguous"):
        endpoint.encode(changed)


def test_batch_string_lists_are_not_silently_serialized() -> None:
    data = _env_result()
    observations = dict(data.observations)
    observations["task_descriptions"] = ["task a", "task b"]
    changed = replace(data, observations=observations)

    with pytest.raises(TypeError, match="cannot contain list"):
        EndpointSchema.from_example(19, 2, changed)

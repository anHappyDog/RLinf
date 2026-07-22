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
from typing import TYPE_CHECKING, Any

import ray
import torch

from rlinf.data.trajectory import PolicyInput, PolicyOutput, ValueRequest
from rlinf.scheduler import Worker
from rlinf.workers.trajectory.route_plan import Route
from rlinf.workers.trajectory.tensor_codec import create_tensor_codec

if TYPE_CHECKING:
    from rlinf.scheduler.worker.worker_group import WorkerGroup
    from rlinf.workers.trajectory.workers import ChannelConfig

_POLICY_MAGIC = 0x504F4C49
_POLICY_VERSION = 2
_POLICY_HEADER_PREFIX = 12
_RLT_PRESENT = 1
_INTERVENE_PRESENT = 2
_IMAGE_RAW = 0
_IMAGE_LZ4 = 1
_IMAGE_XOR_LZ4 = 2


@dataclass(frozen=True)
class PolicyInputLayout:
    """Static tensor layout for direct Env-to-Rollout policy input."""

    batch_size: int
    image_shape: tuple[int, int, int]
    state_shape: tuple[int, ...]
    extra_view_shape: tuple[int, ...] | None = None
    max_description_bytes: int = 256
    compress_images: bool = False
    compression_level: int = 1
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("policy input batch_size must be positive")
        if any(dimension < 1 for dimension in self.image_shape + self.state_shape):
            raise ValueError("policy input tensor dimensions must be positive")
        if self.max_description_bytes < 1:
            raise ValueError("max_description_bytes must be positive")
        if self.compression_level < 1:
            raise ValueError("compression_level must be positive")

    @property
    def header_size(self) -> int:
        """Return the fixed int64 header length."""
        return _POLICY_HEADER_PREFIX + 2 * self.batch_size

    def allocate_buffers(self) -> tuple[torch.Tensor, ...]:
        """Allocate one reusable raw receive buffer set."""
        return (
            torch.empty(
                self.header_size, dtype=torch.int64, pin_memory=self.pin_memory
            ),
            torch.empty(
                (self.batch_size, *self.image_shape),
                dtype=torch.uint8,
                pin_memory=self.pin_memory,
            ),
            torch.empty(
                (self.batch_size, *self.image_shape),
                dtype=torch.uint8,
                pin_memory=self.pin_memory,
            ),
            torch.empty(
                (self.batch_size, *(self.extra_view_shape or (0,))),
                dtype=torch.uint8,
                pin_memory=self.pin_memory,
            ),
            torch.empty(
                (self.batch_size, *self.state_shape),
                dtype=torch.float32,
                pin_memory=self.pin_memory,
            ),
            torch.empty(self.batch_size, dtype=torch.bool, pin_memory=self.pin_memory),
            torch.empty(self.batch_size, dtype=torch.bool, pin_memory=self.pin_memory),
            torch.empty(
                (self.batch_size, self.max_description_bytes),
                dtype=torch.uint8,
                pin_memory=self.pin_memory,
            ),
        )

    def allocate_send_workspace(self) -> tuple[torch.Tensor, ...]:
        """Allocate reusable metadata buffers without duplicating image payloads."""
        return (
            torch.empty(self.header_size, dtype=torch.int64),
            torch.empty(
                (self.batch_size, *(self.extra_view_shape or (0,))),
                dtype=torch.uint8,
            ),
            torch.empty(self.batch_size, dtype=torch.bool),
            torch.empty(self.batch_size, dtype=torch.bool),
            torch.empty(
                (self.batch_size, self.max_description_bytes), dtype=torch.uint8
            ),
        )

    def encode(
        self,
        data: PolicyInput,
        sequence_id: int,
        workspace: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Validate and encode one fixed raw request without copying images."""
        expected_keys = {
            "main_images",
            "wrist_images",
            "extra_view_images",
            "states",
            "task_descriptions",
        }
        if set(data.observations) != expected_keys:
            raise ValueError(
                "direct policy input observations do not match layout: "
                f"expected={sorted(expected_keys)}, actual={sorted(data.observations)}"
            )
        if data.batch_size != self.batch_size:
            raise ValueError("direct policy input requires its fixed batch size")
        main = data.observations["main_images"]
        wrist = data.observations["wrist_images"]
        extra = data.observations["extra_view_images"]
        states = data.observations["states"]
        self._validate_tensor(main, (self.batch_size, *self.image_shape), torch.uint8)
        self._validate_tensor(wrist, (self.batch_size, *self.image_shape), torch.uint8)
        if self.extra_view_shape is None:
            if extra is not None:
                raise ValueError("extra view images are not registered in the layout")
        else:
            self._validate_tensor(
                extra,
                (self.batch_size, *self.extra_view_shape),
                torch.uint8,
            )
        self._validate_tensor(
            states, (self.batch_size, *self.state_shape), torch.float32
        )
        descriptions = data.observations["task_descriptions"]
        if (
            not isinstance(descriptions, (list, tuple))
            or len(descriptions) != self.batch_size
        ):
            raise ValueError("task_descriptions must match the fixed batch size")

        header, empty_extra, rlt, intervene, description_bytes = workspace
        flags = 0
        if data.rlt_switch_flags is None:
            rlt.zero_()
        else:
            self._validate_tensor(data.rlt_switch_flags, (self.batch_size,), torch.bool)
            rlt.copy_(data.rlt_switch_flags)
            flags |= _RLT_PRESENT
        if data.intervene_requested is None:
            intervene.zero_()
        else:
            self._validate_tensor(
                data.intervene_requested, (self.batch_size,), torch.bool
            )
            intervene.copy_(data.intervene_requested)
            flags |= _INTERVENE_PRESENT

        description_bytes.zero_()
        lengths = []
        for index, description in enumerate(descriptions):
            if not isinstance(description, str):
                raise TypeError("task_descriptions must contain strings")
            encoded = description.encode("utf-8")
            if len(encoded) > self.max_description_bytes:
                raise ValueError("task description exceeds fixed byte capacity")
            lengths.append(len(encoded))
            if encoded:
                description_bytes[index, : len(encoded)] = torch.tensor(
                    tuple(encoded), dtype=torch.uint8
                )
        header.copy_(
            torch.tensor(
                (
                    _POLICY_MAGIC,
                    _POLICY_VERSION,
                    sequence_id,
                    data.global_step,
                    data.rollout_epoch,
                    data.chunk_step,
                    self.batch_size,
                    flags,
                    _IMAGE_RAW,
                    main.numel(),
                    _IMAGE_RAW,
                    wrist.numel(),
                    *data.slot_ids,
                    *lengths,
                ),
                dtype=torch.int64,
            )
        )
        return (
            header,
            main,
            wrist,
            extra if extra is not None else empty_extra,
            states,
            rlt,
            intervene,
            description_bytes,
        )

    def decode(
        self,
        buffers: tuple[torch.Tensor, ...],
        expected_sequence: int,
    ) -> PolicyInput:
        """Decode one received fixed buffer set into its business type."""
        header, main, wrist, extra, states, rlt, intervene, description_bytes = buffers
        values = header.tolist()
        if values[:3] != [_POLICY_MAGIC, _POLICY_VERSION, expected_sequence]:
            raise ValueError("direct policy input header or sequence is invalid")
        if values[6] != self.batch_size:
            raise ValueError("direct policy input header batch size is invalid")
        flags = values[7]
        if flags & ~(_RLT_PRESENT | _INTERVENE_PRESENT):
            raise ValueError("direct policy input header flags are invalid")
        slot_start = _POLICY_HEADER_PREFIX
        length_start = slot_start + self.batch_size
        slot_ids = tuple(values[slot_start:length_start])
        lengths = values[length_start:]
        descriptions = []
        for index, length in enumerate(lengths):
            if not 0 <= length <= self.max_description_bytes:
                raise ValueError("task description length is invalid")
            descriptions.append(
                bytes(description_bytes[index, :length]).decode("utf-8")
            )
        return PolicyInput(
            global_step=values[3],
            rollout_epoch=values[4],
            chunk_step=values[5],
            slot_ids=slot_ids,
            observations={
                "main_images": main,
                "wrist_images": wrist,
                "extra_view_images": extra
                if self.extra_view_shape is not None
                else None,
                "states": states,
                "task_descriptions": descriptions,
            },
            rlt_switch_flags=rlt if flags & _RLT_PRESENT else None,
            intervene_requested=(intervene if flags & _INTERVENE_PRESENT else None),
        )

    @staticmethod
    def _validate_tensor(
        tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"direct policy tensor must be contiguous CPU {dtype} with shape {shape}"
            )


class _LiveImageCodec:
    """Maintain lossless image state for one ordered live connection."""

    def __init__(self, layout: PolicyInputLayout) -> None:
        shape = (layout.batch_size, *layout.image_shape)
        image_bytes = torch.empty(shape, dtype=torch.uint8).numel()
        self._codecs = (
            create_tensor_codec("lz4", level=layout.compression_level),
            create_tensor_codec("lz4", level=layout.compression_level),
        )
        self._previous = (
            torch.empty(shape, dtype=torch.uint8),
            torch.empty(shape, dtype=torch.uint8),
        )
        self._xor = (
            torch.empty(shape, dtype=torch.uint8),
            torch.empty(shape, dtype=torch.uint8),
        )
        self._wire = tuple(
            torch.empty(codec.compress_bound(image_bytes), dtype=torch.uint8)
            for codec in self._codecs
        )
        self._has_previous = False

    def encode(
        self, images: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, int, int], ...]:
        """Return payload, mode, and byte count for two image fields."""
        encoded = []
        for index, image in enumerate(images):
            source = image
            mode = _IMAGE_LZ4
            if self._has_previous:
                torch.bitwise_xor(image, self._previous[index], out=self._xor[index])
                source = self._xor[index]
                mode = _IMAGE_XOR_LZ4
            source_bytes = source.numel()
            encoded_bytes = self._codecs[index].compress_into(
                source.reshape(-1), self._wire[index]
            )
            if encoded_bytes < source_bytes:
                encoded.append((self._wire[index][:encoded_bytes], mode, encoded_bytes))
            else:
                encoded.append((image.reshape(-1), _IMAGE_RAW, image.numel()))
            self._previous[index].copy_(image)
        self._has_previous = True
        return tuple(encoded)

    def wire_buffers(self, sizes: tuple[int, int]) -> tuple[torch.Tensor, ...]:
        """Return bounded receive views matching header-declared sizes."""
        if any(size < 1 for size in sizes):
            raise ValueError("live image payload sizes must be positive")
        if any(size > buffer.numel() for size, buffer in zip(sizes, self._wire)):
            raise ValueError("live image payload exceeds registered capacity")
        return tuple(
            buffer[:size] for size, buffer in zip(sizes, self._wire, strict=True)
        )

    def decode(
        self,
        payloads: tuple[torch.Tensor, torch.Tensor],
        modes: tuple[int, int],
        destinations: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Restore two payloads and advance their reference frames."""
        for index, (payload, mode, destination) in enumerate(
            zip(payloads, modes, destinations, strict=True)
        ):
            if mode == _IMAGE_RAW:
                destination.reshape(-1).copy_(payload)
            elif mode in (_IMAGE_LZ4, _IMAGE_XOR_LZ4):
                self._codecs[index].decompress_into(
                    payload,
                    payload.numel(),
                    destination,
                )
                if mode == _IMAGE_XOR_LZ4:
                    if not self._has_previous:
                        raise ValueError("XOR image received before a reference frame")
                    torch.bitwise_xor(
                        destination, self._previous[index], out=destination
                    )
            else:
                raise ValueError(f"invalid live image mode {mode}")
            self._previous[index].copy_(destination)
        self._has_previous = True


@dataclass(frozen=True)
class _PolicyTransfer:
    """Collective-friendly representation with tensors outside the skeleton."""

    data_type: str
    global_step: int
    rollout_epoch: int
    chunk_step: int
    slot_ids: tuple[int, ...]
    skeleton: dict[str, Any]
    tensor_paths: tuple[tuple[str, ...], ...]
    tensors: tuple[torch.Tensor, ...]


class TrajectoryChannel:
    """Typed façade for Env-to-Rollout live communication."""

    def __init__(
        self,
        actor: ray.actor.ActorHandle,
        group_name: str,
        config: "ChannelConfig",
    ) -> None:
        self._actor = actor
        self._group_name = group_name
        self._config = config
        self._current_worker = Worker.current_worker
        self._policy_send_sequences: dict[int, int] = {}
        self._policy_receive_sequences: dict[int, int] = {}
        self._policy_send_workspaces: dict[int, tuple[torch.Tensor, ...]] = {}
        self._policy_receive_buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self._policy_output_buffers: tuple[torch.Tensor, ...] | None = None
        self._policy_send_codecs: dict[int, _LiveImageCodec] = {}
        self._policy_receive_codecs: dict[int, _LiveImageCodec] = {}

    @classmethod
    def from_worker_group(
        cls,
        worker_group: "WorkerGroup",
        config: "ChannelConfig",
    ) -> "TrajectoryChannel":
        """Connect the façade to a single launched channel actor."""
        workers = worker_group.worker_info_list
        if len(workers) != 1:
            raise ValueError("SG-08 TrajectoryChannel requires one channel actor")
        return cls(workers[0].worker, worker_group.worker_group_name, config)

    def publish_policy_input(self, data: PolicyInput) -> None:
        """Publish an Env-owned request and route it to Rollout ranks."""
        if (
            self._config.policy_input_layout is not None
            and self._current_worker is not None
        ):
            self._send_policy_input(data)
            return
        self._publish("publish_policy_input", data, self._config.env_layout)

    def take_policy_input(self) -> PolicyInput:
        """Take the next request routed to the current Rollout rank.

        Direct fixed-frame tensors remain valid until the next call, which
        reuses the receive buffers after policy inference has consumed them.
        """
        if (
            self._config.policy_input_layout is not None
            and self._current_worker is not None
        ):
            return self._receive_policy_input()
        return self._take("take_policy_input", self._config.rollout_layout)

    def publish_policy_output(self, data: PolicyOutput) -> None:
        """Publish Rollout actions and route them back to Env ranks."""
        self._publish("publish_policy_output", data, self._config.rollout_layout)

    def take_policy_output(self) -> PolicyOutput:
        """Take the next actions routed to the current Env rank."""
        return self._take("take_policy_output", self._config.env_layout)

    def publish_value_request(self, request: ValueRequest) -> None:
        """Publish sparse boundary observations to Rollout ranks."""
        self._publish("publish_value_request", request, self._config.env_layout)

    def take_value_request(self) -> ValueRequest:
        """Take the next boundary-value request for this Rollout rank."""
        data = self._take("take_value_request", self._config.rollout_layout)
        if not isinstance(data, ValueRequest):
            raise RuntimeError(f"expected ValueRequest, got {type(data).__name__}")
        return data

    def value_request_count(self) -> int:
        """Return boundary requests currently routed to this Rollout rank."""
        worker = self._current_worker
        worker_rank = 0 if worker is None else worker._rank
        logical_rank = self._config.rollout_layout.logical_rank(worker_rank)
        return ray.get(self._actor.value_request_count.remote(logical_rank))

    def _publish(self, method: str, data, layout) -> None:
        worker = self._current_worker
        worker_rank = 0 if worker is None else worker._rank
        logical_rank = layout.logical_rank(worker_rank)
        if worker is None:
            ray.get(
                getattr(self._actor, f"{method}_via_ray").remote(
                    data, source_rank=logical_rank
                )
            )
            return
        pending = getattr(self._actor, method).remote(
            worker.worker_address, source_rank=logical_rank
        )
        worker.send(
            pack_policy_data(data),
            dst_group_name=self._group_name,
            dst_rank=0,
        )
        ray.get(pending)

    def _take(
        self,
        method: str,
        layout,
    ) -> PolicyInput | PolicyOutput | ValueRequest:
        worker = self._current_worker
        worker_rank = 0 if worker is None else worker._rank
        logical_rank = layout.logical_rank(worker_rank)
        if worker is None:
            return ray.get(
                getattr(self._actor, f"{method}_via_ray").remote(logical_rank)
            )
        pending = getattr(self._actor, method).remote(
            worker.worker_address, destination_rank=logical_rank
        )
        data = unpack_policy_data(worker.recv(self._group_name, 0))
        ray.get(pending)
        return data

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._current_worker = Worker.current_worker
        self._policy_send_sequences = {}
        self._policy_receive_sequences = {}
        self._policy_send_workspaces = {}
        self._policy_receive_buffers = {}
        self._policy_output_buffers = None
        self._policy_send_codecs = {}
        self._policy_receive_codecs = {}

    def _send_policy_input(self, data: PolicyInput) -> None:
        worker = self._current_worker
        layout = self._config.policy_input_layout
        assert worker is not None and layout is not None
        source_rank = self._config.env_layout.logical_rank(worker._rank)
        for route in self._config.direct_policy_routes(source_rank):
            destination_rank = route.destination_rank
            physical_rank = self._config.rollout_layout.data_ranks[destination_rank]
            fragment_layout = replace(layout, batch_size=len(route.slot_ids))
            workspace = self._policy_send_workspaces.get(destination_rank)
            if workspace is None:
                workspace = fragment_layout.allocate_send_workspace()
                self._policy_send_workspaces[destination_rank] = workspace
            sequence = self._policy_send_sequences.get(destination_rank, 0)
            fragment = (
                data
                if route.source_indices == tuple(range(data.batch_size))
                else select_policy_data(data, route.source_indices)
            )
            buffers = fragment_layout.encode(fragment, sequence, workspace)
            if fragment_layout.compress_images:
                codec = self._policy_send_codecs.get(destination_rank)
                if codec is None:
                    codec = _LiveImageCodec(fragment_layout)
                    self._policy_send_codecs[destination_rank] = codec
                images = codec.encode((buffers[1], buffers[2]))
                buffers[0][8:12] = torch.tensor(
                    (images[0][1], images[0][2], images[1][1], images[1][2]),
                    dtype=torch.int64,
                )
                buffers = (
                    buffers[0],
                    images[0][0],
                    images[1][0],
                    *buffers[3:],
                )
            for buffer in buffers:
                if buffer.numel() == 0:
                    continue
                worker.send_tensor(
                    buffer,
                    self._config.rollout_group_name,
                    physical_rank,
                )
            self._policy_send_sequences[destination_rank] = sequence + 1

    def _receive_policy_input(self) -> PolicyInput:
        worker = self._current_worker
        layout = self._config.policy_input_layout
        assert worker is not None and layout is not None
        destination_rank = self._config.rollout_layout.logical_rank(worker._rank)
        fragments = []
        for source_rank, route in self._config.direct_policy_sources(destination_rank):
            physical_rank = self._config.env_layout.data_ranks[source_rank]
            fragment_layout = replace(layout, batch_size=len(route.slot_ids))
            buffers = self._policy_receive_buffers.get(source_rank)
            if buffers is None:
                buffers = fragment_layout.allocate_buffers()
                self._policy_receive_buffers[source_rank] = buffers
            if fragment_layout.compress_images:
                worker.recv_tensor(
                    buffers[0], self._config.env_group_name, physical_rank
                )
                header = buffers[0].tolist()
                modes = (header[8], header[10])
                sizes = (header[9], header[11])
                codec = self._policy_receive_codecs.get(source_rank)
                if codec is None:
                    codec = _LiveImageCodec(fragment_layout)
                    self._policy_receive_codecs[source_rank] = codec
                payloads = codec.wire_buffers(sizes)
                for payload in payloads:
                    worker.recv_tensor(
                        payload, self._config.env_group_name, physical_rank
                    )
                codec.decode(payloads, modes, (buffers[1], buffers[2]))
                remaining = buffers[3:]
            else:
                remaining = buffers
            for buffer in remaining:
                if buffer.numel() == 0:
                    continue
                worker.recv_tensor(buffer, self._config.env_group_name, physical_rank)
            sequence = self._policy_receive_sequences.get(source_rank, 0)
            fragments.append((route, fragment_layout.decode(buffers, sequence)))
            self._policy_receive_sequences[source_rank] = sequence + 1

        start, end = self._config.route_plan.slot_range(
            self._config.rollout_participant, destination_rank
        )
        expected_slots = tuple(range(start, end))
        if len(fragments) == 1 and fragments[0][1].slot_ids == expected_slots:
            return fragments[0][1]
        output_layout = replace(layout, batch_size=end - start)
        if self._policy_output_buffers is None:
            self._policy_output_buffers = output_layout.allocate_buffers()
        return _merge_policy_fragments(
            output_layout,
            self._policy_output_buffers,
            tuple(fragments),
            expected_slots,
        )


def _merge_policy_fragments(
    layout: PolicyInputLayout,
    buffers: tuple[torch.Tensor, ...],
    fragments: tuple[tuple[Route, PolicyInput], ...],
    slot_ids: tuple[int, ...],
) -> PolicyInput:
    """Assemble routed fragments into one reusable Rollout-owned batch."""
    _header, main, wrist, extra, states, rlt, intervene, _descriptions = buffers
    task_descriptions = [""] * layout.batch_size
    first = fragments[0][1]
    has_rlt = first.rlt_switch_flags is not None
    has_intervene = first.intervene_requested is not None
    for route, fragment in fragments:
        if fragment.global_step != first.global_step or (
            fragment.rollout_epoch,
            fragment.chunk_step,
        ) != (first.rollout_epoch, first.chunk_step):
            raise ValueError("direct policy fragments have inconsistent step metadata")
        if (fragment.rlt_switch_flags is not None) != has_rlt or (
            fragment.intervene_requested is not None
        ) != has_intervene:
            raise ValueError(
                "direct policy fragments have inconsistent optional fields"
            )
        indices = list(route.destination_indices)
        main[indices] = fragment.observations["main_images"]
        wrist[indices] = fragment.observations["wrist_images"]
        if layout.extra_view_shape is not None:
            extra[indices] = fragment.observations["extra_view_images"]
        states[indices] = fragment.observations["states"]
        if has_rlt:
            rlt[indices] = fragment.rlt_switch_flags
        if has_intervene:
            intervene[indices] = fragment.intervene_requested
        for index, description in zip(
            indices, fragment.observations["task_descriptions"], strict=True
        ):
            task_descriptions[index] = description
    return PolicyInput(
        global_step=first.global_step,
        rollout_epoch=first.rollout_epoch,
        chunk_step=first.chunk_step,
        slot_ids=slot_ids,
        observations={
            "main_images": main,
            "wrist_images": wrist,
            "extra_view_images": extra if layout.extra_view_shape is not None else None,
            "states": states,
            "task_descriptions": task_descriptions,
        },
        rlt_switch_flags=rlt if has_rlt else None,
        intervene_requested=intervene if has_intervene else None,
    )


def select_policy_data(
    data: PolicyInput | PolicyOutput,
    indices: tuple[int, ...],
) -> PolicyInput | PolicyOutput:
    """Select an ordered slot subset without adding protocol metadata."""
    selected = {
        "slot_ids": tuple(data.slot_ids[index] for index in indices),
    }
    if isinstance(data, PolicyInput):
        selected.update(
            observations=_select(data.observations, indices),
            rlt_switch_flags=_select(data.rlt_switch_flags, indices),
            intervene_requested=_select(data.intervene_requested, indices),
        )
    else:
        selected["actions"] = _select(data.actions, indices)
    return replace(data, **selected)


def select_value_request(
    data: ValueRequest,
    indices: tuple[int, ...],
) -> ValueRequest:
    """Select sparse timeout/tail slots and their observations."""
    return replace(
        data,
        slot_ids=tuple(data.slot_ids[index] for index in indices),
        observations=_select(data.observations, indices),
    )


def pack_policy_data(
    data: PolicyInput | PolicyOutput | ValueRequest,
) -> _PolicyTransfer:
    """Move all live tensors into collective-optimized top-level storage."""
    skeleton: dict[str, Any] = {}
    paths: list[tuple[str, ...]] = []
    tensors: list[torch.Tensor] = []
    for field in fields(data):
        if field.name in {
            "global_step",
            "rollout_epoch",
            "chunk_step",
            "slot_ids",
        }:
            continue
        skeleton[field.name] = _extract_tensors(
            getattr(data, field.name),
            (field.name,),
            paths,
            tensors,
        )
    return _PolicyTransfer(
        data_type=type(data).__name__,
        global_step=data.global_step,
        rollout_epoch=data.rollout_epoch,
        chunk_step=data.chunk_step,
        slot_ids=data.slot_ids,
        skeleton=skeleton,
        tensor_paths=tuple(paths),
        tensors=tuple(tensors),
    )


def unpack_policy_data(
    transfer: _PolicyTransfer,
) -> PolicyInput | PolicyOutput | ValueRequest:
    """Restore a typed live object received through worker collectives."""
    if not isinstance(transfer, _PolicyTransfer):
        raise TypeError(f"expected _PolicyTransfer, got {type(transfer).__name__}")
    values = _copy_tree(transfer.skeleton)
    for path, tensor in zip(transfer.tensor_paths, transfer.tensors, strict=True):
        _assign(values, path, tensor)
    values.update(
        global_step=transfer.global_step,
        rollout_epoch=transfer.rollout_epoch,
        chunk_step=transfer.chunk_step,
        slot_ids=transfer.slot_ids,
    )
    data_types = {
        "PolicyInput": PolicyInput,
        "PolicyOutput": PolicyOutput,
        "ValueRequest": ValueRequest,
    }
    try:
        data_type = data_types[transfer.data_type]
    except KeyError as error:
        raise ValueError(f"unknown live data type {transfer.data_type!r}") from error
    return data_type(**values)


def pack_value_request(data: ValueRequest) -> _PolicyTransfer:
    """Pack one value request for the collective tensor fast path."""
    return pack_policy_data(data)


def unpack_value_request(transfer: _PolicyTransfer) -> ValueRequest:
    """Restore one value request and enforce its business type."""
    data = unpack_policy_data(transfer)
    if not isinstance(data, ValueRequest):
        raise TypeError(f"expected ValueRequest, got {type(data).__name__}")
    return data


def _select(value: Any, indices: tuple[int, ...]) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, dict):
        return {key: _select(child, indices) for key, child in value.items()}
    if isinstance(value, list):
        return [value[index] for index in indices]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    raise TypeError(f"cannot select live field of type {type(value).__name__}")


def _extract_tensors(
    value: Any,
    path: tuple[str, ...],
    paths: list[tuple[str, ...]],
    tensors: list[torch.Tensor],
) -> Any:
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous():
            raise ValueError(f"live tensor {'.'.join(path)!r} must be contiguous")
        paths.append(path)
        tensors.append(value)
        return None
    if isinstance(value, dict):
        return {
            key: _extract_tensors(child, (*path, key), paths, tensors)
            for key, child in value.items()
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if any(isinstance(item, torch.Tensor) for item in value):
            raise TypeError("live tensor lists must be stacked before publish")
        return list(value)
    if isinstance(value, tuple):
        if any(isinstance(item, torch.Tensor) for item in value):
            raise TypeError("live tensor tuples must be stacked before publish")
        return tuple(value)
    raise TypeError(
        f"unsupported live field {'.'.join(path)!r}: {type(value).__name__}"
    )


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def _assign(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

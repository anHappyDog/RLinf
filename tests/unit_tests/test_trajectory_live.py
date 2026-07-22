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

from dataclasses import fields

import pytest
import ray
import torch

from rlinf.data.trajectory import EnvResult, PolicyInput, PolicyOutput
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.workers.trajectory import (
    ChannelConfig,
    PolicyInputLayout,
    RoutePlan,
    TrajectoryChannel,
    TrajectoryChannelWorker,
    WorkerLayout,
    WorkerState,
    select_policy_data,
)
from rlinf.workers.trajectory.live import (
    _IMAGE_LZ4,
    _IMAGE_XOR_LZ4,
    _LiveImageCodec,
    pack_policy_data,
    unpack_policy_data,
)


@pytest.fixture(scope="module")
def cluster():
    cluster = Cluster(num_nodes=1)
    yield cluster


def _policy_input(slot_ids: tuple[int, ...] = (0, 1)) -> PolicyInput:
    batch_size = len(slot_ids)
    return PolicyInput(
        global_step=4,
        rollout_epoch=2,
        chunk_step=3,
        slot_ids=slot_ids,
        observations={
            "main_images": torch.arange(
                batch_size * 2 * 2 * 3, dtype=torch.uint8
            ).reshape(batch_size, 2, 2, 3),
            "states": torch.arange(batch_size * 8, dtype=torch.float32).reshape(
                batch_size, 8
            ),
            "task_descriptions": [f"task {slot_id}" for slot_id in slot_ids],
        },
        rlt_switch_flags=torch.zeros(batch_size, dtype=torch.bool),
    )


def _policy_output(slot_ids: tuple[int, ...] = (0, 1)) -> PolicyOutput:
    return PolicyOutput(
        global_step=4,
        rollout_epoch=2,
        chunk_step=3,
        slot_ids=slot_ids,
        actions=torch.arange(len(slot_ids) * 5 * 7, dtype=torch.float32).reshape(
            len(slot_ids), 5, 7
        ),
    )


def test_select_policy_data_preserves_order_and_static_text():
    selected = select_policy_data(_policy_input((4, 7, 9)), (2, 0))

    assert selected.slot_ids == (9, 4)
    assert selected.observations["task_descriptions"] == ["task 9", "task 4"]
    assert torch.equal(
        selected.observations["states"],
        torch.tensor(
            [[16, 17, 18, 19, 20, 21, 22, 23], [0, 1, 2, 3, 4, 5, 6, 7]],
            dtype=torch.float32,
        ),
    )


def test_live_types_exclude_trajectory_only_fields():
    assert {field.name for field in fields(PolicyInput)} == {
        "global_step",
        "rollout_epoch",
        "chunk_step",
        "slot_ids",
        "observations",
        "rlt_switch_flags",
        "intervene_requested",
    }
    assert {field.name for field in fields(PolicyOutput)} == {
        "global_step",
        "rollout_epoch",
        "chunk_step",
        "slot_ids",
        "actions",
    }


def test_collective_transfer_extracts_nested_tensors_without_copy():
    policy_input = _policy_input()
    transfer = pack_policy_data(policy_input)

    assert transfer.tensor_paths == (
        ("observations", "main_images"),
        ("observations", "states"),
        ("rlt_switch_flags",),
    )
    assert transfer.tensors[0] is policy_input.observations["main_images"]
    assert transfer.tensors[1] is policy_input.observations["states"]
    assert transfer.skeleton["observations"]["main_images"] is None
    assert transfer.skeleton["observations"]["task_descriptions"] == [
        "task 0",
        "task 1",
    ]

    restored = unpack_policy_data(transfer)
    assert restored.slot_ids == policy_input.slot_ids
    assert restored.observations["main_images"] is transfer.tensors[0]
    assert restored.observations["task_descriptions"] == ["task 0", "task 1"]


def test_direct_policy_layout_round_trip_preserves_optional_fields_and_text():
    layout = PolicyInputLayout(
        batch_size=2,
        image_shape=(2, 2, 3),
        state_shape=(8,),
        max_description_bytes=32,
    )
    request = _policy_input()
    request.observations["wrist_images"] = request.observations["main_images"] + 1
    request.observations["extra_view_images"] = None
    request.observations["task_descriptions"] = ["task zero", "任务一"]
    request.intervene_requested = torch.tensor([False, True])
    send_workspace = layout.allocate_send_workspace()
    receive_buffers = layout.allocate_buffers()

    encoded = layout.encode(request, 3, send_workspace)
    for source, destination in zip(encoded, receive_buffers, strict=True):
        destination.copy_(source)
    restored = layout.decode(receive_buffers, 3)

    assert restored.slot_ids == request.slot_ids
    assert restored.observations["task_descriptions"] == ["task zero", "任务一"]
    assert torch.equal(
        restored.observations["main_images"], request.observations["main_images"]
    )
    assert torch.equal(
        restored.observations["wrist_images"], request.observations["wrist_images"]
    )
    assert torch.equal(restored.intervene_requested, request.intervene_requested)


def test_live_image_codec_restores_keyframe_and_xor_frame_bitwise():
    layout = PolicyInputLayout(
        batch_size=2,
        image_shape=(8, 8, 3),
        state_shape=(8,),
        compress_images=True,
    )
    encoder = _LiveImageCodec(layout)
    decoder = _LiveImageCodec(layout)
    first = torch.zeros((2, 8, 8, 3), dtype=torch.uint8)
    second = first.clone()
    second[:, 0, 0] = 7

    first_wire = encoder.encode((first, first))
    first_payloads = decoder.wire_buffers((first_wire[0][2], first_wire[1][2]))
    for encoded, received in zip(first_wire, first_payloads, strict=True):
        received.copy_(encoded[0])
    first_outputs = (torch.empty_like(first), torch.empty_like(first))
    decoder.decode(
        first_payloads,
        (first_wire[0][1], first_wire[1][1]),
        first_outputs,
    )

    second_wire = encoder.encode((second, second))
    second_payloads = decoder.wire_buffers((second_wire[0][2], second_wire[1][2]))
    for encoded, received in zip(second_wire, second_payloads, strict=True):
        received.copy_(encoded[0])
    second_outputs = (torch.empty_like(second), torch.empty_like(second))
    decoder.decode(
        second_payloads,
        (second_wire[0][1], second_wire[1][1]),
        second_outputs,
    )

    assert first_wire[0][1] == _IMAGE_LZ4
    assert second_wire[0][1] == _IMAGE_XOR_LZ4
    assert torch.equal(first_outputs[0], first)
    assert torch.equal(second_outputs[0], second)


def test_direct_policy_routes_support_many_to_one_and_one_to_many():
    layout = PolicyInputLayout(batch_size=2, image_shape=(2, 2, 3), state_shape=(8,))
    many_to_one = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(4, {"env": 2, "rollout": 1}),
        env_layout=WorkerLayout((0, 1)),
        rollout_layout=WorkerLayout((0,)),
        env_group_name="env",
        rollout_group_name="rollout",
        policy_input_layout=layout,
    )
    one_to_many = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(2, {"env": 1, "rollout": 2}),
        env_layout=WorkerLayout((0,)),
        rollout_layout=WorkerLayout((0, 1)),
        env_group_name="env",
        rollout_group_name="rollout",
        policy_input_layout=layout,
    )

    assert [source for source, _route in many_to_one.direct_policy_sources(0)] == [
        0,
        1,
    ]
    assert [
        route.destination_rank for route in one_to_many.direct_policy_routes(0)
    ] == [
        0,
        1,
    ]


def test_direct_policy_capacity_accepts_uneven_env_batches():
    config = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(5, {"env": 2, "rollout": 3}),
        env_layout=WorkerLayout((0, 1)),
        rollout_layout=WorkerLayout((0, 1, 2)),
        env_group_name="env",
        rollout_group_name="rollout",
        policy_input_layout=PolicyInputLayout(
            batch_size=3, image_shape=(2, 2, 3), state_shape=(8,)
        ),
    )

    assert config.route_plan.slot_range("env", 0) == (0, 3)
    assert config.route_plan.slot_range("env", 1) == (3, 5)


def test_channel_routes_policy_input_and_output_by_slots(cluster):
    group = TrajectoryChannelWorker.create_group(maxsize=4).launch(
        cluster,
        NodePlacementStrategy([0]),
        name="trajectory_live_routing",
        max_concurrency=8,
        catch_system_failure=False,
    )
    config = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(7, {"env": 2, "rollout": 3}),
        env_layout=WorkerLayout((0, 2)),
        rollout_layout=WorkerLayout((1, 3, 5)),
    )
    group.configure(config).wait()
    actor = group.worker_info_list[0].worker

    ray.get(actor.publish_policy_input_via_ray.remote(_policy_input((0, 1, 2, 3)), 0))
    rollout_zero = ray.get(actor.take_policy_input_via_ray.remote(0))
    rollout_one = ray.get(actor.take_policy_input_via_ray.remote(1))
    assert rollout_zero.slot_ids == (0, 1, 2)
    assert rollout_one.slot_ids == (3,)

    ray.get(actor.publish_policy_output_via_ray.remote(_policy_output((0, 1, 2)), 0))
    env_zero = ray.get(actor.take_policy_output_via_ray.remote(0))
    assert env_zero.slot_ids == (0, 1, 2)
    assert torch.equal(env_zero.actions, _policy_output((0, 1, 2)).actions)

    with pytest.raises(ray.exceptions.RayTaskError, match="requires PolicyInput"):
        invalid = EnvResult(
            global_step=4,
            rollout_epoch=2,
            chunk_step=3,
            slot_ids=(0,),
            rewards=torch.zeros(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
        )
        ray.get(actor.publish_policy_input_via_ray.remote(invalid, 0))

    group.shutdown().wait()
    group._close()


def test_trajectory_channel_facade_round_trip_and_lifecycle_gate(cluster):
    group = TrajectoryChannelWorker.create_group(maxsize=2).launch(
        cluster,
        NodePlacementStrategy([0]),
        name="trajectory_live_facade",
        max_concurrency=8,
        catch_system_failure=False,
    )
    config = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(2, {"env": 1, "rollout": 1}),
        env_layout=WorkerLayout((0,)),
        rollout_layout=WorkerLayout((0,)),
    )
    group.configure(config).wait()
    channel = TrajectoryChannel.from_worker_group(group, config)

    policy_input = _policy_input()
    channel.publish_policy_input(policy_input)
    received_input = channel.take_policy_input()
    assert received_input.slot_ids == policy_input.slot_ids
    assert torch.equal(
        received_input.observations["main_images"],
        policy_input.observations["main_images"],
    )

    policy_output = _policy_output()
    channel.publish_policy_output(policy_output)
    received_output = channel.take_policy_output()
    assert torch.equal(received_output.actions, policy_output.actions)

    assert group.drain().wait()[0].state is WorkerState.DRAINING
    actor = group.worker_info_list[0].worker
    with pytest.raises(ray.exceptions.RayTaskError, match="not ready"):
        ray.get(actor.publish_policy_input_via_ray.remote(policy_input, 0))
    group.shutdown().wait()
    group._close()

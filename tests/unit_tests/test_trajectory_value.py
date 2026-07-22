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

import pytest
import ray
import torch

from rlinf.data.trajectory import ValueRequest, ValueResult
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.workers.trajectory import (
    ChannelConfig,
    EndpointSchema,
    RoutePlan,
    StorageConfig,
    TrajectoryChannel,
    TrajectoryChannelWorker,
    TrajectoryStorage,
    TransportEndpoint,
    WorkerLayout,
    infer_value_request,
    ingest_storage_data,
)


class FakeValueModel:
    def __init__(self) -> None:
        self.value_head = object()
        self.observations = None

    def predict_action_batch(
        self,
        env_obs,
        mode,
        compute_values,
    ):
        self.observations = env_obs
        assert mode == "train"
        assert compute_values
        batch = env_obs["states"].shape[0]
        return torch.zeros(batch, 5, 7), {
            "prev_values": torch.arange(batch, dtype=torch.float32).reshape(batch, 1)
            + 0.5
        }


@pytest.fixture(scope="module")
def cluster():
    cluster = Cluster(num_nodes=1)
    yield cluster


def _request(
    slot_ids: tuple[int, ...] = (1, 3),
    kind: str = "timeout",
) -> ValueRequest:
    batch = len(slot_ids)
    return ValueRequest(
        global_step=6,
        rollout_epoch=0,
        chunk_step=0 if kind == "timeout" else 1,
        slot_ids=slot_ids,
        kind=kind,
        observations={
            "states": torch.arange(batch * 8, dtype=torch.float32).reshape(batch, 8),
            "main_images": torch.arange(batch * 2 * 2 * 3, dtype=torch.uint8).reshape(
                batch, 2, 2, 3
            ),
            "task_descriptions": [f"terminal {slot_id}" for slot_id in slot_ids],
        },
    )


def test_trajectory_channel_routes_only_selected_sparse_value_slots(cluster):
    group = TrajectoryChannelWorker.create_group(maxsize=4).launch(
        cluster,
        NodePlacementStrategy([0]),
        name="trajectory_value_requests",
        max_concurrency=8,
        catch_system_failure=False,
    )
    config = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(4, {"env": 1, "rollout": 2}),
        env_layout=WorkerLayout((0,)),
        rollout_layout=WorkerLayout((0, 1)),
    )
    group.configure(config).wait()
    actor = group.worker_info_list[0].worker

    request = _request()
    ray_result = actor.publish_value_request_via_ray.remote(request, 0)
    ray.get(ray_result)
    rollout_zero = ray.get(actor.take_value_request_via_ray.remote(0))
    rollout_one = ray.get(actor.take_value_request_via_ray.remote(1))
    assert rollout_zero.slot_ids == (1,)
    assert rollout_one.slot_ids == (3,)
    assert rollout_zero.observations["task_descriptions"] == ["terminal 1"]

    channel = TrajectoryChannel.from_worker_group(group, config)
    tail = _request((0, 2), kind="tail")
    channel.publish_value_request(tail)
    assert channel.take_value_request().slot_ids == (0,)
    assert ray.get(actor.take_value_request_via_ray.remote(1)).slot_ids == (2,)

    group.shutdown().wait()
    group._close()


def test_infer_value_request_uses_real_model_value_path_without_fallback():
    model = FakeValueModel()
    request = _request()

    result = infer_value_request(model, request)

    assert isinstance(result, ValueResult)
    assert result.kind == request.kind
    assert result.slot_ids == request.slot_ids
    assert result.values.shape == (2, 1)
    assert torch.equal(result.values, torch.tensor([[0.5], [1.5]]))
    assert model.observations is request.observations

    model.value_head = None
    del model.value_head
    with pytest.raises(RuntimeError, match="no value_head"):
        infer_value_request(model, request)


def test_value_result_uses_storage_bypass_and_boundary_schema():
    storage = TrajectoryStorage(
        StorageConfig(
            global_step=6,
            rollout_epochs=1,
            chunk_steps=1,
            slot_ids=(0, 1, 2, 3),
            boundary_values=True,
        )
    )
    value = ValueResult(
        global_step=6,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(1, 3),
        kind="timeout",
        values=torch.tensor([[2.0], [4.0]]),
    )
    schema = EndpointSchema.from_example(9, 2, value)
    sender = TransportEndpoint(schema)
    receiver = TransportEndpoint(schema)
    prepared = sender.encode(value)
    buffers = receiver.allocate_receive_buffers()
    buffers.header.copy_(prepared.header)
    for source, destination in zip(
        prepared.payloads, receiver.payload_views(buffers), strict=True
    ):
        destination.copy_(source)

    ack = ingest_storage_data(storage, receiver.decode(buffers))
    assert ack.inserted
    assert not ack.trajectory_ready

#!/usr/bin/env python3
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

import torch

from rlinf.data.trajectory import EnvResult, RewardResult, RolloutResult
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.workers.trajectory import (
    EndpointSchema,
    RoutePlan,
    StorageConfig,
    StorageWorkerConfig,
    SubmitStatus,
    TrajectoryStorageWorker,
    TrajectoryWriter,
    WorkerLayout,
)


class RecordProducer(Worker):
    """Minimal producer used to verify the real fixed-tensor data plane."""

    def write(self, writer: TrajectoryWriter, data):
        return writer.submit(data, wait_for=SubmitStatus.INGESTED)


def records():
    coordinates = {
        "global_step": 1,
        "rollout_epoch": 0,
        "chunk_step": 0,
        "slot_ids": (0, 1),
    }
    env = EnvResult(
        **coordinates,
        rewards=torch.tensor([[1.0], [2.0]]),
        dones=torch.zeros(2, 1, dtype=torch.bool),
        terminations=torch.zeros(2, 1, dtype=torch.bool),
        truncations=torch.zeros(2, 1, dtype=torch.bool),
    )
    rollout = RolloutResult(
        **coordinates,
        actions=torch.arange(70, dtype=torch.float32).reshape(2, 5, 7),
    )
    reward = RewardResult(
        **coordinates,
        rewards=torch.tensor([[3.0], [4.0]]),
        mode="per_step",
    )
    return env, rollout, reward


def main() -> None:
    cluster = Cluster(num_nodes=1)
    placement = NodePlacementStrategy([0])
    route_plan = RoutePlan(
        2,
        {"env": 1, "rollout": 1, "reward": 1, "storage": 1},
    )
    layout = WorkerLayout((0,))
    env, rollout, reward = records()
    schemas = tuple(
        EndpointSchema.from_example(schema_id, 2, data)
        for schema_id, data in enumerate((env, rollout, reward), start=1)
    )
    storage = TrajectoryStorageWorker.create_group().launch(
        cluster,
        placement,
        name="bypass_verify_storage",
        catch_system_failure=False,
    )
    storage.configure(
        StorageWorkerConfig(
            layout=layout,
            route_plan=route_plan,
            storage=StorageConfig(
                global_step=1,
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
                reward_mode="per_step",
                reward_steps=(0,),
            ),
            endpoints=schemas,
        )
    ).wait()

    producers = {}
    for participant in ("env", "rollout", "reward"):
        producers[participant] = RecordProducer.create_group().launch(
            cluster,
            placement,
            name=f"bypass_verify_{participant}",
            catch_system_failure=False,
        )

    writers = {
        participant: TrajectoryWriter.from_worker_group(
            storage,
            route_plan=route_plan,
            source_participant=participant,
            source_layout=layout,
            storage_layout=layout,
            schemas_by_rank={0: (schema,)},
        )
        for participant, schema in zip(
            ("env", "rollout", "reward"), schemas, strict=True
        )
    }
    for participant, data in (
        ("reward", reward),
        ("rollout", rollout),
        ("env", env),
    ):
        ack = producers[participant].write(writers[participant], data).wait()[0][0]
        assert ack.inserted

    retry = producers["env"].write(writers["env"], env).wait()[0][0]
    assert not retry.inserted
    assert storage.trajectory_ready().wait() == [True]

    for producer in producers.values():
        producer._close()
    storage.shutdown().wait()
    storage._close()
    print("trajectory bypass verification passed")


if __name__ == "__main__":
    main()

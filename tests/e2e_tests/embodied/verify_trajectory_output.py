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

import importlib

import ray
import torch

from rlinf.data.trajectory import EnvResult, RolloutResult
from rlinf.models.embodiment.openpi.forward_inputs import OpenPILiberoForwardInputs
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.workers.trajectory import (
    CompressionConfig,
    RoutePlan,
    StorageConfig,
    StorageWorkerConfig,
    TrajectoryReader,
    TrajectoryStorageWorker,
    WorkerLayout,
)


class OutputStorageWorker(TrajectoryStorageWorker):
    """Storage worker with a test-only direct data loader."""

    def load(self, env: EnvResult, rollout: RolloutResult) -> bool:
        assert self._storage is not None
        self._storage.write(env)
        self._storage.write(rollout)
        return self._storage.ready


class OutputActorWorker(Worker):
    """Actor-side receiver exercising the production reader path."""

    def pull(self, reader: TrajectoryReader) -> tuple:
        importlib.import_module("rlinf.models.embodiment.openpi.forward_inputs")
        batch = reader.pull()
        assert batch.forward_inputs is not None
        return (
            batch.slot_ids,
            batch.actions[0, 0, :, 0].tolist(),
            batch.observations["task_descriptions"],
            batch.forward_inputs.action[:, 0].tolist(),
        )


def records() -> tuple[EnvResult, RolloutResult]:
    """Build mixed nested metadata and typed model inputs."""
    slots = (0, 1)
    values = torch.tensor(slots, dtype=torch.float32).reshape(2, 1)
    forward_inputs = OpenPILiberoForwardInputs(
        chains=values[:, None, None, :].expand(2, 5, 2, 1).clone(),
        denoise_inds=torch.arange(4).expand(2, 4).clone(),
        tokenized_prompt=torch.arange(3).expand(2, 3).clone(),
        tokenized_prompt_mask=torch.ones(2, 3, dtype=torch.bool),
        action=values.expand(2, 7).clone(),
        model_action=(values + 1).expand(2, 7).clone(),
        image=torch.zeros(2, 2, 2, 3, dtype=torch.uint8),
        wrist_image=torch.ones(2, 2, 2, 3, dtype=torch.uint8),
        state=values.expand(2, 8).clone(),
    )
    coordinates = {
        "global_step": 3,
        "rollout_epoch": 0,
        "chunk_step": 0,
        "slot_ids": slots,
    }
    env = EnvResult(
        **coordinates,
        rewards=values,
        dones=torch.zeros(2, 1, dtype=torch.bool),
        terminations=torch.zeros(2, 1, dtype=torch.bool),
        truncations=torch.zeros(2, 1, dtype=torch.bool),
        observations={
            "states": values.expand(2, 8).clone(),
            "task_descriptions": ["slot 0", "slot 1"],
        },
    )
    rollout = RolloutResult(
        **coordinates,
        actions=values.expand(2, 7).clone(),
        forward_inputs=forward_inputs,
        state_values=values / 10,
    )
    return env, rollout


def main() -> None:
    cluster = Cluster(num_nodes=1)
    placement = NodePlacementStrategy([0])
    route_plan = RoutePlan(2, {"storage": 1, "actor": 1})
    layout = WorkerLayout((0,))
    compression = CompressionConfig(
        enabled=True,
        codec="lz4",
        min_bytes=1,
        block_bytes=64,
        num_threads=2,
    )
    storage = OutputStorageWorker.create_group().launch(
        cluster,
        placement,
        name="trajectory_output_storage",
        catch_system_failure=False,
    )
    actor = OutputActorWorker.create_group().launch(
        cluster,
        placement,
        name="trajectory_output_actor",
        catch_system_failure=False,
    )
    storage.configure(
        StorageWorkerConfig(
            layout=layout,
            route_plan=route_plan,
            storage=StorageConfig(
                global_step=3,
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0, 1),
                env_fields=frozenset({"observations"}),
                rollout_fields=frozenset({"forward_inputs", "state_values"}),
            ),
            compression=compression,
        )
    ).wait()
    assert storage.load(*records()).wait() == [True]
    reader = TrajectoryReader.from_worker_group(
        storage,
        route_plan=route_plan,
        storage_layout=layout,
        actor_layout=layout,
        compression=compression,
    )

    result = ray.get(actor.worker_info_list[0].worker.pull.remote(reader))
    assert result == (
        (0, 1),
        [0.0, 1.0],
        [[["slot 0", "slot 1"]]],
        [0.0, 1.0],
    )
    metrics = storage.metrics().wait()[0]
    assert metrics["compression_raw_bytes"] > 0
    assert metrics["compression_wire_bytes"] < metrics["compression_raw_bytes"]
    assert metrics["compression_compressed_blocks"] > 0
    assert metrics["compression_workspace_allocations"] > 0

    storage.shutdown().wait()
    storage._close()
    actor._close()
    print("trajectory output verification passed")


if __name__ == "__main__":
    main()

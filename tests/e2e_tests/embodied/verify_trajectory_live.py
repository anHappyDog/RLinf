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

from rlinf.data.trajectory import PolicyInput
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.workers.trajectory import (
    ChannelConfig,
    PolicyInputLayout,
    RoutePlan,
    TrajectoryChannel,
    TrajectoryChannelWorker,
    WorkerLayout,
)


class DirectEnvWorker(Worker):
    """Send fixed policy inputs from one persistent endpoint."""

    def publish(self, channel, requests) -> None:
        for request in requests:
            channel.publish_policy_input(request)

    def publish_generated(
        self, channel, count: int, total_slots: int, env_ranks: int
    ) -> None:
        size, remainder = divmod(total_slots, env_ranks)
        slot_start = self._rank * size + min(self._rank, remainder)
        slot_end = slot_start + size + (self._rank < remainder)
        slot_ids = tuple(range(slot_start, slot_end))
        for chunk_step in range(count):
            channel.publish_policy_input(policy_input(chunk_step, slot_ids))


class DirectRolloutWorker(Worker):
    """Receive fixed policy inputs from one persistent endpoint."""

    def take(self, channel, count):
        received = []
        for _ in range(count):
            request = channel.take_policy_input()
            assert request.observations["main_images"].is_pinned()
            observations = {
                key: (
                    value.clone()
                    if isinstance(value, torch.Tensor)
                    else list(value)
                    if value is not None
                    else None
                )
                for key, value in request.observations.items()
            }
            received.append(
                PolicyInput(
                    global_step=request.global_step,
                    rollout_epoch=request.rollout_epoch,
                    chunk_step=request.chunk_step,
                    slot_ids=request.slot_ids,
                    observations=observations,
                    rlt_switch_flags=(
                        request.rlt_switch_flags.clone()
                        if request.rlt_switch_flags is not None
                        else None
                    ),
                    intervene_requested=(
                        request.intervene_requested.clone()
                        if request.intervene_requested is not None
                        else None
                    ),
                )
            )
        return tuple(received)


def policy_input(chunk_step: int, slot_ids: tuple[int, ...] = (0, 1)) -> PolicyInput:
    """Build one deterministic request with the target observation fields."""
    main = torch.stack(
        [torch.full((2, 2, 3), slot_id, dtype=torch.uint8) for slot_id in slot_ids]
    )
    return PolicyInput(
        global_step=4,
        rollout_epoch=2,
        chunk_step=chunk_step,
        slot_ids=slot_ids,
        observations={
            "main_images": main,
            "wrist_images": main + chunk_step + 1,
            "extra_view_images": main[:, None] + 2,
            "states": torch.stack(
                [torch.full((8,), slot_id, dtype=torch.float32) for slot_id in slot_ids]
            ),
            "task_descriptions": [f"task {slot_id}" for slot_id in slot_ids],
        },
        rlt_switch_flags=torch.tensor(
            [slot_id % 2 == 1 for slot_id in slot_ids], dtype=torch.bool
        ),
    )


def run_case(
    cluster: Cluster, env_ranks: int, rollout_ranks: int, total_slots: int
) -> None:
    suffix = f"{env_ranks}_to_{rollout_ranks}"
    channel_group = TrajectoryChannelWorker.create_group(maxsize=2).launch(
        cluster,
        NodePlacementStrategy([0]),
        name=f"trajectory_live_control_{suffix}",
        max_concurrency=8,
        catch_system_failure=False,
    )
    env_group = DirectEnvWorker.create_group().launch(
        cluster,
        NodePlacementStrategy([0] * env_ranks),
        name=f"trajectory_live_env_{suffix}",
        catch_system_failure=False,
    )
    rollout_group = DirectRolloutWorker.create_group().launch(
        cluster,
        NodePlacementStrategy([0] * rollout_ranks),
        name=f"trajectory_live_rollout_{suffix}",
        catch_system_failure=False,
    )
    env_layout = WorkerLayout(tuple(range(env_ranks)))
    rollout_layout = WorkerLayout(tuple(range(rollout_ranks)))
    env_capacity = (total_slots + env_ranks - 1) // env_ranks
    config = ChannelConfig(
        layout=WorkerLayout((0,)),
        route_plan=RoutePlan(total_slots, {"env": env_ranks, "rollout": rollout_ranks}),
        env_layout=env_layout,
        rollout_layout=rollout_layout,
        env_group_name=env_group.worker_group_name,
        rollout_group_name=rollout_group.worker_group_name,
        policy_input_layout=PolicyInputLayout(
            batch_size=env_capacity,
            image_shape=(2, 2, 3),
            state_shape=(8,),
            extra_view_shape=(1, 2, 2, 3),
            max_description_bytes=32,
            compress_images=True,
            pin_memory=True,
        ),
    )
    channel_group.configure(config).wait()
    channel = TrajectoryChannel.from_worker_group(channel_group, config)
    count = 2

    pending_receive = rollout_group.take(channel, count)
    pending_send = env_group.publish_generated(channel, count, total_slots, env_ranks)
    received_by_rank = pending_receive.wait()
    pending_send.wait()

    for rank, received in enumerate(received_by_rank):
        start, end = config.route_plan.slot_range("rollout", rank)
        for chunk_step, actual in enumerate(received):
            expected = policy_input(chunk_step, tuple(range(start, end)))
            assert actual.chunk_step == expected.chunk_step
            assert actual.slot_ids == expected.slot_ids
            assert (
                actual.observations["task_descriptions"]
                == expected.observations["task_descriptions"]
            )
            for key in (
                "main_images",
                "wrist_images",
                "extra_view_images",
                "states",
            ):
                assert torch.equal(actual.observations[key], expected.observations[key])

    channel_group.shutdown().wait()
    channel_group._close()
    env_group._close()
    rollout_group._close()


def main() -> None:
    cluster = Cluster(num_nodes=1)
    run_case(cluster, env_ranks=2, rollout_ranks=1, total_slots=4)
    run_case(cluster, env_ranks=1, rollout_ranks=2, total_slots=4)
    run_case(cluster, env_ranks=2, rollout_ranks=3, total_slots=5)
    print("trajectory direct live verification passed")


if __name__ == "__main__":
    main()

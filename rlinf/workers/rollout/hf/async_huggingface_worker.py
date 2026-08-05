# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    PolicyInput,
    PolicyOutput,
    TrajectoryEnd,
    merge_policy_inputs,
)
from rlinf.scheduler import Channel, TrajectoryChannel, Worker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        # set the decoupled rollout worker sync weight time
        self.sync_rollout_weight_time = (
            self.num_pipeline_stages * self.n_train_chunk_steps * self.rollout_epoch
        )

        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0
        self._trajectory_step = 0

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        trajectory_channel: TrajectoryChannel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(
                input_channel, output_channel, metric_channel, trajectory_channel
            )
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        trajectory_channel: TrajectoryChannel,
    ):
        if self.env_decoupled_mode:
            await self.decoupled_generate_one_epoch(
                input_channel, output_channel, trajectory_channel
            )
        else:
            while True:
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()

                step_id = self._trajectory_step
                self._trajectory_step += 1
                for epoch_id in range(self.rollout_epoch):
                    await self.generate_one_epoch(
                        input_channel,
                        output_channel,
                        trajectory_channel,
                        step_id,
                        epoch_id,
                    )
                for stage_id in range(self.num_pipeline_stages):
                    trajectory_channel.publish(
                        TrajectoryEnd(
                            step_id=step_id,
                            source=(self._rank, stage_id),
                        )
                    )
                if self.finished_episodes is not None:
                    self.finished_episodes += (
                        self.total_num_train_envs * self.rollout_epoch
                    )
                rollout_metrics = self.pop_execution_times()
                rollout_metrics = {
                    f"time/rollout/{k}": v for k, v in rollout_metrics.items()
                }
                metric_channel.put(
                    {"rank": self._rank, "time": rollout_metrics},
                    async_op=True,
                )

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.total_num_train_envs
                * self.rollout_epoch
            )
            if (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            ):
                break
            await asyncio.sleep(0.01)

    def stop(self):
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    async def _recv_and_apply_actor_sync(self) -> int:
        await super().sync_model_from_actor()
        return self.version

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self._recv_and_apply_actor_sync())

    @Worker.timer("rollout/poll_weight_sync")
    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None:
            return

        if not self._weight_sync_work.done():
            return

        await self._weight_sync_work
        self._weight_sync_work = None
        self._weight_sync_apply_total += 1

        self._start_background_weight_sync_if_needed()

    @Worker.timer("rollout/request_weight_sync")
    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    async def decoupled_generate_one_epoch(
        self,
        input_channel: Channel,
        output_channel: Channel,
        trajectory_channel: TrajectoryChannel,
    ):
        self.update_dagger_beta()
        decoupled_generate_time = 1
        pending = None

        async def receive_policy_input(tag: str) -> tuple[PolicyInput, list[int]]:
            return await self.recv_from_and_record_batch_routes_with_timeout(
                group_name=self.cfg.env.group_name,
                channel=input_channel,
                tag=tag,
                batch_size=self.train_batch_size,
                merge_fn=merge_policy_inputs,
                infer_batch_size_fn=self._infer_policy_input_batch_size,
                timeout_time=0.02,
                recv_queue_size=self.rollout_queue_size,
            )

        receive_tasks = {
            asyncio.create_task(receive_policy_input(tag)): tag
            for tag in ("policy", "policy_final")
        }
        try:
            while True:
                completed_tasks, _ = await asyncio.wait(
                    receive_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in completed_tasks:
                    tag = receive_tasks.pop(task)
                    policy_input, split_sizes = task.result()
                    receive_tasks[asyncio.create_task(receive_policy_input(tag))] = tag

                    if tag == "policy_final":
                        if not policy_input.is_last:
                            raise ValueError("Expected a final policy input.")
                        if pending is None:
                            raise ValueError(
                                "Final policy input has no pending result."
                            )
                        obs, result, sources, step_id = pending
                        _, final_result = self._predict_rollout_actions(
                            policy_input.obs,
                            final_obs=policy_input.env_result.final_obs,
                            rlt_switch_flags=policy_input.env_result.rlt_switch_flags,
                            intervene_requested=policy_input.env_result.intervene_flags,
                        )
                        self._publish_segment(
                            trajectory_channel,
                            step_id,
                            0,
                            sources,
                            obs,
                            result,
                            policy_input,
                            final_result["forward_inputs"],
                        )
                        trajectory_channel.publish(
                            TrajectoryEnd(
                                step_id=step_id,
                                source=(self._rank, 0),
                            )
                        )
                        pending = None
                        self.batch_router[tag].clear()
                        continue
                    if policy_input.is_last:
                        raise ValueError(
                            "Received a final policy input on policy route."
                        )

                    if decoupled_generate_time % self.sync_rollout_weight_time == 0:
                        self.update_dagger_beta()
                        if self._background_weight_sync_active:
                            await self._poll_background_weight_sync()
                        await self.wait_if_stale()
                    decoupled_generate_time += 1

                    actions, result = self._predict_rollout_actions(
                        policy_input.obs,
                        final_obs=policy_input.env_result.final_obs,
                        rlt_switch_flags=policy_input.env_result.rlt_switch_flags,
                        intervene_requested=policy_input.env_result.intervene_flags,
                    )
                    rollout_result = self._build_rollout_result(actions, result)
                    if pending is not None:
                        obs, previous_result, sources, step_id = pending
                        self._publish_segment(
                            trajectory_channel,
                            step_id,
                            0,
                            sources,
                            obs,
                            previous_result,
                            policy_input,
                            rollout_result.forward_inputs,
                        )
                    self.send_to_recorded_batch_routes(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=PolicyOutput(actions=actions.contiguous()),
                        tag=tag,
                        split_fn=self._split_policy_output,
                        split_sizes=split_sizes,
                    )
                    pending = (
                        policy_input.obs,
                        rollout_result,
                        policy_input.sources,
                        self._trajectory_step,
                    )
                    self._trajectory_step += 1
        finally:
            for task in receive_tasks:
                task.cancel()
            await asyncio.gather(*receive_tasks, return_exceptions=True)

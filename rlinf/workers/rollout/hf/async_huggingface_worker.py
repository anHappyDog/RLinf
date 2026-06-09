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
import gc
from typing import Literal

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.scheduler import Channel
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.utils import _build_channel_message, _split_channel_message
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        self.num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )
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

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, metric_channel)
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
    ):
        if self.env_decoupled_mode:
            await self.decoupled_generate_one_epoch(input_channel, output_channel)
        else:
            while True:
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()
                for _ in range(self.rollout_epoch):
                    await self.generate_one_epoch(input_channel, output_channel)
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
        async def recv_func():
            return await self.recv(
                self.actor_group_name,
                src_rank=self.actor_weight_src_rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data):
            await self.send(
                data,
                dst_group_name=self.actor_group_name,
                dst_rank=self.actor_weight_src_rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )

        gc.collect()
        self.torch_platform.empty_cache()
        return applied_version

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self._recv_and_apply_actor_sync())

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

    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    async def decoupled_generate_one_epoch(
        self, input_channel: Channel, output_channel: Channel
    ):
        self.update_dagger_beta()
        decoupled_generate_time = 1
        while True:
            if decoupled_generate_time % self.sync_rollout_weight_time == 0:
                self.update_dagger_beta()
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()
                decoupled_generate_time = decoupled_generate_time + 1
            # env_output = await self.recv_env_output_from_channel(input_channel)
            env_output = await self.recv_env_output_one_moment_from_channel(
                input_channel
            )
            actions, result = self.predict(env_output["obs"])
            save_flags = None
            if result.get("expert_label_flag", False):
                save_flags = torch.full(
                    (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                    True,
                    dtype=torch.bool,
                    device=actions.device,
                )
            rollout_result = RolloutResult(
                actions=actions,
                prev_logprobs=result["prev_logprobs"]
                if self.collect_prev_infos
                else None,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                save_flags=save_flags,
                forward_inputs=result["forward_inputs"],
                versions=torch.full_like(
                    result["prev_logprobs"],
                    float(self.version),
                    dtype=torch.float32,
                ),
            )
            self.send_rollout_result_to_channel(output_channel, rollout_result)

    def send_chunk_actions_to_channel(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to one of the env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        batch_size_map = self.batch_size_map[mode]
        batch_index_map = self.batch_index_map[mode]
        assert len(batch_index_map) == len(batch_size_map), (
            f"batch_index_map and batch_size_map must have the same length, but got {len(batch_index_map)} and {len(batch_size_map)}."
        )
        chunk_actions_split = self._split_actions(chunk_actions, batch_size_map)
        for i, chunk_action_i in enumerate(chunk_actions_split):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation

            batch_index = batch_index_map[i]
            env_rank, batch_idx, _, _ = _split_channel_message(batch_index)

            item = {
                "batch_index": _build_channel_message(
                    env_rank, batch_idx, mode, False, "actions"
                ),
                "batch": chunk_action_i,
            }

            output_channel.put(
                item,
                key=CommMapper.build_channel_key(
                    None, env_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

        # delete the batch index map
        self.batch_index_map[mode] = []
        return

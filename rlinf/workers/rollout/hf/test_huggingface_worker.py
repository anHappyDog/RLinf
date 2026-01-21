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

import torch
from omegaconf.dictconfig import DictConfig

from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class TestMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._pause_lock = asyncio.Lock()
        self._pause = asyncio.Event()

        self.version = 0
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", 0)
        self.finished_episodes = 0
        self.num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )

    async def stage_step(
        self, last_forward_inputs, stage_id, env_output, last_extracted_obs
    ) -> torch.Tensor:
        # print(
        #     f"Rollout Worker {self._rank} stage_step called for stage {stage_id}",
        #     flush=True,
        # )
        if last_forward_inputs[stage_id] is not None:
            last_forward_inputs[stage_id] = self.update_intervene_actions(
                env_output, last_forward_inputs[stage_id]
            )

        # print(f"Rollout Worker {self._rank} preprocessing env obs...", flush=True)
        extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
        # print(f"Rollout Worker {self._rank} getting dones and rewards...", flush=True)
        dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
            env_output, extracted_obs
        )
        await self.wait_if_stale()
        # print(f"Rollout Worker {self._rank} checking if paused...", flush=True)
        await self.wait_if_paused()
        # print(f"Rollout Worker {self._rank} not paused, predicting...", flush=True)
        async with self._pause_lock:
            actions, result = self.predict(extracted_obs)
        # print(f"Rollout Worker {self._rank} prediction done", flush=True)
        chunk_step_result = ChunkStepResult(
            prev_logprobs=result["prev_logprobs"],
            prev_values=result["prev_values"],
            dones=dones,
            truncations=env_output["truncations"],
            terminations=env_output["terminations"],
            rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
            forward_inputs=last_forward_inputs[stage_id],
            versions=torch.Tensor([self.version] * actions.shape[0]),
        )
        self.buffer_list[stage_id].append_result(chunk_step_result)
        if last_extracted_obs[stage_id] is not None and hasattr(
            self.hf_model, "q_head"
        ):
            self.buffer_list[stage_id].add_transition(
                last_extracted_obs[stage_id], real_extracted_obs
            )
        last_extracted_obs[stage_id] = extracted_obs
        last_forward_inputs[stage_id] = result["forward_inputs"]

        return actions

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        if self.enable_offload:
            self.reload_model()

        while not self.should_stop:
            self.buffer_list = [
                EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
                for _ in range(self.num_pipeline_stages)
            ]

            last_extracted_obs = [None for _ in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for _ in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    await self.wait_if_stale()
                    env_output = await self.recv_env_output(input_channel)
                    actions = await self.stage_step(
                        last_forward_inputs,
                        stage_id,
                        env_output,
                        last_extracted_obs,
                    )
                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                if rewards is not None:
                    self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                await self.wait_if_paused()
                async with self._pause_lock:
                    actions, result = self.predict(extracted_obs)
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

            for i in range(self.num_pipeline_stages):
                self.send_rollout_batch(actor_channel, i)
                self.finished_episodes += self.num_envs_per_stage

    async def pause_generation(self):
        async with self._pause_lock:
            self._pause.set()
            # print(f"Rollout Worker {self._rank} generation PAUSED", flush=True)

    async def resume_generation(self):
        # print(f"Rollout Worker {self._rank} resuming generation...", flush=True)
        async with self._pause_lock:
            self._pause.clear()
            # print(f"Rollout Worker {self._rank} generation RESUMED", flush=True)

    async def wait_if_paused(self):
        # if self._pause.is_set():
        # print(f"Rollout Worker {self._rank} waiting due to pause...", flush=True)
        while self._pause.is_set():
            await asyncio.sleep(0.1)
        # if not self._pause.is_set():
        # print(f"Rollout Worker {self._rank} resume from pause", flush=True)

    async def wait_if_stale(self):
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.num_envs_per_stage
                * self.num_pipeline_stages
            )
            if self.finished_episodes + self.num_envs_per_stage <= capacity:
                break
            await asyncio.sleep(0.1)
        # print(f"Rollout Worker {self._rank} resume from staleness wait", flush=True)

    async def set_version(self, version: int):
        self.version = version

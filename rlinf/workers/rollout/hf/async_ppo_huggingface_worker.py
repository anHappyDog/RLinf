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

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncPPOMultiStepRolloutWorker(MultiStepRolloutWorker):
    """Async PPO rollout worker with pause/resume and staleness control."""

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

    def _build_versions_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.full_like(reference, float(self.version), dtype=torch.float32)

    async def _stage_step(
        self,
        last_forward_inputs: list[dict | None],
        last_obs: list[dict | None],
        stage_id: int,
        env_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if last_forward_inputs[stage_id] is not None:
            last_forward_inputs[stage_id] = self.update_intervene_actions(
                env_output, last_forward_inputs[stage_id]
            )

        dones, rewards = self.get_dones_and_rewards(env_output)

        await self.wait_if_paused()
        async with self._pause_lock:
            actions, result = self.predict(env_output["obs"])

        chunk_step_result = ChunkStepResult(
            actions=result["forward_inputs"].get("action", None),
            prev_logprobs=result["prev_logprobs"],
            prev_values=result["prev_values"],
            dones=dones,
            truncations=env_output["truncations"],
            terminations=env_output["terminations"],
            rewards=rewards,
            forward_inputs=last_forward_inputs[stage_id],
            versions=self._build_versions_tensor(result["prev_logprobs"]),
        )
        self.rollout_results[stage_id].append_step_result(chunk_step_result)

        if self.collect_transitions and last_obs[stage_id] is not None:
            curr_obs = last_obs[stage_id]
            next_obs = (
                env_output["final_obs"]
                if dones.any() and self.cfg.env.train.auto_reset
                else env_output["obs"]
            )
            self.rollout_results[stage_id].append_transitions(curr_obs, next_obs)

        last_obs[stage_id] = env_output["obs"]
        last_forward_inputs[stage_id] = result["forward_inputs"]
        return actions

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        assert not self.enable_offload, (
            "Offload not supported in AsyncPPOMultiStepRolloutWorker"
        )

        n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        while not self.should_stop:
            self.rollout_results = [
                EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                    model_weights_id=self.model_weights_id,
                )
                for _ in range(self.num_pipeline_stages)
            ]

            await self.wait_if_stale()

            last_obs = [None for _ in range(self.num_pipeline_stages)]
            last_forward_inputs = [None for _ in range(self.num_pipeline_stages)]

            for _ in range(n_train_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)
                    actions = await self._stage_step(
                        last_forward_inputs=last_forward_inputs,
                        last_obs=last_obs,
                        stage_id=stage_id,
                        env_output=env_output,
                    )
                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                if last_forward_inputs[stage_id] is not None:
                    last_forward_inputs[stage_id] = self.update_intervene_actions(
                        env_output, last_forward_inputs[stage_id]
                    )

                dones, rewards = self.get_dones_and_rewards(env_output)

                await self.wait_if_paused()
                async with self._pause_lock:
                    _, result = self.predict(env_output["obs"])

                chunk_step_result = ChunkStepResult(
                    prev_values=result.get("prev_values", None),
                    dones=dones,
                    truncations=env_output["truncations"],
                    terminations=env_output["terminations"],
                    rewards=rewards,
                    forward_inputs=last_forward_inputs[stage_id],
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)

                if self.collect_transitions and last_obs[stage_id] is not None:
                    curr_obs = last_obs[stage_id]
                    next_obs = (
                        env_output["final_obs"]
                        if dones.any() and self.cfg.env.train.auto_reset
                        else env_output["obs"]
                    )
                    self.rollout_results[stage_id].append_transitions(
                        curr_obs, next_obs
                    )

            for stage_id in range(self.num_pipeline_stages):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
                self.finished_episodes += self.num_envs_per_stage

    async def pause_generation(self) -> None:
        async with self._pause_lock:
            self._pause.set()

    async def resume_generation(self) -> None:
        async with self._pause_lock:
            self._pause.clear()

    async def wait_if_paused(self) -> None:
        while self._pause.is_set():
            await asyncio.sleep(0.1)

    async def wait_if_stale(self) -> None:
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.num_envs_per_stage
                * self.num_pipeline_stages
            )
            if self.finished_episodes + self.num_envs_per_stage <= capacity:
                break
            await asyncio.sleep(0.1)

    async def set_version(self, version: int) -> None:
        self.version = int(version)

    async def stop(self) -> None:
        self.should_stop = True

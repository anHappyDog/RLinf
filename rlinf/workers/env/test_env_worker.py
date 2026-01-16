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

import torch

from rlinf.data.io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class TestEnvWorker(EnvWorker):
    def init_env_outputs(self) -> list[EnvOutput]:
        env_outputs = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = (
                    torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            self.num_done_envs = 0
            self.num_succ_envs = 0
            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=self.last_dones_list[stage_id],
                    terminations=self.last_terminations_list[stage_id],
                    truncations=self.last_truncations_list[stage_id],
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)
        return env_outputs

    def store_env_states(self, env_outputs: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_outputs]
        self.last_dones_list = [env_output.dones for env_output in env_outputs]
        self.last_truncations_list = [
            env_output.truncations for env_output in env_outputs
        ]
        self.last_terminations_list = [
            env_output.terminations for env_output in env_outputs
        ]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_outputs
        ]

    async def start_interacting(self, input_channel: Channel, output_channel: Channel):
        print(f"Env Worker {self._rank} starting environments...", flush=True)
        for env in self.env_list:
            env.start_env()
        print(f"Env Worker {self._rank} started environments.", flush=True)
        while not self.should_stop:
            print(f"Env Worker {self._rank} initializing env outputs...", flush=True)
            env_outputs = self.init_env_outputs()
            print(f"Env Worker {self._rank} initialized env outputs.", flush=True)
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                print(
                    f"Env Worker {self._rank} sending initial env output for stage {stage_id}...",
                    flush=True,
                )
                self.send_env_batch(output_channel, env_output.to_dict())

            for step in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())

                    env_outputs[stage_id] = env_output
            self.store_env_states(env_outputs)

    async def stop(self):
        self.should_stop = True
        for env in self.env_list:
            env.stop_env()

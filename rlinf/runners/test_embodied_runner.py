# Copyright 2026 root
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
from typing import TYPE_CHECKING

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.env.test_env_worker import TestEnvWorker
    from rlinf.workers.rollout.hf.test_huggingface_worker import (
        TestMultiStepRolloutWorker,
    )


class TestEmbodiedRunner(EmbodiedRunner):
    def update_rollout_weights(self):
        self.rollout: "TestMultiStepRolloutWorker"
        self.rollout.pause_generation().wait()
        self.rollout.sync_model_from_actor()
        self.actor.sync_model_to_rollout().wait()
        self.rollout.resume_generation().wait()
        self.rollout.set_version(self.global_step).wait()

    def run(self):
        self.update_rollout_weights()

        self.env: "TestEnvWorker"
        env_handle: Handle = self.env.start_interacting(
            input_channel=self.rollout_channel, output_channel=self.env_channel
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            actor_channel=self.actor_channel,
        )

        while self.global_step < self.max_steps:
            self.actor.recv_rollout_batch(input_channel=self.actor_channel).wait()
            actor_result = self.actor.run_training().wait()

            self.global_step += 1
            self.update_rollout_weights()
            training_metrics = {f"train/{k}": v for k, v in actor_result[0].items()}
            self.metric_logger.log(training_metrics, self.global_step)

            _, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()

        self.env.stop().wait()
        self.rollout.stop().wait()
        env_handle.wait()
        rollout_handle.wait()
        self._save_checkpoint()

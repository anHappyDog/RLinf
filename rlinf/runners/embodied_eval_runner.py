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

import json
import pathlib
import time
import typing

import torch

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        # Channel direction convention (names follow the receiver, not the sender):
        #   rollout_channel: env -> rollout  (env sends obs/RTC requests, rollout receives)
        #   env_channel:     rollout -> env  (rollout sends actions/RTC responses, env receives)
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )

        env_results = env_handle.wait()
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_results = rollout_handle.wait()
            rollout_metrics_list = [
                results for results in rollout_results if results is not None
            ]
            rollout_metrics = compute_evaluate_metrics(rollout_metrics_list)
            rollout_metrics.pop("num_trajectories", None)
        else:
            rollout_metrics = {}

        env_metrics_list = [results for results in env_results if results is not None]
        raw_metrics_path = self.cfg.runner.get("raw_metrics_path", None)
        if raw_metrics_path is not None:
            self._save_raw_metrics(env_metrics_list, raw_metrics_path)
        eval_metrics = compute_evaluate_metrics(env_metrics_list)
        eval_metrics.update(rollout_metrics)
        return eval_metrics

    @staticmethod
    def _save_raw_metrics(metrics_list: list[dict], output_path: str) -> None:
        """Write per-episode numeric metrics before aggregate reduction."""
        merged: dict[str, list] = {}
        for metrics in metrics_list:
            for key, values in metrics.items():
                tensor = torch.as_tensor(values).detach().cpu().reshape(-1)
                merged.setdefault(key, []).extend(tensor.tolist())

        lengths = {len(values) for values in merged.values()}
        if len(lengths) > 1:
            raise ValueError(
                "Raw evaluation metrics do not share one episode dimension: "
                f"{ {key: len(values) for key, values in merged.items()} }."
            )
        num_records = lengths.pop() if lengths else 0
        records = [
            {key: values[index] for key, values in merged.items()}
            for index in range(num_records)
        ]

        path = pathlib.Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {"num_episodes": num_records, "episodes": records}, file, indent=2
            )

    def run(self):
        start_time = time.time()
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)
        print_metrics_table(
            step=0,
            total_steps=1,
            start_time=start_time,
            metrics=eval_metrics,
            log_path=self.metric_logger.log_path,
        )

        self.metric_logger.finish()

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

"""Serve an OpenPI checkpoint with the grounded-control SFT prompt protocol.

Run this entrypoint from the COMET OpenPI checkout so its B1K wrapper and task
metadata remain the source of truth for observation and action processing.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random
import socket

import numpy as np
import torch
import tyro
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from openpi.policies import policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as openpi_config

from rlinf.data.b1k_grounded import (
    ControlProfile,
    ControlSerializer,
    GroundedPromptController,
    ReservedTokenMapping,
)


@dataclasses.dataclass
class Args:
    """Grounded COMET policy-server arguments."""

    checkpoint_dir: str
    token_mapping_path: str
    control_profile: ControlProfile
    task_name: str = "turning_on_radio"
    dataset_root: str = "/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos"
    config_name: str = "pi05_b1k-base"
    max_token_len: int = 512
    port: int = 8000
    control_mode: str = "receeding_horizon"
    max_len: int = 32
    action_horizon: int = 5
    temporal_ensemble_max: int = 3
    seed: int | None = None


class GroundedB1KPolicyWrapper(B1KPolicyWrapper):
    """COMET B1K wrapper that replaces its prompt with the SFT serialization."""

    def __init__(self, *args, prompt_controller: GroundedPromptController, **kwargs):
        super().__init__(*args, **kwargs)
        self._prompt_controller = prompt_controller

    def act(self, input_obs):
        self.task_prompt = self._prompt_controller.prompt(input_obs)
        return super().act(input_obs)


def main(args: Args) -> None:
    """Load the policy, apply grounded prompting, and serve it over websocket."""
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logging.info("Seeded policy process with seed=%d", args.seed)

    train_config = openpi_config.get_config(args.config_name)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model,
            max_token_len=args.max_token_len,
        ),
    )
    policy = policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    token_mapping = ReservedTokenMapping.from_dict(
        json.loads(pathlib.Path(args.token_mapping_path).read_text(encoding="utf-8"))
    )
    task_metadata = json.loads(
        pathlib.Path("scripts/task_mapping.json").read_text(encoding="utf-8")
    )
    prompt_controller = GroundedPromptController(
        ControlSerializer(token_mapping),
        args.control_profile,
        task_metadata[args.task_name]["task"],
    )
    policy = GroundedB1KPolicyWrapper(
        policy,
        task_name=args.task_name,
        control_mode=args.control_mode,
        max_len=args.max_len,
        action_horizon=args.action_horizon,
        temporal_ensemble_max=args.temporal_ensemble_max,
        fine_grained_level=0,
        prompt_controller=prompt_controller,
    )

    hostname = socket.gethostname()
    logging.info("Creating grounded server (host: %s, port: %d)", hostname, args.port)
    WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=train_config.policy_metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

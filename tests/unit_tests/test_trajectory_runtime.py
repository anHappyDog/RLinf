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

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.trajectory import EnvResult, RolloutResult, ValueResult
from rlinf.models.embodiment.openpi.forward_inputs import (
    OpenPILiberoForwardInputs,
)
from rlinf.workers.trajectory.runtime import (
    _env_schemas,
    _openpi_libero_rollout_schemas,
    validate_trajectory_config,
)


def _trajectory_config():
    return OmegaConf.create(
        {
            "trajectory": {"enabled": True},
            "runner": {
                "use_training_pipeline": False,
                "only_eval": False,
                "overlap_env_bootstrap": False,
            },
            "rollout": {"pipeline_stage_num": 1},
            "reward": {"use_reward_model": False},
            "actor": {"model": {"model_type": "openpi"}},
        }
    )


def test_trajectory_config_accepts_owned_execution_mode() -> None:
    validate_trajectory_config(_trajectory_config())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("runner.use_training_pipeline", True, "training pipeline"),
        ("rollout.pipeline_stage_num", 2, "pipeline_stage_num=1"),
        ("runner.only_eval", True, "evaluation-only"),
        ("runner.overlap_env_bootstrap", True, "overlap_env_bootstrap"),
        ("reward.use_reward_model", True, "RewardWorker"),
        ("actor.model.model_type", "openvla", "OpenPI"),
    ],
)
def test_trajectory_config_rejects_unowned_modes(
    path: str, value, message: str
) -> None:
    cfg = _trajectory_config()
    OmegaConf.update(cfg, path, value)

    with pytest.raises(ValueError, match=message):
        validate_trajectory_config(cfg)


def test_disabled_trajectory_config_needs_no_trajectory_fields() -> None:
    validate_trajectory_config(OmegaConf.create({"trajectory": {"enabled": False}}))


def test_configured_schemas_match_openpi_libero_records() -> None:
    batch = 2
    rollout_schema, timeout_schema, tail_schema = _openpi_libero_rollout_schemas(
        0, batch
    )
    forward_inputs = OpenPILiberoForwardInputs(
        chains=torch.zeros(batch, 5, 50, 32),
        denoise_inds=torch.zeros(batch, 4, dtype=torch.int64),
        tokenized_prompt=torch.zeros(batch, 48, dtype=torch.int64),
        tokenized_prompt_mask=torch.zeros(batch, 48, dtype=torch.bool),
        action=torch.zeros(batch, 35, dtype=torch.float64),
        model_action=torch.zeros(batch, 1600),
        image=torch.zeros(batch, 256, 256, 3, dtype=torch.uint8),
        wrist_image=torch.zeros(batch, 256, 256, 3, dtype=torch.uint8),
        state=torch.zeros(batch, 8),
    )
    rollout = RolloutResult(
        global_step=0,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(0, 1),
        actions=torch.zeros(batch, 5, 7, dtype=torch.float64),
        forward_inputs=forward_inputs,
        prev_logprobs=torch.zeros(batch, 5, 7),
        state_values=torch.zeros(batch, 1),
        versions=torch.zeros(batch, 5, 7),
    )
    timeout = ValueResult(
        global_step=0,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(0, 1),
        kind="timeout",
        values=torch.zeros(batch, 1),
    )
    tail = ValueResult(
        global_step=0,
        rollout_epoch=0,
        chunk_step=1,
        slot_ids=(0, 1),
        kind="tail",
        values=torch.zeros(batch, 1),
    )

    rollout_schema.validate(rollout)
    timeout_schema.validate(timeout)
    tail_schema.validate(tail)


def test_configured_env_schema_rejects_layout_changes() -> None:
    schema = _env_schemas(0, 2)[0]
    result = EnvResult(
        global_step=0,
        rollout_epoch=0,
        chunk_step=0,
        slot_ids=(0, 1),
        rewards=torch.zeros(2, 5),
        dones=torch.zeros(2, 5, dtype=torch.bool),
        terminations=torch.zeros(2, 5, dtype=torch.bool),
        truncations=torch.zeros(2, 5, dtype=torch.bool),
    )

    schema.validate(result)
    result.observations = {"unexpected": torch.zeros(2, 1)}
    with pytest.raises(ValueError, match="layout does not match"):
        schema.validate(result)

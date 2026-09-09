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

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
from rlinf.algorithms.registry import policy_loss


def _loss_inputs():
    return {
        "logprobs": torch.zeros(2, 1, requires_grad=True),
        "old_logprobs": torch.zeros(2, 1),
        "advantages": torch.tensor([[1.0], [-1.0]]),
        "values": torch.zeros(2, 1, requires_grad=True),
        "returns": torch.tensor([[1.0], [0.0]]),
        "prev_values": torch.zeros(2, 1),
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "value_clip": 0.2,
        "huber_delta": 1.0,
        "loss_mask": torch.ones(2, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("update_policy", "update_value", "expected_metric", "excluded_metric"),
    [
        (True, False, "actor/policy_loss", "critic/value_loss"),
        (False, True, "critic/value_loss", "actor/policy_loss"),
    ],
)
def test_actor_critic_loss_updates_only_requested_branch(
    update_policy,
    update_value,
    expected_metric,
    excluded_metric,
):
    inputs = _loss_inputs()
    loss, metrics = compute_ppo_actor_critic_loss(
        **inputs,
        update_policy=update_policy,
        update_value=update_value,
    )
    loss.backward()

    assert expected_metric in metrics
    assert excluded_metric not in metrics
    assert (inputs["logprobs"].grad is not None) is update_policy
    assert (inputs["values"].grad is not None) is update_value


def test_actor_critic_loss_rejects_empty_update():
    with pytest.raises(ValueError, match="At least one"):
        compute_ppo_actor_critic_loss(
            **_loss_inputs(),
            update_policy=False,
            update_value=False,
        )


def test_embodied_value_only_update_does_not_require_logprobs():
    inputs = _loss_inputs()
    inputs["logprobs"] = None
    loss, metrics = policy_loss(
        **inputs,
        task_type="embodied",
        loss_type="actor_critic",
        logprob_type="chunk_level",
        reward_type="subtask_chunk_level",
        single_action_dim=1,
        executed_action_mask=torch.ones(2, 1, dtype=torch.bool),
        update_policy=False,
        update_value=True,
    )
    loss.backward()

    assert "critic/value_loss" in metrics
    assert inputs["values"].grad is not None

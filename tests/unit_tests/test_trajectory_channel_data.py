# Copyright 2026 The RLinf Authors.
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

from dataclasses import replace

import numpy as np
import torch

from rlinf.data.schema.embodied_types import (
    EmbodiedRolloutResult,
    EnvResult,
    PolicyInput,
    PolicyOutput,
    merge_policy_inputs,
    split_policy_input,
)


def test_policy_output_supports_tensor_transport_skeleton():
    output = PolicyOutput(actions=torch.ones(2, 3))

    skeleton = replace(output, actions=None)
    restored = replace(skeleton, actions=output.actions)

    assert skeleton.actions is None
    assert torch.equal(restored.actions, output.actions)


def test_transport_results_store_contiguous_cpu_tensors():
    non_contiguous = torch.arange(12).reshape(3, 4).T

    env_result = EnvResult(rewards=non_contiguous)
    rollout_result = EmbodiedRolloutResult(
        actions=non_contiguous,
        forward_inputs={"states": non_contiguous},
    )

    assert env_result.rewards.device.type == "cpu"
    assert env_result.rewards.is_contiguous()
    assert rollout_result.actions.device.type == "cpu"
    assert rollout_result.actions.is_contiguous()
    assert rollout_result.forward_inputs["states"].device.type == "cpu"


def test_policy_input_split_merge_preserves_sources_and_nested_payloads():
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(12).reshape(4, 3),
            "labels": np.arange(4),
        },
        env_result=EnvResult(
            rewards=torch.arange(8).reshape(4, 2),
            dones=torch.zeros(4, 2, dtype=torch.bool),
            reward_assign_lengths=[1, 2, 3, 4],
        ),
        sources=[(0, 0, 2), (1, 0, 2)],
    )

    shards = split_policy_input(policy_input, [1, 3])
    merged = merge_policy_inputs(shards)

    assert shards[0].sources == [(0, 0, 1)]
    assert shards[1].sources == [(0, 0, 1), (1, 0, 2)]
    assert merged.sources == policy_input.sources
    assert torch.equal(merged.obs["states"], policy_input.obs["states"])
    assert np.array_equal(merged.obs["labels"], policy_input.obs["labels"])
    assert merged.env_result.reward_assign_lengths == [1, 2, 3, 4]


def test_online_lerobot_payload_survives_policy_input_routing():
    episode_data = {
        "chunk_actions": torch.arange(12).reshape(3, 2, 2),
        "obs_list": [
            {"images": torch.arange(6).reshape(3, 2)},
            {"images": torch.arange(6, 12).reshape(3, 2)},
        ],
        "terminations": torch.tensor([False, True, False]),
        "truncations": torch.tensor([False, False, True]),
        "infos_list": [
            {"score": np.arange(3)},
            {"score": np.arange(3, 6)},
        ],
    }
    policy_input = PolicyInput(
        obs={"states": torch.arange(9).reshape(3, 3)},
        env_result=EnvResult(episode_data=episode_data),
        sources=[(0, 0, 3)],
    )

    merged = merge_policy_inputs(split_policy_input(policy_input, [1, 2]))
    merged_data = merged.env_result.episode_data

    assert merged_data is not None
    assert torch.equal(merged_data["chunk_actions"], episode_data["chunk_actions"])
    assert torch.equal(
        merged_data["obs_list"][1]["images"],
        episode_data["obs_list"][1]["images"],
    )
    assert np.array_equal(
        merged_data["infos_list"][0]["score"],
        episode_data["infos_list"][0]["score"],
    )
    assert merged.sources == policy_input.sources

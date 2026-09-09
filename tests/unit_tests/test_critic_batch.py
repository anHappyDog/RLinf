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

import torch

from rlinf.utils.critic_batch import (
    export_critic_batch_shard,
    merge_critic_batch_shards,
    summarize_critic_batch,
)


def _rollout_batch(rank: int) -> dict:
    time_size = 3
    batch_size = 2
    shape = (time_size, batch_size, 1)
    terminations = torch.zeros(shape, dtype=torch.bool)
    dones = torch.zeros(shape, dtype=torch.bool)
    dones[1] = True
    terminations[1, rank] = True
    returns = torch.arange(6, dtype=torch.float32).reshape(shape) + rank
    return {
        "returns": returns,
        "advantages": returns - returns.mean(),
        "prev_values": torch.zeros(time_size + 1, batch_size, 1),
        "rewards": torch.zeros(shape),
        "terminations": terminations,
        "dones": dones,
        "loss_mask": torch.ones(shape, dtype=torch.bool),
        "loss_mask_sum": torch.full(shape, 3),
        "sample_weights": torch.ones(shape),
        "actions": torch.randn(time_size, batch_size, 32, 7),
        "forward_inputs": {
            "chains": torch.randn(time_size, batch_size, 2, 32, 7),
            "obs_state": torch.randn(time_size, batch_size, 32),
            "tokenized_prompt": torch.ones(
                time_size, batch_size, 16, dtype=torch.int64
            ),
            "tokenized_prompt_mask": torch.ones(
                time_size, batch_size, 16, dtype=torch.bool
            ),
            "obs_image__base_0_rgb": torch.zeros(
                time_size, batch_size, 3, 8, 8, dtype=torch.uint8
            ),
            "obs_image_mask__base_0_rgb": torch.ones(
                time_size, batch_size, dtype=torch.bool
            ),
        },
    }


def test_export_and_merge_critic_batch_shards(tmp_path):
    paths = [
        export_critic_batch_shard(
            _rollout_batch(rank),
            tmp_path,
            actor_rank=rank,
            actor_world_size=2,
            global_step=7,
        )
        for rank in range(2)
    ]

    shards = [torch.load(path, weights_only=False) for path in paths]
    assert "chains" not in shards[0]["batch"]["forward_inputs"]
    assert "actions" not in shards[0]["batch"]

    merged = merge_critic_batch_shards(shards)
    summary = summarize_critic_batch(merged)

    assert merged["batch"]["returns"].shape == (3, 4, 1)
    assert merged["trajectory_actor_ranks"].tolist() == [0, 0, 1, 1]
    assert merged["trajectory_local_indices"].tolist() == [0, 1, 0, 1]
    assert merged["trajectory_outcomes"].tolist() == [True, False, False, True]
    assert summary["trajectory_count"] == 4
    assert summary["valid_target_count"] == 12
    assert summary["successes"] == 2
    assert summary["failures"] == 2

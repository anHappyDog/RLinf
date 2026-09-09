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

from toolkits.b1k_grounded.cache_critic_features import _episode_metadata
from toolkits.b1k_grounded.fit_critic_offline import (
    MeanTokenCritic,
    StateAttentionCritic,
    StateFusionCritic,
)


@pytest.mark.parametrize(
    "critic_cls",
    [MeanTokenCritic, StateFusionCritic, StateAttentionCritic],
)
def test_offline_critic_head_forward_and_backward(critic_cls):
    batch_size, sequence_length = 3, 5
    feature_dim, state_dim = 8, 4
    pooled = torch.randn(batch_size, feature_dim, dtype=torch.bfloat16)
    state = torch.randn(batch_size, state_dim)
    prefix_out = torch.randn(
        batch_size, sequence_length, feature_dim, dtype=torch.bfloat16
    )
    prefix_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    prefix_mask[:, -1] = False
    model = critic_cls(feature_dim, state_dim)

    prediction = model(pooled, state, prefix_out, prefix_mask)
    prediction.square().mean().backward()

    assert prediction.shape == (batch_size, 1)
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_episode_metadata_splits_stream_after_done_chunk():
    # Online batches carry a final bootstrap row that has no matching critic target.
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    terminations = torch.zeros_like(dones)
    dones[1, 0] = True
    terminations[1, 0] = True
    dones[2, 1] = True
    dones[4] = True
    terminations[4] = True

    episode_ids, outcomes, complete = _episode_metadata(
        dones, terminations, time_size=4
    )

    assert episode_ids.reshape(4, 2).tolist() == [
        [0, 2],
        [0, 2],
        [1, 2],
        [1, 3],
    ]
    assert outcomes.tolist() == [True, False, False, False]
    assert complete.tolist() == [True, False, True, False]

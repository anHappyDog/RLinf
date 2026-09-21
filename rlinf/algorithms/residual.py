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

"""Pure tensor operations for bounded residual off-policy learning."""

import torch


def chunk_target(
    rewards,
    executed,
    terminations,
    truncations,
    next_q,
    gamma,
    bootstrap_truncation=False,
):
    """Discount executed primitive rewards and bootstrap at their actual boundary."""
    rewards = rewards.reshape(rewards.shape[0], -1).float()
    mask = executed.reshape_as(rewards).bool()
    lengths = mask.sum(-1, keepdim=True)
    if (lengths == 0).any() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("Replay requires nonempty prefix execution masks")
    powers = torch.arange(rewards.shape[1], device=rewards.device)
    reward = (torch.where(mask, rewards, 0) * gamma**powers).sum(-1, keepdim=True)
    terminal = (terminations.reshape_as(mask).bool() & mask).any(-1, keepdim=True)
    timeout = (truncations.reshape_as(mask).bool() & mask).any(-1, keepdim=True)
    stop = terminal | (timeout & (not bootstrap_truncation))
    return reward + gamma**lengths * torch.where(stop, 0, next_q)

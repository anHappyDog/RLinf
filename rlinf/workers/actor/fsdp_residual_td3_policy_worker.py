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

"""Residual TD3 losses using the existing FSDP/replay/target-update machinery."""

import torch
import torch.nn.functional as F

from rlinf.algorithms.residual import chunk_target
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3FSDPPolicy


class ResidualTD3FSDPPolicy(RLTTD3FSDPPolicy):
    """Train only the small residual actor and reference-conditioned twin Q."""

    def update_one_epoch(self, train_actor=True):
        warmup = int(self.cfg.algorithm.residual.critic_warmup_updates)
        return super().update_one_epoch(
            train_actor=train_actor and self.update_step >= warmup
        )

    def forward_critic(self, batch):
        cfg = self.cfg.algorithm.residual
        with torch.no_grad():
            actions, _, _ = self.target_model(
                forward_type=ForwardType.SAC,
                obs=batch["next_obs"],
                apply_action_noise=True,
                noise_sigma=cfg.target_noise_sigma,
                noise_clip=cfg.target_noise_clip,
            )
            next_q = (
                self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=batch["next_obs"],
                    actions=actions,
                )
                .min(-1, keepdim=True)
                .values
            )
            target = chunk_target(
                batch["rewards"],
                batch["executed_action_mask"],
                batch["terminations"],
                batch["truncations"],
                next_q,
                self.cfg.algorithm.gamma,
                cfg.bootstrap_truncation,
            )
        values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=batch["curr_obs"],
            actions=batch["actions"],
        )
        loss = F.mse_loss(values, target.expand_as(values))
        return loss, {"q_data": values.mean().item(), "target": target.mean().item()}

    def forward_actor(self, batch):
        self.qf_optimizer.zero_grad(set_to_none=True)
        obs = batch["curr_obs"]
        actions, _, _ = self.model(
            forward_type=ForwardType.SAC, obs=obs, apply_action_noise=False
        )
        q = (
            self.model(forward_type=ForwardType.SAC_Q, obs=obs, actions=actions)
            .min(-1)
            .values
        )
        ref = obs["ref_chunk"].reshape(actions.shape[0], -1)
        epsilon = torch.as_tensor(self.cfg.actor.model.epsilon, device=actions.device)
        delta = (actions - ref).reshape(
            actions.shape[0], -1, self.cfg.actor.model.action_dim
        )
        penalty = (delta / epsilon).square().mean()
        loss = -q.mean() + self.cfg.algorithm.residual.penalty * penalty
        return (
            loss,
            actions.new_zeros(()),
            {
                "q_pi": q.mean().item(),
                "residual_penalty": penalty.item(),
                "residual_abs": delta.abs().mean().item(),
            },
        )

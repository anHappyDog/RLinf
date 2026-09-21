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

"""Bounded residual TD3 policy in environment action units."""

from collections.abc import Sequence

import torch
from torch import nn

from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import (
    RLTTD3MLPPolicy,
    TwinQCritic,
    _make_td3_mlp,
)


class BoundedResidualActor(nn.Module):
    """Zero-initialized correction; noise is relative to the correction bound."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_len: int,
        epsilon: float | Sequence[float],
        hidden_dim: int,
        num_hidden_layers: int,
        exploration_sigma: float,
    ) -> None:
        super().__init__()
        bounds = torch.as_tensor(epsilon, dtype=torch.float32).flatten()
        if bounds.numel() == 1:
            bounds = bounds.expand(action_dim).clone()
        if (
            bounds.numel() != action_dim
            or not torch.isfinite(bounds).all()
            or (bounds <= 0).any()
        ):
            raise ValueError(
                "epsilon must contain positive finite environment-unit bounds (scalar or action_dim values)"
            )
        self.register_buffer("epsilon", bounds.repeat(chunk_len))
        self.sigma = float(exploration_sigma)
        self.mlp = _make_td3_mlp(
            input_dim=state_dim + action_dim * chunk_len,
            output_dim=action_dim * chunk_len,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)

    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic=True,
        apply_action_noise=None,
        noise_sigma=None,
        noise_clip=None,
        **kwargs,
    ) -> torch.Tensor:
        correction = self.mlp(torch.cat((state, reference), dim=-1)).tanh()
        if apply_action_noise is None:
            apply_action_noise = not deterministic
        sigma = self.sigma if noise_sigma is None else float(noise_sigma)
        if apply_action_noise and sigma > 0:
            noise = torch.randn_like(correction) * sigma
            if noise_clip is not None:
                noise = noise.clamp(-noise_clip, noise_clip)
            correction = (correction + noise).clamp(-1, 1)
        # No absolute clipping: a zero correction must preserve the VLA exactly.
        return reference + self.epsilon * correction


class ResidualMLPPolicy(RLTTD3MLPPolicy):
    """Reuse RLT's policy interface, with reference-conditioned twin critics."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        epsilon: float | Sequence[float],
        mlp_hidden_dim: int = 256,
        mlp_num_hidden_layers: int = 2,
        exploration_sigma: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.step_action_dim = self.action_dim = action_dim
        self.chunk_len = self.num_action_chunks = self.ref_chunk_len = num_action_chunks
        self.flat_action_dim = action_dim * num_action_chunks
        self.state_dim = z_dim + proprio_dim
        self.torch_compile_enabled = False
        self.actor = BoundedResidualActor(
            self.state_dim,
            action_dim,
            num_action_chunks,
            epsilon,
            mlp_hidden_dim,
            mlp_num_hidden_layers,
            exploration_sigma,
        )
        self.q_head = TwinQCritic(
            self.state_dim + self.flat_action_dim,
            self.flat_action_dim,
            mlp_hidden_dim,
            mlp_num_hidden_layers,
        )

    def sac_forward(
        self,
        obs,
        *,
        deterministic=True,
        apply_action_noise=False,
        noise_sigma=None,
        noise_clip=None,
        **kwargs,
    ):
        action = self.actor(
            self._state(obs),
            self._get_ref_chunk(obs),
            deterministic=deterministic,
            apply_action_noise=apply_action_noise,
            noise_sigma=noise_sigma,
            noise_clip=noise_clip,
        )
        return action, torch.zeros_like(action), None

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        state = torch.cat((self._state(obs), self._get_ref_chunk(obs)), dim=-1)
        return self.q_head(state.detach(), self._flatten_batch(actions))

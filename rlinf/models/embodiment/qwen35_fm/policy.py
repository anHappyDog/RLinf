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

"""Continuous action flow matching and stateless ODE evaluation."""

import torch
from torch import nn

from .vlm import VLMCondition


class FlowMatchingPolicy(nn.Module):
    """Attach a separately parameterized action vector field to a native VLM."""

    def __init__(self, vlm: nn.Module, expert: nn.Module):
        super().__init__()
        self.vlm = vlm
        self.expert = expert

    def predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        vlm_condition: VLMCondition,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate all action positions at one noise level, without persistent state."""
        return self.expert(
            noisy_actions, timestep, vlm_condition, action_mask=action_mask
        )

    def loss(
        self,
        actions: torch.Tensor,
        condition: VLMCondition,
        action_mask: torch.Tensor | None = None,
        *,
        noise: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute linear-path FM MSE over valid action coordinates.

        Args:
            actions: Normalized actions of shape [B, H, D].
            condition: Native multimodal hidden states and optional GDN states.
            action_mask: Validity mask [B, H] or [B, H, D].
            noise: Optional fixed Gaussian sample for matched validation.
            timestep: Optional fixed interpolation times [B].
        """
        if noise is None:
            noise = torch.randn_like(actions)
        if timestep is None:
            timestep = torch.rand(actions.shape[0], device=actions.device)
        tau = timestep.to(actions.dtype).reshape(-1, 1, 1)
        noisy_actions = (1 - tau) * noise + tau * actions
        token_mask = None
        if action_mask is not None:
            token_mask = action_mask.bool()
            if token_mask.ndim == 3:
                noisy_actions = noisy_actions.masked_fill(~token_mask, 0)
                token_mask = token_mask.any(-1)
            else:
                noisy_actions = noisy_actions.masked_fill(~token_mask[..., None], 0)
        error = (
            self.predict_velocity(
                noisy_actions, timestep, condition, token_mask
            ).float()
            - (actions - noise).float()
        ).square()
        if action_mask is None:
            return error.mean()
        mask = action_mask.to(error)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(error)
        if not torch.any(mask > 0):
            raise ValueError(
                "action_mask must contain at least one valid action coordinate"
            )
        return (error * mask).sum() / mask.sum()

    def forward(
        self, actions: torch.Tensor, condition: VLMCondition, **kwargs
    ) -> torch.Tensor:
        """Return the supervised flow matching loss."""
        return self.loss(actions, condition, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        condition: VLMCondition,
        steps: int = 10,
        method: str = "euler",
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate from Gaussian noise at t=0 to actions at t=1."""
        if steps < 1:
            raise ValueError(f"Expected positive ODE steps, got {steps}")
        if method not in {"euler", "heun"}:
            raise ValueError(f"Expected euler or heun sampler, got {method}")
        if initial_noise is None:
            parameter = next(self.expert.parameters())
            cfg = self.expert.config
            actions = torch.randn(
                condition.hidden_states.shape[0],
                cfg.horizon,
                cfg.action_dim,
                device=parameter.device,
                dtype=parameter.dtype,
                generator=generator,
            )
        else:
            actions = initial_noise.clone()
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((actions.shape[0],), index * dt, device=actions.device)
            velocity = self.predict_velocity(actions, time, condition)
            proposal = actions + dt * velocity
            if method == "heun":
                next_velocity = self.predict_velocity(proposal, time + dt, condition)
                actions = actions + 0.5 * dt * (velocity + next_velocity)
            else:
                actions = proposal
        return actions

    def parameter_report(self, batch_size: int = 1, sequence_length: int = 256) -> dict:
        """Report exact parameter counts and analytical conditioning storage bytes."""
        cfg = self.expert.config
        text = self.vlm.model.config.text_config
        full_layers = sum(kind == "full_attention" for kind in text.layer_types)
        recurrent_elements = sum(h * k * v for h, k, v in self.vlm.state_specs.values())
        dtype_bytes = next(self.vlm.parameters()).element_size()
        report = {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "action_expert_parameters": sum(
                p.numel() for p in self.expert.parameters()
            ),
            "vlm_parameters": sum(p.numel() for p in self.vlm.parameters()),
            "action_query_heads": cfg.num_heads,
            "action_kv_heads": cfg.num_kv_heads,
            "vlm_query_heads": text.num_attention_heads,
            "vlm_kv_heads": text.num_key_value_heads,
            "vlm_attention_head_dim": text.head_dim,
            "gdn_state_specs": self.vlm.state_specs,
            "recurrent_state_bytes_fp32": batch_size * recurrent_elements * 4,
            "native_all_gdn_state_bytes_fp32": batch_size
            * sum(kind == "linear_attention" for kind in text.layer_types)
            * text.linear_num_value_heads
            * text.linear_key_head_dim
            * text.linear_value_head_dim
            * 4,
            "action_gdn_state_shape_per_layer": (
                [
                    batch_size,
                    cfg.gdn_num_value_heads,
                    cfg.gdn_key_head_dim,
                    cfg.gdn_value_head_dim,
                ]
                if cfg.gdn_depth > 0
                else None
            ),
            "action_gdn_state_bytes_fp32": (
                batch_size
                * cfg.gdn_depth
                * cfg.gdn_num_value_heads
                * cfg.gdn_key_head_dim
                * cfg.gdn_value_head_dim
                * 4
                if cfg.gdn_depth > 0
                else 0
            ),
            "vlm_attention_kv_bytes": batch_size
            * sequence_length
            * full_layers
            * 2
            * text.num_key_value_heads
            * text.head_dim
            * dtype_bytes,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "vlm_parameter_dtype_bytes": dtype_bytes,
        }
        report["cross_attention_operations"] = sum(
            module.__class__.__name__ == "GQACrossAttention"
            for module in self.expert.modules()
        )
        return report

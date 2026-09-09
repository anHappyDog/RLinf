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

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, make_attn_mask
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
)


def _mean_valid_action_loss(
    per_timestep_loss: torch.Tensor,
    action_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    """Average flow loss over valid action timesteps only."""
    if action_is_pad is None:
        return per_timestep_loss.mean()
    if action_is_pad.shape != per_timestep_loss.shape:
        raise ValueError(
            "action_is_pad must match per-timestep loss shape; "
            f"got {tuple(action_is_pad.shape)} and "
            f"{tuple(per_timestep_loss.shape)}."
        )
    valid = ~action_is_pad
    if not valid.any():
        raise ValueError("SFT batch contains no valid action timesteps.")
    return per_timestep_loss.masked_select(valid).mean()


class OpenPiPytorchSFTActionModel(OpenPiPytorchActionModel):
    """SFT variant of :class:`OpenPiPytorchActionModel`.

    With ``openpi.use_rlt=False`` this computes the ordinary flow-matching loss.
    With ``openpi.use_rlt=True`` it keeps the same VLA loss and adds the legacy
    RLT-token reconstruction objective:

    ``loss = rlt_loss + rlt_alpha * vla_loss``.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        """Dispatch — SFT variant only supports :attr:`ForwardType.SFT`."""
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}. "
                "Use the RL subclass (actor.model.openpi.task='rl') for PPO."
            )
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: Any) -> torch.Tensor:
        """Compute the flow-matching SFT loss for one batch.

        ``data`` is either a ``(observation, actions)`` tuple or a dict with
        ``observation`` and ``actions`` keys. The data loader has already run
        the openpi transform pipeline, so ``actions`` arrive normalised and
        padded to the model action dim. Returns the scalar mean of the
        ``(B, action_horizon)`` per-timestep loss from :meth:`Pi0.compute_loss`
        (which samples the flow-matching noise/time internally).
        """
        observation, actions, action_is_pad = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        action_is_pad = self._action_padding_to_device(action_is_pad, actions)
        if not self.rlt_cfg.use_rlt:
            per_timestep_loss = self.model.compute_loss(
                observation, actions, train=True
            )
            return _mean_valid_action_loss(per_timestep_loss, action_is_pad)

        per_timestep_loss, prefix_output, prefix_mask = (
            self._sft_forward_with_rlt_prefix(observation, actions)
        )
        vla_loss = _mean_valid_action_loss(per_timestep_loss, action_is_pad)
        rlt_loss, _ = self._rlt_forward(prefix_output, prefix_mask)
        return {
            "loss": rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss,
            "vla_loss": vla_loss,
            "rlt_loss": rlt_loss,
        }

    def compute_loss(self, data: Any) -> torch.Tensor:
        """Alias kept for interface parity with the old action model."""
        return self.sft_forward(data)

    @staticmethod
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any, Any | None]:
        if isinstance(data, (tuple, list)):
            if len(data) not in {2, 3}:
                raise ValueError(
                    "SFT batch tuple must be (observation, actions) or "
                    "(observation, actions, action_is_pad); "
                    f"got length {len(data)}."
                )
            observation, actions = data[:2]
            action_is_pad = data[2] if len(data) == 3 else None
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
            action_is_pad = data.get("action_is_pad")
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions, action_is_pad

    def _action_padding_to_device(
        self,
        action_is_pad: Any | None,
        actions: torch.Tensor,
    ) -> torch.Tensor | None:
        if action_is_pad is None:
            return None
        action_is_pad = torch.as_tensor(action_is_pad, device=self.device).bool()
        if action_is_pad.shape != actions.shape[:2]:
            raise ValueError(
                "action_is_pad must have shape [B, action_horizon]; "
                f"got {tuple(action_is_pad.shape)} for actions "
                f"{tuple(actions.shape)}."
            )
        return action_is_pad

    def _observation_to_device(self, observation: Any) -> Observation:
        observation = Observation.from_observation_like(observation)
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Observation(
            images={k: _move(v) for k, v in observation.images.items()},
            image_masks={k: _move(v) for k, v in observation.image_masks.items()},
            state=_move(observation.state),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
            pcd_xyz=_move(observation.pcd_xyz),
        )

    def _actions_to_device(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        model_action_dim = self.model.action_dim
        if actions.dim() != 3:
            raise ValueError(
                "SFT actions must have shape [B, action_horizon, D]; "
                f"got {tuple(actions.shape)}."
            )
        if actions.shape[-1] == model_action_dim:
            return actions.to(device=self.device, dtype=torch.float32)
        raise ValueError(
            "SFT actions must arrive normalized + padded to the model action "
            f"dim {model_action_dim} (the openpi_rlinf SFT data loader applies the "
            f"openpi transform pipeline before collation); got last dim "
            f"{actions.shape[-1]}."
        )

    def _sft_forward_with_rlt_prefix(
        self,
        observation: Observation,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VLA loss while retaining the prefix hidden states for RLT."""
        batch_size = actions.shape[0]
        device = actions.device

        observation = pi0_model_module.preprocess_observation(observation, train=True)
        embed_dtype = self.model.embed_dtype
        observation = pi0_model_module._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        noise = torch.randn(actions.shape, device=device, dtype=dtype)
        time = (
            torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
            .sample((batch_size,))
            .to(device=device, dtype=dtype)
        )
        time = time * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(
            observation
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
            self.model.embed_suffix(observation, x_t, time)
        )

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        prefix_out, suffix_out = self.model.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]
        v_t = self.model.velocity_from_suffix(
            suffix_out[:, -self.model.action_horizon :]
        )
        loss = torch.mean(torch.square(v_t - u_t), dim=-1)
        prefix_out, prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_out.detach(), prefix_mask, observation.tokenized_prompt
        )
        return loss, prefix_out, prefix_mask

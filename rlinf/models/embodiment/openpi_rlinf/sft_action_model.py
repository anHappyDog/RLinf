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

import dataclasses
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

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

_HISTORY_CONTROLS = ("repeat_current", "shuffle_past")


def control_observation_history(
    observation: Observation, condition: str
) -> Observation:
    """Replace or reorder valid past content while preserving time metadata."""
    if condition not in _HISTORY_CONTROLS:
        raise ValueError(f"Unknown history control: {condition!r}.")
    if observation.history_frame_mask is None or observation.history_states is None:
        raise ValueError("History controls require short-memory tensors.")
    frame_mask = observation.history_frame_mask.bool()

    def _control(values: torch.Tensor) -> torch.Tensor:
        if values.ndim < frame_mask.ndim or values.shape[:2] != frame_mask.shape:
            raise ValueError(
                "Controlled history tensors must start with the same [B, K] "
                f"shape as history_frame_mask; got {tuple(values.shape)} and "
                f"{tuple(frame_mask.shape)}."
            )
        controlled = values.clone()
        if condition == "repeat_current":
            content_mask = frame_mask.reshape(
                *frame_mask.shape,
                *((1,) * (values.ndim - frame_mask.ndim)),
            )
            current = values[:, -1:].expand_as(values)
            return torch.where(content_mask, current, controlled)

        for batch_index, mask in enumerate(frame_mask):
            valid_past = torch.nonzero(mask[:-1], as_tuple=False).flatten()
            controlled[batch_index, valid_past] = values[
                batch_index, valid_past.flip(0)
            ]
        return controlled

    return dataclasses.replace(
        observation,
        images={key: _control(value) for key, value in observation.images.items()},
        history_states=_control(observation.history_states),
    )


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
        history_contrastive_weight: float = 0.0,
        history_contrastive_margin: float = 0.01,
        history_contrastive_conditions: tuple[str, ...] = _HISTORY_CONTROLS,
        history_contrastive_min_valid_frames: int = 6,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )
        self.history_contrastive_weight = float(history_contrastive_weight)
        self.history_contrastive_margin = float(history_contrastive_margin)
        self.history_contrastive_conditions = tuple(
            history_contrastive_conditions
        )
        self.history_contrastive_min_valid_frames = int(
            history_contrastive_min_valid_frames
        )
        if self.history_contrastive_weight < 0:
            raise ValueError("history contrastive weight must be non-negative.")
        if self.history_contrastive_margin < 0:
            raise ValueError("history contrastive margin must be non-negative.")
        unknown_conditions = set(self.history_contrastive_conditions).difference(
            _HISTORY_CONTROLS
        )
        if unknown_conditions:
            raise ValueError(
                f"Unknown history contrastive conditions: {sorted(unknown_conditions)}."
            )
        if not self.history_contrastive_conditions:
            raise ValueError("At least one history contrastive condition is required.")
        if self.history_contrastive_min_valid_frames <= 1:
            raise ValueError(
                "history_contrastive_min_valid_frames must exceed one."
            )
        if self.rlt_cfg.use_rlt and self.history_contrastive_weight > 0:
            raise ValueError("RLT and history contrastive SFT cannot be combined.")

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
        observation, actions = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        if self.history_contrastive_weight > 0:
            return self._history_contrastive_sft_forward(observation, actions)
        if not self.rlt_cfg.use_rlt:
            per_timestep_loss = self.model.compute_loss(
                observation, actions, train=True
            )
            return per_timestep_loss.mean()

        per_timestep_loss, prefix_output, prefix_mask = (
            self._sft_forward_with_rlt_prefix(observation, actions)
        )
        vla_loss = per_timestep_loss.mean()
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
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any]:
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    "SFT batch tuple must be (observation, actions); "
                    f"got length {len(data)}."
                )
            observation, actions = data
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions

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
            history_states=_move(observation.history_states),
            history_frame_mask=_move(observation.history_frame_mask),
            history_time_offsets=_move(observation.history_time_offsets),
            history_contrastive_mask=_move(
                observation.history_contrastive_mask
            ),
        )

    def _history_contrastive_sft_forward(
        self, observation: Observation, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Rank correct history below stop-gradient counterfactual losses."""
        if observation.history_contrastive_mask is None:
            raise ValueError(
                "History contrastive SFT requires history_contrastive_mask from "
                "the memory-critical data pipeline."
            )
        if observation.history_frame_mask is None:
            raise ValueError("History contrastive SFT requires history_frame_mask.")

        batch_size = actions.shape[0]
        noise = torch.randn_like(actions)
        time = (
            torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
            .sample((batch_size,))
            .to(device=actions.device, dtype=actions.dtype)
        )
        time = time * 0.999 + 0.001
        per_timestep_loss = self.model.compute_loss(
            observation,
            actions,
            train=True,
            noise=noise,
            time=time,
        )
        per_sample_loss = per_timestep_loss.mean(dim=-1)
        vla_loss = per_sample_loss.mean()
        selected = observation.history_contrastive_mask.bool()
        selected = selected & (
            observation.history_frame_mask.sum(dim=-1)
            >= self.history_contrastive_min_valid_frames
        )

        zero = vla_loss.detach().new_zeros(())
        metrics: dict[str, torch.Tensor] = {
            "vla_loss": vla_loss,
            "history_contrastive_fraction": selected.float().mean(),
        }
        any_selected = selected.any().to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(any_selected, op=dist.ReduceOp.MAX)
        if not bool(any_selected):
            metrics.update(
                loss=vla_loss,
                history_contrastive_loss=zero,
            )
            for condition in self.history_contrastive_conditions:
                metrics[f"history_delta_{condition}"] = zero
            return metrics

        ranking_losses = []
        for condition in self.history_contrastive_conditions:
            controlled = control_observation_history(observation, condition)
            with torch.no_grad():
                controlled_loss = self.model.compute_loss(
                    controlled,
                    actions,
                    train=True,
                    noise=noise,
                    time=time,
                ).mean(dim=-1)
            if bool(selected.any()):
                delta = controlled_loss - per_sample_loss.detach()
                metrics[f"history_delta_{condition}"] = delta[selected].mean()
                ranking_losses.append(
                    F.relu(
                        self.history_contrastive_margin
                        + per_sample_loss[selected]
                        - controlled_loss[selected]
                    ).mean()
                )
            else:
                metrics[f"history_delta_{condition}"] = zero
                ranking_losses.append(per_sample_loss.sum() * 0.0)

        contrastive_loss = torch.stack(ranking_losses).mean()
        metrics.update(
            loss=vla_loss
            + self.history_contrastive_weight * contrastive_loss,
            history_contrastive_loss=contrastive_loss,
        )
        return metrics

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

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

"""Knowledge-insulated native Qwen action-token and continuous-action training."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .policy import FlowMatchingPolicy
from .vlm import Qwen35VLMWrapper, VLMCondition


class KIFlowMatchingPolicy(FlowMatchingPolicy):
    """Train Qwen by token CE and the independent action expert by FM.

    Prefix conditioning is computed separately from teacher-forced action tokens.
    FM cannot update the VLM through either token features or GDN memories. The
    VLM remains trainable through action-token and optional vision-language CE.
    """

    def __init__(
        self,
        vlm: Qwen35VLMWrapper,
        expert: nn.Module,
        fast_tokenizer: Any = None,
        *,
        fm_weight: float = 1.0,
        action_ce_weight: float = 1.0,
        vl_weight: float = 1.0,
    ) -> None:
        if not vlm.trainable:
            raise ValueError(
                "Knowledge insulation requires a trainable VLM for token CE"
            )
        if fm_weight <= 0 or action_ce_weight <= 0 or vl_weight < 0:
            raise ValueError(
                "KI requires positive FM/action CE weights and nonnegative VL weight"
            )
        super().__init__(vlm, expert)
        self.fast_tokenizer = fast_tokenizer
        self.fm_weight = fm_weight
        self.action_ce_weight = action_ce_weight
        self.vl_weight = vl_weight

    @torch.no_grad()
    def encode_prefix(
        self,
        prefix_inputs: Mapping[str, torch.Tensor],
        *,
        validate_padding: bool = True,
    ) -> VLMCondition:
        """Encode observations only, with fresh native caches and insulated outputs.

        The evaluation context permits native cached GDN extraction even when
        the CE branch uses gradient checkpointing. Its prior module modes are
        restored before the next training forward.
        """
        if "labels" in prefix_inputs:
            raise ValueError(
                "Prefix conditioning must not contain teacher-forced labels"
            )
        training_modes = [(module, module.training) for module in self.vlm.modules()]
        self.vlm.eval()
        try:
            condition = self.vlm.encode(
                **prefix_inputs,
                extract_states=self.expert.config.variant == "gqa6_vlm_state",
                validate_padding=validate_padding,
            )
        finally:
            for module, training in training_modes:
                module.training = training
        return VLMCondition(
            hidden_states=condition.hidden_states.detach(),
            attention_mask=condition.attention_mask,
            gdn_states={
                layer: state.detach() for layer, state in condition.gdn_states.items()
            },
        )

    @staticmethod
    def _labels(inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        labels = inputs.get("labels")
        ids = inputs.get("input_ids")
        if (
            labels is None
            or ids is None
            or labels.ndim != 2
            or labels.shape != ids.shape
        ):
            raise ValueError("Token CE requires matching [B,L] input_ids and labels")
        mask = inputs.get("attention_mask")
        if mask is not None and torch.any((labels != -100) & ~mask.bool()):
            raise ValueError("Token CE labels must ignore padding positions")
        return labels

    @staticmethod
    def prepare_ce_metadata(
        inputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Validate a CPU CE batch and select its supervised logit columns."""
        if inputs["input_ids"].device.type != "cpu":
            raise ValueError(
                "Prepare KI batch metadata before transferring inputs to GPU"
            )
        labels = KIFlowMatchingPolicy._labels(inputs)
        shift_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
        supervised = shift_labels != -100
        positions = supervised.any(0).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            positions = torch.zeros(1, dtype=torch.long)
        return {"ce_positions": positions, "ce_token_count": supervised.sum()}

    @staticmethod
    def prepare_robot_metadata(
        prefix_inputs: Mapping[str, torch.Tensor],
        ce_inputs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Validate one CPU robot batch before it enters the training hot path."""
        if prefix_inputs["input_ids"].device.type != "cpu":
            raise ValueError(
                "Prepare KI batch metadata before transferring inputs to GPU"
            )
        KIFlowMatchingPolicy._validate_prefix(prefix_inputs, ce_inputs)
        if action_mask.shape != actions.shape:
            raise ValueError("Expected robot action_mask to match [B,H,D] actions")
        count = action_mask.sum()
        ce_metadata = KIFlowMatchingPolicy.prepare_ce_metadata(ce_inputs)
        if not bool(count > 0) and bool(ce_metadata["ce_token_count"] > 0):
            raise ValueError(
                "Supervised action tokens require valid continuous actions"
            )
        return {**ce_metadata, "fm_valid_action_scalars": count}

    def _causal_ce_loss(
        self,
        inputs: Mapping[str, torch.Tensor],
        batch_metadata: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        labels = self._labels(inputs) if batch_metadata is None else inputs["labels"]
        model_inputs = dict(inputs)
        for name in (
            "past_key_values",
            "use_cache",
            "logits_to_keep",
            "shift_labels",
            "num_items_in_batch",
            "return_dict",
        ):
            if name in model_inputs:
                raise ValueError(
                    f"KI token CE owns {name}; supply uncached native inputs"
                )
        shift_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
        if batch_metadata is None:
            token_count = (shift_labels != -100).sum()
            positions = (shift_labels != -100).any(0).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                # Empty local supervision still participates in FSDP collectives.
                positions = torch.zeros(1, device=labels.device, dtype=torch.long)
        else:
            token_count = batch_metadata["ce_token_count"]
            positions = batch_metadata["ce_positions"]
        # Native CE accepts already-shifted targets. Project only supervised
        # columns so image/prompt tokens do not allocate vocabulary-sized logits.
        output = self.vlm.model(
            **model_inputs,
            logits_to_keep=positions,
            shift_labels=shift_labels.index_select(1, positions),
            num_items_in_batch=token_count.clamp_min(1),
            use_cache=False,
            return_dict=True,
        )
        return output.loss

    def action_ce_loss(
        self,
        ce_inputs: Mapping[str, torch.Tensor],
        batch_metadata: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute native autoregressive CE over the masked FAST action suffix."""
        return self._causal_ce_loss(ce_inputs, batch_metadata)

    def vl_loss(
        self,
        vlm_batch: Mapping[str, torch.Tensor],
        batch_metadata: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute native masked CE using optional CPU-validated metadata."""
        return self._causal_ce_loss(vlm_batch, batch_metadata)

    @staticmethod
    def _validate_prefix(
        prefix_inputs: Mapping[str, torch.Tensor],
        ce_inputs: Mapping[str, torch.Tensor],
    ) -> None:
        prefix_ids, ce_ids = prefix_inputs["input_ids"], ce_inputs["input_ids"]
        if (
            prefix_ids.ndim != 2
            or ce_ids.ndim != 2
            or prefix_ids.shape[0] != ce_ids.shape[0]
        ):
            raise ValueError(
                "KI prefix and CE inputs must have the same batch dimension"
            )
        prefix_mask = prefix_inputs.get(
            "attention_mask", torch.ones_like(prefix_ids)
        ).bool()
        ce_mask = ce_inputs.get("attention_mask", torch.ones_like(ce_ids)).bool()
        if torch.any(prefix_mask[:, :-1] & ~prefix_mask[:, 1:]) or torch.any(
            ce_mask[:, :-1] & ~ce_mask[:, 1:]
        ):
            raise ValueError("KI native inputs require left padding")
        prefix_lengths, ce_lengths = prefix_mask.sum(-1), ce_mask.sum(-1)
        if torch.any(prefix_lengths == 0) or torch.any(ce_lengths < prefix_lengths):
            raise ValueError(
                "Each CE example must contain its complete observation prefix"
            )
        relative = torch.arange(prefix_ids.shape[1], device=prefix_ids.device)[None]
        relative = relative - (prefix_ids.shape[1] - prefix_lengths[:, None])
        indices = relative + ce_ids.shape[1] - ce_lengths[:, None]
        indices = indices.clamp(0, ce_ids.shape[1] - 1)
        if torch.any((ce_ids.gather(1, indices) != prefix_ids) & prefix_mask):
            raise ValueError(
                "KI CE inputs must start with the identical unpadded observation prefix"
            )
        labels = KIFlowMatchingPolicy._labels(ce_inputs)
        if torch.any((labels.gather(1, indices) != -100) & prefix_mask):
            raise ValueError(
                "Action CE must supervise only suffix tokens after the observation prefix"
            )

    def forward(
        self,
        prefix_inputs: Mapping[str, torch.Tensor],
        ce_inputs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        *,
        vlm_batch: Mapping[str, torch.Tensor] | None = None,
        noise: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        batch_metadata: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return insulated FM, action CE, optional VL CE, and their weighted sum.

        Args:
            prefix_inputs: Native processed observations/instructions without action targets.
            ce_inputs: The same prefix plus teacher-forced action tokens; prefix/padding
                labels are -100. These inputs are never used for FM conditioning.
            actions: Normalized continuous actions [B,H,D].
            action_mask: Valid action coordinates [B,H,D] or timesteps [B,H].
            vlm_batch: Optional native image/text batch with masked next-token labels.
            noise: Optional matched FM Gaussian sample.
            timestep: Optional matched FM times [B].
            batch_metadata: Metadata from prepare_robot_metadata for this exact batch.
                Omit it for unvalidated direct calls.
        """
        if batch_metadata is None:
            self._validate_prefix(prefix_inputs, ce_inputs)
        condition = self.encode_prefix(
            prefix_inputs, validate_padding=batch_metadata is None
        )
        fm_loss = self.loss(
            actions,
            condition,
            action_mask,
            noise=noise,
            timestep=timestep,
            mask_validated=batch_metadata is not None,
        )
        action_ce_loss = self.action_ce_loss(ce_inputs, batch_metadata)
        vl_loss = (
            self.vl_loss(vlm_batch)
            if vlm_batch is not None
            else action_ce_loss.new_zeros(())
        )
        loss = (
            self.fm_weight * fm_loss
            + self.action_ce_weight * action_ce_loss
            + self.vl_weight * vl_loss
        )
        valid_scalars = (
            batch_metadata["fm_valid_action_scalars"]
            if batch_metadata is not None
            else actions.numel()
            if action_mask is None
            else (
                action_mask.sum() * actions.shape[-1]
                if action_mask.ndim == 2
                else action_mask.sum()
            )
        )
        return {
            "loss": loss,
            "fm_loss": fm_loss,
            "action_ce_loss": action_ce_loss,
            "vl_loss": vl_loss,
            "action_ce_tokens": batch_metadata["ce_token_count"]
            if batch_metadata is not None
            else (ce_inputs["labels"][:, 1:] != -100).sum(),
            "vl_tokens": (vlm_batch["labels"][:, 1:] != -100).sum()
            if vlm_batch is not None
            else actions.new_zeros((), dtype=torch.long),
            "fm_valid_action_scalars": torch.as_tensor(
                valid_scalars, device=actions.device
            ),
        }

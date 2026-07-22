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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
from typing_extensions import Self

from rlinf.data.forward_inputs import (
    ForwardInputs,
    TensorFields,
    register_forward_inputs,
)


@register_forward_inputs
@dataclass(frozen=True, kw_only=True)
class OpenPILiberoForwardInputs(ForwardInputs):
    """OpenPI Actor inputs produced by the LIBERO PPO rollout path."""

    schema_name: ClassVar[str] = "openpi_libero"
    schema_version: ClassVar[int] = 1

    chains: torch.Tensor
    denoise_inds: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    action: torch.Tensor
    model_action: torch.Tensor
    image: torch.Tensor
    wrist_image: torch.Tensor
    state: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_model_inputs(
        cls,
        inputs: Mapping[str, torch.Tensor],
    ) -> Self:
        """Build the typed schema from OpenPI's current flat input dictionary."""
        expected = {name for name, _ in cls._field_map()}
        actual = set(inputs)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "OpenPI LIBERO forward inputs do not match schema v1: "
                f"missing={missing}, unexpected={unexpected}."
            )
        return cls(
            chains=inputs["chains"],
            denoise_inds=inputs["denoise_inds"],
            tokenized_prompt=inputs["tokenized_prompt"],
            tokenized_prompt_mask=inputs["tokenized_prompt_mask"],
            action=inputs["action"],
            model_action=inputs["model_action"],
            image=inputs["observation/image"],
            wrist_image=inputs["observation/wrist_image"],
            state=inputs["observation/state"],
        )

    @staticmethod
    def _field_map() -> tuple[tuple[str, str], ...]:
        return (
            ("chains", "chains"),
            ("denoise_inds", "denoise_inds"),
            ("tokenized_prompt", "tokenized_prompt"),
            ("tokenized_prompt_mask", "tokenized_prompt_mask"),
            ("action", "action"),
            ("model_action", "model_action"),
            ("observation/image", "image"),
            ("observation/wrist_image", "wrist_image"),
            ("observation/state", "state"),
        )

    @property
    def batch_size(self) -> int:
        return self.chains.shape[0]

    def validate(self) -> None:
        expected_ndims = {
            "chains": 4,
            "denoise_inds": 2,
            "tokenized_prompt": 2,
            "tokenized_prompt_mask": 2,
            "action": 2,
            "model_action": 2,
            "observation/image": 4,
            "observation/wrist_image": 4,
            "observation/state": 2,
        }
        fields = self.tensor_fields()
        for name, value in fields:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, got {type(value).__name__}."
                )
            if value.ndim != expected_ndims[name]:
                raise ValueError(
                    f"{name} must have {expected_ndims[name]} dimensions, "
                    f"got shape {tuple(value.shape)}."
                )

        batch_size = fields[0][1].shape[0]
        if batch_size == 0:
            raise ValueError("OpenPI LIBERO forward inputs must not be empty.")
        for name, value in fields[1:]:
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} must have leading batch dimension {batch_size}, "
                    f"got shape {tuple(value.shape)}."
                )

        if self.chains.shape[1] != self.denoise_inds.shape[1] + 1:
            raise ValueError(
                "chains must contain one more denoising state than denoise_inds."
            )
        if self.tokenized_prompt.shape != self.tokenized_prompt_mask.shape:
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must have equal shape."
            )
        if self.denoise_inds.dtype not in (torch.int32, torch.int64):
            raise TypeError("denoise_inds must have an integer dtype.")
        if self.tokenized_prompt.dtype not in (torch.int32, torch.int64):
            raise TypeError("tokenized_prompt must have an integer dtype.")
        if self.tokenized_prompt_mask.dtype != torch.bool:
            raise TypeError("tokenized_prompt_mask must have dtype torch.bool.")
        for name, value in (
            ("chains", self.chains),
            ("action", self.action),
            ("model_action", self.model_action),
            ("observation/state", self.state),
        ):
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must have a floating-point dtype.")
        for name, value in (
            ("observation/image", self.image),
            ("observation/wrist_image", self.wrist_image),
        ):
            if value.dtype != torch.uint8:
                raise TypeError(f"{name} must have dtype torch.uint8.")

    def tensor_fields(self) -> TensorFields:
        return tuple((name, getattr(self, attr)) for name, attr in self._field_map())

    def select(self, indices: torch.Tensor | Sequence[int]) -> Self:
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.int64)
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be a one-dimensional integer tensor.")
        if indices.numel() == 0:
            raise ValueError("indices must not be empty.")

        selected = {
            name: value.index_select(
                0, indices.to(device=value.device, dtype=torch.int64)
            )
            for name, value in self.tensor_fields()
        }
        return type(self).from_model_inputs(selected)

    def to_model_kwargs(self) -> dict[str, object]:
        return {"forward_inputs": dict(self.tensor_fields())}

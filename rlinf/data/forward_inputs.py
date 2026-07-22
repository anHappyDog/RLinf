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

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch
from typing_extensions import Self

TensorFields = tuple[tuple[str, torch.Tensor], ...]


class ForwardInputs(ABC):
    """Typed model inputs retained by rollout for Actor recomputation."""

    schema_name: ClassVar[str]
    schema_version: ClassVar[int]

    @classmethod
    @abstractmethod
    def from_model_inputs(cls, inputs: Mapping[str, torch.Tensor]) -> Self:
        """Build this schema from its stable tensor-field mapping."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return the number of samples represented by this object."""

    @abstractmethod
    def validate(self) -> None:
        """Validate field presence, tensor structure, and batch alignment."""

    @abstractmethod
    def tensor_fields(self) -> TensorFields:
        """Return tensor leaves in stable schema order."""

    @abstractmethod
    def select(self, indices: torch.Tensor | Sequence[int]) -> Self:
        """Select samples along the leading batch axis."""

    @abstractmethod
    def to_model_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by the policy model."""


_FORWARD_INPUTS_TYPES: dict[tuple[str, int], type[ForwardInputs]] = {}


def register_forward_inputs(
    forward_inputs_type: type[ForwardInputs],
) -> type[ForwardInputs]:
    """Register one concrete forward-input schema."""
    key = (
        forward_inputs_type.schema_name,
        forward_inputs_type.schema_version,
    )
    if key in _FORWARD_INPUTS_TYPES:
        raise ValueError(
            f"ForwardInputs schema {key[0]!r} version {key[1]} is already registered."
        )
    _FORWARD_INPUTS_TYPES[key] = forward_inputs_type
    return forward_inputs_type


def get_forward_inputs_type(name: str, version: int) -> type[ForwardInputs]:
    """Return the concrete type registered for a schema identity."""
    try:
        return _FORWARD_INPUTS_TYPES[(name, version)]
    except KeyError as error:
        raise ValueError(
            f"Unknown ForwardInputs schema {name!r} version {version}."
        ) from error

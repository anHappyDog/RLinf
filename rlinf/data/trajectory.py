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

from dataclasses import dataclass
from typing import Any, Literal

import torch

from rlinf.data.forward_inputs import ForwardInputs

Observations = dict[str, Any]
ValueKind = Literal["timeout", "tail"]
RewardMode = Literal["per_step", "terminal", "history_buffer"]


def _validate_tensor_batch(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim == 0 or value.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have leading batch dimension {batch_size}, "
            f"got shape {tuple(value.shape)}."
        )


def _validate_batch_tree(value: Any, *, name: str, batch_size: int) -> None:
    if isinstance(value, torch.Tensor):
        _validate_tensor_batch(value, name=name, batch_size=batch_size)
        return
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{name} must not be empty.")
        for key, child in value.items():
            _validate_batch_tree(
                child,
                name=f"{name}.{key}",
                batch_size=batch_size,
            )
        return
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(
                f"{name} must contain {batch_size} batch items, got {len(value)}."
            )
        return
    if value is None:
        return
    raise TypeError(
        f"{name} must contain tensors, nested dictionaries, batch lists, or None; "
        f"got {type(value).__name__}."
    )


@dataclass(kw_only=True)
class TrajectoryData:
    """Shared rollout coordinates for data associated with global environment slots."""

    global_step: int
    rollout_epoch: int
    chunk_step: int
    slot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return len(self.slot_ids)

    def validate(self) -> None:
        for name in ("global_step", "rollout_epoch", "chunk_step"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}."
                )

        if not isinstance(self.slot_ids, tuple) or not self.slot_ids:
            raise ValueError("slot_ids must be a non-empty tuple.")
        if any(
            not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0
            for slot_id in self.slot_ids
        ):
            raise ValueError("slot_ids must contain non-negative integers.")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("slot_ids must not contain duplicates.")


@dataclass(kw_only=True)
class PolicyInput(TrajectoryData):
    """Environment state and controls required for the next policy inference."""

    observations: Observations
    rlt_switch_flags: torch.Tensor | None = None
    intervene_requested: torch.Tensor | None = None

    def validate(self) -> None:
        super().validate()
        _validate_batch_tree(
            self.observations,
            name="observations",
            batch_size=self.batch_size,
        )
        for name in ("rlt_switch_flags", "intervene_requested"):
            value = getattr(self, name)
            if value is not None:
                _validate_tensor_batch(
                    value,
                    name=name,
                    batch_size=self.batch_size,
                )
                if value.dtype != torch.bool:
                    raise TypeError(f"{name} must have dtype torch.bool.")


@dataclass(kw_only=True)
class PolicyOutput(TrajectoryData):
    """Actions required for the next environment interaction."""

    actions: torch.Tensor

    def validate(self) -> None:
        super().validate()
        _validate_tensor_batch(
            self.actions,
            name="actions",
            batch_size=self.batch_size,
        )


@dataclass(kw_only=True)
class EnvResult(TrajectoryData):
    """Raw environment result for one action-chunk transition."""

    rewards: torch.Tensor
    dones: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    observations: Observations | None = None
    next_observations: Observations | None = None
    intervene_actions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None

    def validate(self) -> None:
        super().validate()
        for name in ("rewards", "dones", "terminations", "truncations"):
            _validate_tensor_batch(
                getattr(self, name),
                name=name,
                batch_size=self.batch_size,
            )

        mask_shape = self.dones.shape
        for name in ("terminations", "truncations"):
            value = getattr(self, name)
            if value.shape != mask_shape:
                raise ValueError(
                    f"{name} must have shape {tuple(mask_shape)}, "
                    f"got {tuple(value.shape)}."
                )
        if self.rewards.shape != mask_shape:
            raise ValueError(
                f"rewards must have shape {tuple(mask_shape)}, "
                f"got {tuple(self.rewards.shape)}."
            )
        for name in ("dones", "terminations", "truncations"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool.")

        for name in ("observations", "next_observations"):
            value = getattr(self, name)
            if value is not None:
                _validate_batch_tree(value, name=name, batch_size=self.batch_size)
        for name in ("intervene_actions", "intervene_flags", "rlt_switch_flags"):
            value = getattr(self, name)
            if value is not None:
                _validate_tensor_batch(
                    value,
                    name=name,
                    batch_size=self.batch_size,
                )
                if name != "intervene_actions" and value.dtype != torch.bool:
                    raise TypeError(f"{name} must have dtype torch.bool.")


@dataclass(kw_only=True)
class RolloutResult(TrajectoryData):
    """Policy outputs and Actor training inputs for one transition."""

    actions: torch.Tensor
    forward_inputs: ForwardInputs | None = None
    prev_logprobs: torch.Tensor | None = None
    state_values: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None

    def validate(self) -> None:
        super().validate()
        _validate_tensor_batch(
            self.actions,
            name="actions",
            batch_size=self.batch_size,
        )
        if self.forward_inputs is not None:
            if not isinstance(self.forward_inputs, ForwardInputs):
                raise TypeError("forward_inputs must implement ForwardInputs.")
            self.forward_inputs.validate()
            if self.forward_inputs.batch_size != self.batch_size:
                raise ValueError(
                    "forward_inputs must have leading batch dimension "
                    f"{self.batch_size}, got {self.forward_inputs.batch_size}."
                )
        for name in ("prev_logprobs", "state_values", "versions", "intervene_flags"):
            value = getattr(self, name)
            if value is not None:
                _validate_tensor_batch(
                    value,
                    name=name,
                    batch_size=self.batch_size,
                )
                if name == "intervene_flags" and value.dtype != torch.bool:
                    raise TypeError("intervene_flags must have dtype torch.bool.")


@dataclass(kw_only=True)
class RewardResult(TrajectoryData):
    """External rewards and their alignment mode."""

    rewards: torch.Tensor
    mode: RewardMode = "per_step"
    history_lengths: torch.Tensor | None = None

    def validate(self) -> None:
        super().validate()
        _validate_tensor_batch(
            self.rewards,
            name="rewards",
            batch_size=self.batch_size,
        )
        if self.mode not in ("per_step", "terminal", "history_buffer"):
            raise ValueError(f"Unsupported reward mode {self.mode!r}.")
        if self.mode == "history_buffer" and self.history_lengths is None:
            raise ValueError("history_lengths is required for history_buffer rewards.")
        if self.mode != "history_buffer" and self.history_lengths is not None:
            raise ValueError(
                "history_lengths is only valid for history_buffer rewards."
            )
        if self.history_lengths is not None:
            _validate_tensor_batch(
                self.history_lengths,
                name="history_lengths",
                batch_size=self.batch_size,
            )
            if self.history_lengths.ndim != 1:
                raise ValueError("history_lengths must have shape [batch_size].")
            if self.history_lengths.dtype not in (torch.int32, torch.int64):
                raise TypeError("history_lengths must have an integer dtype.")
            if (self.history_lengths < 0).any():
                raise ValueError("history_lengths must be non-negative.")


@dataclass(kw_only=True)
class ValueRequest(TrajectoryData):
    """Terminal or segment-tail observations that require value inference."""

    kind: ValueKind
    observations: Observations

    def validate(self) -> None:
        super().validate()
        if self.kind not in ("timeout", "tail"):
            raise ValueError(f"Unsupported value kind {self.kind!r}.")
        _validate_batch_tree(
            self.observations,
            name="observations",
            batch_size=self.batch_size,
        )


@dataclass(kw_only=True)
class ValueResult(TrajectoryData):
    """Values produced for truncated or alive segment-boundary slots."""

    kind: ValueKind
    values: torch.Tensor
    versions: torch.Tensor | None = None

    def validate(self) -> None:
        super().validate()
        if self.kind not in ("timeout", "tail"):
            raise ValueError(f"Unsupported value kind {self.kind!r}.")
        _validate_tensor_batch(
            self.values,
            name="values",
            batch_size=self.batch_size,
        )
        if self.values.ndim != 2 or self.values.shape[1] != 1:
            raise ValueError(
                f"values must have shape [batch_size, 1], got {tuple(self.values.shape)}."
            )
        if self.versions is not None:
            _validate_tensor_batch(
                self.versions,
                name="versions",
                batch_size=self.batch_size,
            )

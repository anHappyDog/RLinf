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

from collections.abc import Mapping
from typing import Any

import torch

from rlinf.data.forward_inputs import ForwardInputs
from rlinf.data.trajectory import (
    EnvResult,
    PolicyInput,
    PolicyOutput,
    RolloutResult,
    ValueRequest,
)


def policy_input(
    *,
    global_step: int,
    rollout_epoch: int,
    chunk_step: int,
    slot_ids: tuple[int, ...],
    observations: dict[str, Any],
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
) -> PolicyInput:
    """Build the complete Env-to-Rollout critical-path payload."""
    return PolicyInput(
        global_step=global_step,
        rollout_epoch=rollout_epoch,
        chunk_step=chunk_step,
        slot_ids=slot_ids,
        observations=_cpu(observations),
        rlt_switch_flags=_cpu(rlt_switch_flags),
        intervene_requested=_cpu(intervene_requested),
    )


def rollout_results(
    request: PolicyInput,
    *,
    actions: torch.Tensor,
    forward_inputs: ForwardInputs,
    prev_logprobs: torch.Tensor,
    state_values: torch.Tensor,
    versions: torch.Tensor,
    intervene_flags: torch.Tensor | None = None,
) -> tuple[PolicyOutput, RolloutResult]:
    """Split executable actions from Rollout-owned Actor training data."""
    coordinates = {
        "global_step": request.global_step,
        "rollout_epoch": request.rollout_epoch,
        "chunk_step": request.chunk_step,
        "slot_ids": request.slot_ids,
    }
    actions = _cpu(actions)
    return (
        PolicyOutput(actions=actions, **coordinates),
        RolloutResult(
            actions=actions,
            forward_inputs=_cpu_forward_inputs(forward_inputs),
            prev_logprobs=_cpu(prev_logprobs),
            state_values=_cpu(state_values),
            versions=_cpu(versions),
            intervene_flags=_cpu(intervene_flags),
            **coordinates,
        ),
    )


def env_result(
    request: PolicyInput,
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    observations: dict[str, Any] | None = None,
    next_observations: dict[str, Any] | None = None,
    intervene_actions: torch.Tensor | None = None,
    intervene_flags: torch.Tensor | None = None,
    rlt_switch_flags: torch.Tensor | None = None,
) -> EnvResult:
    """Build raw Env-owned transition data without reward/value correction."""
    return EnvResult(
        global_step=request.global_step,
        rollout_epoch=request.rollout_epoch,
        chunk_step=request.chunk_step,
        slot_ids=request.slot_ids,
        rewards=_cpu(rewards),
        dones=_cpu(dones),
        terminations=_cpu(terminations),
        truncations=_cpu(truncations),
        observations=_cpu(observations),
        next_observations=_cpu(next_observations),
        intervene_actions=_cpu(intervene_actions),
        intervene_flags=_cpu(intervene_flags),
        rlt_switch_flags=_cpu(rlt_switch_flags),
    )


def boundary_request(
    request: PolicyInput,
    *,
    kind: str,
    observations: dict[str, Any],
    mask: torch.Tensor,
    chunk_step: int | None = None,
) -> ValueRequest | None:
    """Select sparse timeout or alive-tail observations for value inference."""
    if (
        mask.ndim != 1
        or mask.shape[0] != request.batch_size
        or mask.dtype != torch.bool
    ):
        raise ValueError("boundary mask must be bool with shape [batch_size]")
    indices = mask.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return ValueRequest(
        global_step=request.global_step,
        rollout_epoch=request.rollout_epoch,
        chunk_step=request.chunk_step if chunk_step is None else chunk_step,
        slot_ids=tuple(request.slot_ids[index] for index in indices.tolist()),
        kind=kind,
        observations=_select(observations, indices),
    )


def _cpu(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, Mapping):
        return {key: _cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def _cpu_forward_inputs(value: ForwardInputs) -> ForwardInputs:
    return type(value).from_model_inputs(
        {name: _cpu(tensor) for name, tensor in value.tensor_fields()}
    )


def _select(value: Any, indices: torch.Tensor) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _cpu(value.index_select(0, indices.to(value.device)))
    if isinstance(value, Mapping):
        return {key: _select(child, indices) for key, child in value.items()}
    if isinstance(value, list):
        return [value[index] for index in indices.tolist()]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices.tolist())
    raise TypeError(f"unsupported observation leaf {type(value).__name__}")

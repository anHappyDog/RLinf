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

import pytest
import torch

from rlinf.models.embodiment.modules.value_head import (
    StateAttentionValueHead,
    StateFusionValueHead,
    ValueHead,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rl_sampler import value_from_prefix


def test_value_head_promotes_features_to_master_parameter_dtype():
    value_head = ValueHead(
        input_dim=8,
        hidden_sizes=(4,),
        output_dim=1,
        activation="relu",
        bias_last=True,
    )
    features = torch.randn(3, 8, dtype=torch.bfloat16)

    values = value_head(features)
    values.sum().backward()

    assert values.dtype == torch.float32
    assert all(
        parameter.dtype == torch.float32 for parameter in value_head.parameters()
    )
    assert all(
        parameter.grad.dtype == torch.float32 for parameter in value_head.parameters()
    )


def test_state_fusion_value_head_uses_state_and_keeps_fp32_master_weights():
    value_head = StateFusionValueHead(
        feature_dim=8,
        state_dim=3,
        state_hidden_dim=4,
        hidden_sizes=(4,),
    )
    prefix = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    state = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)

    values = value_from_prefix(
        value_head,
        prefix,
        mask,
        state=state,
        mode="state_fusion",
    )
    values.sum().backward()

    assert values.shape == (2,)
    assert values.dtype == torch.float32
    assert state.grad is not None
    assert all(
        parameter.dtype == torch.float32 for parameter in value_head.parameters()
    )
    assert all(
        parameter.grad is not None and parameter.grad.dtype == torch.float32
        for parameter in value_head.parameters()
    )


def test_state_fusion_value_head_requires_state():
    value_head = StateFusionValueHead(feature_dim=8, state_dim=3)

    with pytest.raises(ValueError, match="requires robot state"):
        value_from_prefix(
            value_head,
            torch.randn(2, 5, 8),
            torch.ones(2, 5, dtype=torch.bool),
            mode="state_fusion",
        )


def test_state_attention_value_head_uses_only_valid_tokens_and_fp32_master_weights():
    value_head = StateAttentionValueHead(
        feature_dim=8,
        state_dim=3,
        attention_dim=4,
        hidden_sizes=(4,),
    )
    prefix = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    mask = torch.tensor(
        [[True, True, False, False, False], [True, True, True, False, False]]
    )
    state = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)

    values = value_from_prefix(
        value_head,
        prefix,
        mask,
        state=state,
        mode="state_attention",
    )
    changed_prefix = prefix.clone()
    changed_prefix[~mask] = 100
    changed_values = value_from_prefix(
        value_head,
        changed_prefix,
        mask,
        state=state,
        mode="state_attention",
    )
    values.sum().backward()

    assert values.shape == (2,)
    assert values.dtype == torch.float32
    assert torch.allclose(values, changed_values)
    assert state.grad is not None
    assert all(
        parameter.dtype == torch.float32 for parameter in value_head.parameters()
    )
    assert all(
        parameter.grad is not None and parameter.grad.dtype == torch.float32
        for parameter in value_head.parameters()
    )


def test_state_attention_value_head_requires_state():
    value_head = StateAttentionValueHead(feature_dim=8, state_dim=3)

    with pytest.raises(ValueError, match="requires robot state"):
        value_from_prefix(
            value_head,
            torch.randn(2, 5, 8),
            torch.ones(2, 5, dtype=torch.bool),
            mode="state_attention",
        )

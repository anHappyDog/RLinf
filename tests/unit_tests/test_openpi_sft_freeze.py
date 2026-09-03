from __future__ import annotations

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_rlinf.utils.model_builders import (
    _build_sft_model,
)


class _DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.k_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.v_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.o_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _DummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_attention_norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
        self.pre_ffw_norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
        self.mlps = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.attn = _DummyAttention()


class _DummyLlm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([_DummyBlock()])
        self.final_norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])


class _DummyPi0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img = nn.Linear(2, 2)
        self.llm = _DummyLlm()
        self.action_projection = nn.Linear(2, 2)


def test_sft_train_expert_only_freezes_vlm_and_keeps_action_path_trainable() -> None:
    wrapper = _build_sft_model(
        OmegaConf.create({"train_expert_only": True}),
        _DummyPi0(),
        num_steps=5,
        action_env_dim=2,
    )

    assert not any(
        parameter.requires_grad for parameter in wrapper.model.img.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in wrapper.model.llm.embedder.parameters()
    )
    block = wrapper.model.llm.layers[0]
    assert not any(
        parameter.requires_grad for parameter in block.attn.q_proj[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in block.attn.q_proj[1].parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in wrapper.model.action_projection.parameters()
    )


def test_sft_vision_freeze_keeps_both_gemma_experts_trainable() -> None:
    wrapper = _build_sft_model(
        OmegaConf.create({"freeze_vision_encoder": True}),
        _DummyPi0(),
        num_steps=5,
        action_env_dim=2,
    )

    assert not any(
        parameter.requires_grad for parameter in wrapper.model.img.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in wrapper.model.llm.embedder.parameters()
    )
    block = wrapper.model.llm.layers[0]
    assert all(
        parameter.requires_grad for parameter in block.attn.q_proj[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in block.attn.q_proj[1].parameters()
    )


def test_sft_freeze_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_sft_model(
            OmegaConf.create(
                {"train_expert_only": True, "freeze_vision_encoder": True}
            ),
            _DummyPi0(),
            num_steps=5,
            action_env_dim=2,
        )

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_rlinf.pi0_model.short_memory import (
    HistoricalStateEncoder,
    ShortMemoryVisionEncoder,
    fixed_relative_time_encoding,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.siglip import Encoder


class TinyVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 8
        self.pool_type = "none"
        self.dtype_mm = torch.float32
        self.stem = nn.Conv2d(3, width, kernel_size=2, stride=2)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 4, width))
        self.encoder = Encoder(width, depth=3, num_heads=2, mlp_dim=16)
        self.head = nn.Linear(width, 6)

    def forward(self, image: torch.Tensor):
        x = F.conv2d(
            image.permute(0, 3, 1, 2).float(),
            self.stem.weight.float(),
            self.stem.bias.float(),
            stride=self.stem.stride,
        )
        batch, width, height, image_width = x.shape
        x = x.reshape(batch, width, height * image_width).permute(0, 2, 1)
        x = self.encoder(x + self.pos_embedding)
        return self.head(x), None


class RecordingAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.queries = []

    def forward(self, query, _key, _value, **_kwargs):
        self.queries.append(query.detach().clone())
        return torch.zeros_like(query), None


class ZeroMlp(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


class RecordingLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = RecordingAttention()
        self.dropout1 = nn.Identity()
        self.norm2 = nn.LayerNorm(width)
        self.mlp = ZeroMlp()
        self.dropout2 = nn.Identity()


def test_k1_fast_path_matches_original_vision_model() -> None:
    torch.manual_seed(0)
    vision = TinyVisionModel().eval()
    memory = ShortMemoryVisionEncoder((0, 1), drop_history_layer=1).eval()
    images = torch.randn(2, 1, 4, 4, 3)
    expected, _ = vision(images[:, 0])
    actual = memory(vision, images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_temporal_attention_is_causal_before_history_drop() -> None:
    torch.manual_seed(1)
    vision = TinyVisionModel().eval()
    memory = ShortMemoryVisionEncoder((0, 1), drop_history_layer=None).eval()
    images = torch.randn(1, 3, 4, 4, 3)
    changed_future = images.clone()
    changed_future[:, -1] += 10.0

    original = memory(vision, images, return_all_frames=True)
    changed = memory(vision, changed_future, return_all_frames=True)
    torch.testing.assert_close(original[:, :-1], changed[:, :-1])


def test_temporal_position_encoding_is_added_before_layer_norm() -> None:
    width = 8
    memory = ShortMemoryVisionEncoder((0,), drop_history_layer=None)
    layer = RecordingLayer(width)
    tokens = torch.randn(1, 2, 3, width)
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    time_offsets = torch.tensor([[-1.0, 0.0]])

    output = memory._spatial_temporal_layer(layer, tokens, frame_mask, time_offsets)

    temporal_tokens = (
        tokens + fixed_relative_time_encoding(time_offsets, width)[:, :, None, :]
    )
    expected_query = (
        layer.norm1(temporal_tokens).permute(0, 2, 1, 3).reshape(3, 2, width)
    )
    torch.testing.assert_close(layer.attn.queries[1], expected_query)
    torch.testing.assert_close(output, temporal_tokens)


def test_drop_history_and_missing_frame_mask_are_finite() -> None:
    torch.manual_seed(2)
    vision = TinyVisionModel().eval()
    memory = ShortMemoryVisionEncoder((0, 1), drop_history_layer=1).eval()
    images = torch.randn(2, 4, 4, 4, 3)
    frame_mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
    time_offsets = torch.tensor([[-3.0, -2.0, -1.0, 0.0]]).expand(2, -1)
    output = memory(
        vision,
        images,
        frame_mask=frame_mask,
        time_offsets=time_offsets,
    )
    assert output.shape == (2, 4, 6)
    assert torch.isfinite(output).all()


def test_historical_state_encoder_returns_all_k_states_and_masks_padding() -> None:
    encoder = HistoricalStateEncoder(state_dim=3, output_dim=8)
    states = torch.randn(2, 4, 3)
    mask = torch.tensor([[False, True, True, True], [True, True, True, True]])
    offsets = torch.tensor([[-3.0, -2.0, -1.0, 0.0]]).expand(2, -1)
    tokens, token_mask = encoder(states, frame_mask=mask, time_offsets=offsets)
    assert tokens.shape == (2, 4, 8)
    assert torch.equal(token_mask, mask)
    assert torch.count_nonzero(tokens[0, 0]) == 0


def test_current_relative_time_encoding_is_exactly_zero() -> None:
    offsets = torch.tensor([[-2.0, -1.0, 0.0]])
    encoding = fixed_relative_time_encoding(offsets, 8)
    assert torch.count_nonzero(encoding[:, -1]) == 0

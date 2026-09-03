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

"""Functional π0.5 short-memory video and proprioception encoders."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_relative_time_encoding(
    time_offsets: torch.Tensor, width: int
) -> torch.Tensor:
    """Encode seconds relative to the current frame with ``PE(t) - PE(0)``."""
    if width % 2:
        raise ValueError(f"time-encoding width must be even, got {width}.")
    frequencies = torch.exp(
        torch.linspace(
            0.0,
            -math.log(10_000.0),
            width // 2,
            device=time_offsets.device,
            dtype=torch.float32,
        )
    )
    angles = time_offsets.float().unsqueeze(-1) * frequencies
    encoding = torch.cat([angles.sin(), angles.cos() - 1.0], dim=-1)
    return encoding.to(dtype=time_offsets.dtype)


class ShortMemoryVisionEncoder(nn.Module):
    """Extend a SigLIP encoder with causal same-patch temporal attention.

    Spatial and temporal passes reuse each selected SigLIP layer's normalization,
    attention, and projection weights. The adopted residual order is
    ``spatial-attention → temporal-attention → MLP``. After
    ``drop_history_layer``, only the current frame continues through the
    remaining spatial layers, keeping the Gemma prefix length unchanged.

    This module owns no copy of the vision backbone and introduces no temporal
    attention parameters. The caller passes the shared SigLIP module to
    :meth:`forward`.
    """

    def __init__(
        self,
        temporal_layers: Sequence[int] = (3, 7, 11, 15),
        drop_history_layer: int | None = 15,
    ) -> None:
        super().__init__()
        self.temporal_layers = tuple(int(index) for index in temporal_layers)
        self.drop_history_layer = drop_history_layer
        if sorted(set(self.temporal_layers)) != list(self.temporal_layers):
            raise ValueError("temporal_layers must be unique and sorted.")
        if drop_history_layer is not None and any(
            index > drop_history_layer for index in self.temporal_layers
        ):
            raise ValueError(
                "temporal layers cannot occur after the history-drop layer."
            )

    def forward(
        self,
        vision_model: nn.Module,
        images: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        time_offsets: torch.Tensor | None = None,
        return_all_frames: bool = False,
    ) -> torch.Tensor:
        """Encode ``[B, K, H, W, C]`` frames from one camera.

        ``K=1`` explicitly calls the unmodified vision model. For ``K>1``,
        temporal attention is causal and only mixes equal spatial patch indices.
        """
        if images.ndim != 5:
            raise ValueError(
                "short-memory images must have shape [B, K, H, W, C], "
                f"got {tuple(images.shape)}."
            )
        batch_size, history_length = images.shape[:2]
        if history_length == 1:
            tokens, _ = vision_model(images[:, 0])
            return tokens[:, None] if return_all_frames else tokens
        if getattr(vision_model, "pool_type", None) != "none":
            raise ValueError("short memory requires SigLIP pool_type='none'.")

        depth = len(vision_model.encoder.layers)
        invalid_layers = [index for index in self.temporal_layers if index >= depth]
        if invalid_layers:
            raise ValueError(
                f"temporal layer indices {invalid_layers} exceed vision depth {depth}."
            )
        if self.drop_history_layer is not None and self.drop_history_layer >= depth:
            raise ValueError(
                f"drop_history_layer={self.drop_history_layer} exceeds depth {depth}."
            )

        device = images.device
        if frame_mask is None:
            frame_mask = torch.ones(
                batch_size, history_length, dtype=torch.bool, device=device
            )
        else:
            frame_mask = frame_mask.to(device=device, dtype=torch.bool)
        if frame_mask.shape != (batch_size, history_length):
            raise ValueError(
                f"frame_mask must have shape [B, K], got {tuple(frame_mask.shape)}."
            )
        if not torch.all(frame_mask[:, -1]):
            raise ValueError("the current (last) frame must always be valid.")

        if time_offsets is None:
            time_offsets = torch.arange(
                1 - history_length,
                1,
                device=device,
                dtype=torch.float32,
            ).expand(batch_size, -1)
        else:
            time_offsets = time_offsets.to(device=device, dtype=torch.float32)
        if time_offsets.shape != (batch_size, history_length):
            raise ValueError(
                f"time_offsets must have shape [B, K], got {tuple(time_offsets.shape)}."
            )
        if not torch.allclose(
            time_offsets[:, -1], torch.zeros(batch_size, device=device)
        ):
            raise ValueError("the current frame time offset must be zero.")

        tokens = self._embed_frames(vision_model, images)
        tokens = tokens * frame_mask[:, :, None, None]

        for layer_index, layer in enumerate(vision_model.encoder.layers):
            if layer_index in self.temporal_layers and tokens.shape[1] > 1:
                tokens = self._spatial_temporal_layer(
                    layer, tokens, frame_mask, time_offsets
                )
            else:
                shape = tokens.shape
                tokens = layer(tokens.reshape(-1, shape[-2], shape[-1])).reshape(shape)
            tokens = tokens * frame_mask[:, :, None, None]

            if layer_index == self.drop_history_layer:
                tokens = tokens[:, -1:]
                frame_mask = frame_mask[:, -1:]
                time_offsets = time_offsets[:, -1:]

        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        flat_tokens = vision_model.encoder.norm(
            flat_tokens.to(vision_model.encoder.norm.weight.dtype)
        )
        tokens = flat_tokens.reshape(*tokens.shape[:-1], flat_tokens.shape[-1])
        if vision_model.head is not None:
            tokens = vision_model.head(tokens)

        if return_all_frames:
            return tokens
        return tokens[:, -1]

    @staticmethod
    def _embed_frames(vision_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        batch_size, history_length, height, width, channels = images.shape
        flat_images = images.reshape(
            batch_size * history_length, height, width, channels
        )
        x = flat_images.permute(0, 3, 1, 2).float()
        x = F.conv2d(
            x,
            vision_model.stem.weight.float(),
            (
                vision_model.stem.bias.float()
                if vision_model.stem.bias is not None
                else None
            ),
            stride=vision_model.stem.stride,
            padding=vision_model.stem.padding,
        )
        flat_batch, width_dim, patch_height, patch_width = x.shape
        x = x.reshape(flat_batch, width_dim, patch_height * patch_width).permute(
            0, 2, 1
        )
        x = x + vision_model.pos_embedding.float()
        x = x.to(vision_model.dtype_mm)
        return x.reshape(
            batch_size, history_length, patch_height * patch_width, width_dim
        )

    def _spatial_temporal_layer(
        self,
        layer: nn.Module,
        tokens: torch.Tensor,
        frame_mask: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> torch.Tensor:
        shape = tokens.shape
        flat_tokens = tokens.reshape(-1, shape[-2], shape[-1])

        spatial_input = layer.norm1(flat_tokens.to(layer.norm1.weight.dtype))
        spatial_update, _ = layer.attn(
            spatial_input, spatial_input, spatial_input, need_weights=False
        )
        flat_tokens = flat_tokens + layer.dropout1(spatial_update)
        tokens = flat_tokens.reshape(shape)

        time_encoding = fixed_relative_time_encoding(
            time_offsets.to(tokens.dtype), shape[-1]
        )
        temporal_tokens = tokens + time_encoding[:, :, None, :]
        temporal_input = layer.norm1(
            temporal_tokens.reshape(-1, shape[-2], shape[-1]).to(
                layer.norm1.weight.dtype
            )
        ).reshape(shape)
        temporal_update = self._causal_temporal_attention(
            layer.attn, temporal_input, frame_mask
        )
        tokens = temporal_tokens + layer.dropout1(temporal_update)

        flat_tokens = tokens.reshape(-1, shape[-2], shape[-1])
        mlp_input = layer.norm2(flat_tokens.to(layer.norm2.weight.dtype))
        flat_tokens = flat_tokens + layer.dropout2(layer.mlp(mlp_input))
        return flat_tokens.reshape(shape)

    @staticmethod
    def _causal_temporal_attention(
        attention: nn.MultiheadAttention,
        tokens: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_length, patch_count, width = tokens.shape
        causal_mask = torch.triu(
            torch.ones(
                history_length,
                history_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )

        if torch.all(frame_mask):
            sequence = tokens.permute(0, 2, 1, 3).reshape(
                batch_size * patch_count, history_length, width
            )
            update, _ = attention(
                sequence,
                sequence,
                sequence,
                attn_mask=causal_mask,
                need_weights=False,
            )
            return update.reshape(
                batch_size, patch_count, history_length, width
            ).permute(0, 2, 1, 3)

        update = torch.zeros_like(tokens)
        for batch_index in range(batch_size):
            valid_indices = torch.nonzero(
                frame_mask[batch_index], as_tuple=False
            ).squeeze(1)
            sequence = tokens[batch_index, valid_indices].permute(1, 0, 2)
            valid_length = sequence.shape[1]
            sample_update, _ = attention(
                sequence,
                sequence,
                sequence,
                attn_mask=causal_mask[:valid_length, :valid_length],
                need_weights=False,
            )
            update[batch_index, valid_indices] = sample_update.permute(1, 0, 2)
        return update


class HistoricalStateEncoder(nn.Module):
    """Project normalized observation-history states into Gemma-width tokens."""

    def __init__(self, state_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(state_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        states: torch.Tensor,
        *,
        frame_mask: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one continuous state token per observation and its mask."""
        if states.ndim != 3:
            raise ValueError(
                f"historical states must have shape [B, K, D], got {states.shape}."
            )
        if (
            states.shape[:2] != frame_mask.shape
            or frame_mask.shape != time_offsets.shape
        ):
            raise ValueError("states, frame_mask, and time_offsets must share [B, K].")
        tokens = self.norm(self.projection(states.to(self.projection.weight.dtype)))
        tokens = tokens + fixed_relative_time_encoding(
            time_offsets.to(tokens.dtype), tokens.shape[-1]
        )
        state_mask = frame_mask.bool()
        tokens = tokens * state_mask.unsqueeze(-1)
        return tokens, state_mask

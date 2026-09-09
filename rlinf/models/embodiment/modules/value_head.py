# Copyright 2025 The RLinf Authors.
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

import math

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(512, 128),
        output_dim: int = 1,
        activation: str = "gelu",  # 'relu' or 'gelu'
        bias_last: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))

        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="relu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        # Value heads may intentionally retain fp32 master weights while their
        # frozen feature extractor emits bf16. Under FSDP, ``weight.dtype`` is
        # the mixed-precision compute dtype during the wrapped forward; in a
        # bare rollout model it remains fp32.
        x = x.to(self.mlp[0].weight.dtype)
        return self.mlp(x)


class StateFusionValueHead(ValueHead):
    """Predict values from pooled VLM features and proprioceptive state."""

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        state_hidden_dim: int = 128,
        hidden_sizes=(1024, 512, 256),
    ):
        super().__init__(
            input_dim=feature_dim + state_hidden_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation="relu",
            bias_last=True,
        )
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_hidden_dim),
            nn.GELU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
        )

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Fuse masked-pooled prefix features with the current robot state."""
        compute_dtype = self.feature_norm.weight.dtype
        features = self.feature_norm(features.to(compute_dtype))
        state_features = self.state_encoder(state.to(compute_dtype))
        return super().forward(torch.cat((features, state_features), dim=-1))


class StateAttentionValueHead(nn.Module):
    """Use proprioception to attend to visual-language prefix tokens."""

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        attention_dim: int = 256,
        hidden_sizes=(512, 256),
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(feature_dim)
        self.token_projection = nn.Linear(feature_dim, attention_dim)
        self.pooled_projection = nn.Linear(feature_dim, attention_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.output_head = ValueHead(
            input_dim=attention_dim * 3,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation="relu",
            bias_last=True,
        )
        self.scale = math.sqrt(attention_dim)

    def forward(
        self,
        pooled: torch.Tensor,
        state: torch.Tensor,
        prefix_out: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend to valid prefix tokens and fuse them with state features."""
        compute_dtype = self.token_norm.weight.dtype
        tokens = self.token_projection(self.token_norm(prefix_out.to(compute_dtype)))
        state_features = self.state_encoder(state.to(compute_dtype))
        scores = torch.einsum("bsd,bd->bs", tokens, state_features) / self.scale
        scores = scores.masked_fill(~prefix_mask.to(torch.bool), -torch.inf)
        attended = torch.einsum("bs,bsd->bd", scores.softmax(dim=-1), tokens)
        fused = torch.cat(
            (
                attended,
                self.pooled_projection(pooled.to(compute_dtype)),
                state_features,
            ),
            dim=-1,
        )
        return self.output_head(fused)

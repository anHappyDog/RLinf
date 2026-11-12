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

"""Continuous-action temporal mixers and native GDN fast-weight readout."""

import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class VLMCondition(Protocol):
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    gdn_states: dict[int, torch.Tensor]


@dataclass
class ActionExpertConfig:
    """Action architecture, independent of the pretrained VLM dimensions."""

    variant: str = "gqa6"
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    num_kv_heads: int = 4
    vlm_hidden_size: int = 2048
    action_dim: int = 7
    horizon: int = 16
    depth: int | None = None
    action_self_attn: bool = True
    gdn_depth: int | None = None
    gdn_num_key_heads: int = 8
    gdn_num_value_heads: int = 8
    gdn_key_head_dim: int = 128
    gdn_value_head_dim: int = 128
    gdn_conv_kernel_size: int = 4
    gdn_kernel: str = "native"
    state_injection: str = "per_block"
    state_adapter_dim: int = 128
    state_gate_bias: float = -4.0

    def __post_init__(self):
        variants = {
            "gqa6",
            "gqa6_vlm_state",
            "gqa6_gdn1",
            "gqa6_gdn2",
            "gqa6_gdn3",
        }
        if self.variant not in variants:
            raise ValueError(
                f"Expected one of {sorted(variants)}, got {self.variant!r}"
            )
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError(
                "hidden_size must be divisible by num_heads, and num_heads by num_kv_heads"
            )
        if self.gdn_num_value_heads % self.gdn_num_key_heads:
            raise ValueError("GDN value heads must be a multiple of key heads")
        if self.state_injection not in {"per_block", "shared"}:
            raise ValueError("state_injection must be per_block or shared")
        if self.gdn_kernel not in {"native", "reference"}:
            raise ValueError("gdn_kernel must be native or reference")
        if self.depth is None:
            self.depth = 6
        additive_gdn = self.variant.startswith("gqa6_gdn")
        if self.gdn_depth is None:
            self.gdn_depth = int(self.variant[-1]) if additive_gdn else 0
        if (additive_gdn and self.gdn_depth <= 0) or (
            not additive_gdn and self.gdn_depth != 0
        ):
            raise ValueError(
                "GDN depth must be positive for additive variants and zero otherwise"
            )
        if (
            self.depth <= 0
            or min(
                self.horizon,
                self.action_dim,
                self.hidden_size,
                self.intermediate_size,
                self.state_adapter_dim,
            )
            <= 0
        ):
            raise ValueError("Expert depth and dimensions must be positive")


class FlowMatchingTimeEmbedding(nn.Module):
    """Embed a scalar FM time for each example as [batch, hidden_size]."""

    def __init__(self, hidden_size: int):
        super().__init__()
        half = max(hidden_size // 2, 1)
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000) * torch.arange(half).float() / half),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * half, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        phase = timestep.float().reshape(-1, 1) * self.frequencies[None] * 1000
        return self.mlp(
            torch.cat((phase.sin(), phase.cos()), dim=-1).to(self.mlp[0].weight.dtype)
        )


class ActionEmbedding(nn.Module):
    """Project an entire action chunk and add learned physical-time positions."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.projection = nn.Linear(config.action_dim, config.hidden_size)
        self.position = nn.Parameter(torch.empty(config.horizon, config.hidden_size))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if actions.shape[1] > self.position.shape[0]:
            raise ValueError(
                f"Action horizon {actions.shape[1]} exceeds configured {self.position.shape[0]}"
            )
        return (
            self.projection(actions)
            + self.position[None, : actions.shape[1]]
            + time[:, None]
        )


class ActionContextCache:
    """Own fixed VLM K/V for one solve, preserving identity through FSDP casts."""

    def __init__(self) -> None:
        self.key_values: dict[nn.Module, tuple[torch.Tensor, torch.Tensor]] = {}


class GQACrossAttention(nn.Module):
    """Bidirectional grouped-query attention with True indicating valid keys."""

    def __init__(self, config: ActionExpertConfig, context_size: int | None = None):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        context_size = config.vlm_hidden_size if context_size is None else context_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            context_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            context_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_cache: ActionContextCache | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = (
            self.q_proj(hidden)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        cached = (
            context_cache.key_values.get(self) if context_cache is not None else None
        )
        if cached is None:
            key = (
                self.k_proj(context)
                .view(batch, -1, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            value = (
                self.v_proj(context)
                .view(batch, -1, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            if context_cache is not None:
                context_cache.key_values[self] = (key, value)
        else:
            key, value = cached
        mask = (
            attention_mask[:, None, None, :].bool()
            if attention_mask is not None
            else None
        )
        # Masked GQA otherwise defaults to math SDPA, which expands K/V internally.
        backend = (
            sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            )
            if query.is_cuda
            else nullcontext()
        )
        with backend:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=self.num_heads != self.num_kv_heads,
            )
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))


class GQASelfAttention(GQACrossAttention):
    def __init__(self, config: ActionExpertConfig):
        super().__init__(config, context_size=config.hidden_size)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return super().forward(hidden, hidden, attention_mask)


class AdaptiveNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Linear(hidden_size, 2 * hidden_size)

    def forward(self, hidden: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(F.silu(time)).chunk(2, dim=-1)
        return self.norm(hidden) * (1 + scale[:, None]) + shift[:, None]


class FeedForward(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class GDNStateReadAdapter(nn.Module):
    """Read native [B, value_heads, key_dim, value_dim] fast weights.

    Each layer has independent factorized query/output projections. A small
    shared bottleneck controls adapter size without flattening the state.
    """

    def __init__(
        self, config: ActionExpertConfig, state_specs: dict[int, tuple[int, int, int]]
    ):
        super().__init__()
        if not state_specs:
            raise ValueError(
                "State-conditioned experts require at least one selected GDN layer"
            )
        self.state_specs = dict(state_specs)
        grouped = {}
        for index, (layer, spec) in enumerate(state_specs.items()):
            grouped.setdefault(spec, []).append((index, layer))
        self.state_groups = tuple(grouped.items())
        rank = config.state_adapter_dim
        self.query_input = nn.Linear(config.hidden_size, rank, bias=False)
        self.queries = nn.ModuleDict(
            {
                str(layer): nn.Linear(rank, heads * key_dim, bias=False)
                for layer, (heads, key_dim, _) in state_specs.items()
            }
        )
        self.outputs = nn.ModuleDict(
            {
                str(layer): nn.Linear(heads * value_dim, rank, bias=False)
                for layer, (heads, _, value_dim) in state_specs.items()
            }
        )
        self.layer_weights = nn.Parameter(torch.zeros(len(state_specs)))
        self.output = nn.Linear(rank, config.hidden_size, bias=False)
        self.gate = nn.Linear(2 * config.hidden_size, config.hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, config.state_gate_bias)

    def forward(
        self, hidden: torch.Tensor, states: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        latent = self.query_input(hidden)
        aggregate = torch.zeros_like(latent)
        weights = self.layer_weights.softmax(dim=0)
        layer_readouts = {}
        for spec, layers in self.state_groups:
            heads, key_dim, _ = spec
            memories = []
            for _, layer in layers:
                state = states[layer]
                if state.shape[1:] != spec or state.shape[0] != hidden.shape[0]:
                    raise ValueError(
                        f"GDN layer {layer}: expected {(hidden.shape[0], *spec)}, got {tuple(state.shape)}"
                    )
                memories.append(state)
            query_weights = torch.cat(
                [self.queries[str(layer)].weight for _, layer in layers]
            )
            query = (
                F.linear(latent, query_weights)
                .reshape(*hidden.shape[:2], len(layers), heads, key_dim)
                .permute(2, 0, 1, 3, 4)
                .float()
            )
            query = (
                query
                * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
                / math.sqrt(key_dim)
            )
            readout = (
                torch.einsum("lbthk,lbhkv->lbthv", query, torch.stack(memories).float())
                .flatten(-2)
                .to(latent.dtype)
            )
            output_weights = torch.stack(
                [self.outputs[str(layer)].weight for _, layer in layers]
            )
            projected = torch.einsum("lbti,lri->lbtr", readout, output_weights)
            for (index, _), value in zip(layers, projected, strict=True):
                layer_readouts[index] = value
        # Preserve the original low-precision accumulation order across groups.
        for index in range(len(self.state_specs)):
            aggregate = aggregate + weights[index] * layer_readouts[index]
        readout = self.output(aggregate)
        return self.gate(torch.cat((hidden, readout), dim=-1)).sigmoid() * readout


class GQAActionBlock(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.self_attention = (
            GQASelfAttention(config) if config.action_self_attn else None
        )
        self.cross_attention = GQACrossAttention(config)
        self.self_norm = (
            AdaptiveNorm(config.hidden_size) if config.action_self_attn else None
        )
        self.cross_norm = AdaptiveNorm(config.hidden_size)
        self.ffn_norm = AdaptiveNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def forward(
        self,
        hidden: torch.Tensor,
        time: torch.Tensor,
        condition: VLMCondition,
        state_adapter: GDNStateReadAdapter | None = None,
        action_mask: torch.Tensor | None = None,
        context_cache: ActionContextCache | None = None,
    ) -> torch.Tensor:
        if self.self_attention is not None:
            hidden = hidden + self.self_attention(
                self.self_norm(hidden, time), action_mask
            )
        hidden = hidden + self.cross_attention(
            self.cross_norm(hidden, time),
            condition.hidden_states,
            condition.attention_mask,
            context_cache=context_cache,
        )
        if state_adapter is not None:
            hidden = hidden + state_adapter(hidden, condition.gdn_states)
        return hidden + self.ffn(self.ffn_norm(hidden, time))


def reference_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Differentiable token scan matching native Qwen3.5 key-to-value orientation."""
    dtype = query.dtype
    query, key, value = query.float(), key.float(), value.float()
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query / math.sqrt(query.shape[-1])
    state = (
        value.new_zeros(value.shape[0], value.shape[2], key.shape[-1], value.shape[-1])
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for index in range(query.shape[1]):
        state = state * g[:, index].float().exp()[..., None, None]
        key_t = key[:, index]
        prediction = torch.einsum("bhk,bhkv->bhv", key_t, state)
        delta = (value[:, index] - prediction) * beta[:, index, :, None].float()
        state = state + key_t[..., None] * delta[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, index], state))
    return torch.stack(outputs, dim=1).to(dtype), state if output_final_state else None


class ActionGDNBlock(nn.Module):
    """An independently parameterized native mixer; no cache escapes a call."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        native_config = Qwen3_5TextConfig(
            hidden_size=config.hidden_size,
            linear_num_key_heads=config.gdn_num_key_heads,
            linear_num_value_heads=config.gdn_num_value_heads,
            linear_key_head_dim=config.gdn_key_head_dim,
            linear_value_head_dim=config.gdn_value_head_dim,
            linear_conv_kernel_dim=config.gdn_conv_kernel_size,
        )
        self.mixer = Qwen3_5GatedDeltaNet(native_config, layer_idx=0)
        self.reference = config.gdn_kernel == "reference"
        self.mixer_norm = AdaptiveNorm(config.hidden_size)
        self.ffn_norm = AdaptiveNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def forward(
        self,
        hidden: torch.Tensor,
        time: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.mixer_norm(hidden, time)
        mixed = (
            self._kernel_forward(normalized, action_mask)
            if self.reference or normalized.device.type == "cpu"
            else self.mixer(normalized, attention_mask=action_mask)
        )
        hidden = hidden + mixed
        return hidden + self.ffn(self.ffn_norm(hidden, time))

    def _kernel_forward(
        self, hidden: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        mixer = self.mixer
        if mask is not None:
            hidden = hidden * mask[:, :, None].to(hidden.dtype)
        batch, length, _ = hidden.shape
        mixed = mixer.in_proj_qkv(hidden).transpose(1, 2)
        mixed = F.silu(mixer.conv1d(mixed)[:, :, :length]).transpose(1, 2)
        query, key, value = mixed.split(
            (mixer.key_dim, mixer.key_dim, mixer.value_dim), dim=-1
        )
        query = query.reshape(batch, length, mixer.num_k_heads, mixer.head_k_dim)
        key = key.reshape(batch, length, mixer.num_k_heads, mixer.head_k_dim)
        value = value.reshape(batch, length, mixer.num_v_heads, mixer.head_v_dim)
        repeats = mixer.num_v_heads // mixer.num_k_heads
        query, key = (
            query.repeat_interleave(repeats, dim=2),
            key.repeat_interleave(repeats, dim=2),
        )
        decay = -mixer.A_log.float().exp() * F.softplus(
            mixer.in_proj_a(hidden).float() + mixer.dt_bias
        )
        beta = mixer.in_proj_b(hidden).sigmoid()
        if self.reference:
            kernel = reference_gated_delta_rule
        else:
            # Native decorators select installed CUDA-only FLA even for CPU input.
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                torch_chunk_gated_delta_rule,
            )

            kernel = inspect.unwrap(torch_chunk_gated_delta_rule)
        output, _ = kernel(query, key, value, decay, beta, use_qk_l2norm_in_kernel=True)
        gate = mixer.in_proj_z(hidden).reshape(-1, mixer.head_v_dim)
        output = mixer.norm(output.reshape(-1, mixer.head_v_dim), gate)
        return mixer.out_proj(output.reshape(batch, length, mixer.value_dim))


class GQAActionExpert(nn.Module):
    def __init__(
        self,
        config: ActionExpertConfig,
        state_specs: dict[int, tuple[int, int, int]] | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = ActionEmbedding(config)
        self.time_embedding = FlowMatchingTimeEmbedding(config.hidden_size)
        self.blocks = nn.ModuleList(
            [GQAActionBlock(config) for _ in range(config.depth)]
        )
        self.state_adapters = nn.ModuleList()
        if config.variant == "gqa6_vlm_state":
            count = config.depth if config.state_injection == "per_block" else 1
            self.state_adapters.extend(
                [GDNStateReadAdapter(config, state_specs or {}) for _ in range(count)]
            )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.action_dim)
        self.cross_attention_count = config.depth

    def forward_hidden(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: VLMCondition,
        action_mask: torch.Tensor | None = None,
        context_cache: ActionContextCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode actions with the complete GQA backbone and return FM conditioning."""
        time = self.time_embedding(timestep.expand(noisy_actions.shape[0]))
        hidden = self.embedding(noisy_actions, time)
        for index, block in enumerate(self.blocks):
            adapter = (
                self.state_adapters[index if len(self.state_adapters) > 1 else 0]
                if len(self.state_adapters)
                else None
            )
            hidden = block(hidden, time, condition, adapter, action_mask, context_cache)
        return hidden, time

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: VLMCondition,
        action_mask: torch.Tensor | None = None,
        context_cache: ActionContextCache | None = None,
    ) -> torch.Tensor:
        hidden, _ = self.forward_hidden(
            noisy_actions, timestep, condition, action_mask, context_cache
        )
        return self.output(self.output_norm(hidden))

    def predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        vlm_condition: VLMCondition,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(noisy_actions, timestep, vlm_condition, action_mask)


class GQAStateActionExpert(GQAActionExpert):
    def __init__(
        self, config: ActionExpertConfig, state_specs: dict[int, tuple[int, int, int]]
    ):
        if config.variant != "gqa6_vlm_state":
            raise ValueError("GQAStateActionExpert requires variant='gqa6_vlm_state'")
        super().__init__(config, state_specs)


class GDNActionExpert(GQAActionExpert):
    """Keep the complete GQA expert and append independent action GDN blocks."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__(config)
        self.gdn_blocks = nn.ModuleList(
            [ActionGDNBlock(config) for _ in range(config.gdn_depth)]
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: VLMCondition,
        action_mask: torch.Tensor | None = None,
        context_cache: ActionContextCache | None = None,
    ) -> torch.Tensor:
        hidden, time = self.forward_hidden(
            noisy_actions, timestep, condition, action_mask, context_cache
        )
        for block in self.gdn_blocks:
            hidden = block(hidden, time, action_mask)
        return self.output(self.output_norm(hidden))


def build_action_expert(
    config: ActionExpertConfig,
    state_specs: dict[int, tuple[int, int, int]] | None = None,
) -> nn.Module:
    """Construct a temporal mixer with the common velocity-field interface."""
    if config.variant.startswith("gqa6_gdn"):
        return GDNActionExpert(config)
    if config.variant == "gqa6_vlm_state":
        return GQAStateActionExpert(config, state_specs or {})
    return GQAActionExpert(config)

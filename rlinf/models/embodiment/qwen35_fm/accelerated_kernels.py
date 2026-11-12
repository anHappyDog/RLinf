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

"""Opt-in Hopper FlashAttention-3 for native Qwen and action expert attention."""

import sys
import types

import torch


def install_fa3(policy: torch.nn.Module, package_root: str) -> dict:
    """Install FA3 before FSDP wrapping without changing parameters or state keys."""
    sys.path.insert(0, str(package_root))
    import flash_attn3

    sys.modules["flash_attn_interface"] = flash_attn3
    from rlinf.models.embodiment.qwen35_fm.action_expert import (
        GQACrossAttention,
        GQASelfAttention,
    )

    # The native Transformers wrapper owns Qwen causal padding and packed vision boundaries.
    configs = [
        policy.vlm.model.config,
        policy.vlm.model.config.text_config,
        policy.vlm.model.config.vision_config,
    ]
    configs += [m.config for m in policy.vlm.model.modules() if hasattr(m, "config")]
    for config in configs:
        config._attn_implementation = "flash_attention_3"

    def cross(self, hidden, context, attention_mask=None, context_cache=None):
        batch, length, _ = hidden.shape
        keys = context.shape[1]
        q = self.q_proj(hidden).reshape(batch * length, self.num_heads, self.head_dim)
        cached = (
            context_cache.key_values.get(self) if context_cache is not None else None
        )
        if cached is None:
            k = self.k_proj(context).reshape(
                batch, keys, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(context).reshape(
                batch, keys, self.num_kv_heads, self.head_dim
            )
            if context_cache is not None:
                context_cache.key_values[self] = (k.transpose(1, 2), v.transpose(1, 2))
        else:
            k, v = (tensor.transpose(1, 2) for tensor in cached)
        if attention_mask is None:
            mask = torch.ones((batch, keys), dtype=torch.bool, device=hidden.device)
        else:
            mask = attention_mask.bool()
        lengths = mask.sum(-1, dtype=torch.int32)
        cu_q = torch.arange(batch + 1, device=hidden.device, dtype=torch.int32) * length
        cu_k = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        out = flash_attn3.flash_attn_varlen_func(
            q,
            k[mask],
            v[mask],
            cu_q,
            cu_k,
            length,
            keys,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return self.out_proj(out.reshape(batch, length, -1))

    def self_attention(self, hidden, attention_mask=None):
        return cross(self, hidden, hidden, attention_mask)

    count = 0
    for module in policy.expert.modules():
        if isinstance(module, GQASelfAttention):
            module.forward = types.MethodType(self_attention, module)
            count += 1
        elif isinstance(module, GQACrossAttention):
            module.forward = types.MethodType(cross, module)
            count += 1
    return {
        "backend": "fa3",
        "expert_attention_modules": count,
        "kernel": "kernels-community/flash-attn3@v1 torch27-cxx11-cu126",
    }

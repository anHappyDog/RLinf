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

"""Opt-in FlashQLA chunk GDN backend for native Qwen3.5 layers."""

import sys
from pathlib import Path


def install_flashqla(policy, package_root: str | Path) -> dict:
    """Install the chunk kernel before FSDP wrapping, retaining native decode.

    Args:
        policy: KI policy containing the native Qwen3.5 backbone.
        package_root: Directory containing the isolated flash_qla and tilelang packages.

    Returns:
        Kernel versions and the number of Qwen GDN layers affected in this process.
    """
    package_root = Path(package_root).resolve()
    if not (package_root / "flash_qla").is_dir():
        raise ValueError(f"Expected isolated flash_qla package in {package_root}")
    sys.path.insert(0, str(package_root))
    import flash_qla
    import tilelang
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    layers = sum(
        isinstance(module, native.Qwen3_5GatedDeltaNet) for module in policy.modules()
    )
    if not layers:
        raise ValueError("FlashQLA requires native Qwen3.5 GatedDeltaNet layers")

    def chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        # Qwen already masks padded inputs before the chunk kernel. Do not repack
        # here: left-padding and returned recurrent-state boundaries must agree.
        if chunk_size != 64:
            raise ValueError(f"FlashQLA expects chunk_size=64, found {chunk_size}")
        return flash_qla.chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=kwargs.get("scale"),
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=kwargs.get("cu_seqlens"),
            head_first=kwargs.get("head_first", False),
            state_v_first=kwargs.get("state_v_first", False),
        )

    native.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    return {
        "backend": "flashqla",
        "flashqla_version": flash_qla.__version__,
        "tilelang_version": tilelang.__version__,
        "qwen_gdn_layers": layers,
        "package_root": str(package_root),
        "scope": "native Qwen chunk GDN; recurrent single-token decode unchanged",
    }

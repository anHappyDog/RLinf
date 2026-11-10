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

"""Export an RLinf actor checkpoint for the original OpenPI policy loader."""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def export_core_weights(
    core: nn.Module,
    actor_state: Mapping[str, torch.Tensor],
    output_path: str | Path,
) -> None:
    """Strictly restore ``core.*`` actor tensors and save shared-weight metadata."""
    from safetensors.torch import save_model

    unexpected = [name for name in actor_state if not name.startswith("core.")]
    if unexpected:
        raise ValueError(f"Expected only core.* actor state keys, found {unexpected}")
    state = {name.removeprefix("core."): value for name, value in actor_state.items()}
    core.load_state_dict(state, strict=True)
    aliases = defaultdict(list)
    for name, parameter in core.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
    for names in aliases.values():
        for name in names[1:]:
            if not torch.equal(state[names[0]], state[name]):
                raise ValueError(f"Shared core parameter values differ: {names}")
    output_path = Path(output_path)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    save_model(core, str(partial_path))
    os.replace(partial_path, output_path)


def export_checkpoint(actor_dir: str | Path, output_dir: str | Path) -> dict:
    """Create an exclusive official checkpoint, retaining failed output for inspection.

    The actor directory must contain ``model_state_dict/full_weights.pt`` and
    ``upstream_assets``.
    ``EXPORT.json`` is written only after the official weight file is complete.
    Model construction and restoration run on CPU and do not read base weights.
    """
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors import safe_open

    from .config import load_train_config

    actor_dir = Path(actor_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(actor_dir / "upstream_assets", output_dir / "assets")
    config = load_train_config(str(output_dir))
    core = PI0Pytorch(config.model)
    core.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    source = actor_dir / "model_state_dict" / "full_weights.pt"
    state = torch.load(source, map_location="cpu", weights_only=True)
    weights_path = output_dir / "model.safetensors"
    export_core_weights(core, state, weights_path)
    with weights_path.open("rb") as stream:
        weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with safe_open(weights_path, framework="pt", device="cpu") as stored:
        stored_tensor_count = len(stored.keys())
        alias_metadata = stored.metadata()
    result = {
        "status": "OFFICIAL_OPENPI_CHECKPOINT_EXPORTED",
        "source_actor_dir": str(actor_dir.resolve()),
        "source_weights": str(source.resolve()),
        "source_weights_bytes": source.stat().st_size,
        "output_dir": str(output_dir.resolve()),
        "model_name": config.name,
        "action_horizon": config.model.action_horizon,
        "action_dim": config.model.action_dim,
        "state_key_count": len(state),
        "stored_tensor_count": stored_tensor_count,
        "shared_tensor_metadata": alias_metadata,
        "model_bytes": weights_path.stat().st_size,
        "model_sha256": weight_sha,
        "assets_sha256": {
            str(path.relative_to(output_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((output_dir / "assets").rglob("*"))
            if path.is_file()
        },
        "model_device": "cpu",
        "precision": "upstream selective bfloat16; original float32 projections retained",
        "base_weights_loaded": False,
    }
    trajectory_path = output_dir / "assets" / "realized_trajectory.json"
    if trajectory_path.is_file():
        trajectory = json.loads(trajectory_path.read_text())
        result["action_target"] = "future measured joint positions and gripper targets"
        result["target_recipe"] = trajectory["recipe"]
        result["normalization"] = "train-only realized trajectory statistics"
        result["inference_loader"] = "rlinf.models.embodiment.openpi_official.get_model"
    qtarget_path = output_dir / "assets" / "canonical_qtarget.json"
    if qtarget_path.is_file():
        qtarget = json.loads(qtarget_path.read_text())
        result["action_target"] = "absolute joint qtarget rad7 and gripper closedness01"
        result["target_recipe"] = qtarget["recipe"]
        result["normalization"] = "source-routed train-only canonical statistics"
        result["inference_loader"] = "rlinf.models.embodiment.openpi_official.get_model"
        result["closed_loop_environment_adapter"] = False
    (output_dir / "EXPORT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Export one actor directory without overwriting an existing output."""
    from rlinf.utils.logging import get_logger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = export_checkpoint(args.actor_dir, args.output_dir)
    get_logger().info(f"Exported official checkpoint to {result['output_dir']}")


if __name__ == "__main__":
    main()

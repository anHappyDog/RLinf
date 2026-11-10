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

"""Optional upstream OpenPI integration, separate from RLinf's in-tree port."""

import json
from pathlib import Path
from typing import Any


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Load an original PyTorch checkpoint without changing parameter dtypes."""
    from openpi.policies.policy_config import create_trained_policy

    from rlinf.models.embodiment.openpi_official.config import load_train_config
    from rlinf.models.embodiment.openpi_official.policy import (
        OfficialOpenPiPolicy,
        load_exported_policy,
    )

    if torch_dtype is not None:
        raise ValueError(
            "openpi_official requires precision=null to retain checkpoint dtypes"
        )
    train_config = load_train_config(cfg.model_path, cfg.openpi_official.config_name)
    qtarget_path = Path(cfg.model_path) / "assets" / "canonical_qtarget.json"
    qtarget_manifest = cfg.openpi_official.get("qtarget_manifest_dir")
    recipe_assets = Path(cfg.model_path) / "assets" / "realized_trajectory.json"
    manifest_dir = cfg.openpi_official.get("trajectory_manifest_dir")
    if qtarget_path.is_file() or qtarget_manifest is not None:
        from types import SimpleNamespace

        from safetensors.torch import load_model

        from rlinf.data.datasets.openpi_official.qtarget import (
            QtargetInputTransform,
            QtargetOutputTransform,
            qtarget_assets,
        )

        assets = (
            qtarget_assets(qtarget_manifest)
            if qtarget_manifest is not None
            else json.loads(qtarget_path.read_text())
        )
        if qtarget_path.is_file() and assets != json.loads(qtarget_path.read_text()):
            raise ValueError("Checkpoint and requested canonical qtarget assets differ")
        weight_path = str(Path(cfg.model_path) / "model.safetensors")
        core = train_config.model.load_pytorch(train_config, weight_path)
        core.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        if (Path(cfg.model_path) / "EXPORT.json").is_file():
            load_model(core, weight_path, strict=True, device="cpu")
        core.to("cpu").eval()
        policy = SimpleNamespace(
            _model=core,
            _input_transform=QtargetInputTransform(train_config, assets),
            _output_transform=QtargetOutputTransform(assets),
            _sample_kwargs={"num_steps": cfg.num_steps},
        )
    elif recipe_assets.is_file() or manifest_dir is not None:
        from rlinf.models.embodiment.openpi_official.trajectory import (
            TrajectoryInputTransform,
            TrajectoryOutputTransform,
            trajectory_assets,
        )

        assets = (
            trajectory_assets(manifest_dir)
            if manifest_dir is not None
            else json.loads(recipe_assets.read_text())
        )
        if recipe_assets.is_file() and assets != json.loads(recipe_assets.read_text()):
            raise ValueError(
                "Checkpoint and requested realized trajectory assets differ"
            )
        from types import SimpleNamespace

        from safetensors.torch import load_model

        weight_path = str(Path(cfg.model_path) / "model.safetensors")
        core = train_config.model.load_pytorch(train_config, weight_path)
        core.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        if (Path(cfg.model_path) / "EXPORT.json").is_file():
            load_model(core, weight_path, strict=True, device="cpu")
        core.to("cpu").eval()
        policy = SimpleNamespace(
            _model=core,
            _input_transform=TrajectoryInputTransform(train_config, assets),
            _output_transform=TrajectoryOutputTransform(assets),
            _sample_kwargs={"num_steps": cfg.num_steps},
        )
    else:
        loader = (
            load_exported_policy
            if (Path(cfg.model_path) / "EXPORT.json").is_file()
            else create_trained_policy
        )
        policy = loader(
            train_config,
            cfg.model_path,
            pytorch_device="cpu",
            sample_kwargs={"num_steps": cfg.num_steps},
        )
    return OfficialOpenPiPolicy(policy, freeze_vision=cfg.openpi_official.freeze_vision)

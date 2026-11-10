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

"""Checkpoint-owned configuration for the upstream OpenPI implementation."""

import dataclasses
import pickle
from pathlib import Path
from typing import Any


def load_train_config(model_path: str, config_name: str | None = None) -> Any:
    """Read a trusted checkpoint config, or an explicitly named upstream config."""
    from openpi.training import config

    root = Path(model_path)
    paths = (
        root / "assets" / "train_config.pkl",
        root / "upstream_assets" / "train_config.pkl",
        root / "actor" / "upstream_assets" / "train_config.pkl",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is not None:
        with path.open("rb") as stream:
            train_config = pickle.load(stream)
        if config_name is not None and train_config.name != config_name:
            raise ValueError(
                f"Expected checkpoint config {config_name}, found {train_config.name}"
            )
    elif config_name is not None:
        train_config = config.get_config(config_name)
    else:
        raise ValueError(
            "A checkpoint train_config.pkl or explicit config_name is required"
        )
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
    )

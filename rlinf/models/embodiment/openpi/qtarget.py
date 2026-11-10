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

"""Canonical qtarget transforms around RLinf's Pi0 implementation."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.pi0 import Pi0


def load_qtarget_assets(model_path: Path, manifest_dir: str | None) -> dict | None:
    """Load normalization from a prepared mixture or an exported checkpoint."""
    assets = None
    for directory in ("assets", "upstream_assets", "actor/upstream_assets"):
        path = model_path / directory / "canonical_qtarget.json"
        if path.is_file():
            assets = json.loads(path.read_text())
            break
    if manifest_dir is not None:
        from rlinf.data.datasets.openpi_official.qtarget import qtarget_assets

        requested = qtarget_assets(manifest_dir)
        if assets is not None and requested != assets:
            raise ValueError("Checkpoint and requested canonical qtarget assets differ")
        assets = requested
    return assets


class CanonicalQtargetPi0(Pi0):
    """Use source-specific normalization without changing the Pi0 model core."""

    def __init__(self, config: Any, *, qtarget_assets: dict, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.qtarget_assets = qtarget_assets
        self._input_transform = None
        self._output_transform = None

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs: Any):
        if forward_type == ForwardType.DEFAULT:
            observation = self._observation_to_device(kwargs["observation"])
            return self.sample_actions(
                observation,
                noise=kwargs.get("noise"),
                num_steps=kwargs.get("num_steps", self.num_steps),
            )
        return super().forward(forward_type, **kwargs)

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: list[dict], *, noise: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """Return physical canonical actions for explicitly routed observations."""
        if self._input_transform is None:
            from openpi.models.pi0_config import Pi0Config

            from rlinf.data.datasets.openpi_official.qtarget import (
                QtargetInputTransform,
                QtargetOutputTransform,
            )

            config = SimpleNamespace(
                model=Pi0Config(
                    pi05=self.pi05,
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                    max_token_len=self.max_token_len,
                    discrete_state_input=self._config.discrete_state_input,
                )
            )
            self._input_transform = QtargetInputTransform(config, self.qtarget_assets)
            self._output_transform = QtargetOutputTransform(self.qtarget_assets)
        inputs = [self._input_transform(obs) for obs in env_obs]
        batch = tree_map(
            lambda *values: torch.as_tensor(np.stack(values), device=self.device),
            *inputs,
        )
        actions = self(
            forward_type=ForwardType.DEFAULT,
            observation=batch,
            noise=None if noise is None else torch.as_tensor(noise, device=self.device),
        )
        outputs = [
            self._output_transform(
                {
                    "actions": action.detach().cpu().numpy(),
                    "source_schema": obs["source_schema"],
                }
            )["actions"]
            for action, obs in zip(actions, env_obs, strict=True)
        ]
        return np.stack(outputs), {}

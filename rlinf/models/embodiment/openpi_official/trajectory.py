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

"""Assets and transforms for future measured-joint trajectory targets."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.openpi.control_schema import normalize, unnormalize


def trajectory_assets(manifest_dir: str | Path) -> dict:
    """Bind the realized-trajectory recipe to its train-only normalization."""
    root = Path(manifest_dir)
    metadata = json.loads((root / "normalization.json").read_text())
    recipe = metadata["target_recipe"]
    if recipe["name"] != "realized_joint_trajectory":
        raise ValueError("Expected a realized_joint_trajectory preparation")
    if recipe["target_hz"] != 15 or recipe["horizon"] != 16:
        raise ValueError("Realized trajectory requires 15 Hz and horizon 16")
    schemas = list(metadata["schemas"].values())
    for key in ("state", "action"):
        expected = schemas[0][key]
        if len(expected["mean"]) != 8 or expected["transform"] != "mean_std":
            raise ValueError(
                "Realized trajectory requires shared mean/std state8/action8"
            )
        if any(schema[key] != expected for schema in schemas[1:]):
            raise ValueError(
                "Realized trajectory requires shared train-only normalization"
            )
    return {"version": 1, "recipe": recipe, "normalization": metadata}


def save_trajectory_assets(
    manifest_dir: str | Path, output_assets_dir: str | Path
) -> None:
    """Save the recipe and train-only statistics alongside the model assets."""
    root = Path(output_assets_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "realized_trajectory.json").write_text(
        json.dumps(trajectory_assets(manifest_dir), indent=2, sort_keys=True) + "\n"
    )


def build_trajectory_prompt(
    task: str,
    source_schema: str,
    robot_identity: str,
    gripper_identity: str,
    state_gripper_present: bool,
) -> str:
    """Identify robot embodiment and missing proprioception without command proxies."""
    presence = "present" if state_gripper_present else "missing"
    return (
        f"{source_schema} {robot_identity} {gripper_identity}; "
        f"measured gripper {presence}. {task}"
    )


def trajectory_model_transform(train_config: Any) -> Any:
    """Retain checkpoint image, text and padding transforms without native adapters."""
    from openpi import transforms
    from openpi.training.config import ModelTransformFactory

    model = train_config.model
    if model.action_dim != 8 or model.action_horizon != 16:
        raise ValueError(
            "Realized trajectory requires the checkpoint action8/H16 architecture"
        )
    return transforms.compose(ModelTransformFactory()(model).inputs)


class TrajectoryInputTransform:
    """Normalize canonical measured state8 once and apply checkpoint model transforms.

    Inputs use canonical ``state``, ``image``, ``image_mask`` and task ``prompt``.
    ``source_schema``, ``robot_identity`` and ``gripper_identity`` identify the
    embodiment. ``state_mask`` marks missing measured values, which become zero.
    """

    def __init__(self, train_config: Any, assets: dict) -> None:
        self.stats = next(iter(assets["normalization"]["schemas"].values()))
        self.model_transform = trajectory_model_transform(train_config)

    def __call__(self, observation: dict) -> dict:
        observation = dict(observation)
        state_mask = observation.pop("state_mask")
        observation["prompt"] = build_trajectory_prompt(
            observation["prompt"],
            observation.pop("source_schema"),
            observation.pop("robot_identity"),
            observation.pop("gripper_identity"),
            bool(state_mask[7]),
        )
        observation["state"] = np.where(
            state_mask, normalize(observation["state"], self.stats["state"]), 0
        ).astype(np.float32)
        if "actions" in observation:
            observation["actions"] = normalize(
                observation["actions"], self.stats["action"]
            )
        return self.model_transform(observation)


class TrajectoryOutputTransform:
    """Return future measured joint positions and gripper targets in recipe units."""

    def __init__(self, assets: dict) -> None:
        self.stats = next(iter(assets["normalization"]["schemas"].values()))["action"]

    def __call__(self, output: dict) -> dict:
        return {**output, "actions": unnormalize(output["actions"], self.stats)}

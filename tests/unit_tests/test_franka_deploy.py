# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the physical Pi0.5 pickup observation/action contract."""

import numpy as np
import pytest

from toolkits.franka_deploy import pi05_policy


def stats(lo=-1.0, hi=1.0):
    return {
        "transform": "quantile",
        "q01": [lo] * 8,
        "q99": [hi] * 8,
        "count": [4096] * 8,
    }


def policy():
    value = pi05_policy.Pi05QtargetPolicy.__new__(pi05_policy.Pi05QtargetPolicy)
    value.stats = {"state": stats(0, 0.1), "action": stats()}
    value.horizon = 16
    value.metadata = {"gripper_feedback_used": True}
    return value


def test_quantiles_preserve_outliers_constants_and_passthrough():
    config = stats(0, 2)
    config["q99"][1] = 0
    config["passthrough_indices"] = [7]
    raw = np.array([3, 4, -1, 0, 1, 2, 5, 0.25], dtype=np.float32)
    encoded = pi05_policy.quantile_transform(raw, config)
    assert encoded[0] == 2
    assert encoded[1] == 4
    assert encoded[2] == -2
    assert encoded[7] == 0.25
    np.testing.assert_allclose(
        pi05_policy.quantile_transform(encoded, config, inverse=True), raw
    )


def test_measured_width_reaches_transform_before_padding(monkeypatch):
    model = policy()
    seen = []

    def transform(observation):
        seen.append(observation["state"].copy())
        return observation

    model.transform = transform
    monkeypatch.setattr(pi05_policy, "preprocess_image", lambda image: image.copy())
    request = {
        "qpos": np.arange(7, dtype=np.float32) / 100,
        "gripper_width_m": 0.085,
        "prompt": "Pick up the orange cube",
        "images": {
            role: np.zeros((224, 224, 3), np.uint8) for role in ("external", "wrist")
        },
    }
    first = model.prepare_observation(request)
    request["gripper_width_m"] = 0.01
    second = model.prepare_observation(request)
    assert seen[0].shape == seen[1].shape == (8,)
    assert seen[0][7] == pytest.approx(0.7)
    assert seen[1][7] == pytest.approx(-0.8)
    np.testing.assert_array_equal(seen[0][:7], seen[1][:7])
    assert first["prompt"] == request["prompt"]
    assert first["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    assert not second["image"]["right_wrist_0_rgb"].any()


def test_joint_delta_uses_one_anchor_and_preserves_continuous_gripper():
    model = policy()
    anchor = np.arange(7, dtype=np.float32) / 10
    predicted = np.full((16, 32), np.nan, dtype=np.float32)
    predicted[:, :7] = np.arange(16, dtype=np.float32)[:, None] / 100
    predicted[:, 7] = 0.25
    result = model.decode_actions(predicted, anchor)
    expected = anchor[None, :] + np.arange(16, dtype=np.float32)[:, None] / 100
    np.testing.assert_allclose(result["qtarget"], expected)
    np.testing.assert_allclose(result["gripper_closedness"], 0.25)
    assert result["qtarget"].shape == (16, 7)
    assert result["gripper_closedness"].shape == (16,)


def test_invalid_physical_output_rejected():
    model = policy()
    predicted = np.zeros((16, 32), np.float32)
    predicted[4, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.decode_actions(predicted, np.zeros(7))

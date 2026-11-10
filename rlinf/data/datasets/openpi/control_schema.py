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

"""Reversible native-control transforms shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _float_copy(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array.astype(np.result_type(array.dtype, np.float32), copy=True)


@dataclass(frozen=True)
class ControlAdapter:
    """Map native controls to chunk-relative joints and canonical grippers.

    Actions have shape ``[..., horizon, action_dim]`` and the anchor state has
    shape ``[..., state_dim]``, broadcastable across the action batch dimensions.
    Every future joint target subtracts the same anchor joint position. Gripper
    endpoints map to zero (open) and one (closed), without clipping.
    """

    kind: str = "identity"
    joint_action_indices: tuple[int, ...] = ()
    joint_state_indices: tuple[int, ...] = ()
    gripper_action_indices: tuple[int, ...] = ()
    gripper_state_indices: tuple[int, ...] = ()
    gripper_open: float | tuple[float, ...] = 0.0
    gripper_closed: float | tuple[float, ...] = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"identity", "joint_relative"}:
            raise ValueError(f"Unknown control adapter kind: {self.kind}")
        if self.kind == "joint_relative":
            if not self.joint_action_indices or len(self.joint_action_indices) != len(
                self.joint_state_indices
            ):
                raise ValueError(
                    "Joint action/state indices must be nonempty and paired"
                )
        elif self.joint_action_indices or self.joint_state_indices:
            raise ValueError("Joint indices require kind='joint_relative'")
        for joints, grippers in (
            (self.joint_action_indices, self.gripper_action_indices),
            (self.joint_state_indices, self.gripper_state_indices),
        ):
            indices = (*joints, *grippers)
            if any(not isinstance(index, int) or index < 0 for index in indices) or len(
                set(indices)
            ) != len(indices):
                raise ValueError(
                    "Control indices must be nonnegative and nonoverlapping"
                )
            if grippers:
                endpoints = np.broadcast_arrays(
                    np.asarray(self.gripper_open), np.asarray(self.gripper_closed)
                )
                for endpoint in endpoints:
                    np.broadcast_to(endpoint, (len(grippers),))
                if not all(np.isfinite(endpoint).all() for endpoint in endpoints):
                    raise ValueError("Gripper endpoints must be finite")
                if np.any(endpoints[0] == endpoints[1]):
                    raise ValueError("Gripper open and closed endpoints must differ")

    @classmethod
    def from_config(cls, config: dict | None) -> ControlAdapter:
        """Build from JSON-compatible fields; None preserves native controls."""
        if config is None:
            return cls()
        fields = dict(config)
        for name in (
            "joint_action_indices",
            "joint_state_indices",
            "gripper_action_indices",
            "gripper_state_indices",
        ):
            if name in fields:
                fields[name] = tuple(fields[name])
        for name in ("gripper_open", "gripper_closed"):
            if name in fields and isinstance(fields[name], list):
                fields[name] = tuple(fields[name])
        return cls(**fields)

    def _gripper(
        self, values: np.ndarray, indices: tuple[int, ...], inverse: bool = False
    ) -> None:
        if indices:
            opened = np.asarray(self.gripper_open, dtype=values.dtype)
            scale = np.asarray(self.gripper_closed, dtype=values.dtype) - opened
            gripper = values[..., indices]
            values[..., indices] = (
                gripper * scale + opened if inverse else (gripper - opened) / scale
            )

    def _anchor(self, actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
        if actions.ndim < 2 or np.ndim(anchor_state) < 1:
            raise ValueError("Expected actions [..., H, D] and anchor_state [..., S]")
        joints = np.asarray(anchor_state)[..., self.joint_state_indices]
        return np.broadcast_to(
            joints, (*actions.shape[:-2], len(self.joint_action_indices))
        )[..., None, :]

    def encode_actions(
        self, actions: np.ndarray, anchor_state: np.ndarray
    ) -> np.ndarray:
        """Encode a native chunk using one reference state; do not mutate inputs."""
        encoded = _float_copy(actions)
        if encoded.ndim < 2:
            raise ValueError("Expected actions with shape [..., horizon, action_dim]")
        if self.kind == "joint_relative":
            encoded[..., self.joint_action_indices] -= self._anchor(
                encoded, anchor_state
            )
        self._gripper(encoded, self.gripper_action_indices)
        return encoded

    def decode_actions(
        self, actions: np.ndarray, anchor_state: np.ndarray
    ) -> np.ndarray:
        """Recover native joint targets and grippers from an encoded chunk."""
        decoded = _float_copy(actions)
        if decoded.ndim < 2:
            raise ValueError("Expected actions with shape [..., horizon, action_dim]")
        self._gripper(decoded, self.gripper_action_indices, inverse=True)
        if self.kind == "joint_relative":
            decoded[..., self.joint_action_indices] += self._anchor(
                decoded, anchor_state
            )
        return decoded

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """Preserve native state fields except canonical gripper coordinates."""
        encoded = _float_copy(state)
        self._gripper(encoded, self.gripper_state_indices)
        return encoded


def _normalization_parameters(
    values: np.ndarray, stats: dict
) -> tuple[np.ndarray, np.ndarray]:
    transform = stats.get("transform", "mean_std")
    if transform == "quantile":
        lo = np.asarray(stats["q01"], dtype=values.dtype)
        hi = np.asarray(stats["q99"], dtype=values.dtype)
        if np.any(hi < lo):
            raise ValueError("Expected q99 >= q01 for every normalization dimension")
        center, scale = (lo + hi) / 2, (hi - lo) / 2
    elif transform in {"mean_std", "tanh_mean_std"}:
        center = np.asarray(stats["mean"], dtype=values.dtype)
        scale = np.asarray(stats["std"], dtype=values.dtype)
        if np.any(scale < 0):
            raise ValueError("Normalization standard deviations must be nonnegative")
    else:
        raise ValueError(f"Unknown normalization transform: {transform}")
    if center.shape != values.shape[-1:] or scale.shape != values.shape[-1:]:
        raise ValueError(
            "Normalization statistics must match the final feature dimension"
        )
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Normalization statistics must be finite")
    return center, np.where(scale == 0, 1, scale)


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize the final feature axis without clipping quantiles.

    ``transform`` defaults to legacy ``mean_std`` (``mean``, ``std``). Other
    choices are ``tanh_mean_std`` and ``quantile`` (``q01``, ``q99``). Constant
    dimensions use unit scale. ``passthrough_indices`` remain unchanged, which
    preserves canonical binary gripper targets.
    """
    source = _float_copy(values)
    center, scale = _normalization_parameters(source, stats)
    result = (source - center) / scale
    if stats.get("transform") == "tanh_mean_std":
        result = np.tanh(result)
    indices = stats.get("passthrough_indices", [])
    result[..., indices] = source[..., indices]
    return result


def unnormalize(values: np.ndarray, stats: dict) -> np.ndarray:
    """Invert normalize; tanh values must remain strictly between -1 and 1."""
    source = _float_copy(values)
    center, scale = _normalization_parameters(source, stats)
    result = source.copy()
    indices = stats.get("passthrough_indices", [])
    result[..., indices] = 0
    if stats.get("transform") == "tanh_mean_std":
        if np.any(np.abs(result) >= 1):
            raise ValueError("Inverse tanh normalization requires values in (-1, 1)")
        result = np.arctanh(result)
    result = result * scale + center
    result[..., indices] = source[..., indices]
    return result

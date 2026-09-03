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

"""Validated simulator-state pools for correctness-first BEHAVIOR RL."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

SUBPOOL_FORMAT_VERSION = 2
SUBPOOL_TYPES = ("canonical", "predecessor_success", "recovery")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as state_file:
        for block in iter(lambda: state_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_state_bytes(state: Mapping[str, Any]) -> bytes:
    """Serialize a complete OmniGibson state, including grasp constraints."""
    if not isinstance(state, Mapping):
        raise TypeError("A full simulator state must be a mapping.")
    buffer = io.BytesIO()
    # OmniGibson's flat ``serialized=True`` representation omits assisted-grasp
    # state.  The legacy torch format is deterministic for a fixed state and
    # retains the nested tensors / strings required by ``serialized=False``.
    torch.save(state, buffer, _use_new_zipfile_serialization=False)
    return buffer.getvalue()


def full_state_sha256(state: Mapping[str, Any]) -> str:
    """Return the checksum of a complete OmniGibson simulator state."""
    return hashlib.sha256(_full_state_bytes(state)).hexdigest()


@dataclass(frozen=True)
class SubpoolSnapshot:
    """One restorable BEHAVIOR simulator state and its training condition."""

    snapshot_id: str
    state_path: str
    state_sha256: str
    activity_name: str
    scene_model: str
    asset_fingerprint: str
    subtask_id: int
    skill: str
    pool_type: str
    task_description: str
    control_json: str
    episode_index: int | None = None
    frame_index: int | None = None
    format_version: int = SUBPOOL_FORMAT_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format_version != SUBPOOL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported subpool format_version={self.format_version}; "
                f"expected {SUBPOOL_FORMAT_VERSION}."
            )
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be empty.")
        if Path(self.state_path).is_absolute():
            raise ValueError("state_path must be relative to the catalog directory.")
        if Path(self.state_path).suffix != ".pt":
            raise ValueError(
                "Subpool format v2 requires a .pt full-state checkpoint; flat "
                ".npy states lose assisted-grasp constraints."
            )
        if len(self.state_sha256) != 64:
            raise ValueError("state_sha256 must be a hexadecimal SHA-256 digest.")
        if self.pool_type not in SUBPOOL_TYPES:
            raise ValueError(
                f"Unsupported pool_type={self.pool_type!r}; expected one of "
                f"{SUBPOOL_TYPES}."
            )
        if self.subtask_id < 0:
            raise ValueError("subtask_id must be non-negative.")
        if not self.skill.strip():
            raise ValueError("skill must not be empty.")
        if not self.task_description.strip():
            raise ValueError("task_description must not be empty.")
        try:
            control = json.loads(self.control_json)
        except json.JSONDecodeError as exc:
            raise ValueError("control_json must contain valid JSON.") from exc
        if not isinstance(control, dict):
            raise ValueError("control_json must encode a JSON object.")
        if control.get("skill") != self.skill:
            raise ValueError(
                "control_json skill must match the manifest skill, got "
                f"{control.get('skill')!r} and {self.skill!r}."
            )
        reward_spec = self.metadata.get("reward")
        if not isinstance(reward_spec, Mapping):
            raise ValueError("metadata.reward must contain a reward specification.")
        instance_id = self.metadata.get("instance_id")
        if not isinstance(instance_id, int) or isinstance(instance_id, bool):
            raise ValueError("metadata.instance_id must be an integer.")
        if instance_id < 0:
            raise ValueError("metadata.instance_id must be non-negative.")
        from rlinf.envs.behavior.subpool_reward import SubtaskRewardSpec

        SubtaskRewardSpec.from_mapping(reward_spec)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubpoolSnapshot":
        """Build a snapshot record from one manifest entry."""
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest entry."""
        value = asdict(self)
        value["metadata"] = dict(self.metadata)
        return value


class SubpoolCatalog:
    """Immutable view over a validated JSONL subpool manifest."""

    def __init__(self, root: Path, records: Sequence[SubpoolSnapshot]) -> None:
        if not records:
            raise ValueError("A subpool catalog must contain at least one snapshot.")
        self.root = root.resolve()
        self.records = tuple(records)
        ids = [record.snapshot_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Subpool manifest contains duplicate snapshot_id values.")

        self._by_subtask: dict[int, dict[str, tuple[SubpoolSnapshot, ...]]] = {}
        for subtask_id in sorted({record.subtask_id for record in self.records}):
            self._by_subtask[subtask_id] = {}
            for pool_type in SUBPOOL_TYPES:
                selected = tuple(
                    record
                    for record in self.records
                    if record.subtask_id == subtask_id and record.pool_type == pool_type
                )
                if selected:
                    self._by_subtask[subtask_id][pool_type] = selected
            if "canonical" not in self._by_subtask[subtask_id]:
                raise ValueError(f"subtask_id={subtask_id} has no canonical snapshot.")

        from rlinf.envs.behavior.subpool_reward import SubtaskRewardSpec

        reward_scale_signatures = set()
        for record in self.records:
            spec = SubtaskRewardSpec.from_mapping(record.metadata["reward"])
            reward_scale_signatures.add(
                (
                    spec.success_bonus,
                    spec.timeout_penalty,
                    spec.progress_clip,
                    round(spec.step_penalty * spec.max_steps, 6),
                )
            )
        if len(reward_scale_signatures) != 1:
            raise ValueError(
                "All subtasks must share success/timeout/progress scales and the "
                "same cumulative step-penalty budget, got "
                f"{sorted(reward_scale_signatures)}."
            )

    @classmethod
    def from_jsonl(
        cls,
        manifest_path: str | os.PathLike[str],
        *,
        verify_states: bool = True,
    ) -> "SubpoolCatalog":
        """Load and validate a subpool manifest.

        Args:
            manifest_path: JSONL manifest path.
            verify_states: Verify that every state exists and matches its checksum.

        Returns:
            The validated catalog.
        """
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Subpool manifest does not exist: {path}")

        records = []
        with path.open("r", encoding="utf-8") as manifest_file:
            fcntl.flock(manifest_file.fileno(), fcntl.LOCK_SH)
            try:
                for line_number, line in enumerate(manifest_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(SubpoolSnapshot.from_dict(json.loads(line)))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid subpool manifest entry at {path}:{line_number}."
                        ) from exc
            finally:
                fcntl.flock(manifest_file.fileno(), fcntl.LOCK_UN)

        catalog = cls(path.parent, records)
        if verify_states:
            for record in catalog.records:
                catalog.load_state(record)
        return catalog

    @property
    def subtask_ids(self) -> tuple[int, ...]:
        """Return sorted subtask ids represented by the catalog."""
        return tuple(self._by_subtask)

    @property
    def runtime_signature(self) -> tuple[str, str, int]:
        """Return the one activity, scene, and instance this process may load."""
        signatures = {
            (
                record.activity_name,
                record.scene_model,
                int(record.metadata["instance_id"]),
            )
            for record in self.records
        }
        if len(signatures) != 1:
            raise ValueError(
                "One persistent BEHAVIOR simulator can only consume snapshots "
                "from one (activity_name, scene_model, instance_id), got "
                f"{sorted(signatures)}."
            )
        return next(iter(signatures))

    def state_path(self, record: SubpoolSnapshot) -> Path:
        """Resolve a record path without allowing it to escape the catalog."""
        path = (self.root / record.state_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(
                f"Snapshot {record.snapshot_id!r} escapes catalog root: {path}"
            )
        return path

    def load_state(self, record: SubpoolSnapshot) -> Mapping[str, Any]:
        """Load one complete simulator state after checksum validation."""
        path = self.state_path(record)
        if not path.is_file():
            raise FileNotFoundError(
                f"State for snapshot {record.snapshot_id!r} does not exist: {path}"
            )
        actual_digest = _sha256_file(path)
        if actual_digest != record.state_sha256:
            raise ValueError(
                f"State checksum mismatch for snapshot {record.snapshot_id!r}: "
                f"expected {record.state_sha256}, got {actual_digest}."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError(
                f"Snapshot {record.snapshot_id!r} must contain a full mapping state."
            )
        return state

    def sample(
        self,
        rng: np.random.Generator,
        *,
        subtask_id: int | None = None,
        pool_weights: Mapping[str, float] | None = None,
    ) -> SubpoolSnapshot:
        """Sample a subtask uniformly, then a configured pool and snapshot.

        Missing pools are removed and the remaining weights are renormalized. This
        makes the subtask distribution independent of how many snapshots each skill
        has, which prevents easy or over-collected skills from dominating a batch.
        """
        if subtask_id is None:
            subtask_id = int(rng.choice(self.subtask_ids))
        if subtask_id not in self._by_subtask:
            raise KeyError(f"Unknown subtask_id={subtask_id}.")

        available = self._by_subtask[subtask_id]
        weights = dict.fromkeys(SUBPOOL_TYPES, 1.0)
        if pool_weights is not None:
            unknown = set(pool_weights) - set(SUBPOOL_TYPES)
            if unknown:
                raise ValueError(f"Unknown subpool weight keys: {sorted(unknown)}")
            weights.update(pool_weights)

        pool_names = tuple(available)
        probabilities = np.asarray([weights[name] for name in pool_names], dtype=float)
        if np.any(probabilities < 0) or not np.any(probabilities > 0):
            raise ValueError(
                f"Available pool weights for subtask_id={subtask_id} must include "
                "at least one positive finite value."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Subpool weights must be finite.")
        probabilities /= probabilities.sum()
        pool_name = str(rng.choice(pool_names, p=probabilities))
        records = available[pool_name]
        return records[int(rng.integers(len(records)))]


class SubpoolStore:
    """Append-only writer for states produced by online rollouts."""

    def __init__(self, manifest_path: str | os.PathLike[str]) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.manifest_path.parent / "states"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: SubpoolSnapshot, state: Mapping[str, Any]) -> None:
        """Atomically write a complete state and append its manifest entry."""
        state_bytes = _full_state_bytes(state)

        target = self.state_dir / f"{record.snapshot_id}.pt"
        expected_relative = str(target.relative_to(self.manifest_path.parent))
        if record.state_path != expected_relative:
            raise ValueError(
                f"record.state_path must be {expected_relative!r}, got "
                f"{record.state_path!r}."
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.state_dir, prefix=".state-", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(state_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            actual_digest = _sha256_file(temporary_path)
            if actual_digest != record.state_sha256:
                raise ValueError(
                    f"state_sha256 for snapshot {record.snapshot_id!r} does not "
                    "match the supplied state."
                )
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

        with self.manifest_path.open("a", encoding="utf-8") as manifest_file:
            fcntl.flock(manifest_file.fileno(), fcntl.LOCK_EX)
            manifest_file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
            fcntl.flock(manifest_file.fileno(), fcntl.LOCK_UN)


def validate_subpool_env_config(
    cfg: Any,
    *,
    num_envs: int,
    pipeline_stage_num: int,
) -> None:
    """Reject BEHAVIOR optimizations that violate subpool PPO semantics."""

    def select(key: str, default: Any) -> Any:
        return OmegaConf.select(cfg, key, default=default)

    errors = []
    if num_envs != 1:
        errors.append(f"num_envs must be 1, got {num_envs}")
    if int(select("num_env_subprocess", 1)) != 1:
        errors.append("num_env_subprocess must be 1")
    if bool(select("skip_intermediate_obs_in_chunk", False)):
        errors.append("skip_intermediate_obs_in_chunk must be false")
    if pipeline_stage_num != 1:
        errors.append(f"pipeline_stage_num must be 1, got {pipeline_stage_num}")
    if bool(select("auto_reset", False)):
        errors.append("auto_reset must be false")
    if bool(select("enable_offload", False)):
        errors.append("enable_offload must be false")
    if str(select("renderer_mode", "rlinf")) != "official":
        errors.append("renderer_mode must be official")
    if errors:
        raise ValueError(
            "Invalid correctness-first BEHAVIOR subpool config: " + "; ".join(errors)
        )


def validate_subpool_rollout_horizons(
    reward_horizons: Sequence[int],
    *,
    episode_horizon: int,
    rollout_horizon: int,
) -> None:
    """Require the fixed rollout to cover every task-specific timeout."""
    if not reward_horizons:
        raise ValueError("At least one subtask reward horizon is required.")
    longest = max(int(horizon) for horizon in reward_horizons)
    if episode_horizon < longest:
        raise ValueError(
            "env.max_episode_steps must cover the longest subtask reward "
            f"horizon ({longest}), got {episode_horizon}."
        )
    if rollout_horizon < longest:
        raise ValueError(
            "env.max_steps_per_rollout_epoch must cover the longest subtask "
            f"reward horizon ({longest}), got {rollout_horizon}."
        )


def validate_round_robin_coverage(
    subtask_ids: Sequence[int],
    *,
    env_world_size: int,
    fixed_subtask_id: int | None,
) -> None:
    """Require one subtask per env rank for an exactly balanced PPO batch."""
    unique_subtask_ids = tuple(sorted(set(subtask_ids)))
    if fixed_subtask_id is not None:
        if fixed_subtask_id not in unique_subtask_ids:
            raise ValueError(f"Unknown fixed_subtask_id={fixed_subtask_id}.")
        return
    if len(unique_subtask_ids) != env_world_size:
        raise ValueError(
            "Correctness-first round-robin requires exactly one subtask per env "
            f"rank, got {len(unique_subtask_ids)} subtasks and "
            f"env_world_size={env_world_size}."
        )


def validate_subpool_export_request(
    *,
    instance_reward_mode: str,
    run_episode_idx: int | None,
    run_episode_indices: Sequence[int] | None,
) -> None:
    """Require unambiguous episode IDs and task-specific reward stages."""
    if instance_reward_mode != "task":
        raise ValueError(
            "Subpool export requires instance_reward_mode=task so direct "
            "task-specific reward stages are installed."
        )
    if run_episode_idx is not None:
        raise ValueError(
            "Subpool export rejects positional run_episode_idx; use the explicit "
            "run_episode_indices list to prevent silently selecting another episode."
        )
    if not run_episode_indices:
        raise ValueError(
            "Subpool export requires a non-empty explicit run_episode_indices list."
        )


__all__ = [
    "SUBPOOL_FORMAT_VERSION",
    "SUBPOOL_TYPES",
    "SubpoolCatalog",
    "SubpoolSnapshot",
    "SubpoolStore",
    "full_state_sha256",
    "validate_round_robin_coverage",
    "validate_subpool_export_request",
    "validate_subpool_env_config",
    "validate_subpool_rollout_horizons",
]

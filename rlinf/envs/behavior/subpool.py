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
import shutil
import socket
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

SUBPOOL_FORMAT_VERSION = 2
SUBPOOL_TYPES = ("canonical", "predecessor_success", "recovery")
FAILURE_STATE_FORMAT_VERSION = 2
RECOVERY_STATUSES = ("eligible", "ineligible", "not_needed", "unknown")


def relative_tilt_angle_deg(
    reference_quaternion: Sequence[float],
    current_quaternion: Sequence[float],
) -> float:
    """Measure how far an object's original up direction has tilted.

    OmniGibson quaternions use ``(x, y, z, w)`` order. The reference pose
    determines which object-local direction was vertical at reset, so this
    remains valid for assets whose local z-axis is not their physical up axis.
    """

    def rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
        value = np.asarray(quaternion, dtype=np.float64)
        if value.shape != (4,):
            raise ValueError(
                f"Expected an xyzw quaternion with shape (4,), got {value.shape}."
            )
        norm = np.linalg.norm(value)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Quaternion must have a finite, non-zero norm.")
        x, y, z, w = value / norm
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    world_up = np.asarray([0.0, 0.0, 1.0])
    reference_rotation = rotation_matrix(reference_quaternion)
    current_rotation = rotation_matrix(current_quaternion)
    reference_local_up = reference_rotation.T @ world_up
    current_world_up = current_rotation @ reference_local_up
    cosine = float(np.clip(np.dot(current_world_up, world_up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


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
    def runtime_signature(self) -> tuple[str, str]:
        """Return the activity and scene shared by this process's snapshots."""
        signatures = {
            (record.activity_name, record.scene_model) for record in self.records
        }
        if len(signatures) != 1:
            raise ValueError(
                "One persistent BEHAVIOR simulator can only consume snapshots "
                "from one (activity_name, scene_model), got "
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
        target = self._state_target(record)

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

        self._append_record(record)

    def append_from_file(
        self, record: SubpoolSnapshot, source_path: str | os.PathLike[str]
    ) -> None:
        """Atomically copy an already validated state and append its record."""
        target = self._state_target(record)
        source = Path(source_path).resolve()
        if _sha256_file(source) != record.state_sha256:
            raise ValueError(
                f"state_sha256 for snapshot {record.snapshot_id!r} does not "
                "match the source file."
            )
        with (
            source.open("rb") as source_file,
            tempfile.NamedTemporaryFile(
                mode="wb", dir=self.state_dir, prefix=".state-", delete=False
            ) as temporary_file,
        ):
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(source_file, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

        self._append_record(record)

    def _state_target(self, record: SubpoolSnapshot) -> Path:
        target = self.state_dir / f"{record.snapshot_id}.pt"
        expected_relative = str(target.relative_to(self.manifest_path.parent))
        if record.state_path != expected_relative:
            raise ValueError(
                f"record.state_path must be {expected_relative!r}, got "
                f"{record.state_path!r}."
            )
        return target

    def _append_record(self, record: SubpoolSnapshot) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as manifest_file:
            fcntl.flock(manifest_file.fileno(), fcntl.LOCK_EX)
            manifest_file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
            fcntl.flock(manifest_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class FailureAnalysis:
    """Structured, skill-relative interpretation of one failed state."""

    termination_reason: str | None
    failure_tags: tuple[str, ...]
    recovery_status: str
    recovery_reason: str
    facts: Mapping[str, Any]
    analyzer: str

    def __post_init__(self) -> None:
        if self.recovery_status not in RECOVERY_STATUSES:
            raise ValueError(
                f"Unsupported recovery_status={self.recovery_status!r}; expected "
                f"one of {RECOVERY_STATUSES}."
            )
        if not self.failure_tags:
            raise ValueError("failure_tags must not be empty.")
        if not self.recovery_reason:
            raise ValueError("recovery_reason must not be empty.")
        if not self.analyzer:
            raise ValueError("analyzer must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable analysis record."""
        return {
            "termination_reason": self.termination_reason,
            "failure_tags": list(self.failure_tags),
            "recovery": {
                "status": self.recovery_status,
                "reason": self.recovery_reason,
            },
            "facts": dict(self.facts),
            "analyzer": self.analyzer,
        }


def analyze_subtask_failure(
    skill: str,
    facts: Mapping[str, Any],
    *,
    termination_reason: str | None,
) -> FailureAnalysis:
    """Classify simulator facts relative to the active primitive skill.

    Unknown skills retain their facts for later analyzers instead of being
    admitted to recovery training without evidence.
    """
    normalized_skill = skill.strip().lower()
    if normalized_skill != "pick up from":
        return FailureAnalysis(
            termination_reason=termination_reason,
            failure_tags=("subtask_incomplete",),
            recovery_status="unknown",
            recovery_reason=f"No failure analyzer is registered for {skill!r}.",
            facts=facts,
            analyzer="generic-v1",
        )

    required = ("in_hand", "on_original_support", "tilt_angle_deg")
    missing = [key for key in required if key not in facts]
    if missing:
        raise KeyError(f"Pickup failure facts are missing required keys: {missing}.")

    in_hand = bool(facts["in_hand"])
    on_support = bool(facts["on_original_support"])
    tipped = bool(facts.get("tipped", False))
    common_tags = ["target_in_hand" if in_hand else "gripper_empty"]
    if on_support:
        common_tags.append("target_on_original_support")
    else:
        common_tags.append("target_off_original_support")

    if in_hand:
        return FailureAnalysis(
            termination_reason=termination_reason,
            failure_tags=tuple(common_tags),
            recovery_status="unknown",
            recovery_reason=(
                "The target is already in hand although pickup did not terminate."
            ),
            facts=facts,
            analyzer="pickup-v1",
        )
    if not on_support:
        return FailureAnalysis(
            termination_reason=termination_reason,
            failure_tags=tuple(common_tags),
            recovery_status="ineligible",
            recovery_reason="The target has left its audited source support.",
            facts=facts,
            analyzer="pickup-v1",
        )
    if tipped:
        return FailureAnalysis(
            termination_reason=termination_reason,
            failure_tags=tuple(common_tags + ["target_tipped"]),
            recovery_status="eligible",
            recovery_reason=(
                "The tipped target remains on its audited source support."
            ),
            facts=facts,
            analyzer="pickup-v1",
        )
    return FailureAnalysis(
        termination_reason=termination_reason,
        failure_tags=tuple(common_tags + ["target_upright"]),
        recovery_status="not_needed",
        recovery_reason="The original pickup policy can retry an upright target.",
        facts=facts,
        analyzer="pickup-v1",
    )


@dataclass(frozen=True)
class FailureStateRecord:
    """Metadata for one failed rollout's restorable terminal state."""

    failure_id: str
    state_path: str
    state_sha256: str
    captured_at_utc: str
    hostname: str
    process_id: int
    run_id: str | None
    policy_global_step: int | None
    collection_index: int | None
    sampling_group: int | None
    capture_kind: str
    source_snapshot: Mapping[str, Any]
    outcome: Mapping[str, Any]
    analysis: Mapping[str, Any]
    format_version: int = FAILURE_STATE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FAILURE_STATE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported failure-state format_version={self.format_version}; "
                f"expected {FAILURE_STATE_FORMAT_VERSION}."
            )
        if not self.failure_id:
            raise ValueError("failure_id must not be empty.")
        if Path(self.state_path).is_absolute() or Path(self.state_path).suffix != ".pt":
            raise ValueError("state_path must be a relative .pt path.")
        if len(self.state_sha256) != 64:
            raise ValueError("state_sha256 must be a hexadecimal SHA-256 digest.")
        if self.policy_global_step is not None and self.policy_global_step < 0:
            raise ValueError("policy_global_step must be non-negative.")
        if not self.source_snapshot.get("snapshot_id"):
            raise ValueError("source_snapshot.snapshot_id must not be empty.")
        if bool(self.outcome.get("success", False)):
            raise ValueError("A failure-state record cannot describe a success.")
        if self.capture_kind not in ("terminal", "stable_recovery_event"):
            raise ValueError(f"Unsupported capture_kind={self.capture_kind!r}.")
        if not self.analysis.get("failure_tags"):
            raise ValueError("analysis.failure_tags must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact record."""
        value = asdict(self)
        value["source_snapshot"] = dict(self.source_snapshot)
        value["outcome"] = dict(self.outcome)
        return value


class FailureStateStore:
    """Atomically persist failed rollout terminal states and sidecar metadata."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        run_id: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        self.run_id = run_id

    def capture(
        self,
        state: Mapping[str, Any],
        *,
        policy_global_step: int | None,
        collection_index: int | None,
        sampling_group: int | None,
        capture_kind: str,
        source_snapshot: Mapping[str, Any],
        outcome: Mapping[str, Any],
        analysis: Mapping[str, Any],
    ) -> Path:
        """Save one full simulator state and return its metadata path."""
        state_bytes = _full_state_bytes(state)
        state_sha256 = hashlib.sha256(state_bytes).hexdigest()
        failure_id = f"failure-{uuid.uuid4().hex}"
        step_name = (
            "global_step_unknown"
            if policy_global_step is None
            else f"global_step_{policy_global_step:06d}"
        )
        artifact_dir = self.output_dir / self.hostname / step_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state_path = artifact_dir / f"{failure_id}.pt"
        metadata_path = artifact_dir / f"{failure_id}.json"
        record = FailureStateRecord(
            failure_id=failure_id,
            state_path=state_path.name,
            state_sha256=state_sha256,
            captured_at_utc=datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            hostname=self.hostname,
            process_id=self.process_id,
            run_id=self.run_id,
            policy_global_step=policy_global_step,
            collection_index=collection_index,
            sampling_group=sampling_group,
            capture_kind=capture_kind,
            source_snapshot=dict(source_snapshot),
            outcome=dict(outcome),
            analysis=dict(analysis),
        )

        self._atomic_write(state_path, state_bytes)
        self._atomic_write(
            metadata_path,
            (json.dumps(record.to_dict(), sort_keys=True) + "\n").encode("utf-8"),
        )
        return metadata_path

    @staticmethod
    def _atomic_write(path: Path, contents: bytes) -> None:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def merge_subpool_manifests(
    input_manifests: Sequence[str | os.PathLike[str]],
    output_manifest: str | os.PathLike[str],
) -> SubpoolCatalog:
    """Merge validated catalogs into one self-contained catalog.

    The output is created only after all source manifests and destination paths
    pass validation. Snapshot identifiers must be unique across sources.

    Args:
        input_manifests: Source JSONL manifests to merge.
        output_manifest: Destination JSONL manifest in a fresh catalog directory.

    Returns:
        The merged, checksum-verified catalog.
    """
    if not input_manifests:
        raise ValueError("At least one input manifest is required.")

    output_path = Path(output_manifest).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}.")

    sources = [SubpoolCatalog.from_jsonl(path) for path in input_manifests]
    entries = [
        (
            source,
            record,
            replace(record, state_path=f"states/{record.snapshot_id}.pt"),
        )
        for source in sources
        for record in source.records
    ]
    normalized_records = [output_record for _, _, output_record in entries]
    candidate = SubpoolCatalog(output_path.parent, normalized_records)
    _ = candidate.runtime_signature

    state_paths = [candidate.state_path(record) for record in candidate.records]
    existing = [path for path in state_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite destination states: "
            + ", ".join(str(path) for path in existing)
        )

    store = SubpoolStore(output_path)
    for source, source_record, output_record in entries:
        store.append_from_file(output_record, source.state_path(source_record))
    return SubpoolCatalog.from_jsonl(output_path)


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
    if pipeline_stage_num != 1:
        errors.append(f"pipeline_stage_num must be 1, got {pipeline_stage_num}")
    if bool(select("enable_offload", False)):
        errors.append("enable_offload must be false")
    if str(select("renderer_mode", "rlinf")) != "official":
        errors.append("renderer_mode must be official")
    if bool(select("subpool.failure_state_capture.enabled", False)) and not select(
        "subpool.failure_state_capture.output_dir", None
    ):
        errors.append(
            "subpool.failure_state_capture.output_dir is required when enabled"
        )
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
    "FAILURE_STATE_FORMAT_VERSION",
    "RECOVERY_STATUSES",
    "SUBPOOL_FORMAT_VERSION",
    "SUBPOOL_TYPES",
    "FailureAnalysis",
    "FailureStateRecord",
    "FailureStateStore",
    "SubpoolCatalog",
    "SubpoolSnapshot",
    "SubpoolStore",
    "analyze_subtask_failure",
    "full_state_sha256",
    "merge_subpool_manifests",
    "relative_tilt_angle_deg",
    "validate_round_robin_coverage",
    "validate_subpool_export_request",
    "validate_subpool_env_config",
    "validate_subpool_rollout_horizons",
]

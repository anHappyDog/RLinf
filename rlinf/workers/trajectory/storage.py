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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch

from rlinf.data.forward_inputs import ForwardInputs
from rlinf.data.trajectory import (
    EnvResult,
    RewardMode,
    RewardResult,
    RolloutResult,
    TrajectoryData,
    ValueResult,
)

_ENV_FIELDS = frozenset(
    {
        "observations",
        "next_observations",
        "intervene_actions",
        "intervene_flags",
        "rlt_switch_flags",
    }
)
_ROLLOUT_FIELDS = frozenset(
    {
        "forward_inputs",
        "prev_logprobs",
        "state_values",
        "versions",
        "intervene_flags",
    }
)
_VALUE_FIELDS = frozenset({"versions"})


@dataclass(frozen=True, kw_only=True)
class StorageConfig:
    """Immutable schema and coordinate bounds for one trajectory shard."""

    global_step: int
    rollout_epochs: int
    chunk_steps: int
    slot_ids: tuple[int, ...]
    env_fields: frozenset[str] = frozenset()
    rollout_fields: frozenset[str] = frozenset()
    value_fields: frozenset[str] = frozenset()
    reward_mode: RewardMode | None = None
    reward_steps: tuple[int, ...] = ()
    boundary_values: bool = False

    def __post_init__(self) -> None:
        for name in ("global_step", "rollout_epochs", "chunk_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative.")
        if self.rollout_epochs < 1 or self.chunk_steps < 1:
            raise ValueError("rollout_epochs and chunk_steps must be positive.")
        if not isinstance(self.slot_ids, tuple):
            raise TypeError("slot_ids must be a tuple.")
        if not self.slot_ids or len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("slot_ids must be non-empty and unique.")
        if any(
            not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0
            for slot_id in self.slot_ids
        ):
            raise ValueError("slot_ids must contain non-negative integers.")

        self._validate_fields("env_fields", self.env_fields, _ENV_FIELDS)
        self._validate_fields("rollout_fields", self.rollout_fields, _ROLLOUT_FIELDS)
        self._validate_fields("value_fields", self.value_fields, _VALUE_FIELDS)

        if self.reward_mode is None:
            if self.reward_steps:
                raise ValueError("reward_steps requires reward_mode.")
        else:
            if self.reward_mode not in ("per_step", "terminal", "history_buffer"):
                raise ValueError(f"Unsupported reward_mode {self.reward_mode!r}.")
            if not self.reward_steps:
                raise ValueError(
                    "reward_steps must not be empty when reward is enabled."
                )
            if len(set(self.reward_steps)) != len(self.reward_steps):
                raise ValueError("reward_steps must not contain duplicates.")
            if any(
                not isinstance(step, int)
                or isinstance(step, bool)
                or not 0 <= step < self.chunk_steps
                for step in self.reward_steps
            ):
                raise ValueError("reward_steps must be valid transition indices.")

        if not isinstance(self.boundary_values, bool):
            raise TypeError("boundary_values must be a bool.")
        if not self.boundary_values and self.value_fields:
            raise ValueError("value_fields requires boundary_values=True.")

    @staticmethod
    def _validate_fields(
        name: str,
        configured: frozenset[str],
        allowed: frozenset[str],
    ) -> None:
        if not isinstance(configured, frozenset):
            raise TypeError(f"{name} must be a frozenset.")
        unexpected = configured - allowed
        if unexpected:
            raise ValueError(f"Unsupported {name}: {sorted(unexpected)}.")


@dataclass(frozen=True, kw_only=True)
class TrajectoryBatch:
    """A complete local-slot trajectory in Actor training order."""

    global_step: int
    slot_ids: tuple[int, ...]
    env_rewards: torch.Tensor
    dones: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    actions: torch.Tensor
    observations: Any | None = None
    next_observations: Any | None = None
    intervene_actions: torch.Tensor | None = None
    env_intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None
    forward_inputs: ForwardInputs | None = None
    prev_logprobs: torch.Tensor | None = None
    state_values: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    rollout_intervene_flags: torch.Tensor | None = None
    external_rewards: torch.Tensor | None = None
    reward_mask: torch.Tensor | None = None
    reward_history_lengths: torch.Tensor | None = None
    timeout_values: torch.Tensor | None = None
    timeout_mask: torch.Tensor | None = None
    tail_values: torch.Tensor | None = None
    tail_mask: torch.Tensor | None = None
    timeout_versions: torch.Tensor | None = None
    tail_versions: torch.Tensor | None = None

    @property
    def rollout_epochs(self) -> int:
        return self.actions.shape[0]

    @property
    def chunk_steps(self) -> int:
        return self.actions.shape[1]


_GRID_FIELDS = (
    "env_rewards",
    "dones",
    "terminations",
    "truncations",
    "actions",
    "observations",
    "next_observations",
    "intervene_actions",
    "env_intervene_flags",
    "rlt_switch_flags",
    "prev_logprobs",
    "state_values",
    "versions",
    "rollout_intervene_flags",
    "external_rewards",
    "reward_mask",
    "reward_history_lengths",
    "timeout_values",
    "timeout_mask",
    "timeout_versions",
)
_TAIL_FIELDS = ("tail_values", "tail_mask", "tail_versions")


def select_trajectory_batch(
    batch: TrajectoryBatch,
    indices: tuple[int, ...],
) -> TrajectoryBatch:
    """Select Actor-owned slots while preserving trajectory time order."""
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= len(batch.slot_ids) for index in indices
    ):
        raise ValueError("trajectory slot indices must be unique and in range")
    index = torch.tensor(indices, dtype=torch.long)
    updates = {
        name: _select_axis(getattr(batch, name), index, axis=2) for name in _GRID_FIELDS
    }
    updates.update(
        {
            name: _select_axis(getattr(batch, name), index, axis=1)
            for name in _TAIL_FIELDS
        }
    )
    updates["forward_inputs"] = _select_forward_inputs(batch, index)
    return replace(
        batch,
        slot_ids=tuple(batch.slot_ids[item] for item in indices),
        **updates,
    )


def merge_trajectory_batches(
    batches: Sequence[TrajectoryBatch],
    slot_ids: tuple[int, ...],
) -> TrajectoryBatch:
    """Merge Storage shards into one Actor shard in requested slot order."""
    if not batches:
        raise ValueError("at least one trajectory batch is required")
    if not slot_ids or len(set(slot_ids)) != len(slot_ids):
        raise ValueError("actor slot_ids must be non-empty and unique")
    first = batches[0]
    for batch in batches[1:]:
        if (
            batch.global_step != first.global_step
            or batch.rollout_epochs != first.rollout_epochs
            or batch.chunk_steps != first.chunk_steps
        ):
            raise ValueError("trajectory shards have incompatible coordinates")

    combined_slots = tuple(slot_id for batch in batches for slot_id in batch.slot_ids)
    if len(set(combined_slots)) != len(combined_slots):
        raise ValueError("trajectory shards contain overlapping slots")
    if set(combined_slots) != set(slot_ids):
        raise ValueError("trajectory shards do not exactly cover actor slots")

    updates = {
        name: _concat_axis([getattr(batch, name) for batch in batches], axis=2)
        for name in _GRID_FIELDS
    }
    updates.update(
        {
            name: _concat_axis([getattr(batch, name) for batch in batches], axis=1)
            for name in _TAIL_FIELDS
        }
    )
    updates["forward_inputs"] = _concat_forward_inputs(batches)
    combined = replace(first, slot_ids=combined_slots, **updates)
    positions = {slot_id: index for index, slot_id in enumerate(combined_slots)}
    return select_trajectory_batch(
        combined,
        tuple(positions[slot_id] for slot_id in slot_ids),
    )


class TrajectoryStorage:
    """Deterministically assemble one trajectory shard in memory."""

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self._slot_indices = {
            slot_id: index for index, slot_id in enumerate(config.slot_ids)
        }
        self._records: dict[tuple[str, int, int, str], list[TrajectoryData]] = {}
        self._identities: dict[
            tuple[str, int, int, str, tuple[int, ...]], TrajectoryData
        ] = {}
        self._signatures: dict[tuple[str, str], Any] = {}

    def write(
        self,
        result: EnvResult | RolloutResult | RewardResult | ValueResult,
    ) -> bool:
        """Store one routed result; return false for an exact duplicate retry."""
        role, discriminator = self._role(result)
        self._validate_coordinates(result, role)
        self._validate_schema(result)

        bucket = (role, result.rollout_epoch, result.chunk_step, discriminator)
        identity = (*bucket, result.slot_ids)
        previous = self._identities.get(identity)
        if previous is not None:
            if _equal(previous, result):
                return False
            raise ValueError("Conflicting content for an existing trajectory identity.")

        covered = {
            slot_id
            for stored in self._records.get(bucket, ())
            for slot_id in stored.slot_ids
        }
        overlap = covered.intersection(result.slot_ids)
        if overlap:
            raise ValueError(
                f"Overlapping {role} coverage for slots {sorted(overlap)} at "
                f"epoch={result.rollout_epoch}, step={result.chunk_step}."
            )

        signature_key = (role, "" if role == "value" else discriminator)
        signature = _payload_signature(result)
        previous_signature = self._signatures.get(signature_key)
        if previous_signature is not None and signature != previous_signature:
            raise ValueError(f"Inconsistent tensor schema for {role} results.")

        self._signatures.setdefault(signature_key, signature)
        self._identities[identity] = result
        self._records.setdefault(bucket, []).append(result)
        return True

    @property
    def ready(self) -> bool:
        """Whether all configured data and conditional values are present."""
        return not self.missing()

    def missing(self) -> tuple[str, ...]:
        """Describe absent or unexpected coverage that prevents export."""
        problems: list[str] = []
        expected = set(self.config.slot_ids)
        for epoch in range(self.config.rollout_epochs):
            for step in range(self.config.chunk_steps):
                for role in ("env", "rollout"):
                    actual = self._coverage((role, epoch, step, ""))
                    self._append_coverage_problem(
                        problems, role, epoch, step, expected, actual
                    )
                if step in self.config.reward_steps:
                    actual = self._coverage(
                        ("reward", epoch, step, self.config.reward_mode or "")
                    )
                    self._append_coverage_problem(
                        problems, "reward", epoch, step, expected, actual
                    )

        if self.config.boundary_values:
            self._append_value_problems(problems)
        return tuple(problems)

    def export(self) -> TrajectoryBatch:
        """Export the complete shard in epoch, step, then local-slot order."""
        problems = self.missing()
        if problems:
            raise RuntimeError("Trajectory is not ready: " + "; ".join(problems))

        env = self._grid("env", "")
        rollout = self._grid("rollout", "")
        env_rewards = self._assemble_grid(env, "rewards")
        optional_env = {
            name: self._assemble_grid(env, name) for name in self.config.env_fields
        }
        optional_rollout = {
            name: self._assemble_forward_inputs(rollout)
            if name == "forward_inputs"
            else self._assemble_grid(rollout, name)
            for name in self.config.rollout_fields
        }

        reward_data = self._assemble_rewards(env_rewards)
        value_data = self._assemble_values(env_rewards)
        return TrajectoryBatch(
            global_step=self.config.global_step,
            slot_ids=self.config.slot_ids,
            env_rewards=env_rewards,
            dones=self._assemble_grid(env, "dones"),
            terminations=self._assemble_grid(env, "terminations"),
            truncations=self._assemble_grid(env, "truncations"),
            actions=self._assemble_grid(rollout, "actions"),
            observations=optional_env.get("observations"),
            next_observations=optional_env.get("next_observations"),
            intervene_actions=optional_env.get("intervene_actions"),
            env_intervene_flags=optional_env.get("intervene_flags"),
            rlt_switch_flags=optional_env.get("rlt_switch_flags"),
            forward_inputs=optional_rollout.get("forward_inputs"),
            prev_logprobs=optional_rollout.get("prev_logprobs"),
            state_values=optional_rollout.get("state_values"),
            versions=optional_rollout.get("versions"),
            rollout_intervene_flags=optional_rollout.get("intervene_flags"),
            **reward_data,
            **value_data,
        )

    @staticmethod
    def _role(result: TrajectoryData) -> tuple[str, str]:
        if isinstance(result, EnvResult):
            return "env", ""
        if isinstance(result, RolloutResult):
            return "rollout", ""
        if isinstance(result, RewardResult):
            return "reward", result.mode
        if isinstance(result, ValueResult):
            return "value", result.kind
        raise TypeError(f"Unsupported storage input {type(result).__name__}.")

    def _validate_coordinates(self, result: TrajectoryData, role: str) -> None:
        if result.global_step != self.config.global_step:
            raise ValueError(
                f"Expected global_step {self.config.global_step}, "
                f"got {result.global_step}."
            )
        if not 0 <= result.rollout_epoch < self.config.rollout_epochs:
            raise ValueError("rollout_epoch is outside this storage generation.")
        expected_step = self.config.chunk_steps
        if role == "value" and isinstance(result, ValueResult):
            valid = (result.kind == "tail" and result.chunk_step == expected_step) or (
                result.kind == "timeout" and 0 <= result.chunk_step < expected_step
            )
        else:
            valid = 0 <= result.chunk_step < expected_step
        if not valid:
            raise ValueError("chunk_step is invalid for this result role.")

        unknown = set(result.slot_ids) - self._slot_indices.keys()
        if unknown:
            raise ValueError(f"Result contains unowned slots {sorted(unknown)}.")

    def _validate_schema(self, result: TrajectoryData) -> None:
        if isinstance(result, EnvResult):
            self._validate_optional_fields(result, self.config.env_fields, _ENV_FIELDS)
        elif isinstance(result, RolloutResult):
            self._validate_optional_fields(
                result, self.config.rollout_fields, _ROLLOUT_FIELDS
            )
        elif isinstance(result, RewardResult):
            if self.config.reward_mode is None:
                raise ValueError("RewardResult is not enabled by this storage schema.")
            if result.mode != self.config.reward_mode:
                raise ValueError(
                    f"Expected reward mode {self.config.reward_mode!r}, "
                    f"got {result.mode!r}."
                )
            if result.chunk_step not in self.config.reward_steps:
                raise ValueError("RewardResult is not expected at this chunk_step.")
        elif isinstance(result, ValueResult):
            if not self.config.boundary_values:
                raise ValueError("ValueResult is not enabled by this storage schema.")
            self._validate_optional_fields(
                result, self.config.value_fields, _VALUE_FIELDS
            )

    @staticmethod
    def _validate_optional_fields(
        result: TrajectoryData,
        configured: frozenset[str],
        allowed: frozenset[str],
    ) -> None:
        for name in allowed:
            present = getattr(result, name) is not None
            if present != (name in configured):
                expectation = "required" if name in configured else "not configured"
                raise ValueError(f"{name} is {expectation} for this storage schema.")

    def _coverage(self, bucket: tuple[str, int, int, str]) -> set[int]:
        return {
            slot_id
            for result in self._records.get(bucket, ())
            for slot_id in result.slot_ids
        }

    @staticmethod
    def _append_coverage_problem(
        problems: list[str],
        role: str,
        epoch: int,
        step: int,
        expected: set[int],
        actual: set[int],
    ) -> None:
        if missing := expected - actual:
            problems.append(
                f"missing {role} epoch={epoch} step={step} slots={sorted(missing)}"
            )
        if unexpected := actual - expected:
            problems.append(
                f"unexpected {role} epoch={epoch} step={step} "
                f"slots={sorted(unexpected)}"
            )

    def _append_value_problems(self, problems: list[str]) -> None:
        for epoch in range(self.config.rollout_epochs):
            for step in range(self.config.chunk_steps):
                env_bucket = self._records.get(("env", epoch, step, ""), ())
                if set(self._slots(env_bucket)) != set(self.config.slot_ids):
                    continue
                truncated = self._masked_slots(env_bucket, "truncations")
                terminated = self._masked_slots(env_bucket, "terminations")
                expected = truncated - terminated
                actual = self._coverage(("value", epoch, step, "timeout"))
                self._append_coverage_problem(
                    problems, "timeout value", epoch, step, expected, actual
                )

            final_env = self._records.get(
                ("env", epoch, self.config.chunk_steps - 1, ""), ()
            )
            if set(self._slots(final_env)) != set(self.config.slot_ids):
                continue
            done = self._masked_slots(final_env, "dones")
            expected = set(self.config.slot_ids) - done
            actual = self._coverage(("value", epoch, self.config.chunk_steps, "tail"))
            self._append_coverage_problem(
                problems,
                "tail value",
                epoch,
                self.config.chunk_steps,
                expected,
                actual,
            )

    @staticmethod
    def _slots(results: Sequence[TrajectoryData]) -> tuple[int, ...]:
        return tuple(slot_id for result in results for slot_id in result.slot_ids)

    @staticmethod
    def _masked_slots(results: Sequence[TrajectoryData], field_name: str) -> set[int]:
        selected: set[int] = set()
        for result in results:
            mask = getattr(result, field_name).reshape(result.batch_size, -1).any(dim=1)
            selected.update(
                slot_id
                for slot_id, include in zip(result.slot_ids, mask.tolist(), strict=True)
                if include
            )
        return selected

    def _grid(self, role: str, discriminator: str) -> list[list[list[TrajectoryData]]]:
        return [
            [
                self._records[(role, epoch, step, discriminator)]
                for step in range(self.config.chunk_steps)
            ]
            for epoch in range(self.config.rollout_epochs)
        ]

    def _assemble_grid(
        self,
        grid: list[list[list[TrajectoryData]]],
        field_name: str,
    ) -> Any:
        merged = [
            [self._merge_field(results, field_name) for results in epoch]
            for epoch in grid
        ]
        return _stack_grid(merged)

    def _merge_field(self, results: Sequence[TrajectoryData], field_name: str) -> Any:
        return self._merge_values(
            [(result.slot_ids, getattr(result, field_name)) for result in results]
        )

    def _merge_values(self, parts: Sequence[tuple[tuple[int, ...], Any]]) -> Any:
        first = parts[0][1]
        if isinstance(first, torch.Tensor):
            output = first.new_empty((len(self.config.slot_ids), *first.shape[1:]))
            for slot_ids, value in parts:
                indices = torch.tensor(
                    [self._slot_indices[slot_id] for slot_id in slot_ids],
                    device=value.device,
                )
                output.index_copy_(0, indices, value)
            return output
        if isinstance(first, Mapping):
            return {
                key: self._merge_values(
                    [(slot_ids, value[key]) for slot_ids, value in parts]
                )
                for key in first
            }
        if isinstance(first, (list, tuple)):
            output: list[Any] = [None] * len(self.config.slot_ids)
            for slot_ids, value in parts:
                for slot_id, item in zip(slot_ids, value, strict=True):
                    output[self._slot_indices[slot_id]] = item
            return type(first)(output) if isinstance(first, tuple) else output
        raise TypeError(
            f"Cannot assemble batched value of type {type(first).__name__}."
        )

    def _assemble_forward_inputs(
        self, grid: list[list[list[TrajectoryData]]]
    ) -> ForwardInputs:
        first = grid[0][0][0].forward_inputs
        assert isinstance(first, ForwardInputs)
        fields: dict[str, torch.Tensor] = {}
        for name, _ in first.tensor_fields():
            merged = [
                [
                    self._merge_values(
                        [
                            (
                                result.slot_ids,
                                dict(result.forward_inputs.tensor_fields())[name],
                            )
                            for result in results
                        ]
                    )
                    for results in epoch
                ]
                for epoch in grid
            ]
            stacked = _stack_grid(merged)
            fields[name] = stacked.flatten(0, 2)
        return type(first).from_model_inputs(fields)

    def _assemble_rewards(self, env_rewards: torch.Tensor) -> dict[str, Any]:
        if self.config.reward_mode is None:
            return {
                "external_rewards": None,
                "reward_mask": None,
                "reward_history_lengths": None,
            }

        first_results = self._records[
            ("reward", 0, self.config.reward_steps[0], self.config.reward_mode)
        ]
        first_rewards = self._merge_field(first_results, "rewards")
        rewards = first_rewards.new_zeros(
            (
                self.config.rollout_epochs,
                self.config.chunk_steps,
                len(self.config.slot_ids),
                *first_rewards.shape[1:],
            )
        )
        mask = torch.zeros(
            env_rewards.shape[:3], dtype=torch.bool, device=env_rewards.device
        )
        history_lengths = None
        if self.config.reward_mode == "history_buffer":
            history_lengths = torch.zeros(
                env_rewards.shape[:3], dtype=torch.int64, device=env_rewards.device
            )
        for epoch in range(self.config.rollout_epochs):
            for step in self.config.reward_steps:
                results = self._records[
                    ("reward", epoch, step, self.config.reward_mode)
                ]
                rewards[epoch, step] = self._merge_field(results, "rewards")
                mask[epoch, step] = True
                if history_lengths is not None:
                    history_lengths[epoch, step] = self._merge_field(
                        results, "history_lengths"
                    )
        return {
            "external_rewards": rewards,
            "reward_mask": mask,
            "reward_history_lengths": history_lengths,
        }

    def _assemble_values(self, reference: torch.Tensor) -> dict[str, Any]:
        if not self.config.boundary_values:
            return {
                "timeout_values": None,
                "timeout_mask": None,
                "tail_values": None,
                "tail_mask": None,
                "timeout_versions": None,
                "tail_versions": None,
            }
        value_results = [
            result
            for (role, _, _, _), results in self._records.items()
            if role == "value"
            for result in results
        ]
        value_reference = value_results[0].values if value_results else reference
        shape = (
            self.config.rollout_epochs,
            self.config.chunk_steps,
            len(self.config.slot_ids),
            1,
        )
        timeout_values = torch.zeros(
            shape, dtype=value_reference.dtype, device=value_reference.device
        )
        timeout_mask = torch.zeros(
            shape[:-1], dtype=torch.bool, device=reference.device
        )
        tail_values = torch.zeros(
            (self.config.rollout_epochs, len(self.config.slot_ids), 1),
            dtype=value_reference.dtype,
            device=value_reference.device,
        )
        tail_mask = torch.zeros(
            tail_values.shape[:-1], dtype=torch.bool, device=reference.device
        )
        has_versions = "versions" in self.config.value_fields
        timeout_versions = None
        tail_versions = None
        if has_versions and value_results:
            version_reference = value_results[0].versions
            assert version_reference is not None
            version_suffix = version_reference.shape[1:]
            timeout_versions = version_reference.new_zeros(
                (*shape[:-1], *version_suffix)
            )
            tail_versions = version_reference.new_zeros(
                (
                    self.config.rollout_epochs,
                    len(self.config.slot_ids),
                    *version_suffix,
                )
            )

        for epoch in range(self.config.rollout_epochs):
            for step in range(self.config.chunk_steps):
                results = self._records.get(("value", epoch, step, "timeout"), ())
                self._place_sparse(timeout_values[epoch, step], results, "values")
                self._mark_sparse(timeout_mask[epoch, step], results)
                if timeout_versions is not None:
                    self._place_sparse(
                        timeout_versions[epoch, step], results, "versions"
                    )
            results = self._records.get(
                ("value", epoch, self.config.chunk_steps, "tail"), ()
            )
            self._place_sparse(tail_values[epoch], results, "values")
            self._mark_sparse(tail_mask[epoch], results)
            if tail_versions is not None:
                self._place_sparse(tail_versions[epoch], results, "versions")

        return {
            "timeout_values": timeout_values,
            "timeout_mask": timeout_mask,
            "tail_values": tail_values,
            "tail_mask": tail_mask,
            "timeout_versions": timeout_versions,
            "tail_versions": tail_versions,
        }

    def _place_sparse(
        self,
        output: torch.Tensor,
        results: Sequence[TrajectoryData],
        field_name: str,
    ) -> None:
        for result in results:
            indices = torch.tensor(
                [self._slot_indices[slot_id] for slot_id in result.slot_ids],
                device=output.device,
            )
            output.index_copy_(0, indices, getattr(result, field_name))

    def _mark_sparse(
        self, output: torch.Tensor, results: Sequence[TrajectoryData]
    ) -> None:
        for result in results:
            indices = torch.tensor(
                [self._slot_indices[slot_id] for slot_id in result.slot_ids],
                device=output.device,
            )
            output[indices] = True


def _select_forward_inputs(
    batch: TrajectoryBatch,
    index: torch.Tensor,
) -> ForwardInputs | None:
    inputs = batch.forward_inputs
    if inputs is None:
        return None
    selected = {}
    for name, tensor in inputs.tensor_fields():
        shaped = tensor.unflatten(
            0,
            (batch.rollout_epochs, batch.chunk_steps, len(batch.slot_ids)),
        )
        selected[name] = shaped.index_select(2, index.to(tensor.device)).flatten(0, 2)
    return type(inputs).from_model_inputs(selected)


def _concat_forward_inputs(
    batches: Sequence[TrajectoryBatch],
) -> ForwardInputs | None:
    inputs = [batch.forward_inputs for batch in batches]
    if all(value is None for value in inputs):
        return None
    if any(value is None for value in inputs):
        raise ValueError("trajectory shards disagree on forward_inputs presence")
    typed_inputs = [value for value in inputs if value is not None]
    first = typed_inputs[0]
    if any(type(value) is not type(first) for value in typed_inputs[1:]):
        raise ValueError("trajectory shards use different forward-input schemas")
    names = tuple(name for name, _ in first.tensor_fields())
    if any(
        tuple(name for name, _ in value.tensor_fields()) != names
        for value in typed_inputs[1:]
    ):
        raise ValueError("trajectory shards use different forward-input fields")

    merged = {}
    for name in names:
        tensors = []
        for batch, value in zip(batches, typed_inputs, strict=True):
            tensor = dict(value.tensor_fields())[name]
            tensors.append(
                tensor.unflatten(
                    0,
                    (batch.rollout_epochs, batch.chunk_steps, len(batch.slot_ids)),
                )
            )
        merged[name] = torch.cat(tensors, dim=2).flatten(0, 2)
    return type(first).from_model_inputs(merged)


def _select_axis(value: Any, index: torch.Tensor, axis: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.index_select(axis, index.to(value.device))
    if isinstance(value, Mapping):
        return {key: _select_axis(child, index, axis) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        if axis == 0:
            selected = [value[item] for item in index.tolist()]
            return tuple(selected) if isinstance(value, tuple) else selected
        selected = [_select_axis(child, index, axis - 1) for child in value]
        return tuple(selected) if isinstance(value, tuple) else selected
    raise TypeError(f"cannot select slot axis from {type(value).__name__}")


def _concat_axis(values: Sequence[Any], axis: int) -> Any:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("trajectory shards disagree on optional field presence")
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=axis)
    if isinstance(first, Mapping):
        if any(value.keys() != first.keys() for value in values[1:]):
            raise ValueError("trajectory shards use different mapping fields")
        return {
            key: _concat_axis([value[key] for value in values], axis) for key in first
        }
    if isinstance(first, (list, tuple)):
        if any(type(value) is not type(first) for value in values[1:]):
            raise ValueError("trajectory shards use different sequence types")
        if axis == 0:
            merged = [item for value in values for item in value]
            return tuple(merged) if isinstance(first, tuple) else merged
        if any(len(value) != len(first) for value in values[1:]):
            raise ValueError("trajectory shards use different time dimensions")
        merged = [
            _concat_axis([value[index] for value in values], axis - 1)
            for index in range(len(first))
        ]
        return tuple(merged) if isinstance(first, tuple) else merged
    raise TypeError(f"cannot concatenate slot axis for {type(first).__name__}")


def _stack_grid(grid: list[list[Any]]) -> Any:
    first = grid[0][0]
    if isinstance(first, torch.Tensor):
        return torch.stack([torch.stack(epoch) for epoch in grid])
    if isinstance(first, Mapping):
        return {
            key: _stack_grid([[value[key] for value in epoch] for epoch in grid])
            for key in first
        }
    return grid


def _payload_signature(result: TrajectoryData) -> Any:
    excluded = {
        "global_step",
        "rollout_epoch",
        "chunk_step",
        "slot_ids",
        "kind",
    }
    return tuple(
        (field.name, _value_signature(getattr(result, field.name)))
        for field in fields(result)
        if field.name not in excluded
    )


def _value_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape[1:]), value.dtype, value.device)
    if isinstance(value, ForwardInputs):
        return (
            "forward_inputs",
            type(value),
            tuple(
                (name, _value_signature(tensor))
                for name, tensor in value.tensor_fields()
            ),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (key, _value_signature(value[key])) for key in sorted(value, key=str)
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("batch_sequence", type(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return ("scalar", type(value), value)
    return ("object", type(value))


def _equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, ForwardInputs):
        return all(
            left_name == right_name and torch.equal(left_value, right_value)
            for (left_name, left_value), (right_name, right_value) in zip(
                left.tensor_fields(), right.tensor_fields(), strict=True
            )
        )
    if is_dataclass(left):
        return all(
            _equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right

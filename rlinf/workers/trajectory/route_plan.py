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
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    """One destination shard of an ordered source slot batch."""

    destination_rank: int
    source_indices: tuple[int, ...]
    destination_indices: tuple[int, ...]
    slot_ids: tuple[int, ...]


class RoutePlan:
    """Deterministic global-slot partitions shared by trajectory participants."""

    def __init__(self, total_slots: int, world_sizes: Mapping[str, int]) -> None:
        if not isinstance(total_slots, int) or isinstance(total_slots, bool):
            raise TypeError("total_slots must be an integer.")
        if total_slots < 1:
            raise ValueError("total_slots must be positive.")
        if not world_sizes:
            raise ValueError("world_sizes must not be empty.")

        normalized: dict[str, int] = {}
        for participant, world_size in world_sizes.items():
            if not isinstance(participant, str) or not participant:
                raise ValueError("participant names must be non-empty strings.")
            if (
                not isinstance(world_size, int)
                or isinstance(world_size, bool)
                or world_size < 1
            ):
                raise ValueError(
                    f"world size for {participant!r} must be a positive integer."
                )
            normalized[participant] = world_size

        self.total_slots = total_slots
        self._world_sizes = normalized
        self._ranges = {
            participant: tuple(
                self._balanced_range(total_slots, world_size, rank)
                for rank in range(world_size)
            )
            for participant, world_size in normalized.items()
        }
        self._owners = {
            participant: tuple(
                rank
                for rank, (start, end) in enumerate(ranges)
                for _ in range(start, end)
            )
            for participant, ranges in self._ranges.items()
        }

    @property
    def world_sizes(self) -> dict[str, int]:
        """Return a copy of the configured participant rank counts."""
        return self._world_sizes.copy()

    @staticmethod
    def _balanced_range(total: int, parts: int, rank: int) -> tuple[int, int]:
        size, remainder = divmod(total, parts)
        start = rank * size + min(rank, remainder)
        end = start + size + (rank < remainder)
        return start, end

    def slot_range(self, participant: str, rank: int) -> tuple[int, int]:
        """Return the participant rank's half-open global slot range."""
        self._validate_rank(participant, rank)
        return self._ranges[participant][rank]

    def owner(self, participant: str, slot_id: int) -> int:
        """Return the participant rank that owns a global slot."""
        self._validate_slot(slot_id)
        self._validate_participant(participant)
        return self._owners[participant][slot_id]

    def local_index(self, participant: str, rank: int, slot_id: int) -> int:
        """Convert an owned global slot into a participant-rank local index."""
        self._validate_slot(slot_id)
        start, end = self.slot_range(participant, rank)
        if not start <= slot_id < end:
            raise ValueError(
                f"slot {slot_id} is not owned by {participant!r} rank {rank}."
            )
        return slot_id - start

    def global_slot(self, participant: str, rank: int, local_index: int) -> int:
        """Convert a participant-rank local index into a global slot."""
        start, end = self.slot_range(participant, rank)
        if (
            not isinstance(local_index, int)
            or isinstance(local_index, bool)
            or not 0 <= local_index < end - start
        ):
            raise ValueError(
                f"invalid local index {local_index!r} for {participant!r} rank {rank}."
            )
        return start + local_index

    def routes(
        self,
        source: str,
        source_rank: int,
        destination: str,
    ) -> tuple[Route, ...]:
        """Route a source rank's complete contiguous slot batch."""
        start, end = self.slot_range(source, source_rank)
        return self._route_slots(tuple(range(start, end)), destination)

    def route_slots(
        self,
        source: str,
        source_rank: int,
        slot_ids: Sequence[int],
        destination: str,
    ) -> tuple[Route, ...]:
        """Route an ordered, possibly sparse slot batch to destination ranks."""
        source_start, source_end = self.slot_range(source, source_rank)
        slots = tuple(slot_ids)
        for slot_id in slots:
            self._validate_slot(slot_id)
            if not source_start <= slot_id < source_end:
                raise ValueError(
                    f"slot {slot_id} is not owned by {source!r} rank {source_rank}."
                )
        if len(set(slots)) != len(slots):
            raise ValueError("slot_ids must not contain duplicates.")
        return self._route_slots(slots, destination)

    def _route_slots(
        self,
        slot_ids: tuple[int, ...],
        destination: str,
    ) -> tuple[Route, ...]:
        self._validate_participant(destination)
        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for source_index, slot_id in enumerate(slot_ids):
            destination_rank = self.owner(destination, slot_id)
            grouped.setdefault(destination_rank, []).append(
                (
                    source_index,
                    self.local_index(destination, destination_rank, slot_id),
                    slot_id,
                )
            )

        return tuple(
            Route(
                destination_rank=rank,
                source_indices=tuple(item[0] for item in items),
                destination_indices=tuple(item[1] for item in items),
                slot_ids=tuple(item[2] for item in items),
            )
            for rank, items in sorted(grouped.items())
        )

    def _validate_participant(self, participant: str) -> None:
        if participant not in self._world_sizes:
            raise ValueError(f"unknown route participant {participant!r}.")

    def _validate_rank(self, participant: str, rank: int) -> None:
        self._validate_participant(participant)
        world_size = self._world_sizes[participant]
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < world_size
        ):
            raise ValueError(
                f"invalid rank {rank!r} for {participant!r} world size {world_size}."
            )

    def _validate_slot(self, slot_id: int) -> None:
        if (
            not isinstance(slot_id, int)
            or isinstance(slot_id, bool)
            or not 0 <= slot_id < self.total_slots
        ):
            raise ValueError(
                f"slot_id must be in [0, {self.total_slots}), got {slot_id!r}."
            )

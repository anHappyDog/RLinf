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

"""Map-style OpenPI input dataset for grounded-control pilot sidecars."""

from __future__ import annotations

import collections
import dataclasses
import itertools
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from rlinf.data.b1k_grounded import (
    ControlProfile,
    ControlSerializer,
    GroundedControlSpec,
    ReservedTokenMapping,
)

_REQUIRED_COLUMNS = (
    "sample_id",
    "frame_index",
    "control_json",
    "state",
    "actions",
    "action_is_pad",
    "rgb_head_path",
    "rgb_left_wrist_path",
    "rgb_right_wrist_path",
)
_SAMPLING_COLUMNS = (
    "task_index",
    "episode_index",
    "segment_index",
    "interval_index",
)
_VIDEO_PATH_COLUMNS = (
    "rgb_head_path",
    "rgb_left_wrist_path",
    "rgb_right_wrist_path",
)


@dataclasses.dataclass(frozen=True)
class GroundedEpisodeGroup:
    """Sampler metadata for one episode and its annotated stage instances."""

    task_index: int
    episode_index: int
    stage_groups: tuple[tuple[int, ...], ...]


class TaskEpisodeStageSampler(torch.utils.data.Sampler[int]):
    """Sample tasks and episodes uniformly with softened stage duration.

    Every global optimizer batch has task counts that differ by at most one.
    Inside a task, episodes are uniform and stage probability is proportional
    to ``num_stage_samples ** stage_exponent``. Frames are uniform inside the
    selected stage.

    Rank-local task runs share one sampled episode. This preserves the target
    marginal distribution while avoiding a video reopen for every sample.
    """

    def __init__(
        self,
        episode_groups: tuple[GroundedEpisodeGroup, ...],
        *,
        num_replicas: int,
        rank: int,
        global_batch_size: int,
        samples_per_epoch: int,
        stage_exponent: float,
        seed: int,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas}).")
        if global_batch_size < num_replicas:
            raise ValueError("global_batch_size must be at least num_replicas.")
        if global_batch_size % num_replicas:
            raise ValueError("global_batch_size must be divisible by num_replicas.")
        if samples_per_epoch <= 0 or samples_per_epoch % global_batch_size:
            raise ValueError(
                "samples_per_epoch must be positive and divisible by global_batch_size."
            )
        if not 0.0 <= stage_exponent <= 1.0:
            raise ValueError("stage_exponent must be in [0, 1].")
        if not episode_groups:
            raise ValueError("episode_groups must not be empty.")

        episodes_by_task: dict[int, list[GroundedEpisodeGroup]] = (
            collections.defaultdict(list)
        )
        for episode in episode_groups:
            if not episode.stage_groups or any(
                not stage for stage in episode.stage_groups
            ):
                raise ValueError("Every episode and stage group must be non-empty.")
            episodes_by_task[episode.task_index].append(episode)

        self._episodes_by_task = {
            task_index: tuple(sorted(episodes, key=lambda value: value.episode_index))
            for task_index, episodes in sorted(episodes_by_task.items())
        }
        self._task_indices = tuple(self._episodes_by_task)
        if global_batch_size < len(self._task_indices):
            raise ValueError(
                "global_batch_size must be at least the number of tasks so every "
                "task is represented in each global batch."
            )

        self._num_replicas = num_replicas
        self._rank = rank
        self._global_batch_size = global_batch_size
        self._samples_per_epoch = samples_per_epoch
        self._stage_exponent = stage_exponent
        self._seed = seed
        self._epoch = 0
        self._start_global_batch = 0
        self._local_batch_size = global_batch_size // num_replicas
        self._num_global_batches = samples_per_epoch // global_batch_size

    def __iter__(self):
        layout_rng = random.Random(self._seed + self._epoch)
        sample_rng = random.Random(
            self._seed + self._epoch * self._num_replicas + self._rank
        )
        indices: list[int] = []
        global_batch_offset = self._epoch * self._num_global_batches

        for batch_index in range(self._num_global_batches):
            task_sequence = self._task_sequence(
                global_batch_offset + batch_index, layout_rng
            )
            local_start = self._rank * self._local_batch_size
            local_tasks = task_sequence[
                local_start : local_start + self._local_batch_size
            ]
            batch_indices: list[int] = []
            for task_index, task_run in itertools.groupby(local_tasks):
                run_length = sum(1 for _ in task_run)
                episode = sample_rng.choice(self._episodes_by_task[task_index])
                stage_weights = [
                    math.pow(len(stage), self._stage_exponent)
                    for stage in episode.stage_groups
                ]
                for _ in range(run_length):
                    stage = sample_rng.choices(
                        episode.stage_groups, weights=stage_weights, k=1
                    )[0]
                    batch_indices.append(sample_rng.choice(stage))

            if batch_index >= self._start_global_batch:
                indices.extend(batch_indices)

        return iter(indices)

    def __len__(self) -> int:
        return self._samples_per_epoch // self._num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic sampling epoch."""
        self._epoch = epoch

    @property
    def num_global_batches(self) -> int:
        """Return the number of optimizer batches in one sampling epoch."""
        return self._num_global_batches

    def set_start_global_batch(self, batch_index: int) -> None:
        """Skip completed global batches while preserving the RNG stream."""
        if not 0 <= batch_index < self._num_global_batches:
            raise ValueError(
                f"batch_index must be in [0, {self._num_global_batches}); "
                f"got {batch_index}."
            )
        self._start_global_batch = batch_index

    def _task_sequence(self, global_batch_index: int, rng: random.Random) -> list[int]:
        task_count = len(self._task_indices)
        base_count, remainder = divmod(self._global_batch_size, task_count)
        extra_start = global_batch_index % task_count
        task_counts = {
            task_index: base_count
            + int((offset - extra_start) % task_count < remainder)
            for offset, task_index in enumerate(self._task_indices)
        }
        task_order = list(self._task_indices)
        rng.shuffle(task_order)
        return [
            task_index
            for task_index in task_order
            for _ in range(task_counts[task_index])
        ]


class EpisodeShardedSampler(torch.utils.data.Sampler[int]):
    """Assign episodes to ranks and preserve frame order inside each episode.

    The smaller rank partitions repeat a few local samples so every distributed
    worker executes the same number of steps without splitting an episode across
    ranks. Shuffling episode order each epoch rotates which samples are repeated.
    """

    def __init__(
        self,
        episode_groups: tuple[tuple[int, ...], ...],
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas}).")
        if not episode_groups or any(not group for group in episode_groups):
            raise ValueError("episode_groups must contain non-empty groups.")
        if len(episode_groups) < num_replicas:
            raise ValueError(
                f"Cannot shard {len(episode_groups)} episodes across "
                f"{num_replicas} ranks."
            )

        groups_by_rank: list[list[tuple[int, ...]]] = [[] for _ in range(num_replicas)]
        sample_counts = [0] * num_replicas
        for group in sorted(episode_groups, key=lambda value: (-len(value), value[0])):
            target_rank = min(
                range(num_replicas), key=lambda value: (sample_counts[value], value)
            )
            groups_by_rank[target_rank].append(group)
            sample_counts[target_rank] += len(group)

        self._groups = tuple(tuple(groups) for groups in groups_by_rank)
        self._rank = rank
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._num_samples = max(sample_counts)

    def __iter__(self):
        groups = list(self._groups[self._rank])
        if self._shuffle:
            random.Random(self._seed + self._epoch).shuffle(groups)
        indices = [index for group in groups for index in group]
        padding = self._num_samples - len(indices)
        if padding:
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        return iter(indices)

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic episode shuffle epoch."""
        self._epoch = epoch


class GroundedBehaviorSftDataset(torch.utils.data.Dataset):
    """Read a grounded sidecar while leaving RGB in the original B1K videos.

    Rows expose random access, while :class:`EpisodeShardedSampler` keeps each
    rank on disjoint episodes and orders their frames so the three OpenCV decoders
    can advance rather than seek for every sample.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        sidecar_path: str | Path,
        token_mapping: ReservedTokenMapping,
        profile: ControlProfile,
    ) -> None:
        if not isinstance(profile, ControlProfile):
            raise TypeError("profile must be a ControlProfile.")
        self._dataset_root = Path(dataset_root).resolve()
        self._sidecar_path = Path(sidecar_path).resolve()
        schema = pq.read_schema(self._sidecar_path)
        columns = _REQUIRED_COLUMNS + _SAMPLING_COLUMNS
        missing = sorted(set(columns).difference(schema.names))
        if missing:
            raise ValueError(f"Grounded sidecar is missing columns: {missing}.")
        self._table = pq.read_table(self._sidecar_path, columns=list(columns))
        if self._table.num_rows == 0:
            raise ValueError("Grounded sidecar contains no samples.")
        self._serializer = ControlSerializer(token_mapping)
        self._profile = profile
        self._captures: dict[Path, Any] = {}
        self._capture_positions: dict[Path, int] = {}
        self._active_episode: tuple[str, str, str] | None = None
        grouped_indices: dict[tuple[str, str, str], list[int]] = (
            collections.defaultdict(list)
        )
        for index in range(self._table.num_rows):
            grouped_indices[self._episode_key(index)].append(index)
        self._episode_groups = tuple(
            tuple(
                sorted(
                    indices,
                    key=lambda index: self._table["frame_index"][index].as_py(),
                )
            )
            for _, indices in sorted(grouped_indices.items())
        )
        grouped_stages: dict[tuple[int, int], dict[tuple[int, int], list[int]]] = (
            collections.defaultdict(lambda: collections.defaultdict(list))
        )
        for index in range(self._table.num_rows):
            episode_key = (
                self._table["task_index"][index].as_py(),
                self._table["episode_index"][index].as_py(),
            )
            stage_key = (
                self._table["segment_index"][index].as_py(),
                self._table["interval_index"][index].as_py(),
            )
            grouped_stages[episode_key][stage_key].append(index)
        self._sampling_episode_groups = tuple(
            GroundedEpisodeGroup(
                task_index=episode_key[0],
                episode_index=episode_key[1],
                stage_groups=tuple(
                    tuple(indices)
                    for _, indices in sorted(grouped_stages[episode_key].items())
                ),
            )
            for episode_key in sorted(grouped_stages)
        )

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = {name: self._table[name][index].as_py() for name in _REQUIRED_COLUMNS}
        self._activate_episode(tuple(row[name] for name in _VIDEO_PATH_COLUMNS))
        control = GroundedControlSpec.from_json(row["control_json"])
        if control.timestep != row["frame_index"]:
            raise ValueError(
                f"Sidecar sample {row['sample_id']} has frame/control timestep "
                f"mismatch: {row['frame_index']} != {control.timestep}."
            )
        frame_index = row["frame_index"]
        return {
            "sample_id": row["sample_id"],
            "observation.images.rgb.head": self._read_rgb_frame(
                self._source_path(row["rgb_head_path"]), frame_index
            ),
            "observation.images.rgb.left_wrist": self._read_rgb_frame(
                self._source_path(row["rgb_left_wrist_path"]), frame_index
            ),
            "observation.images.rgb.right_wrist": self._read_rgb_frame(
                self._source_path(row["rgb_right_wrist_path"]), frame_index
            ),
            "observation.state": np.asarray(row["state"], dtype=np.float32),
            "action": np.asarray(row["actions"], dtype=np.float32),
            "action_is_pad": np.asarray(row["action_is_pad"], dtype=np.bool_),
            "prompt": self._serializer.serialize(control, self._profile),
        }

    def close(self) -> None:
        """Release video handles owned by this dataset instance."""
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._capture_positions.clear()
        self._active_episode = None

    @property
    def episode_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return row indices grouped by episode and ordered by frame."""
        return self._episode_groups

    @property
    def sampling_episode_groups(self) -> tuple[GroundedEpisodeGroup, ...]:
        """Return the task, episode, and stage hierarchy used for sampling."""
        return self._sampling_episode_groups

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle process-local OpenCV handles into DataLoader workers."""
        state = self.__dict__.copy()
        state["_captures"] = {}
        state["_capture_positions"] = {}
        state["_active_episode"] = None
        return state

    def _episode_key(self, index: int) -> tuple[str, str, str]:
        return tuple(self._table[name][index].as_py() for name in _VIDEO_PATH_COLUMNS)

    def _activate_episode(self, episode: tuple[str, str, str]) -> None:
        if episode != self._active_episode:
            self.close()
            self._active_episode = episode

    def _source_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(f"Sidecar source path must be relative: {path}.")
        resolved = (self._dataset_root / path).resolve()
        try:
            resolved.relative_to(self._dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Sidecar source path escapes the dataset root: {path}."
            ) from error
        return resolved

    def _read_rgb_frame(self, path: Path, frame_index: int) -> np.ndarray:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Grounded sidecar RGB decoding requires OpenCV."
            ) from error

        capture = self._captures.get(path)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise OSError(f"Could not open RGB video {path}.")
            self._captures[path] = capture
            self._capture_positions[path] = 0

        current_position = self._capture_positions[path]
        gap = frame_index - current_position
        if gap < 0 or gap > 64:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            for _ in range(gap):
                if not capture.grab():
                    raise IndexError(
                        f"Could not advance to frame {frame_index} in {path}."
                    )
        success, frame = capture.read()
        if not success:
            raise IndexError(f"Could not read frame {frame_index} from {path}.")
        self._capture_positions[path] = frame_index + 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

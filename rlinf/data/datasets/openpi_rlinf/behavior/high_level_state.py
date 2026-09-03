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

"""Current-state loading and train-only normalization for B1K high-level text."""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from rlinf.models.embodiment.openpi.policies.behavior_policy import (
    extract_state_from_proprio,
)


class ManifestFrame(Protocol):
    """Manifest fields required to load one B1K state."""

    episode_index: int
    frame_index: int


@dataclasses.dataclass(frozen=True)
class StateNormStats:
    """Per-dimension statistics for OpenPI-compatible quantile normalization."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray
    sample_count: int

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Apply the quantile normalization used by the π0.5 Behavior config."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape[-1] != self.q01.shape[-1]:
            raise ValueError(
                f"Expected state dim {self.q01.shape[-1]}, got {state.shape[-1]}."
            )
        return (state - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "state_dim": int(self.q01.shape[-1]),
            "sample_count": self.sample_count,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> StateNormStats:
        """Construct statistics from their JSON representation."""
        stats = cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            q01=np.asarray(data["q01"], dtype=np.float32),
            q99=np.asarray(data["q99"], dtype=np.float32),
            sample_count=int(data["sample_count"]),
        )
        if int(data["state_dim"]) != stats.q01.shape[-1]:
            raise ValueError("state_dim does not match the stored quantile vectors.")
        return stats


class BehaviorStateReader:
    """Read exact-frame 23D policy states from episode parquet files."""

    def __init__(self, dataset_root: str | Path, *, episode_cache_size: int = 32):
        if episode_cache_size <= 0:
            raise ValueError("episode_cache_size must be positive.")
        self.dataset_root = Path(dataset_root)
        info = json.loads((self.dataset_root / "meta" / "info.json").read_text())
        self._data_path_template = str(info["data_path"])
        self._chunk_size = int(info["chunks_size"])
        self._episode_cache_size = episode_cache_size
        self._episode_states: OrderedDict[int, np.ndarray] = OrderedDict()

    def read(self, episode_index: int, frame_index: int) -> np.ndarray:
        """Return the extracted 23D state for one zero-based episode frame."""
        if episode_index in self._episode_states:
            states = self._episode_states.pop(episode_index)
            self._episode_states[episode_index] = states
        else:
            states = self._load_episode(episode_index)
            self._episode_states[episode_index] = states
            if len(self._episode_states) > self._episode_cache_size:
                self._episode_states.popitem(last=False)
        if frame_index < 0 or frame_index >= len(states):
            raise IndexError(
                f"Frame {frame_index} is outside episode {episode_index} "
                f"with {len(states)} states."
            )
        return states[frame_index]

    def _load_episode(self, episode_index: int) -> np.ndarray:
        import pyarrow.parquet as pq

        relative_path = self._data_path_template.format(
            episode_chunk=episode_index // self._chunk_size,
            episode_index=episode_index,
        )
        parquet_path = self.dataset_root / relative_path
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        column = (
            pq.read_table(parquet_path, columns=["observation.state"])
            .column(0)
            .combine_chunks()
        )
        if column.null_count:
            raise ValueError(f"State column contains null rows: {parquet_path}")
        row_lengths = np.diff(column.offsets.to_numpy(zero_copy_only=False))
        if not np.all(row_lengths == 256):
            raise ValueError(f"Expected 256D proprio rows in {parquet_path}.")
        proprio = column.values.to_numpy(zero_copy_only=False).reshape(len(column), 256)
        return np.asarray(extract_state_from_proprio(proprio), dtype=np.float32)


def compute_state_norm_stats(
    reader: BehaviorStateReader,
    entries: Iterable[ManifestFrame],
) -> StateNormStats:
    """Compute state statistics from training-manifest frames only."""
    states = np.stack(
        [reader.read(entry.episode_index, entry.frame_index) for entry in entries]
    )
    if len(states) < 2:
        raise ValueError("At least two training states are required for statistics.")
    return StateNormStats(
        mean=states.mean(axis=0),
        std=states.std(axis=0),
        q01=np.quantile(states, 0.01, axis=0).astype(np.float32),
        q99=np.quantile(states, 0.99, axis=0).astype(np.float32),
        sample_count=len(states),
    )


def write_state_norm_stats(stats: StateNormStats, output_path: str | Path) -> None:
    """Write state normalization statistics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats.to_json_dict(), indent=2) + "\n", encoding="utf-8"
    )


def read_state_norm_stats(stats_path: str | Path) -> StateNormStats:
    """Read state normalization statistics from JSON."""
    return StateNormStats.from_json_dict(json.loads(Path(stats_path).read_text()))

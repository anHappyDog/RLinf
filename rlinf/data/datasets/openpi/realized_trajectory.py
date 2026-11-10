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

"""Future measured-joint labels on an explicit control-time grid."""

from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np

RECIPE = {
    "name": "realized_joint_trajectory",
    "version": 1,
    "target_hz": 15.0,
    "horizon": 16,
    "joint_state_indices": list(range(7)),
    "timestamp_feature": "source.timestamp",
    "timestamp_unit": "seconds",
    "target_offsets": "(k+1)/target_hz",
    "joint_interpolation": "linear measured joint position; no extrapolation",
    "gripper_interpolation": "zero_order_hold; no extrapolation",
}


class RealizedTrajectoryCache:
    """Cache episode-local measured trajectories independently from packed files."""

    def __init__(self, numeric_cache, max_episodes: int = 64) -> None:
        self.numeric_cache = numeric_cache
        self.max_episodes = max_episodes
        self._episodes: OrderedDict = OrderedDict()

    def read(self, row: dict) -> dict:
        """Read the source observation clock and declared gripper signals."""
        recipe = row["target_recipe"]
        key = (
            row["data_file"],
            row["row_offset"],
            row["frames"],
            json.dumps(recipe, sort_keys=True),
        )
        if key in self._episodes:
            self._episodes.move_to_end(key)
            return self._episodes[key]
        mask_feature = row.get("supervision_mask_feature")
        columns = {"observation.state", recipe["timestamp_feature"]}
        if mask_feature:
            columns.add(mask_feature)
        for field in ("gripper_target", "gripper_state"):
            signal = recipe.get(field)
            if signal is not None:
                columns.add(signal["feature"])
        arrays = self.numeric_cache.read_columns(row, sorted(columns))
        time = arrays[recipe["timestamp_feature"]].reshape(-1).astype(np.float64)
        if len(time) != row["frames"]:
            raise ValueError(
                f"Trajectory exceeds its Parquet file: {row['catalog_id']}"
            )
        finite_time = np.isfinite(time)
        if np.any(np.diff(time[finite_time]) < 0):
            raise ValueError(f"Observation timestamps decrease: {row['catalog_id']}")
        supervised = np.zeros(len(time), dtype=bool)
        lo, hi = row["core_range"]
        supervised[lo:hi] = True
        if mask_feature:
            supervised &= arrays[mask_feature].reshape(-1).astype(bool)
        q = arrays["observation.state"][:, recipe["joint_state_indices"]].astype(
            np.float64
        )
        result = {"time": time, "q": q, "supervised": supervised}
        for field in ("gripper_target", "gripper_state"):
            signal = recipe.get(field)
            if signal is None:
                result[field] = np.full(len(time), np.nan)
                continue
            values = arrays[signal["feature"]][:, signal["index"]].astype(np.float64)
            values = (values - signal["open"]) / (signal["closed"] - signal["open"])
            if signal.get("clip", False):
                values = np.clip(values, 0, 1)
            result[field] = values
        self._episodes[key] = result
        if len(self._episodes) > self.max_episodes:
            self._episodes.popitem(last=False)
        return result

    def valid_anchors(self, row: dict) -> np.ndarray:
        """Exclude anchors whose complete future grid has no supervised dimension."""
        data = self.read(row)
        recipe = row["target_recipe"]
        time, q, supervised = data["time"], data["q"], data["supervised"]
        result = np.zeros(len(time), dtype=bool)
        available = np.flatnonzero(np.isfinite(time))
        if not len(available):
            return result
        clock = time[available]
        targets = (
            time[:, None] + (np.arange(recipe["horizon"]) + 1) / recipe["target_hz"]
        )
        upper = np.searchsorted(clock, targets, side="left")
        upper_safe = np.minimum(upper, len(clock) - 1)
        exact = (upper < len(clock)) & np.isclose(
            clock[upper_safe], targets, rtol=0, atol=1e-9
        )
        lower = np.where(exact, upper_safe, np.maximum(upper - 1, 0))
        lo, hi = available[lower], available[upper_safe]
        valid = (
            (targets >= clock[0])
            & (targets <= clock[-1])
            & ((hi == lo) | (hi == lo + 1))
            & supervised[lo]
            & supervised[hi]
            & np.any(np.isfinite(q[lo]) & np.isfinite(q[hi]), axis=-1)
        )
        result |= valid.any(axis=1)
        signal = recipe.get("gripper_target")
        if signal is not None:
            grip_clock = clock + signal.get("time_offset_s", 0.0)
            index = np.searchsorted(grip_clock, targets, side="right") - 1
            selected = available[np.clip(index, 0, len(clock) - 1)]
            valid = (
                (targets >= grip_clock[0])
                & (targets <= grip_clock[-1])
                & supervised[selected]
                & np.isfinite(data["gripper_target"][selected])
            )
            result |= valid.any(axis=1)
        return result & supervised & np.isfinite(time) & np.isfinite(q).all(axis=1)

    def sample(self, row: dict, frame: int) -> dict:
        """Return raw state8, future targets, dimension masks, and source times."""
        data = self.read(row)
        recipe = row["target_recipe"]
        time, q, supervised = data["time"], data["q"], data["supervised"]
        anchor_time = time[frame]
        targets = anchor_time + (np.arange(recipe["horizon"]) + 1) / recipe["target_hz"]
        actions = np.zeros((recipe["horizon"], 8), dtype=np.float64)
        mask = np.zeros_like(actions, dtype=bool)
        available = np.flatnonzero(np.isfinite(time))
        if len(available) and np.isfinite(anchor_time):
            clock = time[available]
            upper = np.searchsorted(clock, targets, side="left")
            upper_safe = np.minimum(upper, len(clock) - 1)
            exact = (upper < len(clock)) & np.isclose(
                clock[upper_safe], targets, rtol=0, atol=1e-9
            )
            lower = np.where(exact, upper_safe, np.maximum(upper - 1, 0))
            upper = upper_safe
            lo, hi = available[lower], available[upper]
            in_bounds = (targets >= clock[0]) & (targets <= clock[-1])
            adjacent = (hi == lo) | (hi == lo + 1)
            dt = time[hi] - time[lo]
            alpha = np.divide(
                targets - time[lo], dt, out=np.zeros_like(targets), where=dt > 0
            )
            values = q[lo] + alpha[:, None] * (q[hi] - q[lo])
            valid = in_bounds & adjacent & supervised[lo] & supervised[hi]
            mask[:, :7] = valid[:, None] & np.isfinite(q[lo]) & np.isfinite(q[hi])
            actions[:, :7] = np.where(mask[:, :7], values, 0)
            signal = recipe.get("gripper_target")
            if signal is not None:
                grip_clock = clock + signal.get("time_offset_s", 0.0)
                grip_index = np.searchsorted(grip_clock, targets, side="right") - 1
                grip_safe = available[np.clip(grip_index, 0, len(clock) - 1)]
                grip = data["gripper_target"][grip_safe]
                mask[:, 7] = (
                    (targets >= grip_clock[0])
                    & (targets <= grip_clock[-1])
                    & supervised[grip_safe]
                    & np.isfinite(grip)
                )
                actions[:, 7] = np.where(mask[:, 7], grip, 0)
        state = np.r_[q[frame], data["gripper_state"][frame]]
        state_mask = np.isfinite(state)
        state = np.where(state_mask, state, 0)
        return {
            "state": state,
            "state_mask": state_mask,
            "actions": actions,
            "action_mask": mask,
            "anchor_timestamp": float(anchor_time),
            "target_timestamps": targets,
            "time_feature": recipe["timestamp_feature"],
        }


class MaskedMoments:
    """Merge finite, supervised dimensions without treating missing grippers as zero."""

    def __init__(self, dim: int = 8) -> None:
        self.count = np.zeros(dim, dtype=np.int64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def add(self, values: np.ndarray, mask: np.ndarray) -> None:
        for dim in range(values.shape[-1]):
            selected = values[..., dim][mask[..., dim]]
            if not len(selected):
                continue
            count = len(selected)
            mean = selected.mean()
            m2 = np.square(selected - mean).sum()
            total = self.count[dim] + count
            delta = mean - self.mean[dim]
            self.m2[dim] += m2 + delta * delta * self.count[dim] * count / total
            self.mean[dim] += delta * count / total
            self.count[dim] = total

    def result(self) -> dict:
        variance = np.divide(
            self.m2, self.count, out=np.zeros_like(self.m2), where=self.count > 0
        )
        std = np.sqrt(variance)
        std[std == 0] = 1
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "transform": "mean_std",
        }


class RealizedTrajectorySampler:
    """Seeded source -> original episode -> anchor draws with rank-stride sharding."""

    def __init__(
        self,
        dataset,
        *,
        samples_per_epoch: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        import math

        if not 0 <= rank < world_size or batch_size < 1 or samples_per_epoch < 1:
            raise ValueError(
                "Expected valid rank/world_size, positive batch size and sample budget"
            )
        self.dataset, self.rank, self.world_size = dataset, rank, world_size
        self.seed, self.shuffle, self.epoch, self.position = seed, shuffle, 0, 0
        self.num_samples = (
            math.ceil(samples_per_epoch / (world_size * batch_size)) * batch_size
            if shuffle
            else len(range(rank, len(dataset), world_size))
        )
        from rlinf.data.datasets.openpi.manifest_index import ManifestIndex

        if isinstance(dataset.episodes, ManifestIndex):
            self.sources = dataset.episodes.metadata["sources"]
            self.groups = dataset.episodes.sampler_groups()
        else:
            grouped = {}
            start = 0
            for row in dataset.episodes:
                count = len(row["anchor_indices"])
                grouped.setdefault(row["schema_id"], {}).setdefault(
                    row["source_group_id"], []
                ).append((start, count))
                start += count
            self.sources = sorted(grouped)
            self.groups = []
            for source in self.sources:
                group_offsets, clip_starts, clip_ends = [0], [], []
                for clips in grouped[source].values():
                    total = 0
                    for start, count in clips:
                        total += count
                        clip_starts.append(start)
                        clip_ends.append(total)
                    group_offsets.append(len(clip_starts))
                self.groups.append(
                    tuple(
                        np.asarray(values, dtype=np.int64)
                        for values in (group_offsets, clip_starts, clip_ends)
                    )
                )
        weights = np.array(
            [dataset.schemas[name]["weight"] for name in self.sources], dtype=np.float64
        )
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Source weights must be finite and positive")
        self.source_cdf = np.cumsum(weights / weights.sum())
        self.source_cdf[-1] = 1.0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch, self.position = epoch, 0

    def __iter__(self):
        import torch

        if not self.shuffle:
            for position in range(self.position, self.num_samples):
                self.position = position + 1
                yield self.rank + position * self.world_size
            return
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.num_samples * self.world_size
        for block_start in range(0, total, 4096):
            draws = torch.rand(
                (min(4096, total - block_start), 3),
                generator=generator,
                dtype=torch.float64,
            ).numpy()
            local_start = (self.rank - block_start) % self.world_size
            for offset in range(local_start, len(draws), self.world_size):
                position = (block_start + offset - self.rank) // self.world_size
                if position < self.position:
                    continue
                source_u, episode_u, anchor_u = draws[offset]
                source = int(np.searchsorted(self.source_cdf, source_u, side="right"))
                offsets, starts, ends = self.groups[source]
                group = min(int(episode_u * (len(offsets) - 1)), len(offsets) - 2)
                begin, end = offsets[group], offsets[group + 1]
                anchor = min(int(anchor_u * ends[end - 1]), int(ends[end - 1]) - 1)
                clip = int(
                    begin + np.searchsorted(ends[begin:end], anchor, side="right")
                )
                index = int(
                    starts[clip] + anchor - (ends[clip - 1] if clip > begin else 0)
                )
                self.position = position + 1
                yield index

    def state_dict(self) -> dict:
        return {
            "algorithm": "realized_source_episode_anchor_v1",
            "epoch": self.epoch,
            "position": self.position,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "world_size": self.world_size,
            "rank": self.rank,
            "num_samples": self.num_samples,
            "manifest_fingerprint": self.dataset.manifest_fingerprint,
        }

    def migrate_world_size(self, states: list[dict]) -> dict:
        """Resume a fully consumed global prefix with a new rank-stride topology."""
        old_world_size = len(states)
        if not states or old_world_size == self.world_size:
            raise ValueError(
                "Sampler migration requires a different nonempty rank topology"
            )
        reference = states[0]
        expected = self.state_dict()
        for rank, state in enumerate(states):
            for key in ("algorithm", "seed", "shuffle", "manifest_fingerprint"):
                if state.get(key) != expected[key]:
                    raise ValueError(
                        f"Realized sampler migration contract differs: {key}"
                    )
            if (
                state["rank"] != rank
                or state["world_size"] != old_world_size
                or state["epoch"] != reference["epoch"]
                or state["num_samples"] != reference["num_samples"]
                or state["position"] != reference["position"]
                or not 0 <= state["position"] <= state["num_samples"]
            ):
                raise ValueError(
                    "Sampler migration requires aligned checkpoint rank prefixes"
                )
        budget = reference["num_samples"] * old_world_size
        consumed = reference["position"] * old_world_size
        if (
            budget != self.num_samples * self.world_size
            or consumed % self.world_size
            or not self.shuffle
        ):
            raise ValueError(
                "Sampler migration requires the same global budget and a divisible consumed prefix"
            )
        self.epoch = reference["epoch"]
        self.position = consumed // self.world_size
        return {
            "old_world_size": old_world_size,
            "new_world_size": self.world_size,
            "epoch": self.epoch,
            "global_samples_consumed_in_epoch": consumed,
            "global_samples_per_epoch": budget,
            "old_rank_position": reference["position"],
            "new_rank_position": self.position,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        for key in expected:
            if key not in {"epoch", "position"} and state.get(key) != expected[key]:
                raise ValueError(f"Realized sampler resume contract differs: {key}")
        if not 0 <= state["position"] <= self.num_samples:
            raise ValueError("Realized sampler consumed position exceeds epoch budget")
        self.epoch, self.position = state["epoch"], state["position"]

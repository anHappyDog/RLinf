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

"""Small, deterministic MolmoBot LeRobot v3 SFT subsets with native splits."""

import json
from pathlib import Path

import numpy as np
import torch


class MolmoBotSubset:
    """Load bounded episode prefixes from an existing MolmoBot v3 conversion.

    Chunks never cross episode boundaries. The conversion already pairs obs[i]
    with commanded_action[i+1], so no additional action shift is applied.
    """

    def __init__(
        self,
        root: str | Path,
        horizon: int,
        max_episodes: int,
        chunks_per_episode: int,
        cameras: list[str],
        include_state: bool = False,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.cameras = cameras
        self.horizon = horizon
        self.include_state = include_state
        for camera in cameras:
            if self.info["features"][camera]["dtype"] != "video":
                raise ValueError(f"Expected video feature, found {camera}")
        metadata_columns = [
            "episode_index",
            "length",
            "tasks",
            "data/chunk_index",
            "data/file_index",
            "pipeline/provenance",
        ]
        for camera in cameras:
            metadata_columns += [
                f"videos/{camera}/chunk_index",
                f"videos/{camera}/file_index",
                f"videos/{camera}/from_timestamp",
            ]
        episodes = []
        for path in sorted((self.root / "meta/episodes").rglob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=max_episodes, columns=metadata_columns
            ):
                episodes.extend(batch.to_pylist()[: max_episodes - len(episodes)])
                if len(episodes) == max_episodes:
                    break
            if len(episodes) == max_episodes:
                break
        if not episodes:
            raise ValueError(f"Expected episodes in {self.root}")
        self.samples = []
        self.manifest = []
        loaded_path = None
        table = None
        episode_slices = {}
        for episode in episodes:
            path = self.root / self.info["data_path"].format(
                chunk_index=episode["data/chunk_index"],
                file_index=episode["data/file_index"],
            )
            if path != loaded_path:
                columns = [
                    "episode_index",
                    "action",
                    "source.extra.pipeline_supervision_mask",
                    "source.observation_index",
                    "source.action_index",
                ]
                if include_state:
                    columns.append("observation.state")
                table = pq.read_table(path, columns=columns)
                loaded_path = path
                episode_slices = {}
                episode_ids = table["episode_index"].to_numpy()
                if len(episode_ids):
                    boundaries = np.r_[
                        0,
                        np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1,
                        len(episode_ids),
                    ]
                    for start, stop in zip(
                        boundaries[:-1], boundaries[1:], strict=True
                    ):
                        episode_slices.setdefault(int(episode_ids[start]), []).append(
                            (int(start), int(stop - start))
                        )
            slices = episode_slices.get(episode["episode_index"], [])
            if len(slices) == 1:
                rows = table.slice(*slices[0])
            elif slices:
                rows = pa.concat_tables([table.slice(*part) for part in slices])
            else:
                rows = table.slice(0, 0)
            actions = np.array(rows["action"].to_pylist(), dtype=np.float32)
            states = (
                np.array(rows["observation.state"].to_pylist(), dtype=np.float32)
                if include_state
                else None
            )
            valid = np.array(rows["source.extra.pipeline_supervision_mask"].to_pylist())
            if len(actions) != episode["length"]:
                raise ValueError(
                    f"Expected {episode['length']} frames, found {len(actions)}"
                )
            obs_index = rows["source.observation_index"].to_numpy()
            act_index = rows["source.action_index"].to_numpy()
            if not np.array_equal(act_index, obs_index + 1):
                raise ValueError("Expected native obs[i]/command[i+1] alignment")
            starts = np.unique(
                np.linspace(
                    0, max(0, len(actions) - horizon), chunks_per_episode, dtype=int
                )
            )
            provenance = json.loads(episode["pipeline/provenance"])
            for start in starts:
                count = min(horizon, len(actions) - start)
                mask = np.zeros((horizon, actions.shape[-1]), dtype=bool)
                mask[:count] = valid[start : start + count, None]
                if not mask.any():
                    continue
                chunk = np.zeros((horizon, actions.shape[-1]), dtype=np.float32)
                chunk[:count] = actions[start : start + count]
                self.samples.append(
                    {
                        "actions": torch.from_numpy(chunk),
                        "mask": torch.from_numpy(mask),
                        "episode": episode,
                        "start": int(start),
                    }
                )
                if include_state:
                    self.samples[-1]["state"] = torch.from_numpy(states[start])
                self.manifest.append(
                    {
                        "episode_index": episode["episode_index"],
                        "start": int(start),
                        "valid_action_scalars": int(mask.sum()),
                        "task": episode["tasks"][0],
                        "provenance": provenance,
                        "data_file": str(path),
                    }
                )
        if not self.samples:
            raise ValueError("Expected at least one supervised action chunk")
        # Spawned workers share two storages rather than one file descriptor per chunk.
        self.actions = torch.stack([sample["actions"] for sample in self.samples])
        self.action_masks = torch.stack([sample["mask"] for sample in self.samples])
        for index, sample in enumerate(self.samples):
            sample["actions"] = self.actions[index]
            sample["mask"] = self.action_masks[index]
        if include_state:
            self.states = torch.stack([sample["state"] for sample in self.samples])
            for index, sample in enumerate(self.samples):
                sample["state"] = self.states[index]

    def normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit per-dimension moments on this selected training subset only."""
        actions = torch.stack([sample["actions"] for sample in self.samples])
        mask = torch.stack([sample["mask"] for sample in self.samples])
        count = mask.sum((0, 1)).clamp_min(1)
        mean = (actions * mask).sum((0, 1)) / count
        variance = ((actions - mean).square() * mask).sum((0, 1)) / count
        return mean, variance.sqrt().clamp_min(1e-3)

    def images(self, index: int):
        """Decode the selected observation using exact encoded video timestamps."""
        import av

        sample = self.samples[index]
        episode = sample["episode"]
        result = []
        for camera in self.cameras:
            path = self.root / self.info["video_path"].format(
                video_key=camera,
                chunk_index=episode[f"videos/{camera}/chunk_index"],
                file_index=episode[f"videos/{camera}/file_index"],
            )
            timestamp = episode[f"videos/{camera}/from_timestamp"] + (
                sample["start"] / self.info["fps"]
            )
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                container.seek(max(0, int(timestamp / stream.time_base)), stream=stream)
                for frame in container.decode(stream):
                    if (
                        float(frame.pts * stream.time_base)
                        >= timestamp - 0.5 / self.info["fps"]
                    ):
                        result.append(frame.to_image())
                        break
                else:
                    raise ValueError(f"No video frame at {timestamp}: {path}")
        return result

    def __len__(self) -> int:
        return len(self.samples)

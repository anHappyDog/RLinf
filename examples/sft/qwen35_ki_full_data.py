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

"""Prepare and lazily read full native Molmo supervision retained by qtarget data."""

import argparse
import hashlib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FullMolmoDataset(Dataset):
    """Read physical action8/state9 and synchronized observations without eager chunks."""

    def __init__(
        self,
        root,
        prepared,
        cameras=None,
        horizon=16,
        limit=None,
        require_complete=True,
    ):
        self.root = Path(root).resolve()
        self.prepared = Path(prepared).resolve()
        self.metadata = json.loads((self.prepared / "metadata.json").read_text())
        if (require_complete and not self.metadata["complete"]) or self.metadata[
            "root"
        ] != str(self.root):
            raise ValueError("Prepared full Molmo root or completeness differs")
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.cameras = cameras or self.metadata["cameras"]
        if (
            self.cameras != self.metadata["cameras"]
            or horizon != self.metadata["horizon"]
        ):
            raise ValueError("Prepared full Molmo camera/horizon contract differs")
        self.horizon = horizon
        self.limit = limit
        self.action_quantiles = self.metadata.get("action_quantiles")
        self.state_quantiles = self.metadata.get("state_quantiles")
        self.manifest_descriptor = {
            key: self.metadata[key]
            for key in ("root", "cameras", "horizon", "episodes", "anchors")
        }
        self.manifest_descriptor.update(
            index_sha256=self.metadata.get("index_sha256"), limit=limit
        )
        self.manifest_file = str(self.prepared / "episodes.jsonl")
        self._arrays = OrderedDict()
        self._video_cache = None
        self._rows = None
        self._open_index()

    def _open_index(self):
        import mmap

        self.ends = np.load(self.prepared / "ends.npy", mmap_mode="r")
        self.offsets = np.load(self.prepared / "offsets.npy", mmap_mode="r")
        self.anchors = np.memmap(self.prepared / "anchors.bin", mode="r", dtype="int32")
        with (self.prepared / "episodes.jsonl").open("rb") as stream:
            self._rows = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("ends", "offsets", "anchors", "_rows", "_video_cache"):
            state[key] = None
        state["_arrays"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_index()

    def __len__(self):
        return (
            min(self.limit, self.metadata["anchors"])
            if self.limit is not None
            else self.metadata["anchors"]
        )

    def normalization(self):
        return torch.tensor(self.metadata["mean"]), torch.tensor(self.metadata["std"])

    def _sample(self, index):
        episode = int(np.searchsorted(self.ends, index, side="right"))
        row = json.loads(self._rows[self.offsets[episode] : self.offsets[episode + 1]])
        start = int(self.anchors[index])
        name = row["numeric"]
        if name not in self._arrays:
            self._arrays[name] = np.load(self.prepared / name, mmap_mode="r")
            if len(self._arrays) > 8:
                self._arrays.popitem(last=False)
        else:
            self._arrays.move_to_end(name)
        array = self._arrays[name]
        selected = start + np.arange(self.horizon)
        valid = selected < row["length"]
        selected = np.minimum(selected, row["length"] - 1)
        values = array[row["row_offset"] + selected]
        mask = valid & values[:, 17].astype(bool)
        actions = values[:, :8].copy()
        state = array[row["row_offset"] + start, 8:17].copy()
        return (
            row,
            start,
            actions,
            np.broadcast_to(mask[:, None], actions.shape).copy(),
            state,
        )

    def __getitem__(self, index):
        index, valid = index if isinstance(index, tuple) else (index, True)
        row, start, actions, mask, state = self._sample(index)
        if self._video_cache is None:
            from rlinf.data.datasets.video import VideoFrameCache

            self._video_cache = VideoFrameCache(32)
        from PIL import Image

        timestamp = start / self.info["fps"]
        images = []
        for camera in self.cameras:
            video = row["videos"][camera]
            pixels = self._video_cache.read(
                video["path"],
                video["from_timestamp"] + timestamp,
                tolerance_s=0.5 / self.info["fps"] + 1e-5,
            )
            images.append(Image.fromarray(pixels).resize((224, 224)))
        return {
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask & valid),
            "state": torch.from_numpy(state),
            "images": images,
            "cameras": self.cameras,
            "instruction": row["task"],
        }


def prepare_full_molmo(root, output, cameras, horizon=16, workers=8):
    """Publish sidecars and an episode index once; retain original native columns."""
    import pyarrow.parquet as pq

    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        result = json.loads(metadata_path.read_text())
        if (
            result["root"] != str(root)
            or result["cameras"] != cameras
            or result["horizon"] != horizon
        ):
            raise ValueError("Existing preparation has a different contract")
        if result["complete"]:
            return result
    (output / "numeric").mkdir(exist_ok=True)
    info = json.loads((root / "meta/info.json").read_text())
    files = sorted((root / "data").rglob("*.parquet"))

    def numeric(item):
        number, path = item
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "source.extra.original.action",
                "source.extra.original.state",
                "source.extra.pipeline_supervision_mask",
                "source.observation_index",
                "source.action_index",
            ],
            use_threads=False,
        )
        obs = table["source.observation_index"].to_numpy()
        if not np.array_equal(table["source.action_index"].to_numpy(), obs + 1):
            raise ValueError(f"Native observation/action alignment differs: {path}")
        actions = np.asarray(
            table["source.extra.original.action"]
            .combine_chunks()
            .values.to_numpy()
            .reshape(len(table), 8),
            dtype="float32",
        )
        states = np.asarray(
            table["source.extra.original.state"]
            .combine_chunks()
            .values.to_numpy()
            .reshape(len(table), 9),
            dtype="float32",
        )
        if actions.shape != (len(table), 8) or states.shape != (len(table), 9):
            raise ValueError(f"Expected native action8/state9: {path}")
        valid = table["source.extra.pipeline_supervision_mask"].to_numpy().astype(bool)
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError(f"Native numeric data contains nonfinite values: {path}")
        array = np.column_stack((actions, states, valid)).astype("float32")
        name = f"numeric/{number:04d}.npy"
        np.save(output / name, array)
        ids = table["episode_index"].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        ranges = {
            int(ids[a]): (int(a), int(b - a))
            for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
        }
        if len(ranges) != len(boundaries) - 1:
            raise ValueError(f"Expected contiguous native episodes: {path}")
        selected = actions[valid].astype("float64")
        return (
            str(path.relative_to(root)),
            name,
            ranges,
            selected.sum(0),
            np.square(selected).sum(0),
            len(selected),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(numeric, enumerate(files)))
    descriptors = {name: (sidecar, ranges) for name, sidecar, ranges, *_ in prepared}
    moments = (
        sum(x[3] for x in prepared),
        sum(x[4] for x in prepared),
        sum(x[5] for x in prepared),
    )
    mean = moments[0] / moments[2]
    std = np.sqrt(np.maximum(moments[1] / moments[2] - mean**2, 1e-6))
    columns = [
        "episode_index",
        "length",
        "tasks",
        "data/chunk_index",
        "data/file_index",
        "pipeline/provenance",
    ]
    for camera in cameras:
        columns.extend(
            f"videos/{camera}/{key}"
            for key in ["chunk_index", "file_index", "from_timestamp"]
        )
    offsets, ends, total, count = [0], [], 0, 0
    indexed_array, indexed_name = None, None
    with (
        (output / "episodes.jsonl").open("wb") as rows,
        (output / "anchors.bin").open("wb") as anchors,
    ):
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=8192, columns=columns
            ):
                for episode in batch.to_pylist():
                    relative = info["data_path"].format(
                        chunk_index=episode["data/chunk_index"],
                        file_index=episode["data/file_index"],
                    )
                    name, ranges = descriptors[relative]
                    offset, length = ranges[episode["episode_index"]]
                    if length != episode["length"]:
                        raise ValueError(
                            f"Episode length differs: {episode['episode_index']}"
                        )
                    if indexed_name != name:
                        indexed_array = np.load(output / name, mmap_mode="r")
                        indexed_name = name
                    array = indexed_array
                    selected = np.flatnonzero(
                        array[offset : offset + length, 17]
                    ).astype("int32")
                    if not len(selected):
                        continue
                    provenance = json.loads(episode["pipeline/provenance"])
                    row = {
                        "episode_index": episode["episode_index"],
                        "source_episode_id": provenance["source_episode_id"],
                        "numeric": name,
                        "row_offset": offset,
                        "length": length,
                        "task": episode["tasks"][0],
                        "videos": {},
                    }
                    for camera in cameras:
                        video = root / info["video_path"].format(
                            video_key=camera,
                            chunk_index=episode[f"videos/{camera}/chunk_index"],
                            file_index=episode[f"videos/{camera}/file_index"],
                        )
                        row["videos"][camera] = {
                            "path": str(video),
                            "from_timestamp": episode[
                                f"videos/{camera}/from_timestamp"
                            ],
                        }
                    rows.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
                    offsets.append(rows.tell())
                    anchors.write(selected.tobytes())
                    total += len(selected)
                    ends.append(total)
                    count += 1
    np.save(output / "offsets.npy", np.asarray(offsets, dtype="int64"))
    np.save(output / "ends.npy", np.asarray(ends, dtype="int64"))
    result = {
        "complete": False,
        "root": str(root),
        "cameras": cameras,
        "horizon": horizon,
        "episodes": count,
        "source_episodes": info["total_episodes"],
        "anchors": total,
        "fps": info["fps"],
        "action_semantics": "native absolute joint qtarget rad7 + gripper command0..255",
        "state_semantics": "native qpos rad7 + left/right measured driver rad",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    metadata_path.write_text(json.dumps(result, indent=2))
    dataset = FullMolmoDataset(root, output, cameras, horizon, require_complete=False)
    states, actions = [], []
    for index in np.unique(
        np.linspace(0, len(dataset) - 1, min(4096, len(dataset)), dtype="int64")
    ):
        _, _, a, mask, state = dataset._sample(int(index))
        states.append(state)
        actions.append(a[mask[:, 0]])

    def quantiles(values):
        return {
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    result["action_quantiles"] = quantiles(np.concatenate(actions))
    result["state_quantiles"] = quantiles(np.stack(states))
    result["index_sha256"] = hashlib.sha256(
        (output / "offsets.npy").read_bytes() + (output / "ends.npy").read_bytes()
    ).hexdigest()
    result["complete"] = True
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(metadata_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_full_molmo(
                args.root, args.output, args.cameras, workers=args.workers
            )
        )
    )

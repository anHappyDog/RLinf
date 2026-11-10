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

"""Explicitly prepared, read-only shared indexes for mixture manifests."""

import hashlib
import json
import mmap
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def build_manifest_indexes(root: str | Path) -> None:
    """Build split indexes once, before training ranks open the prepared mixture."""
    root = Path(root)
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text())
    files = {}
    for name, expected in contract["files"].items():
        path = root / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Prepared manifest changed: {name}; regenerate the preparation contract"
            )
        stat = path.stat()
        files[name] = [stat.st_size, stat.st_mtime_ns]
    fingerprint = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    for split in ("train", "val", "test"):
        directory = root / f"{split}.index"
        directory.mkdir(exist_ok=True)
        offsets, ends, grouped = [0], [], {}
        fps_by_schema = {}
        total = 0
        with (
            (root / f"{split}.jsonl").open("rb") as source,
            (directory / "rows.jsonl").open("wb") as rows,
            (directory / "anchors.bin").open("wb") as anchors,
        ):
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                fps_by_schema.setdefault(row["schema_id"], set()).add(row["fps"])
                frames = np.asarray(row.pop("anchor_indices"), dtype=np.int64)
                anchors.write(frames.tobytes())
                rows.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
                offsets.append(rows.tell())
                grouped.setdefault(row["schema_id"], {}).setdefault(
                    row["source_group_id"], []
                ).append((total, len(frames)))
                total += len(frames)
                ends.append(total)
        np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(directory / "ends.npy", np.asarray(ends, dtype=np.int64))
        sources = sorted(grouped)
        for index, source in enumerate(sources):
            group_offsets, starts, clip_ends = [0], [], []
            for clips in grouped[source].values():
                count = 0
                for start, length in clips:
                    count += length
                    starts.append(start)
                    clip_ends.append(count)
                group_offsets.append(len(starts))
            for name, values in zip(
                ("groups", "starts", "clip_ends"),
                (group_offsets, starts, clip_ends),
                strict=True,
            ):
                np.save(
                    directory / f"{index}.{name}.npy",
                    np.asarray(values, dtype=np.int64),
                )
        metadata = {
            "version": 1,
            "files": files,
            "manifest_fingerprint": fingerprint,
            "sources": sources,
            "fps_by_schema": {
                name: sorted(values) for name, values in fps_by_schema.items()
            },
            "episodes": len(ends),
            "anchors": total,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata))


class ManifestIndex(Sequence):
    """Episode rows and anchors mapped read-only; pickle contains only the directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        root = self.directory.parent
        for name, expected in self.metadata["files"].items():
            stat = (root / name).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise ValueError(
                    f"Prepared manifest changed: {name}; rebuild manifest indexes"
                )
        fingerprint = hashlib.sha256((root / "contract.json").read_bytes()).hexdigest()
        if fingerprint != self.metadata["manifest_fingerprint"]:
            raise ValueError("Prepared contract changed; rebuild manifest indexes")
        self._open()

    def _open(self) -> None:
        self.ends = np.load(self.directory / "ends.npy", mmap_mode="r")
        self.offsets = np.load(self.directory / "offsets.npy", mmap_mode="r")
        self.anchors = (
            np.memmap(self.directory / "anchors.bin", dtype=np.int64, mode="r")
            if self.metadata["anchors"]
            else np.empty(0, dtype=np.int64)
        )
        with (self.directory / "rows.jsonl").open("rb") as stream:
            self.rows = (
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                if self.metadata["episodes"]
                else None
            )

    def __len__(self) -> int:
        return self.metadata["episodes"]

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = json.loads(self.rows[self.offsets[index] : self.offsets[index + 1]])
        start = int(self.ends[index - 1]) if index else 0
        row["anchor_indices"] = self.anchors[start : int(self.ends[index])]
        return row

    def __getstate__(self) -> dict:
        return {"directory": self.directory}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["directory"])

    def sampler_groups(self) -> list:
        """Return source/group arrays in the original sampler's insertion order."""
        return [
            tuple(
                np.load(self.directory / f"{index}.{name}.npy", mmap_mode="r")
                for name in ("groups", "starts", "clip_ends")
            )
            for index in range(len(self.metadata["sources"]))
        ]

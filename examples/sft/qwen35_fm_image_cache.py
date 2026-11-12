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

"""Cache the exact selected MolmoBot RGB observations as memory-mapped arrays."""

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_CACHE = None
_IMAGE_SIZE = None


def selection_signature(subset) -> str:
    """Identify the ordered action/observation selection used by a dataset."""
    serialized = json.dumps(
        {"manifest": subset.manifest, "video_selections": _episode_tasks(subset)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _initialize_worker(path: str, image_size: int) -> None:
    global _CACHE, _IMAGE_SIZE
    _CACHE = np.load(path, mmap_mode="r+")
    _IMAGE_SIZE = image_size


def _decode_episode(task) -> int:
    """Decode each episode once per camera and write disjoint cache rows."""
    import av

    camera_tasks, fps = task
    for camera_index, (path, targets) in enumerate(camera_tasks):
        pending = iter(targets)
        row, timestamp = next(pending)
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            container.seek(max(0, int(timestamp / stream.time_base)), stream=stream)
            for frame in container.decode(stream):
                if float(frame.pts * stream.time_base) < timestamp - 0.5 / fps:
                    continue
                image = np.asarray(frame.to_image().resize((_IMAGE_SIZE, _IMAGE_SIZE)))
                # Closely spaced selected observations can choose the same frame.
                while float(frame.pts * stream.time_base) >= timestamp - 0.5 / fps:
                    _CACHE[row, camera_index] = image
                    try:
                        row, timestamp = next(pending)
                    except StopIteration:
                        break
                else:
                    continue
                break
            else:
                raise ValueError(f"No selected video frame at {timestamp}: {path}")
    return len(camera_tasks[0][1])


def _episode_tasks(subset):
    episodes = {}
    for index, sample in enumerate(subset.samples):
        episodes.setdefault(sample["episode"]["episode_index"], []).append(index)
    tasks = []
    for indices in episodes.values():
        episode = subset.samples[indices[0]]["episode"]
        cameras = []
        for camera in subset.cameras:
            path = subset.root / subset.info["video_path"].format(
                video_key=camera,
                chunk_index=episode[f"videos/{camera}/chunk_index"],
                file_index=episode[f"videos/{camera}/file_index"],
            )
            targets = [
                (
                    index,
                    episode[f"videos/{camera}/from_timestamp"]
                    + subset.samples[index]["start"] / subset.info["fps"],
                )
                for index in indices
            ]
            cameras.append((str(path), targets))
        tasks.append((cameras, subset.info["fps"]))
    return tasks


def _build_split(subset, destination: Path, image_size: int, workers: int, logger):
    shape = (len(subset), len(subset.cameras), image_size, image_size, 3)
    array = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.uint8, shape=shape
    )
    del array
    tasks = _episode_tasks(subset)
    completed = 0
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(str(destination), image_size),
    ) as pool:
        futures = [pool.submit(_decode_episode, task) for task in tasks]
        for future in as_completed(futures):
            completed += future.result()
            if completed % 1024 < 8 or completed == len(subset):
                logger.info(
                    f"{destination.stem}: cached {completed}/{len(subset)} "
                    f"observations in {time.perf_counter() - start:.1f}s"
                )
    array = np.load(destination, mmap_mode="r+")
    array.flush()
    # Check first/last, episode boundaries, and interior samples using the original seek path.
    check_indices = np.unique(
        np.linspace(0, len(subset) - 1, min(16, len(subset)), dtype=int)
    )
    for index in check_indices:
        expected = subset.images(int(index))
        for camera, image in enumerate(expected):
            original = np.asarray(image.resize((image_size, image_size)))
            if not np.array_equal(original, array[index, camera]):
                raise RuntimeError(
                    f"Cached image differs from native selection at row {index}, camera {camera}"
                )
    return {
        "count": len(subset),
        "selection_sha256": selection_signature(subset),
        "filename": destination.name,
        "shape": list(shape),
        "dtype": "uint8",
        "exact_pixel_checks": len(check_indices) * len(subset.cameras),
        "elapsed_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-episodes", type=int, default=4096)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--chunks-per-episode", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cameras", nargs="+", default=["observation.images.exterior"])
    args = parser.parse_args()
    if (
        min(
            args.train_episodes,
            args.val_episodes,
            args.chunks_per_episode,
            args.horizon,
            args.image_size,
            args.workers,
        )
        < 1
    ):
        raise ValueError(
            "Cache selection, image dimensions and worker count must be positive"
        )
    from examples.sft.qwen35_fm_data import MolmoBotSubset
    from rlinf.utils.logging import get_logger

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(args.data_path).resolve()
    metadata = {
        "schema_version": 1,
        "complete": False,
        "data_path": str(data_root),
        "cameras": args.cameras,
        "horizon": args.horizon,
        "image_size": args.image_size,
        "splits": {},
    }
    for split, source, episode_count in [
        ("train", "molmobot", args.train_episodes),
        ("validation", "molmobot_val", args.val_episodes),
    ]:
        subset = MolmoBotSubset(
            data_root / source,
            args.horizon,
            episode_count,
            args.chunks_per_episode,
            args.cameras,
        )
        metadata["splits"][split] = _build_split(
            subset, output / f"{split}.npy", args.image_size, args.workers, get_logger()
        )
    metadata["complete"] = True
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(output / "metadata.json")
    get_logger().info(f"Completed exact selected-observation cache: {output}")


if __name__ == "__main__":
    main()

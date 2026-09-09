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

"""Create and read deterministic, exposure-aware B1K episode splits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

FORMAT_VERSION = "b1k_episode_split_v1"
SELECTION_POLICY = "sha256_rank_unexposed_eval_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _task_records(dataset_root: Path) -> dict[int, dict[str, Any]]:
    records = {}
    with (dataset_root / "meta" / "tasks.jsonl").open() as file:
        for line in file:
            record = json.loads(line)
            records[int(record["task_index"])] = record
    return records


def _available_episodes(dataset_root: Path, task_index: int) -> list[int]:
    annotation_dir = dataset_root / "annotations" / f"task-{task_index:04d}"
    return sorted(
        int(path.stem.removeprefix("episode_"))
        for path in annotation_dir.glob("episode_*.json")
    )


def _rank(seed: int, task_index: int, episode_index: int) -> bytes:
    value = f"{seed}:{task_index}:{episode_index}".encode()
    return hashlib.sha256(value).digest()


def _episode_record(
    task: dict[str, Any], task_index: int, episode_index: int
) -> dict[str, Any]:
    run_episode_index = episode_index - task_index * 10_000
    if run_episode_index <= 0:
        raise ValueError(
            f"Episode {episode_index} is invalid for task {task_index}; "
            "expected the B1K global episode-index convention."
        )
    return {
        "task_index": task_index,
        "task_name": task["task_name"],
        "episode_index": episode_index,
        "run_episode_index": run_episode_index,
    }


def create_episode_split(
    dataset_root: str | Path,
    *,
    task_indices: Iterable[int],
    train_per_task: int,
    validation_per_task: int,
    test_per_task: int,
    seed: int,
    exclude_manifests: Iterable[str | Path] = (),
    exclude_episodes: Iterable[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Return a deterministic split with exposed episodes confined to train."""
    dataset_root = Path(dataset_root).resolve()
    selected_task_indices = sorted(set(task_indices))
    if not selected_task_indices:
        raise ValueError("task_indices must not be empty.")
    split_sizes = {
        "train": train_per_task,
        "validation": validation_per_task,
        "test": test_per_task,
    }
    if any(size < 0 for size in split_sizes.values()):
        raise ValueError("Split sizes must be non-negative.")

    tasks = _task_records(dataset_root)
    unknown_tasks = sorted(set(selected_task_indices).difference(tasks))
    if unknown_tasks:
        raise ValueError(f"Unknown task indices: {unknown_tasks}.")

    available = {
        task_index: _available_episodes(dataset_root, task_index)
        for task_index in selected_task_indices
    }
    required_per_task = sum(split_sizes.values())
    for task_index, episodes in available.items():
        if len(episodes) != required_per_task:
            raise ValueError(
                f"Task {task_index} has {len(episodes)} episodes, but split sizes "
                f"require exactly {required_per_task}."
            )

    exposure_reasons: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    exposure_sources = []
    for raw_path in exclude_manifests:
        path = Path(raw_path).resolve()
        manifest = _read_json(path)
        source_episodes = manifest.get("source_episodes")
        if not isinstance(source_episodes, list):
            raise ValueError(f"Manifest has no source_episodes list: {path}.")
        source_label = f"manifest:{path}"
        for record in source_episodes:
            task_index = int(record["task_index"])
            episode_index = int(record["episode_index"])
            if task_index in available and episode_index in available[task_index]:
                exposure_reasons[(task_index, episode_index)].add(source_label)
        exposure_sources.append(
            {
                "type": "manifest",
                "path": str(path),
                "sha256": _file_sha256(path),
            }
        )

    manual_exclusions = list(exclude_episodes)
    for task_index, episode_index in manual_exclusions:
        if task_index not in available:
            raise ValueError(f"Excluded episode uses unknown task {task_index}.")
        if episode_index not in available[task_index]:
            raise ValueError(
                f"Excluded episode {episode_index} is unavailable for task {task_index}."
            )
        exposure_reasons[(task_index, episode_index)].add("manual_exclusion")
    if manual_exclusions:
        exposure_sources.append({"type": "manual_cli"})

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    task_counts = {}
    for task_index in selected_task_indices:
        exposed = {
            episode_index
            for exposed_task, episode_index in exposure_reasons
            if exposed_task == task_index
        }
        clean = [
            episode_index
            for episode_index in available[task_index]
            if episode_index not in exposed
        ]
        clean.sort(key=lambda episode_index: _rank(seed, task_index, episode_index))
        clean_eval_count = validation_per_task + test_per_task
        if len(clean) < clean_eval_count:
            raise ValueError(
                f"Task {task_index} has only {len(clean)} unexposed episodes; "
                f"validation and test require {clean_eval_count}."
            )

        test = set(clean[:test_per_task])
        validation = set(clean[test_per_task : test_per_task + validation_per_task])
        train = set(available[task_index]).difference(test, validation)
        if len(train) != train_per_task:
            raise RuntimeError(
                f"Task {task_index} produced {len(train)} train episodes, "
                f"expected {train_per_task}."
            )
        if exposed.difference(train):
            raise RuntimeError("An exposed episode escaped the train split.")

        for split_name, episode_indices in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            splits[split_name].extend(
                _episode_record(tasks[task_index], task_index, episode_index)
                for episode_index in sorted(episode_indices)
            )
        task_counts[str(task_index)] = {
            "available": len(available[task_index]),
            "exposed": len(exposed),
            **split_sizes,
        }

    exposed_records = []
    for (task_index, episode_index), reasons in sorted(exposure_reasons.items()):
        record = _episode_record(tasks[task_index], task_index, episode_index)
        record["reasons"] = sorted(reasons)
        exposed_records.append(record)

    return {
        "format_version": FORMAT_VERSION,
        "dataset_root": str(dataset_root),
        "seed": seed,
        "selection_policy": SELECTION_POLICY,
        "exposure_policy": "exposed_episodes_are_train_only",
        "task_indices": selected_task_indices,
        "split_sizes_per_task": split_sizes,
        "exposure_sources": exposure_sources,
        "exposed_episodes": exposed_records,
        "task_counts": task_counts,
        "splits": splits,
    }


def load_episode_split(
    path: str | Path,
    split_name: str,
    *,
    task_indices: Iterable[int] | None = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Load and validate episode pairs from one split manifest section."""
    path = Path(path).resolve()
    manifest = _read_json(path)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported split format {manifest.get('format_version')!r} in {path}."
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise ValueError(f"Unknown split {split_name!r} in {path}.")
    records = splits[split_name]
    if not isinstance(records, list):
        raise ValueError(f"Split {split_name!r} must be a list.")

    selected_tasks = None if task_indices is None else set(task_indices)
    episodes = []
    for record in records:
        task_index = int(record["task_index"])
        episode_index = int(record["episode_index"])
        if selected_tasks is None or task_index in selected_tasks:
            episodes.append((task_index, episode_index))
    episodes.sort()
    if not episodes:
        raise ValueError(f"Split {split_name!r} selected no episodes.")
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"Split {split_name!r} contains duplicate episodes.")
    if selected_tasks is not None:
        missing_tasks = selected_tasks.difference(task for task, _ in episodes)
        if missing_tasks:
            raise ValueError(
                f"Split {split_name!r} has no episodes for tasks "
                f"{sorted(missing_tasks)}."
            )

    metadata = {
        "mode": "split_manifest",
        "split_name": split_name,
        "split_manifest": str(path),
        "split_manifest_sha256": _file_sha256(path),
        "selection_policy": manifest.get("selection_policy"),
    }
    return episodes, metadata


def _parse_excluded_episode(value: str) -> tuple[int, int]:
    try:
        task_index, episode_index = value.split(":", maxsplit=1)
        return int(task_index), int(episode_index)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Excluded episodes must use TASK_INDEX:EPISODE_INDEX."
        ) from error


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-indices", type=int, nargs="+", required=True)
    parser.add_argument("--train-per-task", type=int, required=True)
    parser.add_argument("--validation-per-task", type=int, required=True)
    parser.add_argument("--test-per-task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Sidecar manifest whose source episodes must remain train-only.",
    )
    parser.add_argument(
        "--exclude-episode",
        type=_parse_excluded_episode,
        action="append",
        default=[],
        help="Additional train-only episode as TASK_INDEX:EPISODE_INDEX.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}.")
    manifest = create_episode_split(
        args.dataset_root,
        task_indices=args.task_indices,
        train_per_task=args.train_per_task,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        seed=args.seed,
        exclude_manifests=args.exclude_manifest,
        exclude_episodes=args.exclude_episode,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest["task_counts"], indent=2))


if __name__ == "__main__":
    main()

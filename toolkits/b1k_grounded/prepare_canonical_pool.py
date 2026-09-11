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

"""Assemble and success-stratify canonical BEHAVIOR subpool snapshots."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rlinf.envs.behavior.subpool import (
    SubpoolCatalog,
    SubpoolSnapshot,
    SubpoolStore,
)


@dataclasses.dataclass(frozen=True)
class SnapshotScore:
    """Baseline success counts for one canonical snapshot."""

    snapshot_id: str
    successes: int
    attempts: int

    @property
    def success_rate(self) -> float:
        """Return the empirical success rate."""
        return self.successes / self.attempts


def _terminal_time_record(
    record: SubpoolSnapshot,
    *,
    horizon: int,
) -> SubpoolSnapshot:
    metadata = dict(record.metadata)
    reward = dict(metadata["reward"])
    reward.update(
        {
            "max_steps": horizon,
            "potential_terms": [],
            "step_penalty": -1.0 / horizon,
            "definition": "terminal_time_v1",
        }
    )
    metadata["reward"] = reward
    metadata["timeout_policy"] = {
        "max_steps": horizon,
        "name": f"init_only_h{horizon}_v1",
        "step_penalty_budget": -1.0,
    }
    return dataclasses.replace(record, metadata=metadata)


def assemble_canonical_pool(
    input_manifests: Sequence[Path],
    output_manifest: Path,
    *,
    subtask_id: int,
    horizon: int,
    require_gt_success: bool = True,
) -> tuple[SubpoolSnapshot, ...]:
    """Copy unique, validated canonical records into one normalized catalog."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if output_manifest.exists() or output_manifest.parent.exists():
        raise FileExistsError(
            f"Output catalog must be a fresh path: {output_manifest.parent}"
        )

    selected: dict[str, tuple[SubpoolCatalog, SubpoolSnapshot]] = {}
    for manifest in input_manifests:
        catalog = SubpoolCatalog.from_jsonl(manifest)
        for record in catalog.records:
            if record.pool_type != "canonical" or record.subtask_id != subtask_id:
                continue
            try:
                control = json.loads(record.control_json)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(control, dict) or not control.get("subgoal"):
                continue
            gt_validation = record.metadata.get("gt_validation", {})
            if require_gt_success and not gt_validation.get("success", False):
                continue
            start_frame = gt_validation.get("start_frame")
            end_frame = gt_validation.get("end_frame")
            if start_frame is not None and end_frame is not None:
                duration = int(end_frame) - int(start_frame)
                if duration <= 0 or duration > horizon:
                    continue
            if record.snapshot_id in selected:
                continue
            selected[record.snapshot_id] = (catalog, record)

    if not selected:
        raise ValueError("No canonical snapshots matched the requested filters.")
    store = SubpoolStore(output_manifest)
    output_records = []
    for snapshot_id in sorted(selected):
        catalog, record = selected[snapshot_id]
        normalized = _terminal_time_record(record, horizon=horizon)
        store.append_from_file(normalized, catalog.state_path(record))
        output_records.append(normalized)
    return tuple(output_records)


def load_snapshot_scores(path: Path) -> dict[str, SnapshotScore]:
    """Load ``snapshot_id -> {successes, attempts}`` baseline measurements."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_scores = payload.get("snapshots", payload)
    if not isinstance(raw_scores, dict):
        raise ValueError("Score JSON must contain a snapshot mapping.")

    scores = {}
    for snapshot_id, raw_score in raw_scores.items():
        successes = int(raw_score["successes"])
        attempts = int(raw_score["attempts"])
        if attempts <= 0 or not 0 <= successes <= attempts:
            raise ValueError(
                f"Invalid score for {snapshot_id}: {successes}/{attempts}."
            )
        scores[snapshot_id] = SnapshotScore(
            snapshot_id=snapshot_id,
            successes=successes,
            attempts=attempts,
        )
    return scores


def collect_snapshot_scores(
    source_manifest: Path,
    metrics_paths: Sequence[Path],
    output_path: Path,
    *,
    expected_attempts: int,
) -> dict[str, Any]:
    """Convert per-episode RLinf evaluation metrics to snapshot score counts."""
    if expected_attempts <= 0:
        raise ValueError("expected_attempts must be positive.")
    if output_path.exists():
        raise FileExistsError(f"Score output already exists: {output_path}")
    catalog = SubpoolCatalog.from_jsonl(source_manifest)
    by_episode = {record.episode_index: record for record in catalog.records}
    if len(by_episode) != len(catalog.records):
        raise ValueError("Canonical baseline requires unique episode indices.")

    raw_metrics = {}
    for metrics_path in metrics_paths:
        raw_metrics.update(json.loads(metrics_path.read_text(encoding="utf-8")))
    snapshots = {}
    for episode_index, record in sorted(by_episode.items()):
        prefix = f"eval/snapshot/episode_{episode_index}"
        attempts = int(raw_metrics[f"{prefix}/attempts"])
        success_rate = float(raw_metrics[f"{prefix}/success"])
        if attempts != expected_attempts:
            raise ValueError(
                f"Episode {episode_index} has {attempts} attempts; expected "
                f"{expected_attempts}."
            )
        successes = int(round(success_rate * attempts))
        if not np.isclose(success_rate, successes / attempts, atol=1e-6):
            raise ValueError(
                f"Episode {episode_index} success rate {success_rate} is not an "
                f"integer count over {attempts} attempts."
            )
        snapshots[record.snapshot_id] = {
            "episode_index": episode_index,
            "successes": successes,
            "attempts": attempts,
            "success_rate": successes / attempts,
        }
    payload = {
        "format_version": "b1k_canonical_sft_baseline_scores_v1",
        "source_manifest": str(source_manifest.resolve()),
        "metrics_paths": [str(path.resolve()) for path in metrics_paths],
        "snapshots": snapshots,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def stratified_snapshot_split(
    records: Sequence[SubpoolSnapshot],
    scores: dict[str, SnapshotScore],
    *,
    split_size: int,
    seed: int,
) -> tuple[tuple[SubpoolSnapshot, ...], tuple[SubpoolSnapshot, ...]]:
    """Create matched train/eval splits across the baseline-success range."""
    if split_size <= 0:
        raise ValueError("split_size must be positive.")
    scored_records = [record for record in records if record.snapshot_id in scores]
    required = split_size * 2
    if len(scored_records) < required:
        raise ValueError(
            f"Need at least {required} scored snapshots, got {len(scored_records)}."
        )
    scored_records.sort(
        key=lambda record: (
            scores[record.snapshot_id].success_rate,
            record.snapshot_id,
        )
    )
    if len(scored_records) > required:
        indices = [
            int((index + 0.5) * len(scored_records) / required)
            for index in range(required)
        ]
        scored_records = [scored_records[index] for index in indices]

    rng = np.random.default_rng(seed)
    train = []
    evaluation = []
    for pair_index in range(split_size):
        pair = scored_records[pair_index * 2 : pair_index * 2 + 2]
        if bool(rng.integers(2)):
            pair.reverse()
        train.append(pair[0])
        evaluation.append(pair[1])
    return tuple(train), tuple(evaluation)


def select_duration_stratified_candidates(
    records: Sequence[SubpoolSnapshot],
    *,
    count: int,
    seed: int,
) -> tuple[SubpoolSnapshot, ...]:
    """Select candidates across the demonstrated completion-time range."""
    if count <= 0 or len(records) < count:
        raise ValueError(
            f"count must be positive and at most {len(records)}, got {count}."
        )
    rng = np.random.default_rng(seed)
    tie_breakers = {record.snapshot_id: float(rng.random()) for record in records}

    def sort_key(record: SubpoolSnapshot) -> tuple[int, float, str]:
        gt_validation = record.metadata.get("gt_validation", {})
        duration = int(gt_validation.get("end_frame", 0)) - int(
            gt_validation.get("start_frame", 0)
        )
        return duration, tie_breakers[record.snapshot_id], record.snapshot_id

    ordered = sorted(records, key=sort_key)
    if len(ordered) == count:
        return tuple(ordered)
    indices = [int((index + 0.5) * len(ordered) / count) for index in range(count)]
    return tuple(ordered[index] for index in indices)


def _write_subset(
    source: SubpoolCatalog,
    records: Sequence[SubpoolSnapshot],
    output_manifest: Path,
) -> None:
    if output_manifest.exists() or output_manifest.parent.exists():
        raise FileExistsError(
            f"Output catalog must be a fresh path: {output_manifest.parent}"
        )
    store = SubpoolStore(output_manifest)
    for record in sorted(records, key=lambda item: item.snapshot_id):
        store.append_from_file(record, source.state_path(record))


def write_catalog_partitions(
    source_manifest: Path,
    output_dir: Path,
    *,
    partition_size: int,
) -> tuple[Path, ...]:
    """Write deterministic self-contained catalog partitions."""
    if partition_size <= 0:
        raise ValueError("partition_size must be positive.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be fresh: {output_dir}")
    source = SubpoolCatalog.from_jsonl(source_manifest)
    records = sorted(source.records, key=lambda record: record.snapshot_id)
    output_manifests = []
    for start in range(0, len(records), partition_size):
        output_manifest = (
            output_dir / f"batch_{start // partition_size:02d}" / "manifest.jsonl"
        )
        _write_subset(
            source,
            records[start : start + partition_size],
            output_manifest,
        )
        output_manifests.append(output_manifest)
    return tuple(output_manifests)


def _score_summary(
    records: Sequence[SubpoolSnapshot],
    scores: dict[str, SnapshotScore],
) -> dict[str, Any]:
    rates = [scores[record.snapshot_id].success_rate for record in records]
    return {
        "count": len(records),
        "mean_success_rate": float(np.mean(rates)),
        "min_success_rate": float(np.min(rates)),
        "max_success_rate": float(np.max(rates)),
        "snapshots": [
            {
                "snapshot_id": record.snapshot_id,
                "episode_index": record.episode_index,
                "successes": scores[record.snapshot_id].successes,
                "attempts": scores[record.snapshot_id].attempts,
                "success_rate": scores[record.snapshot_id].success_rate,
            }
            for record in sorted(
                records,
                key=lambda item: (
                    scores[item.snapshot_id].success_rate,
                    item.snapshot_id,
                ),
            )
        ],
    }


def write_stratified_split(
    source_manifest: Path,
    scores_path: Path,
    output_dir: Path,
    *,
    split_size: int,
    seed: int,
) -> dict[str, Any]:
    """Write matched canonical train and held-out evaluation catalogs."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be fresh: {output_dir}")
    source = SubpoolCatalog.from_jsonl(source_manifest)
    scores = load_snapshot_scores(scores_path)
    train, evaluation = stratified_snapshot_split(
        source.records,
        scores,
        split_size=split_size,
        seed=seed,
    )
    _write_subset(source, train, output_dir / "train" / "manifest.jsonl")
    _write_subset(source, evaluation, output_dir / "heldout_eval" / "manifest.jsonl")
    summary = {
        "format_version": "b1k_canonical_success_stratified_split_v1",
        "source_manifest": str(source_manifest.resolve()),
        "scores_path": str(scores_path.resolve()),
        "seed": seed,
        "train": _score_summary(train, scores),
        "heldout_eval": _score_summary(evaluation, scores),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Run canonical-pool assembly or success-stratified splitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--input-manifest", type=Path, nargs="+", required=True)
    assemble.add_argument("--output-manifest", type=Path, required=True)
    assemble.add_argument("--subtask-id", type=int, default=1)
    assemble.add_argument("--horizon", type=int, default=1280)
    assemble.add_argument("--allow-unverified", action="store_true")

    split = subparsers.add_parser("split")
    split.add_argument("--source-manifest", type=Path, required=True)
    split.add_argument("--scores", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--split-size", type=int, default=20)
    split.add_argument("--seed", type=int, default=20260911)

    select = subparsers.add_parser("select")
    select.add_argument("--source-manifest", type=Path, required=True)
    select.add_argument("--output-manifest", type=Path, required=True)
    select.add_argument("--count", type=int, default=40)
    select.add_argument("--seed", type=int, default=20260911)

    partition = subparsers.add_parser("partition")
    partition.add_argument("--source-manifest", type=Path, required=True)
    partition.add_argument("--output-dir", type=Path, required=True)
    partition.add_argument("--partition-size", type=int, default=20)

    scores = subparsers.add_parser("scores")
    scores.add_argument("--source-manifest", type=Path, required=True)
    scores.add_argument("--metrics", type=Path, nargs="+", required=True)
    scores.add_argument("--output", type=Path, required=True)
    scores.add_argument("--expected-attempts", type=int, default=20)

    args = parser.parse_args()
    if args.command == "assemble":
        records = assemble_canonical_pool(
            args.input_manifest,
            args.output_manifest,
            subtask_id=args.subtask_id,
            horizon=args.horizon,
            require_gt_success=not args.allow_unverified,
        )
        print(json.dumps({"canonical_snapshots": len(records)}, indent=2))
        return
    if args.command == "select":
        source = SubpoolCatalog.from_jsonl(args.source_manifest)
        records = select_duration_stratified_candidates(
            source.records,
            count=args.count,
            seed=args.seed,
        )
        _write_subset(source, records, args.output_manifest)
        print(json.dumps({"canonical_snapshots": len(records)}, indent=2))
        return
    if args.command == "partition":
        manifests = write_catalog_partitions(
            args.source_manifest,
            args.output_dir,
            partition_size=args.partition_size,
        )
        print(
            json.dumps(
                {"partitions": [str(manifest) for manifest in manifests]},
                indent=2,
            )
        )
        return
    if args.command == "scores":
        payload = collect_snapshot_scores(
            args.source_manifest,
            args.metrics,
            args.output,
            expected_attempts=args.expected_attempts,
        )
        print(json.dumps({"snapshots": len(payload["snapshots"])}, indent=2))
        return
    summary = write_stratified_split(
        args.source_manifest,
        args.scores,
        args.output_dir,
        split_size=args.split_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

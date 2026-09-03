#!/usr/bin/env python3
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

"""Aggregate repeated-seed R1 validation metrics and verify split identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_METRIC_NAMES = (
    "loss",
    "token_accuracy",
    "exact_match",
    "macro_task_exact_match",
    "verb_accuracy",
    "object_accuracy",
    "destination_accuracy",
    "step_count_accuracy",
)
_ACTION_METRIC_NAMES = (
    "relative_rmse",
    "mean_cosine_similarity",
    "max_absolute_drift",
    "first_action_relative_rmse",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write grouped mean/std metrics for completed R1 runs."""
    args = parse_args()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run_dir in args.run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        grouped[str(metrics["input_mode"])].append(
            {
                "run_dir": str(run_dir),
                "selection_sha256": _sha256(run_dir / "val_selection.jsonl"),
                "metrics": metrics["final"]["val"],
            }
        )

    summary = {}
    for input_mode, runs in sorted(grouped.items()):
        selection_hashes = {str(run["selection_sha256"]) for run in runs}
        aggregates = {}
        for metric_name in _METRIC_NAMES:
            values = [
                float(run["metrics"][metric_name])
                for run in runs
                if metric_name in run["metrics"]
            ]
            if values:
                aggregates[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
        action_metrics = []
        for run in runs:
            action_metrics_path = (
                Path(str(run["run_dir"]))
                / "action_preservation"
                / "metrics.json"
            )
            if action_metrics_path.is_file():
                action_metrics.append(json.loads(action_metrics_path.read_text()))
        action_summary = None
        if action_metrics:
            action_summary = {
                "run_count": len(action_metrics),
                "pass_rate": sum(
                    bool(item["action_preservation_pass"])
                    for item in action_metrics
                )
                / len(action_metrics),
                "aggregates": {
                    metric_name: {
                        "mean": float(
                            np.mean(
                                [float(item[metric_name]) for item in action_metrics]
                            )
                        ),
                        "std": float(
                            np.std(
                                [float(item[metric_name]) for item in action_metrics]
                            )
                        ),
                        "values": [
                            float(item[metric_name]) for item in action_metrics
                        ],
                    }
                    for metric_name in _ACTION_METRIC_NAMES
                },
            }
        summary[input_mode] = {
            "run_count": len(runs),
            "identical_validation_selection": len(selection_hashes) == 1,
            "validation_selection_sha256": sorted(selection_hashes),
            "aggregates": aggregates,
            "action_preservation": action_summary,
            "run_dirs": [str(run["run_dir"]) for run in runs],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

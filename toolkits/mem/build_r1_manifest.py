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

"""Build deterministic episode-level R1 high-level text manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    build_r1_manifest,
    write_r1_manifests,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-primitive", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--task-index",
        type=int,
        action="append",
        dest="task_indices",
        help="Restrict to one or more tasks. By default all tasks are included.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and write all R1 manifest splits."""
    args = parse_args()
    entries, report = build_r1_manifest(
        args.dataset_root,
        samples_per_primitive=args.samples_per_primitive,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        task_indices=args.task_indices,
    )
    write_r1_manifests(entries, args.output_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

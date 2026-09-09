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

"""Merge validated BEHAVIOR subpool manifests into a standalone catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlinf.envs.behavior.subpool import merge_subpool_manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Merge all input catalogs and report the resulting coverage."""
    args = _parse_args()
    catalog = merge_subpool_manifests(args.input_manifest, args.output_manifest)
    episode_indices = sorted(
        record.episode_index
        for record in catalog.records
        if record.episode_index is not None
    )
    print(
        f"Merged {len(catalog.records)} snapshots from episodes "
        f"{episode_indices} into {args.output_manifest.resolve()}"
    )


if __name__ == "__main__":
    main()

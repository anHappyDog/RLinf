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

"""Prepare grouped splits and native-schema statistics without copying media."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.data.datasets.openpi.mixture import prepare_mixture
from rlinf.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path(__file__).parent / "config/data/franka_pickup_sources.yaml",
    )
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-development", action="store_true")
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--max-val-episodes", type=int)
    args = parser.parse_args()

    profiles = OmegaConf.to_container(OmegaConf.load(args.source_config).sources)
    if args.sources is not None:
        known = {profile["name"] for profile in profiles}
        unknown = set(args.sources) - known
        if unknown:
            raise ValueError(f"Unknown source profiles: {sorted(unknown)}")
        profiles = [p for p in profiles if p["name"] in args.sources]
    summary = prepare_mixture(
        args.catalog_root,
        args.output_dir,
        profiles,
        val_fraction=args.val_fraction,
        seed=args.seed,
        allow_development=args.allow_development,
        max_train_episodes=args.max_train_episodes,
        max_val_episodes=args.max_val_episodes,
    )
    get_logger().info("Prepared native mixture: %s", summary)


if __name__ == "__main__":
    main()

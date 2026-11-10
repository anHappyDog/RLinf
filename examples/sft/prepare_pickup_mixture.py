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

from rlinf.data.datasets.openpi.manifest_index import build_manifest_indexes
from rlinf.data.datasets.openpi.mixture import prepare_mixture
from rlinf.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", "--dataset-root", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-index-only", action="store_true")
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
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument(
        "--sampling-unit", choices=("frame", "episode"), default="frame"
    )
    parser.add_argument("--statistics-max-anchors", type=int, default=8192)
    parser.add_argument("--parquet-threads", type=int, default=8)
    parser.add_argument("--target-recipe", choices=("realized_joint_trajectory",))
    parser.add_argument(
        "--source-weight-policy", choices=("molmobot_60_sqrt_episodes",)
    )
    args = parser.parse_args()

    if args.build_index_only:
        build_manifest_indexes(args.output_dir)
        return
    if args.catalog_root is None:
        parser.error("--catalog-root is required unless --build-index-only is used")

    import pyarrow as pa

    if args.parquet_threads < 1:
        raise ValueError("Parquet thread count must be positive")
    pa.set_cpu_count(args.parquet_threads)
    profiles = OmegaConf.to_container(OmegaConf.load(args.source_config).sources)
    if args.sources is not None:
        known = {profile["name"] for profile in profiles}
        unknown = set(args.sources) - known
        if unknown:
            raise ValueError(f"Unknown source profiles: {sorted(unknown)}")
        profiles = [p for p in profiles if p["name"] in args.sources]
    groups = {p["training_group"] for p in profiles if "training_group" in p}
    if len(groups) > 1:
        raise ValueError(
            "Native final sources have different action semantics; select sources "
            "from one training_group with --sources and prepare separate recipes"
        )
    summary = prepare_mixture(
        args.catalog_root,
        args.output_dir,
        profiles,
        val_fraction=args.val_fraction,
        seed=args.seed,
        allow_development=args.allow_development,
        max_train_episodes=args.max_train_episodes,
        max_val_episodes=args.max_val_episodes,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        sampling_unit=args.sampling_unit,
        statistics_max_anchors=args.statistics_max_anchors,
        target_recipe=args.target_recipe,
        source_weight_policy=args.source_weight_policy,
    )
    get_logger().info("Prepared native mixture: %s", summary)


if __name__ == "__main__":
    main()

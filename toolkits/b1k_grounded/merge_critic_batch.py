#!/usr/bin/env python3
"""Merge and validate actor-local critic batch shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlinf.utils.critic_batch import (
    atomic_torch_save,
    merge_critic_batch_shards,
    summarize_critic_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shard_dir",
        type=Path,
        help="Directory containing actor_rank_*.pt files for one global step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Merged output path (default: <shard_dir>/critic_batch_merged.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.shard_dir.glob("actor_rank_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No actor shards found in {args.shard_dir}.")

    shards = [
        torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths
    ]
    merged = merge_critic_batch_shards(shards)
    output = args.output or args.shard_dir / "critic_batch_merged.pt"
    atomic_torch_save(merged, output)

    summary = summarize_critic_batch(merged)
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

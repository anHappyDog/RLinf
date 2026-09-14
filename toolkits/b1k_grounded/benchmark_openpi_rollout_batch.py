#!/usr/bin/env python3
"""Measure OpenPI rollout latency across candidate dynamic batch sizes."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models import get_model

DEFAULT_PROMPT = (
    "<unused0> Turn on the radio receiver that's on the table in the living room. "
    "<unused1> pick up radio from coffee table <unused2> pick up from <unused3> "
    "<unused6> <unused12> radio <unused15> <unused18> <loc0833> <loc0543> "
    "<loc1023> <loc0683> <unused4> <unused3> <unused7> <unused12> coffee table "
    "<unused15> <unused18> <loc0708> <loc0493> <loc1023> <loc1023> <unused4> "
    "<unused22>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_observation(batch_size: int) -> dict[str, object]:
    """Create shape-faithful BEHAVIOR observations without changing model work."""
    return {
        "main_images": torch.zeros(batch_size, 720, 720, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(batch_size, 2, 480, 480, 3, dtype=torch.uint8),
        "states": torch.zeros(batch_size, 256, dtype=torch.float32),
        "task_descriptions": [DEFAULT_PROMPT] * batch_size,
    }


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    model_cfg = copy.deepcopy(cfg.actor.model)
    model_cfg.model_path = str(args.model_path)
    model_cfg.precision = "bf16"

    torch.manual_seed(args.seed)
    model = get_model(model_cfg)
    model.eval().cuda()

    results = []
    for batch_size in args.batch_sizes:
        obs = make_observation(batch_size)
        timings = []
        error = None
        try:
            for index in range(args.warmup + args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.predict_action_batch(obs, mode="train")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if index >= args.warmup:
                    timings.append(elapsed)
        except torch.OutOfMemoryError as exc:
            error = str(exc)
            torch.cuda.empty_cache()

        record = {
            "batch_size": batch_size,
            "latency_seconds": timings,
            "mean_seconds": statistics.mean(timings) if timings else None,
            "median_seconds": statistics.median(timings) if timings else None,
            "samples_per_second": (
                batch_size / statistics.mean(timings) if timings else None
            ),
            "error": error,
        }
        results.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "config": str(args.config),
        "model_path": str(args.model_path),
        "device": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

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

"""Benchmark real BEHAVIOR chunk stepping from a deterministic subpool state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
import ray
import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior.behavior_env import BehaviorSubpoolEnv
from rlinf.envs.behavior.subpool import SubpoolCatalog


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--subtask-id", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=0.0)
    parser.add_argument("--skip-intermediate-obs", action="store_true")
    return parser.parse_args()


def _compose_env_cfg(args: argparse.Namespace, asset_fingerprint: str):
    repo = Path(__file__).resolve().parents[2]
    config_dir = repo / "examples" / "embodiment" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    os.environ.setdefault("B1K_SUBPOOL_RESULT_DIR", str(args.report.parent))
    os.environ.setdefault("B1K_SUBPOOL_MODEL_PATH", "/unused-by-env-benchmark")
    os.environ.setdefault("B1K_SUBPOOL_MANIFEST", str(args.manifest))
    os.environ.setdefault("B1K_GROUNDED_TOKEN_MAPPING", str(args.token_mapping))
    os.environ.setdefault("B1K_ASSET_FINGERPRINT", asset_fingerprint)

    with hydra.initialize_config_dir(str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            "behavior_subpool_ppo_openpi_pi05",
            overrides=[
                "env.train.total_num_envs=1",
                f"env.train.subpool.manifest_path={args.manifest}",
                f"env.train.subpool.token_mapping_path={args.token_mapping}",
                f"env.train.subpool.asset_fingerprint={asset_fingerprint}",
                f"env.train.subpool.fixed_subtask_id={args.subtask_id}",
                "env.train.subpool.dynamic_updates=false",
                "env.train.auto_reset=false",
                "env.train.skip_intermediate_obs_in_chunk="
                f"{str(args.skip_intermediate_obs).lower()}",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg.env.train


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _image_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    """Run zero-action chunks and emit timings plus correctness fingerprints."""
    args = _parse_args()
    if args.chunk_size <= 0 or args.num_chunks <= 0:
        raise ValueError("chunk-size and num-chunks must be positive.")
    if args.seed_offset < 0:
        raise ValueError("seed-offset must be non-negative.")
    if args.action_scale < 0:
        raise ValueError("action-scale must be non-negative.")

    catalog = SubpoolCatalog.from_jsonl(args.manifest)
    records = [
        record
        for record in catalog.records
        if record.subtask_id == args.subtask_id and record.pool_type == "canonical"
    ]
    if not records:
        raise KeyError(f"No canonical snapshot for subtask_id={args.subtask_id}.")
    env_cfg = _compose_env_cfg(args, records[0].asset_fingerprint)

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    env = None
    try:
        start = time.perf_counter()
        env = BehaviorSubpoolEnv(
            env_cfg,
            num_envs=1,
            seed_offset=args.seed_offset,
            total_num_processes=1,
            worker_info=SimpleNamespace(group_world_size=1),
        )
        init_seconds = time.perf_counter() - start

        start = time.perf_counter()
        initial_obs, _ = env.reset()
        reset_seconds = time.perf_counter() - start
        snapshot = env.current_snapshot

        generator = torch.Generator().manual_seed(args.action_seed)
        actions = (
            torch.rand(
                args.num_chunks,
                1,
                args.chunk_size,
                23,
                generator=generator,
            )
            * 2
            - 1
        ) * args.action_scale
        chunk_seconds = []
        executed_actions = 0
        reward_rows = []
        termination_rows = []
        truncation_rows = []
        final_obs = initial_obs
        for chunk_index in range(args.num_chunks):
            start = time.perf_counter()
            observations, rewards, terminations, truncations, _ = env.chunk_step(
                actions[chunk_index]
            )
            chunk_seconds.append(time.perf_counter() - start)
            executed_actions += int(env.last_executed_action_mask.sum().item())
            reward_rows.extend(_tensor_list(rewards))
            termination_rows.extend(_tensor_list(terminations))
            truncation_rows.extend(_tensor_list(truncations))
            final_obs = observations[-1]
            if bool((terminations | truncations).any()):
                break

        runtime_seconds = sum(chunk_seconds)
        report = {
            "skip_intermediate_obs": args.skip_intermediate_obs,
            "snapshot_id": snapshot.snapshot_id,
            "episode_index": snapshot.episode_index,
            "seed_offset": args.seed_offset,
            "action_seed": args.action_seed,
            "action_scale": args.action_scale,
            "action_sha256": _image_digest(actions),
            "chunk_size": args.chunk_size,
            "requested_chunks": args.num_chunks,
            "completed_chunks": len(chunk_seconds),
            "executed_actions": executed_actions,
            "init_seconds": init_seconds,
            "reset_seconds": reset_seconds,
            "chunk_seconds": chunk_seconds,
            "runtime_seconds": runtime_seconds,
            "actions_per_second": executed_actions / runtime_seconds,
            "rewards": reward_rows,
            "terminations": termination_rows,
            "truncations": truncation_rows,
            "final_state": _tensor_list(final_obs["states"]),
            "main_image_shape": list(final_obs["main_images"].shape),
            "wrist_image_shape": list(final_obs["wrist_images"].shape),
            "main_image_sha256": _image_digest(final_obs["main_images"]),
            "wrist_image_sha256": _image_digest(final_obs["wrist_images"]),
            "task_descriptions": final_obs["task_descriptions"],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if env is not None:
            env.close()
        ray.shutdown()


if __name__ == "__main__":
    main()

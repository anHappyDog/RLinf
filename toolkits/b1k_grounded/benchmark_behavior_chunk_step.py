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
from rlinf.envs.behavior.subpool import SubpoolCatalog, full_state_sha256

# B1K 2025 explicitly marks this R1Pro observation field as invalid replay data.
_R1PRO_JOINT_EFFORT_SLICE = slice(112, 140)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--subtask-id", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=0.0)
    parser.add_argument("--num-resets", type=int, default=1)
    parser.add_argument("--state-cache-size", type=int, default=0)
    parser.add_argument("--skip-intermediate-obs", action="store_true")
    parser.add_argument("--skip-official-task-termination", action="store_true")
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
                f"env.train.total_num_envs={args.num_envs}",
                f"env.train.subpool.manifest_path={args.manifest}",
                f"env.train.subpool.token_mapping_path={args.token_mapping}",
                f"env.train.subpool.asset_fingerprint={asset_fingerprint}",
                f"env.train.subpool.fixed_subtask_id={args.subtask_id}",
                f"env.train.subpool.outcome_group_size={args.num_envs}",
                "env.train.subpool.dynamic_updates=false",
                f"env.train.subpool.state_cache_size={args.state_cache_size}",
                "env.train.subpool.fixed_snapshot_per_env=true",
                "env.train.subpool.skip_official_task_termination="
                f"{str(args.skip_official_task_termination).lower()}",
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


def _difference_from_first(value: torch.Tensor) -> list[dict[str, float | int]]:
    """Summarize each vector slot against slot zero without storing pixels."""
    reference = value[0].detach().cpu().to(torch.float64)
    summaries = []
    for slot_index, candidate in enumerate(value):
        difference = (candidate.detach().cpu().to(torch.float64) - reference).abs()
        summaries.append(
            {
                "slot": slot_index,
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "different_values": int(torch.count_nonzero(difference).item()),
            }
        )
    return summaries


def _without_joint_effort(value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            value[..., : _R1PRO_JOINT_EFFORT_SLICE.start],
            value[..., _R1PRO_JOINT_EFFORT_SLICE.stop :],
        ),
        dim=-1,
    )


def main() -> None:
    """Run zero-action chunks and emit timings plus correctness fingerprints."""
    args = _parse_args()
    if (
        args.num_envs <= 0
        or args.chunk_size <= 0
        or args.num_chunks <= 0
        or args.num_resets <= 0
    ):
        raise ValueError(
            "num-envs, chunk-size, num-chunks, and num-resets must be positive."
        )
    if args.seed_offset < 0:
        raise ValueError("seed-offset must be non-negative.")
    if args.action_scale < 0:
        raise ValueError("action-scale must be non-negative.")
    if args.state_cache_size < 0:
        raise ValueError("state-cache-size must be non-negative.")

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
    ray_options = {}
    if ray_temp_dir := os.environ.get("RAY_TMPDIR"):
        ray_options["_temp_dir"] = ray_temp_dir
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        **ray_options,
    )
    env = None
    try:
        start = time.perf_counter()
        env = BehaviorSubpoolEnv(
            env_cfg,
            num_envs=args.num_envs,
            seed_offset=args.seed_offset,
            total_num_processes=1,
            worker_info=SimpleNamespace(group_world_size=1),
        )
        init_seconds = time.perf_counter() - start

        reset_seconds_all = []
        reset_snapshot_ids = []
        reset_main_image_sha256 = []
        reset_policy_observations = []
        for _ in range(args.num_resets):
            start = time.perf_counter()
            initial_obs, _ = env.reset()
            reset_seconds_all.append(time.perf_counter() - start)
            reset_snapshot_ids.append(
                [snapshot.snapshot_id for snapshot in env.current_snapshots]
            )
            reset_main_image_sha256.append(
                [_image_digest(image) for image in initial_obs["main_images"]]
            )
            reset_policy_observations.append(
                {
                    key: initial_obs[key].detach().cpu().clone()
                    for key in ("states", "main_images", "wrist_images")
                }
            )
        reset_seconds = reset_seconds_all[-1]
        snapshots = env.current_snapshots

        generator = torch.Generator().manual_seed(args.action_seed)
        actions = (
            torch.rand(
                args.num_chunks,
                args.num_envs,
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
        simulator_state = ray.get(
            env.pool.env_processes[0].dump_serialized_state.remote()
        )
        canonical_states = ray.get(
            env.pool.env_processes[0].dump_subpool_states.remote()
        )
        report = {
            "num_envs": args.num_envs,
            "skip_intermediate_obs": args.skip_intermediate_obs,
            "skip_official_task_termination": (args.skip_official_task_termination),
            "state_cache_size": args.state_cache_size,
            "state_cache_info": dict(env.catalog.state_cache_info),
            "snapshot_ids": [snapshot.snapshot_id for snapshot in snapshots],
            "episode_indices": [snapshot.episode_index for snapshot in snapshots],
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
            "reset_seconds_all": reset_seconds_all,
            "reset_snapshot_ids": reset_snapshot_ids,
            "reset_main_image_sha256": reset_main_image_sha256,
            "slot_zero_reset_state_difference_from_first": _difference_from_first(
                torch.stack([obs["states"][0] for obs in reset_policy_observations])
            ),
            "slot_zero_reset_main_image_difference_from_first": (
                _difference_from_first(
                    torch.stack(
                        [obs["main_images"][0] for obs in reset_policy_observations]
                    )
                )
            ),
            "slot_zero_reset_wrist_image_difference_from_first": (
                _difference_from_first(
                    torch.stack(
                        [obs["wrist_images"][0] for obs in reset_policy_observations]
                    )
                )
            ),
            "initial_state_difference_from_slot_zero": _difference_from_first(
                initial_obs["states"]
            ),
            "initial_valid_state_difference_from_slot_zero": (
                _difference_from_first(_without_joint_effort(initial_obs["states"]))
            ),
            "initial_main_image_difference_from_slot_zero": _difference_from_first(
                initial_obs["main_images"]
            ),
            "initial_wrist_image_difference_from_slot_zero": _difference_from_first(
                initial_obs["wrist_images"]
            ),
            "chunk_seconds": chunk_seconds,
            "runtime_seconds": runtime_seconds,
            "actions_per_second": executed_actions / runtime_seconds,
            "environment_steps_per_second": executed_actions / runtime_seconds,
            "physics_steps_per_second": (
                executed_actions / args.num_envs / runtime_seconds
            ),
            "simulator_state": _tensor_list(simulator_state),
            "simulator_state_shape": list(simulator_state.shape),
            "simulator_state_sha256": _image_digest(simulator_state),
            "canonical_state_sha256": [
                full_state_sha256(state) for state in canonical_states
            ],
            "rewards": reward_rows,
            "terminations": termination_rows,
            "truncations": truncation_rows,
            "final_state": _tensor_list(final_obs["states"]),
            "final_state_difference_from_slot_zero": _difference_from_first(
                final_obs["states"]
            ),
            "final_valid_state_difference_from_slot_zero": _difference_from_first(
                _without_joint_effort(final_obs["states"])
            ),
            "final_main_image_difference_from_slot_zero": _difference_from_first(
                final_obs["main_images"]
            ),
            "final_wrist_image_difference_from_slot_zero": _difference_from_first(
                final_obs["wrist_images"]
            ),
            "main_image_shape": list(final_obs["main_images"].shape),
            "wrist_image_shape": list(final_obs["wrist_images"].shape),
            "main_image_sha256": [
                _image_digest(image) for image in final_obs["main_images"]
            ],
            "wrist_image_sha256": [
                _image_digest(image) for image in final_obs["wrist_images"]
            ],
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

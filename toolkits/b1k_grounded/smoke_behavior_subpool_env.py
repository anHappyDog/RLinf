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

"""Exercise a real ``BehaviorSubpoolEnv`` restore, terminal, and freeze cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import hydra
import ray
import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior.behavior_env import BehaviorSubpoolEnv
from rlinf.envs.behavior.subpool import (
    SubpoolCatalog,
    SubpoolStore,
    full_state_sha256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subtask-id", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--reset-count", type=int, default=32)
    parser.add_argument("--skip-intermediate-obs", action="store_true")
    parser.add_argument("--skip-official-task-termination", action="store_true")
    parser.add_argument("--require-all-snapshots", action="store_true")
    parser.add_argument("--verify-failure-state-save", action="store_true")
    parser.add_argument("--verify-dynamic-updates", action="store_true")
    parser.add_argument("--verify-pickup-progress-reward", action="store_true")
    parser.add_argument("--staggered-timeouts", action="store_true")
    return parser.parse_args()


def _make_one_step_catalog(args: argparse.Namespace):
    source = SubpoolCatalog.from_jsonl(args.manifest)
    selected = [
        record
        for record in source.records
        if record.subtask_id == args.subtask_id and record.pool_type == "canonical"
    ]
    if not selected:
        raise KeyError(f"No canonical snapshot for subtask_id={args.subtask_id}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    if manifest.exists() and manifest.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite smoke catalog {manifest}.")

    if args.staggered_timeouts:
        selected = [selected[0]] * args.num_envs

    smoke_records = []
    store = SubpoolStore(manifest)
    for record_index, source_record in enumerate(selected):
        state = source.load_state(source_record)
        reward = dict(source_record.metadata["reward"])
        step_penalty_budget = reward["step_penalty"] * reward["max_steps"]
        timeout_steps = (
            record_index + 1
            if args.staggered_timeouts
            else 4
            if args.verify_dynamic_updates
            else 1
        )
        reward["max_steps"] = timeout_steps
        reward["step_penalty"] = step_penalty_budget / timeout_steps
        metadata = dict(source_record.metadata)
        metadata["reward"] = reward
        metadata["smoke_only"] = True
        snapshot_id = f"smoke-{source_record.snapshot_id}"
        if args.staggered_timeouts:
            snapshot_id = f"{snapshot_id}-timeout{timeout_steps:02d}"
        smoke_record = replace(
            source_record,
            snapshot_id=snapshot_id,
            state_path=f"states/{snapshot_id}.pt",
            state_sha256=full_state_sha256(state),
            metadata=metadata,
        )
        store.append(smoke_record, state)
        smoke_records.append(smoke_record)
    return manifest, smoke_records


def _compose_env_cfg(args: argparse.Namespace, manifest: Path, record):
    repo = Path(__file__).resolve().parents[2]
    config_dir = repo / "examples" / "embodiment" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    os.environ.setdefault("B1K_SUBPOOL_RESULT_DIR", str(args.output_dir))
    os.environ.setdefault("B1K_SUBPOOL_MODEL_PATH", "/unused-by-env-smoke")
    os.environ.setdefault("B1K_SUBPOOL_MANIFEST", str(manifest))
    os.environ.setdefault("B1K_GROUNDED_TOKEN_MAPPING", str(args.token_mapping))
    os.environ.setdefault("B1K_ASSET_FINGERPRINT", record.asset_fingerprint)

    overrides = [
        f"env.train.total_num_envs={args.num_envs}",
        f"env.train.subpool.manifest_path={manifest}",
        f"env.train.subpool.token_mapping_path={args.token_mapping}",
        f"env.train.subpool.asset_fingerprint={record.asset_fingerprint}",
        f"env.train.subpool.fixed_subtask_id={record.subtask_id}",
        f"env.train.subpool.outcome_group_size={args.num_envs}",
        f"env.train.subpool.dynamic_updates={str(args.verify_dynamic_updates).lower()}",
        "env.train.subpool.state_capture_interval=1",
        "env.train.subpool.recovery_min_lag_states=1",
        "env.train.subpool.recovery_max_lag_states=2",
        # This smoke test checks the explicit frozen-terminal contract.
        # Training enables auto-reset, which intentionally starts a new
        # episode on the next chunk instead of returning frozen output.
        "env.train.auto_reset=false",
        "env.train.skip_intermediate_obs_in_chunk="
        f"{str(args.skip_intermediate_obs).lower()}",
        "env.train.subpool.skip_official_task_termination="
        f"{str(args.skip_official_task_termination).lower()}",
        "env.train.subpool.failure_state_capture.enabled="
        f"{str(args.verify_failure_state_save).lower()}",
        "env.train.subpool.failure_state_capture.output_dir="
        f"{args.output_dir / 'failure_states'}",
        "env.train.subpool.failure_state_capture.run_id=vector-smoke",
        "env.train.subpool.failure_state_capture.policy_global_step=0",
    ]
    if args.verify_pickup_progress_reward:
        overrides.extend(
            [
                "+env.train.subpool.reward_overrides.step_penalty=0.0",
                "+env.train.subpool.reward_overrides.potential_terms="
                "[{key:pickup_progress_score,scale:2.0,direction:increase}]",
                "+env.train.subpool.reward_overrides.progress_clip=2.0",
            ]
        )
    if args.staggered_timeouts:
        overrides.extend(
            [
                "env.train.subpool.outcome_snapshot_schedule=shuffled_round_robin",
                "env.train.subpool.pool_weights.canonical=1.0",
                "env.train.subpool.pool_weights.predecessor_success=0.0",
                "env.train.subpool.pool_weights.recovery=0.0",
            ]
        )

    with hydra.initialize_config_dir(str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            "behavior_subpool_ppo_openpi_pi05",
            overrides=overrides,
        )
    OmegaConf.resolve(cfg)
    return cfg.env.train


def _same_observation(left: dict, right: dict) -> bool:
    return (
        all(
            torch.equal(left[key], right[key])
            for key in ("main_images", "wrist_images", "states")
        )
        and left["task_descriptions"] == right["task_descriptions"]
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as state_file:
        for chunk in iter(lambda: state_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _policy_state(proprio: torch.Tensor) -> torch.Tensor:
    """Extract the exact 23-D R1Pro state consumed by the pi0.5 policy."""
    return torch.cat(
        (
            proprio[..., 253:256],
            proprio[..., 236:240],
            proprio[..., 158:165],
            proprio[..., 197:204],
            proprio[..., 193:195].sum(dim=-1, keepdim=True),
            proprio[..., 232:234].sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )


def _difference_from_first(value: torch.Tensor) -> list[dict[str, float | int]]:
    reference = value[0].detach().cpu().to(torch.float64)
    summaries = []
    for index, candidate in enumerate(value):
        difference = (candidate.detach().cpu().to(torch.float64) - reference).abs()
        summaries.append(
            {
                "index": index,
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "different_values": int(torch.count_nonzero(difference).item()),
            }
        )
    return summaries


def main() -> None:
    """Run the one-step timeout and post-terminal freeze checks."""
    args = _parse_args()
    if args.chunk_size <= 1:
        raise ValueError("chunk-size must exceed one to test prefix masking.")
    if args.staggered_timeouts and args.verify_dynamic_updates:
        raise ValueError(
            "staggered-timeouts and verify-dynamic-updates are separate smoke "
            "contracts and cannot be combined."
        )
    timeout_steps = (
        args.num_envs
        if args.staggered_timeouts
        else 4
        if args.verify_dynamic_updates
        else 1
    )
    if args.chunk_size <= timeout_steps:
        raise ValueError(
            f"chunk-size must exceed the smoke timeout of {timeout_steps}."
        )
    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive.")
    if args.reset_count <= 0:
        raise ValueError("reset-count must be positive.")
    manifest, records = _make_one_step_catalog(args)
    env_cfg = _compose_env_cfg(args, manifest, records[0])

    # The nested BehaviorProcess shares the driver's selected rendering GPU but
    # intentionally requests no additional Ray GPU resource. Without this flag,
    # Ray masks CUDA_VISIBLE_DEVICES for the child and Isaac Sim exits at startup.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    env = None
    try:
        env = BehaviorSubpoolEnv(
            env_cfg,
            num_envs=args.num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=SimpleNamespace(group_world_size=1),
        )
        actions = torch.zeros(args.num_envs, args.chunk_size, 23)
        reset_results = []
        reset_policy_states = []
        for reset_index in range(args.reset_count):
            if args.staggered_timeouts:
                env.prepare_outcome_group_reset(
                    reset_index,
                    logical_group_index=list(range(args.num_envs)),
                    update_index=reset_index,
                )
            initial_obs, _ = env.reset()
            if not all(initial_obs["task_descriptions"]):
                raise AssertionError(
                    "Online P2 task description is empty after restore."
                )
            policy_states = _policy_state(initial_obs["states"])
            reset_policy_states.append(policy_states.detach().cpu().clone())

            terminal_obs, rewards, terminations, truncations, _ = env.chunk_step(
                actions
            )
            first_mask = env.last_executed_action_mask.clone()
            slot_timeout_steps = [
                int(snapshot.metadata["reward"]["max_steps"])
                for snapshot in env.current_snapshots
            ]
            expected_mask = [
                [True] * slot_steps
                + [False] * (args.chunk_size - slot_steps)
                for slot_steps in slot_timeout_steps
            ]
            if first_mask.tolist() != expected_mask:
                raise AssertionError(f"Unexpected terminal prefix mask: {first_mask}.")
            done_flags = terminations | truncations
            for env_index, slot_steps in enumerate(slot_timeout_steps):
                if not bool(done_flags[env_index, slot_steps - 1]):
                    raise AssertionError(
                        f"Slot {env_index} did not terminate at step {slot_steps}."
                    )
                if bool(done_flags[env_index, slot_steps:].any()):
                    raise AssertionError(
                        f"Slot {env_index} has terminal flags after its valid prefix."
                    )

            frozen_obs, frozen_rewards, frozen_terms, frozen_truncs, _ = env.chunk_step(
                actions
            )
            frozen_mask = env.last_executed_action_mask.clone()
            if bool(frozen_mask.any()):
                raise AssertionError("Post-terminal chunk executed simulator actions.")
            if bool(frozen_rewards.any() or frozen_terms.any() or frozen_truncs.any()):
                raise AssertionError(
                    "Post-terminal chunk changed rewards or done flags."
                )
            if not _same_observation(terminal_obs[-1], frozen_obs[-1]):
                raise AssertionError(
                    "Post-terminal observation was not frozen exactly."
                )

            reset_results.append(
                {
                    "snapshot_ids": [
                        snapshot.snapshot_id for snapshot in env.current_snapshots
                    ],
                    "episode_indices": [
                        snapshot.episode_index for snapshot in env.current_snapshots
                    ],
                    "first_chunk_executed_mask": first_mask.tolist(),
                    "first_chunk_rewards": rewards.tolist(),
                    "first_chunk_terminations": terminations.tolist(),
                    "first_chunk_truncations": truncations.tolist(),
                    "slot_timeout_steps": slot_timeout_steps,
                    "frozen_chunk_executed_mask": frozen_mask.tolist(),
                    "online_prompts": initial_obs["task_descriptions"],
                    "policy_state_sha256": [
                        _tensor_sha256(value) for value in policy_states
                    ],
                    "main_image_sha256": [
                        _tensor_sha256(value) for value in initial_obs["main_images"]
                    ],
                    "wrist_image_sha256": [
                        _tensor_sha256(value) for value in initial_obs["wrist_images"]
                    ],
                    "policy_state_difference_from_slot_zero": (
                        _difference_from_first(policy_states)
                    ),
                }
            )

        expected_snapshot_ids = {record.snapshot_id for record in records}
        sampled_snapshot_ids = {
            snapshot_id
            for result in reset_results
            for snapshot_id in result["snapshot_ids"]
        }
        if args.require_all_snapshots and sampled_snapshot_ids != expected_snapshot_ids:
            missing = sorted(expected_snapshot_ids - sampled_snapshot_ids)
            raise AssertionError(
                f"Smoke did not sample every snapshot; missing={missing}."
            )

        failure_metadata = []
        if args.verify_failure_state_save:
            metadata_paths = sorted(
                (args.output_dir / "failure_states").rglob("*.json")
            )
            expected_count = args.reset_count * args.num_envs
            if len(metadata_paths) != expected_count:
                raise AssertionError(
                    f"Expected {expected_count} terminal failure states, found "
                    f"{len(metadata_paths)}."
                )
            for metadata_path in metadata_paths:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                state_path = metadata_path.parent / metadata["state_path"]
                if _file_sha256(state_path) != metadata["state_sha256"]:
                    raise AssertionError(
                        f"Failure state checksum mismatch for {state_path}."
                    )
                torch.load(state_path, map_location="cpu", weights_only=False)
                failure_metadata.append(str(metadata_path))

        dynamic_snapshot_ids = []
        if args.verify_dynamic_updates:
            refreshed_catalog = SubpoolCatalog.from_jsonl(manifest)
            dynamic_records = [
                record
                for record in refreshed_catalog.records
                if record.snapshot_id.startswith("online-")
                and record.pool_type == "recovery"
            ]
            expected_count = args.reset_count * args.num_envs
            if len(dynamic_records) != expected_count:
                raise AssertionError(
                    f"Expected {expected_count} dynamic recovery states, found "
                    f"{len(dynamic_records)}."
                )
            for record in dynamic_records:
                state_path = refreshed_catalog.state_path(record)
                if _file_sha256(state_path) != record.state_sha256:
                    raise AssertionError(
                        f"Dynamic state checksum mismatch for {record.snapshot_id}."
                    )
                refreshed_catalog.load_state(record)
            dynamic_snapshot_ids = [record.snapshot_id for record in dynamic_records]

        report = {
            "passed": True,
            "num_envs": args.num_envs,
            "subtask_id": records[0].subtask_id,
            "skill": records[0].skill,
            "source_snapshot_count": len(records),
            "reset_count": args.reset_count,
            "skip_intermediate_obs": args.skip_intermediate_obs,
            "skip_official_task_termination": (args.skip_official_task_termination),
            "sampled_snapshot_ids": sorted(sampled_snapshot_ids),
            "sampled_episode_indices": sorted(
                {
                    episode_index
                    for result in reset_results
                    for episode_index in result["episode_indices"]
                    if episode_index is not None
                }
            ),
            "all_snapshots_sampled": sampled_snapshot_ids == expected_snapshot_ids,
            "failure_state_save_verified": args.verify_failure_state_save,
            "failure_state_metadata": failure_metadata,
            "dynamic_updates_verified": args.verify_dynamic_updates,
            "pickup_progress_reward_verified": (args.verify_pickup_progress_reward),
            "staggered_timeouts_verified": args.staggered_timeouts,
            "dynamic_snapshot_ids": dynamic_snapshot_ids,
            "slot_zero_policy_state_difference_across_resets": (
                _difference_from_first(
                    torch.stack([states[0] for states in reset_policy_states])
                )
            ),
            "reset_results": reset_results,
        }
        report_path = args.output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if env is not None:
            env.close()
        ray.shutdown()


if __name__ == "__main__":
    main()

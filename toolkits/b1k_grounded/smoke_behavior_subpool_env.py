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
    parser.add_argument("--chunk-size", type=int, default=8)
    return parser.parse_args()


def _make_one_step_catalog(args: argparse.Namespace):
    source = SubpoolCatalog.from_jsonl(args.manifest)
    selected = next(
        (
            record
            for record in source.records
            if record.subtask_id == args.subtask_id and record.pool_type == "canonical"
        ),
        None,
    )
    if selected is None:
        raise KeyError(f"No canonical snapshot for subtask_id={args.subtask_id}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    if manifest.exists() and manifest.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite smoke catalog {manifest}.")

    state = source.load_state(selected)
    reward = dict(selected.metadata["reward"])
    reward["max_steps"] = 1
    metadata = dict(selected.metadata)
    metadata["reward"] = reward
    metadata["smoke_only"] = True
    snapshot_id = f"smoke-{selected.snapshot_id}"
    record = replace(
        selected,
        snapshot_id=snapshot_id,
        state_path=f"states/{snapshot_id}.pt",
        state_sha256=full_state_sha256(state),
        metadata=metadata,
    )
    SubpoolStore(manifest).append(record, state)
    return manifest, record


def _compose_env_cfg(args: argparse.Namespace, manifest: Path, record):
    repo = Path(__file__).resolve().parents[2]
    config_dir = repo / "examples" / "embodiment" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    os.environ.setdefault("B1K_SUBPOOL_RESULT_DIR", str(args.output_dir))
    os.environ.setdefault("B1K_SUBPOOL_MODEL_PATH", "/unused-by-env-smoke")
    os.environ.setdefault("B1K_SUBPOOL_MANIFEST", str(manifest))
    os.environ.setdefault("B1K_GROUNDED_TOKEN_MAPPING", str(args.token_mapping))
    os.environ.setdefault("B1K_ASSET_FINGERPRINT", record.asset_fingerprint)

    with hydra.initialize_config_dir(str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            "behavior_subpool_ppo_openpi_pi05",
            overrides=[
                "env.train.total_num_envs=1",
                f"env.train.subpool.manifest_path={manifest}",
                f"env.train.subpool.token_mapping_path={args.token_mapping}",
                f"env.train.subpool.asset_fingerprint={record.asset_fingerprint}",
                f"env.train.subpool.fixed_subtask_id={record.subtask_id}",
                "env.train.subpool.dynamic_updates=false",
            ],
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


def main() -> None:
    """Run the one-step timeout and post-terminal freeze checks."""
    args = _parse_args()
    if args.chunk_size <= 1:
        raise ValueError("chunk-size must exceed one to test prefix masking.")
    manifest, record = _make_one_step_catalog(args)
    env_cfg = _compose_env_cfg(args, manifest, record)

    # The nested BehaviorProcess shares the driver's selected rendering GPU but
    # intentionally requests no additional Ray GPU resource. Without this flag,
    # Ray masks CUDA_VISIBLE_DEVICES for the child and Isaac Sim exits at startup.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    env = None
    try:
        env = BehaviorSubpoolEnv(
            env_cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=SimpleNamespace(group_world_size=1),
        )
        initial_obs, _ = env.reset()
        if not initial_obs["task_descriptions"][0]:
            raise AssertionError("Online P2 task description is empty after restore.")

        actions = torch.zeros(1, args.chunk_size, 23)
        terminal_obs, rewards, terminations, truncations, _ = env.chunk_step(actions)
        first_mask = env.last_executed_action_mask.clone()
        if first_mask.tolist() != [[True] + [False] * (args.chunk_size - 1)]:
            raise AssertionError(f"Unexpected terminal prefix mask: {first_mask}.")
        if not bool((terminations | truncations)[0, 0]):
            raise AssertionError("The one-step smoke subtask did not terminate.")
        if bool((terminations | truncations)[0, 1:].any()):
            raise AssertionError("Unexecuted chunk suffix contains terminal flags.")

        frozen_obs, frozen_rewards, frozen_terms, frozen_truncs, _ = env.chunk_step(
            actions
        )
        frozen_mask = env.last_executed_action_mask.clone()
        if bool(frozen_mask.any()):
            raise AssertionError("Post-terminal chunk executed simulator actions.")
        if bool(frozen_rewards.any() or frozen_terms.any() or frozen_truncs.any()):
            raise AssertionError("Post-terminal chunk changed rewards or done flags.")
        if not _same_observation(terminal_obs[-1], frozen_obs[-1]):
            raise AssertionError("Post-terminal observation was not frozen exactly.")

        report = {
            "passed": True,
            "snapshot_id": record.snapshot_id,
            "subtask_id": record.subtask_id,
            "skill": record.skill,
            "first_chunk_executed_mask": first_mask.tolist(),
            "first_chunk_rewards": rewards.tolist(),
            "first_chunk_terminations": terminations.tolist(),
            "first_chunk_truncations": truncations.tolist(),
            "frozen_chunk_executed_mask": frozen_mask.tolist(),
            "online_prompt": initial_obs["task_descriptions"][0],
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

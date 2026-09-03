#!/usr/bin/env python3
"""Launch a short-memory pickup eval from raw demonstration states."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

from rlinf.envs.behavior.demonstration_reset import read_demonstration_instance_id
from toolkits.mem.grasp_event_types import GraspEventSample
from toolkits.mem.short_memory_paired_eval import parse_int_list

BEHAVIOR_ACTION_FREQUENCY_HZ = 30
BEHAVIOR_PHYSICS_FREQUENCY_HZ = 120


def build_command(
    *,
    python: str,
    repo_root: Path,
    checkpoint: str,
    norm_stats_path: str,
    instance_dir: str,
    instance_ids: list[int],
    demonstration_paths: list[str],
    frame_indices: list[int],
    output_dir: Path,
    seed: int,
    max_episode_steps: int,
    valid_history_lengths: list[int] | None = None,
    observation_override_path: Path | None = None,
    model_action_override_path: Path | None = None,
) -> list[str]:
    """Build one reproducible mid-stage reset evaluation command."""
    count = len(instance_ids)
    if len(demonstration_paths) != count or len(frame_indices) != count:
        raise ValueError(
            "Instance ids, demonstration paths, and frame indices must align."
        )
    if count == 0:
        raise ValueError("At least one mid-stage reset is required.")
    if any("," in path or "[" in path or "]" in path for path in demonstration_paths):
        raise ValueError("Demonstration paths cannot contain commas or brackets.")

    instance_list = ",".join(str(value) for value in instance_ids)
    path_list = ",".join(demonstration_paths)
    frame_list = ",".join(str(value) for value in frame_indices)
    command = [
        python,
        str(repo_root / "evaluations/eval_embodied_agent.py"),
        "--config-path",
        str(repo_root / "evaluations/behavior"),
        "--config-name",
        "behavior_pi05_short_memory_eval",
        f"runner.logger.log_path={output_dir}",
        f"runner.raw_metrics_path={output_dir / 'raw_metrics.json'}",
        f"rollout.model.model_path={checkpoint}",
        f"rollout.model.openpi_data.norm_stats_path={norm_stats_path}",
        f"rollout.seed={seed}",
        f"env.eval.seed={seed}",
        f"env.eval.total_num_envs={count}",
        f"env.eval.max_episode_steps={max_episode_steps}",
        f"env.eval.max_steps_per_rollout_epoch={max_episode_steps}",
        "env.eval.skip_intermediate_obs_in_chunk=false",
        f"env.eval.omni_config.env.action_frequency={BEHAVIOR_ACTION_FREQUENCY_HZ}",
        f"env.eval.omni_config.env.rendering_frequency={BEHAVIOR_ACTION_FREQUENCY_HZ}",
        f"env.eval.omni_config.env.physics_frequency={BEHAVIOR_PHYSICS_FREQUENCY_HZ}",
        f"env.eval.video_cfg.fps={BEHAVIOR_ACTION_FREQUENCY_HZ}",
        f"env.eval.activity_instance_ids=[{instance_list}]",
        f"env.eval.demonstration_reset_paths=[{path_list}]",
        f"env.eval.demonstration_reset_frame_indices=[{frame_list}]",
        "env.eval.demonstration_reset_history_stride=30",
        "env.eval.demonstration_reset_stage=pickup_from_support",
        "env.eval.prompt_mode=oracle_stage",
        "env.eval.oracle_initial_stage=pickup_from_support",
        "env.eval.history_ablation=none",
        f"env.eval.decision_trace_dir={output_dir / 'decision_trace'}",
        f"rollout.model.openpi.eval_snapshot_dir={output_dir / 'eval_snapshot'}",
        f"env.eval.omni_config.task.activity_instance_dir={instance_dir}",
    ]
    if valid_history_lengths is not None:
        if len(valid_history_lengths) != count:
            raise ValueError(
                "Valid history lengths and demonstration paths must align."
            )
        if any(value < 1 or value > 6 for value in valid_history_lengths):
            raise ValueError("Valid history lengths must be within [1, 6].")
        valid_list = ",".join(str(value) for value in valid_history_lengths)
        command.append(
            f"env.eval.demonstration_reset_valid_history_lengths=[{valid_list}]"
        )
    if observation_override_path is not None:
        command.append(
            "rollout.model.openpi.eval_observation_override_path="
            f"{observation_override_path}"
        )
    if model_action_override_path is not None:
        command.append(
            "rollout.model.openpi.eval_model_action_override_path="
            f"{model_action_override_path}"
        )
    return command


def write_observation_overrides(
    *,
    selection_cache: str | Path,
    episode_indices: list[int],
    frame_indices: list[int],
    output_path: Path,
) -> None:
    """Export ranked, tensor-only normalized observations from an event cache."""
    samples: list[GraspEventSample] = torch.load(
        Path(selection_cache).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    by_frame = {
        (sample.episode_index, sample.frame_index): sample for sample in samples
    }
    payload = {}
    for rank, key in enumerate(zip(episode_indices, frame_indices, strict=True)):
        if key not in by_frame:
            raise KeyError(
                f"Selection cache has no observation for episode/frame {key}."
            )
        observation = by_frame[key].observation.to_dict()
        payload[f"rank_{rank}"] = tree_map(
            lambda value: value.detach().cpu() if torch.is_tensor(value) else value,
            observation,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def write_model_action_overrides(
    *,
    selection_cache: str | Path,
    episode_indices: list[int],
    frame_indices: list[int],
    output_path: Path,
) -> None:
    """Export ranked normalized action chunks from an event cache."""
    samples: list[GraspEventSample] = torch.load(
        Path(selection_cache).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    by_frame = {
        (sample.episode_index, sample.frame_index): sample for sample in samples
    }
    payload = {}
    for rank, key in enumerate(zip(episode_indices, frame_indices, strict=True)):
        if key not in by_frame:
            raise KeyError(f"Selection cache has no actions for episode/frame {key}.")
        payload[f"rank_{rank}"] = by_frame[key].actions.detach().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    """Write a manifest and optionally execute the mid-stage evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument(
        "--instance-ids", type=parse_int_list, default=parse_int_list("0,0,0,0")
    )
    parser.add_argument(
        "--episode-indices",
        type=parse_int_list,
        default=parse_int_list("10,20,30,40"),
    )
    parser.add_argument(
        "--frame-indices",
        type=parse_int_list,
        default=parse_int_list("1064,1133,898,1030"),
    )
    parser.add_argument("--max-episode-steps", type=int, default=512)
    parser.add_argument(
        "--valid-history-lengths",
        type=parse_int_list,
        help="Optional per-episode valid K-frame counts for loader-parity tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-cache",
        help=(
            "Optional grasp-event selection_samples.pt. Matching normalized "
            "observations replace each worker's first live model input."
        ),
    )
    parser.add_argument(
        "--oracle-actions",
        action="store_true",
        help="Replay each selected sample's normalized action chunk once.",
    )
    parser.add_argument(
        "--python", default="/mnt/public/daibo/venv/behavior_openpi/bin/python"
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    if len(args.episode_indices) != len(args.instance_ids):
        raise ValueError("Episode indices and instance ids must align.")
    demonstration_paths = [
        str(raw_data_dir / f"episode_{episode_index:08d}.hdf5")
        for episode_index in args.episode_indices
    ]
    missing = [path for path in demonstration_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw demonstration files: {missing}")
    recorded_instance_ids = [
        read_demonstration_instance_id(path) for path in demonstration_paths
    ]
    if recorded_instance_ids != args.instance_ids:
        raise ValueError(
            "Raw demonstration instance ids do not match --instance-ids: "
            f"recorded={recorded_instance_ids}, requested={args.instance_ids}."
        )

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_override_path = None
    model_action_override_path = None
    if args.oracle_actions and not args.selection_cache:
        raise ValueError("--oracle-actions requires --selection-cache.")
    if args.selection_cache:
        observation_override_path = output_dir / "exact_observation_overrides.pt"
        write_observation_overrides(
            selection_cache=args.selection_cache,
            episode_indices=args.episode_indices,
            frame_indices=args.frame_indices,
            output_path=observation_override_path,
        )
        if args.oracle_actions:
            model_action_override_path = output_dir / "exact_model_action_overrides.pt"
            write_model_action_overrides(
                selection_cache=args.selection_cache,
                episode_indices=args.episode_indices,
                frame_indices=args.frame_indices,
                output_path=model_action_override_path,
            )
    command = build_command(
        python=args.python,
        repo_root=repo_root,
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats_path,
        instance_dir=args.instance_dir,
        instance_ids=args.instance_ids,
        demonstration_paths=demonstration_paths,
        frame_indices=args.frame_indices,
        output_dir=output_dir,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        valid_history_lengths=args.valid_history_lengths,
        observation_override_path=observation_override_path,
        model_action_override_path=model_action_override_path,
    )
    manifest = {
        "protocol": "behavior_midstage_pickup_v1",
        "episode_indices": args.episode_indices,
        "instance_ids": args.instance_ids,
        "demonstration_paths": demonstration_paths,
        "frame_indices": args.frame_indices,
        "history_length": 6,
        "history_stride": 30,
        "valid_history_lengths": args.valid_history_lengths,
        "action_frequency_hz": BEHAVIOR_ACTION_FREQUENCY_HZ,
        "physics_frequency_hz": BEHAVIOR_PHYSICS_FREQUENCY_HZ,
        "max_episode_steps": args.max_episode_steps,
        "selection_cache": args.selection_cache,
        "observation_override_path": (
            str(observation_override_path) if observation_override_path else None
        ),
        "oracle_actions": args.oracle_actions,
        "model_action_override_path": (
            str(model_action_override_path) if model_action_override_path else None
        ),
        "command": command,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(" ".join(command))

    if not args.run:
        return
    env = os.environ.copy()
    env.setdefault("EMBODIED_PATH", str(repo_root / "examples/embodiment"))
    env.setdefault("REPO_PATH", str(repo_root))
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    subprocess.run(command, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create and optionally run the paired short-memory evaluation matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

BEHAVIOR_ACTION_FREQUENCY_HZ = 30
BEHAVIOR_PHYSICS_FREQUENCY_HZ = 120


@dataclass(frozen=True)
class ShortMemoryCondition:
    """One member of the short-memory causal comparison."""

    name: str
    model_path: str
    config_name: str
    history_ablation: str | None


def parse_int_list(value: str) -> list[int]:
    """Parse a non-empty comma-separated integer list."""
    values = [int(item) for item in value.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def build_command(
    *,
    python: str,
    repo_root: Path,
    condition: ShortMemoryCondition,
    seed: int,
    instance_ids: list[int],
    instance_dir: str,
    norm_stats_path: str,
    output_dir: Path,
    max_episode_steps: int,
    skip_intermediate_obs_in_chunk: bool = False,
) -> list[str]:
    """Build one command while preserving the paired instance contract."""
    instance_list = ",".join(str(instance_id) for instance_id in instance_ids)
    command = [
        python,
        str(repo_root / "evaluations/eval_embodied_agent.py"),
        "--config-path",
        str(repo_root / "evaluations/behavior"),
        "--config-name",
        condition.config_name,
        f"runner.logger.log_path={output_dir}",
        f"runner.raw_metrics_path={output_dir / 'raw_metrics.json'}",
        f"rollout.model.model_path={condition.model_path}",
        f"rollout.model.openpi_data.norm_stats_path={norm_stats_path}",
        f"rollout.seed={seed}",
        f"env.eval.seed={seed}",
        "env.eval.prompt_mode=task",
        f"env.eval.total_num_envs={len(instance_ids)}",
        f"env.eval.max_episode_steps={max_episode_steps}",
        f"env.eval.max_steps_per_rollout_epoch={max_episode_steps}",
        "env.eval.skip_intermediate_obs_in_chunk="
        f"{str(skip_intermediate_obs_in_chunk).lower()}",
        f"env.eval.omni_config.env.action_frequency={BEHAVIOR_ACTION_FREQUENCY_HZ}",
        f"env.eval.omni_config.env.rendering_frequency={BEHAVIOR_ACTION_FREQUENCY_HZ}",
        f"env.eval.omni_config.env.physics_frequency={BEHAVIOR_PHYSICS_FREQUENCY_HZ}",
        f"env.eval.video_cfg.fps={BEHAVIOR_ACTION_FREQUENCY_HZ}",
        f"env.eval.activity_instance_ids=[{instance_list}]",
        f"env.eval.omni_config.task.activity_instance_dir={instance_dir}",
    ]
    if condition.history_ablation is not None:
        command.append("env.eval.history_decision_stride=1")
        command.append(f"env.eval.history_ablation={condition.history_ablation}")
    return command


def main() -> None:
    """Write the paired manifest and optionally execute all conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memoryless-checkpoint", required=True)
    parser.add_argument("--short-memory-checkpoint", required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument(
        "--instance-ids",
        type=parse_int_list,
        default=parse_int_list("242,295,211,203"),
    )
    parser.add_argument("--eval-seeds", type=parse_int_list, default=[42])
    parser.add_argument("--max-episode-steps", type=int, default=2048)
    parser.add_argument(
        "--skip-intermediate-obs-in-chunk",
        action="store_true",
        help=(
            "Skip rendering and observation collection inside each action chunk. "
            "Disabled by default for the canonical BEHAVIOR evaluation protocol."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--python",
        default="/mnt/public/daibo/venv/behavior_openpi/bin/python",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = [
        ShortMemoryCondition(
            "memoryless_task",
            args.memoryless_checkpoint,
            "behavior_pi05_mem_r2_eval",
            None,
        ),
        *(
            ShortMemoryCondition(
                f"short_{history_ablation}",
                args.short_memory_checkpoint,
                "behavior_pi05_short_memory_eval",
                history_ablation,
            )
            for history_ablation in ("none", "repeat_current", "shuffle_past")
        ),
    ]

    jobs = []
    for seed in args.eval_seeds:
        for condition in conditions:
            job_output = output_root / condition.name / f"seed_{seed}"
            command = build_command(
                python=args.python,
                repo_root=repo_root,
                condition=condition,
                seed=seed,
                instance_ids=args.instance_ids,
                instance_dir=args.instance_dir,
                norm_stats_path=args.norm_stats_path,
                output_dir=job_output,
                max_episode_steps=args.max_episode_steps,
                skip_intermediate_obs_in_chunk=(args.skip_intermediate_obs_in_chunk),
            )
            jobs.append(
                {
                    "condition": asdict(condition),
                    "seed": seed,
                    "instance_ids": args.instance_ids,
                    "output_dir": str(job_output),
                    "command": command,
                }
            )

    manifest = {
        "protocol": "short_memory_paired_v1",
        "max_episode_steps": args.max_episode_steps,
        "action_frequency_hz": BEHAVIOR_ACTION_FREQUENCY_HZ,
        "physics_frequency_hz": BEHAVIOR_PHYSICS_FREQUENCY_HZ,
        "history_decision_stride": 1,
        "skip_intermediate_obs_in_chunk": args.skip_intermediate_obs_in_chunk,
        "comparison": [
            "memoryless_task controls for the original K=1 architecture",
            "short_none uses the correctly ordered K-frame history",
            "short_repeat_current removes past content without changing metadata",
            "short_shuffle_past tests sensitivity to temporal order",
        ],
        "jobs": jobs,
    }
    manifest_path = output_root / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    for job in jobs:
        print(" ".join(job["command"]))

    if not args.run:
        return

    env = os.environ.copy()
    env.setdefault("EMBODIED_PATH", str(repo_root / "examples/embodiment"))
    env.setdefault("REPO_PATH", str(repo_root))
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    for job in jobs:
        Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(job["command"], check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()

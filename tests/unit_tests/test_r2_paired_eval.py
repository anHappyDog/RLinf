from __future__ import annotations

import json

import torch

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from toolkits.mem.r2_paired_eval import (
    CONDITION_NAMES,
    EvaluationCondition,
    build_command,
)


def test_condition_names_have_stable_causal_order() -> None:
    assert CONDITION_NAMES == (
        "base_task",
        "mixed_task",
        "mixed_oracle_stage",
    )


def test_paired_command_preserves_instance_order(tmp_path) -> None:
    command = build_command(
        python="python",
        repo_root=tmp_path,
        condition=EvaluationCondition(
            "mixed_oracle_stage", "/checkpoint", "oracle_stage"
        ),
        seed=43,
        instance_ids=[12, 24, 38, 54],
        instance_dir="/instances",
        norm_stats_path="/norm_stats.json",
        output_dir=tmp_path / "result",
        max_episode_steps=2048,
    )
    assert "env.eval.activity_instance_ids=[12,24,38,54]" in command
    assert not any("task.activity_instance_id=" in item for item in command)
    assert "env.eval.prompt_mode=oracle_stage" in command
    assert "env.eval.max_episode_steps=2048" in command
    assert "env.eval.max_steps_per_rollout_epoch=2048" in command
    assert "env.eval.skip_intermediate_obs_in_chunk=false" in command
    assert "env.eval.omni_config.env.action_frequency=30" in command
    assert "env.eval.omni_config.env.rendering_frequency=30" in command
    assert "env.eval.omni_config.env.physics_frequency=120" in command
    assert "env.eval.video_cfg.fps=30" in command
    assert "rollout.model.model_path=/checkpoint" in command


def test_raw_eval_metrics_are_written_per_episode(tmp_path) -> None:
    output_path = tmp_path / "raw_metrics.json"
    EmbodiedEvalRunner._save_raw_metrics(
        [
            {
                "success": torch.tensor([False, True]),
                "activity_instance_id": torch.tensor([12, 24]),
            },
            {
                "success": torch.tensor([True]),
                "activity_instance_id": torch.tensor([38]),
            },
        ],
        str(output_path),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["num_episodes"] == 3
    assert [item["activity_instance_id"] for item in payload["episodes"]] == [
        12,
        24,
        38,
    ]

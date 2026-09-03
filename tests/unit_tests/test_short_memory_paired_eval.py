from __future__ import annotations

from toolkits.mem.short_memory_paired_eval import (
    ShortMemoryCondition,
    build_command,
    parse_int_list,
)


def test_integer_list_parser_is_standalone() -> None:
    assert parse_int_list("242,295,211,203") == [242, 295, 211, 203]


def test_short_memory_command_selects_history_control(tmp_path) -> None:
    command = build_command(
        python="python",
        repo_root=tmp_path,
        condition=ShortMemoryCondition(
            "short_shuffle_past",
            "/short",
            "behavior_pi05_short_memory_eval",
            "shuffle_past",
        ),
        seed=42,
        instance_ids=[242, 295, 211, 203],
        instance_dir="/instances",
        norm_stats_path="/norm.json",
        output_dir=tmp_path / "output",
        max_episode_steps=2048,
    )

    assert "behavior_pi05_short_memory_eval" in command
    assert "rollout.model.model_path=/short" in command
    assert "env.eval.history_ablation=shuffle_past" in command
    assert "env.eval.activity_instance_ids=[242,295,211,203]" in command
    assert "env.eval.skip_intermediate_obs_in_chunk=false" in command
    assert "env.eval.omni_config.env.action_frequency=30" in command
    assert "env.eval.omni_config.env.rendering_frequency=30" in command
    assert "env.eval.omni_config.env.physics_frequency=120" in command
    assert "env.eval.video_cfg.fps=30" in command
    assert "env.eval.history_decision_stride=1" in command


def test_memoryless_command_has_no_history_override(tmp_path) -> None:
    command = build_command(
        python="python",
        repo_root=tmp_path,
        condition=ShortMemoryCondition(
            "memoryless_task",
            "/memoryless",
            "behavior_pi05_mem_r2_eval",
            None,
        ),
        seed=42,
        instance_ids=[242],
        instance_dir="/instances",
        norm_stats_path="/norm.json",
        output_dir=tmp_path / "output",
        max_episode_steps=2048,
    )

    assert "behavior_pi05_mem_r2_eval" in command
    assert not any("history_ablation" in argument for argument in command)
    assert not any("history_decision_stride" in argument for argument in command)


def test_short_memory_command_can_opt_into_render_skipping(tmp_path) -> None:
    command = build_command(
        python="python",
        repo_root=tmp_path,
        condition=ShortMemoryCondition(
            "memoryless_task",
            "/memoryless",
            "behavior_pi05_mem_r2_eval",
            None,
        ),
        seed=42,
        instance_ids=[242],
        instance_dir="/instances",
        norm_stats_path="/norm.json",
        output_dir=tmp_path / "output",
        max_episode_steps=4096,
        skip_intermediate_obs_in_chunk=True,
    )

    assert "env.eval.skip_intermediate_obs_in_chunk=true" in command

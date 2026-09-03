from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlinf.envs.behavior.instance_loader import (
    ActivityInstanceFile,
    ActivityInstanceLoader,
)
from rlinf.envs.behavior.oracle_prompt import (
    StagePromptController,
    extract_sequential_reward_info,
)

STAGE_PROMPTS = {
    # B1K annotates navigation/reaching as part of the pickup primitive.
    "move_to_radio": "pick up radio from coffee table",
    "pickup_from_support": "pick up radio from coffee table",
    "press_radio": "press radio",
}


def _stage_info(stage: str) -> dict:
    return {
        "reward": {
            "task_specific": {
                "current_stage_name": stage,
                "current_stage_idx": 1,
                "completed_stage_count": 1,
                "all_stages_completed": False,
            }
        }
    }


def test_task_prompt_mode_ignores_stage_updates() -> None:
    controller = StagePromptController(
        task_prompt="turn on the radio", num_envs=2, mode="task"
    )
    assert controller.update([_stage_info("press_radio"), {}]) == [
        "turn on the radio",
        "turn on the radio",
    ]


def test_oracle_stage_prompts_follow_simulator_reward_state() -> None:
    controller = StagePromptController(
        task_prompt="turn on the radio",
        num_envs=2,
        mode="oracle_stage",
        stage_prompts=STAGE_PROMPTS,
        initial_stage="move_to_radio",
    )
    assert controller.prompts() == [
        "pick up radio from coffee table",
        "pick up radio from coffee table",
    ]
    assert controller.update(
        [_stage_info("pickup_from_support"), _stage_info("press_radio")]
    ) == ["pick up radio from coffee table", "press radio"]

    # A completed sequential reward reports "done"; retain the last useful prompt.
    assert controller.update([_stage_info("done"), {}]) == [
        "pick up radio from coffee table",
        "press radio",
    ]


def test_extract_sequential_reward_info_rejects_unrelated_rewards() -> None:
    assert extract_sequential_reward_info({"reward": {"potential": 1.0}}) == {}


def test_fixed_instance_ids_are_ordered_and_validated(monkeypatch) -> None:
    loader = ActivityInstanceLoader(
        omni_cfg=None,
        activity_name="turning_on_radio",
        activity_instance_id=12,
        instance_resample_mode="disabled",
        activity_instances=(
            ActivityInstanceFile(12, "/tmp/12.json", "tro_state"),
            ActivityInstanceFile(24, "/tmp/24.json", "tro_state"),
        ),
    ).with_fixed_instance_ids([24, 12])
    monkeypatch.setattr(loader, "_apply_instance_files", lambda *_args: None)
    vec_env = SimpleNamespace(
        envs=[
            SimpleNamespace(task=SimpleNamespace(activity_instance_id=0)),
            SimpleNamespace(task=SimpleNamespace(activity_instance_id=0)),
        ]
    )
    assert loader.prepare_reset(vec_env) == (24, 12)

    with pytest.raises(ValueError, match="undiscovered"):
        loader.with_fixed_instance_ids([99])

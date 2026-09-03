import pytest

from rlinf.envs.behavior.subpool_reward import (
    SubtaskRewardSpec,
    SubtaskRewardTracker,
    get_stage_info,
    validate_demo_horizon,
)


def test_potential_difference_has_common_bonus_and_terminal_status():
    spec = SubtaskRewardSpec.from_mapping(
        {
            "potential_terms": [
                {"key": "distance", "scale": 2.0, "direction": "decrease"}
            ],
            "success_bonus": 10.0,
            "step_penalty": -0.1,
            "progress_clip": 1.0,
            "max_steps": 3,
        }
    )
    tracker = SubtaskRewardTracker(spec)
    first = tracker.step({"distance": 1.0, "completed": False})
    second = tracker.step({"distance": 0.25, "completed": True})
    assert first.reward == pytest.approx(-0.1)
    assert second.progress == pytest.approx(1.0)
    assert second.reward == pytest.approx(10.9)
    assert second.success and not second.timeout


def test_timeout_uses_same_failure_scale():
    tracker = SubtaskRewardTracker(
        SubtaskRewardSpec.from_mapping(
            {"potential_terms": [], "max_steps": 1, "timeout_penalty": -2.0}
        )
    )
    outcome = tracker.step({"completed": False})
    assert outcome.reward == pytest.approx(-2.01)
    assert outcome.timeout and not outcome.success


def test_missing_potential_metric_fails_loudly():
    tracker = SubtaskRewardTracker(
        SubtaskRewardSpec.from_mapping(
            {"potential_terms": [{"key": "distance", "direction": "decrease"}]}
        )
    )
    with pytest.raises(KeyError, match="distance"):
        tracker.step({"completed": False})


def test_stage_info_selection_is_ordered_and_strict():
    info = {
        "reward": {
            "task_specific": {
                "stage_infos": {
                    "move": {"completed": True},
                    "pickup": {"completed": False},
                }
            }
        }
    }
    assert get_stage_info(info, 1) == {"completed": False}
    with pytest.raises(IndexError):
        get_stage_info(info, 2)


def test_demo_horizon_must_fit_before_export():
    spec = SubtaskRewardSpec.from_mapping({"potential_terms": [], "max_steps": 16})
    validate_demo_horizon(spec, start_frame=10, end_frame=26)
    with pytest.raises(ValueError, match="shorter than the audited GT suffix"):
        validate_demo_horizon(spec, start_frame=10, end_frame=27)

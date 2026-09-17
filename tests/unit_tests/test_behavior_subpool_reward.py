import pytest

from rlinf.envs.behavior.subpool_reward import (
    SubtaskRewardSpec,
    SubtaskRewardTracker,
    compute_pickup_potential_v2,
    get_stage_info,
    resolve_demo_reward_spec,
    validate_demo_horizon,
)


def test_pickup_potential_v2_has_ordered_phases_and_penalizes_drops():
    far = compute_pickup_potential_v2(
        eef_distance=0.3,
        target_contact=False,
        in_hand=False,
        on_support=True,
        lift_clearance=0.0,
    )
    contact = compute_pickup_potential_v2(
        eef_distance=0.0,
        target_contact=True,
        in_hand=False,
        on_support=True,
        lift_clearance=0.0,
    )
    held = compute_pickup_potential_v2(
        eef_distance=0.0,
        target_contact=True,
        in_hand=True,
        on_support=True,
        lift_clearance=0.02,
    )
    completed = compute_pickup_potential_v2(
        eef_distance=0.1,
        target_contact=False,
        in_hand=True,
        on_support=False,
        lift_clearance=0.04,
    )
    dropped = compute_pickup_potential_v2(
        eef_distance=0.1,
        target_contact=False,
        in_hand=False,
        on_support=False,
        lift_clearance=0.04,
    )

    assert far["potential"] == pytest.approx(0.0)
    assert contact["potential"] == pytest.approx(0.55)
    assert held["potential"] == pytest.approx(0.775)
    assert completed["potential"] == pytest.approx(1.0)
    assert dropped["potential"] == pytest.approx(0.0)


def test_discounted_potential_is_primed_and_terminal_potential_is_zero():
    tracker = SubtaskRewardTracker(
        SubtaskRewardSpec.from_mapping(
            {
                "potential_terms": [{"key": "potential"}],
                "step_penalty": 0.0,
                "progress_clip": 1.0,
                "max_steps": 3,
                "potential_discount": 0.9,
                "prime_potential_at_reset": True,
                "zero_terminal_potential": True,
            }
        )
    )
    assert tracker.prime({"potential": 0.2}) == pytest.approx(0.2)

    approach = tracker.step({"potential": 0.5, "completed": False})
    success = tracker.step({"potential": 1.0, "completed": True})

    assert approach.progress == pytest.approx(0.25)
    assert success.progress == pytest.approx(-0.5)
    assert success.continuation_potential == pytest.approx(0.0)
    assert success.reward == pytest.approx(9.5)


def test_discounted_shaping_return_is_path_independent():
    def shaping_return(potentials):
        tracker = SubtaskRewardTracker(
            SubtaskRewardSpec.from_mapping(
                {
                    "potential_terms": [{"key": "potential"}],
                    "step_penalty": 0.0,
                    "progress_clip": 1.0,
                    "max_steps": len(potentials),
                    "potential_discount": 0.9,
                    "prime_potential_at_reset": True,
                    "zero_terminal_potential": True,
                }
            )
        )
        tracker.prime({"potential": 0.2})
        rewards = [
            tracker.step(
                {
                    "potential": potential,
                    "completed": index == len(potentials) - 1,
                }
            ).progress
            for index, potential in enumerate(potentials)
        ]
        return sum(0.9**index * reward for index, reward in enumerate(rewards))

    assert shaping_return([0.3, 0.5, 1.0]) == pytest.approx(-0.2)
    assert shaping_return([0.8, 0.1, 1.0]) == pytest.approx(-0.2)


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
    assert second.cumulative_progress == pytest.approx(1.0)
    assert second.cumulative_step_penalty == pytest.approx(-0.2)
    assert second.cumulative_terminal_reward == pytest.approx(10.0)


def test_timeout_uses_same_failure_scale():
    tracker = SubtaskRewardTracker(
        SubtaskRewardSpec.from_mapping(
            {"potential_terms": [], "max_steps": 1, "timeout_penalty": -2.0}
        )
    )
    outcome = tracker.step({"completed": False})
    assert outcome.reward == pytest.approx(-2.01)
    assert outcome.timeout and not outcome.success


def test_bounded_pickup_potential_rewards_change_not_occupancy():
    tracker = SubtaskRewardTracker(
        SubtaskRewardSpec.from_mapping(
            {
                "potential_terms": [
                    {
                        "key": "pickup_progress_score",
                        "scale": 2.0,
                        "direction": "increase",
                    }
                ],
                "step_penalty": 0.0,
                "progress_clip": 2.0,
                "max_steps": 5,
            }
        )
    )

    approach = tracker.step({"pickup_progress_score": 0.4, "completed": False})
    unchanged = tracker.step({"pickup_progress_score": 0.4, "completed": False})
    lifted = tracker.step({"pickup_progress_score": 0.8, "completed": False})
    dropped = tracker.step({"pickup_progress_score": 0.0, "completed": False})
    success = tracker.step({"pickup_progress_score": 1.0, "completed": True})

    assert approach.progress == 0.0
    assert unchanged.progress == 0.0
    assert lifted.progress == pytest.approx(0.8)
    assert dropped.progress == pytest.approx(-1.6)
    assert success.progress == pytest.approx(2.0)
    assert success.reward == pytest.approx(12.0)


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


def test_demo_reward_template_uses_per_snapshot_official_horizon():
    resolved = resolve_demo_reward_spec(
        {
            "potential_terms": [],
            "max_steps_multiplier": 3.0,
            "step_penalty_budget": -1.0,
        },
        start_frame=100,
        end_frame=761,
    )

    assert resolved["max_steps"] == 1983
    assert resolved["step_penalty"] == pytest.approx(-1 / 1983)
    assert "max_steps_multiplier" not in resolved
    assert "step_penalty_budget" not in resolved


def test_demo_reward_template_rejects_ambiguous_derived_fields():
    with pytest.raises(ValueError, match="either max_steps or"):
        resolve_demo_reward_spec(
            {
                "potential_terms": [],
                "max_steps": 100,
                "max_steps_multiplier": 3.0,
            },
            start_frame=0,
            end_frame=10,
        )

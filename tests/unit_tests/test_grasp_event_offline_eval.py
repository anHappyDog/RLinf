from __future__ import annotations

import json

import torch

from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from toolkits.mem.grasp_event_offline_eval import (
    GraspEventSample,
    GraspEventSelector,
    assemble_history_metrics,
    classify_gripper_phase,
    summarize_history_comparison,
    summarize_records,
    summarize_sample_prediction,
)


def _actions(left: list[float], right: list[float]) -> torch.Tensor:
    actions = torch.zeros(len(left), 23)
    actions[:, 14] = torch.tensor(left)
    actions[:, 22] = torch.tensor(right)
    return actions


def _sample(phase: str, actions: torch.Tensor) -> GraspEventSample:
    observation = Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 32),
    )
    return GraspEventSample(phase, 10, 100, 6, observation, actions)


def test_gripper_phase_classification_distinguishes_event_windows() -> None:
    assert classify_gripper_phase(_actions([1, 1, 1], [1, 1, 1])) == "open"
    assert classify_gripper_phase(_actions([1, 1, -1], [1, 1, 1])) == "close_onset"
    assert classify_gripper_phase(_actions([-1, -1, -1], [1, 1, 1])) == "closed_hold"
    assert classify_gripper_phase(_actions([-1, 1, 1], [1, 1, 1])) == "other"


def test_selector_keeps_open_controls_away_from_close_onset() -> None:
    observation = Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 32),
        history_frame_mask=torch.ones(1, 6, dtype=torch.bool),
    )
    selector = GraspEventSelector(
        samples_per_phase=2,
        max_samples_per_episode=2,
        open_control_margin=2,
        closed_sample_stride=1,
        gripper_indices=(14, 22),
        close_threshold=0.0,
    )
    open_actions = _actions([1, 1, 1, 1], [1, 1, 1, 1])
    for frame_index in range(5):
        selector.consider(
            episode_index=10,
            frame_index=frame_index,
            observation=observation,
            actions=open_actions,
        )
    selector.consider(
        episode_index=10,
        frame_index=5,
        observation=observation,
        actions=_actions([1, 1, 1, -1], [1, 1, 1, 1]),
    )

    assert [sample.frame_index for sample in selector.selected["open_control"]] == [
        1,
        2,
    ]


def test_sample_summary_measures_draw_level_close_and_timing() -> None:
    sample = _sample("close_onset", _actions([1, 1, -1, -1], [1, 1, 1, 1]))
    predictions = torch.stack(
        [
            _actions([1, 1, -1, -1], [1, 1, 1, 1]),
            _actions([1, 1, 1, 1], [1, 1, 1, 1]),
        ]
    )

    summary = summarize_sample_prediction(
        sample=sample,
        predicted_actions=predictions,
    )

    assert summary["pred_any_close_draw_rate"] == 0.5
    assert summary["correct_hand_close_draw_rate"] == 0.5
    assert summary["closed_step_recall"] == 0.5
    assert summary["first_close_index_mae_with_miss_penalty"] == 1.0
    assert summary["action_mae_left_arm"] == 0.0
    assert summary["action_mae_left_gripper"] == 0.5


def test_history_comparison_reports_paired_action_wins() -> None:
    correct = {
        **{f"action_mae_{name}": 0.1 for name in (
            "base",
            "trunk",
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        )},
        "non_gripper_action_cosine": 0.9,
    }
    controlled = {
        **{key: value + 0.1 for key, value in correct.items() if key.startswith("action_mae_")},
        "non_gripper_action_cosine": 0.8,
    }

    summary = summarize_history_comparison(
        {"correct": [correct], "repeat_current": [controlled]}
    )

    arm = summary["repeat_current"]["action_mae_right_arm"]
    assert round(arm["mean_delta_controlled_minus_correct"], 6) == 0.1
    assert arm["correct_win_rate"] == 1.0
    cosine = summary["repeat_current"]["non_gripper_action_cosine"]
    assert round(cosine["mean_delta_controlled_minus_correct"], 6) == -0.1
    assert cosine["correct_win_rate"] == 1.0

    assembled = assemble_history_metrics(
        {
            "correct": {"offline_grasp_gate": True},
            "repeat_current": {"offline_grasp_gate": False},
        },
        {"correct": [correct], "repeat_current": [controlled]},
    )
    assert assembled["history_conditions"]["correct"] is not assembled
    json.dumps(assembled)


def test_grasp_gate_requires_event_recall_without_open_false_positives() -> None:
    open_record = {
        "phase": "open_control",
        "valid_history_frames": 6,
        "gt_closed_step_fraction": 0.0,
        "pred_any_close_draw_rate": 0.0,
        "pred_closed_step_fraction": 0.0,
        "pred_left_close_draw_rate": 0.0,
        "pred_right_close_draw_rate": 0.0,
        "correct_hand_close_draw_rate": None,
        "closed_step_recall": None,
        "first_close_index_mae_with_miss_penalty": None,
    }
    event_record = {
        **open_record,
        "phase": "close_onset",
        "gt_closed_step_fraction": 0.5,
        "pred_any_close_draw_rate": 1.0,
        "correct_hand_close_draw_rate": 1.0,
        "closed_step_recall": 0.75,
        "first_close_index_mae_with_miss_penalty": 1.0,
    }
    hold_record = {**event_record, "phase": "closed_hold"}

    summary = summarize_records(
        [open_record, event_record, hold_record],
        min_event_any_close_rate=0.5,
        min_correct_hand_close_rate=0.5,
        min_closed_step_recall=0.5,
        max_open_any_close_rate=0.25,
    )

    assert summary["offline_grasp_gate"] is True

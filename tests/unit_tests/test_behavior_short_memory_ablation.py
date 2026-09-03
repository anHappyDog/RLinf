from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from rlinf.envs.behavior.behavior_env import (
    BehaviorEnv,
    apply_short_memory_history_ablation,
)


class _PromptController:
    def update(self, _infos) -> None:
        pass

    def prompts(self) -> list[str]:
        return ["turn on the radio"]


def _history() -> dict[str, torch.Tensor]:
    values = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    return {
        "history_main_images": values[:, :, None],
        "history_wrist_images": values[:, :, None, None],
        "history_states": values[:, :, None] + 10.0,
        "history_frame_mask": torch.tensor(
            [[False, True, True, True], [True, True, True, True]]
        ),
        "history_time_offsets": torch.tensor(
            [[0.0, -2.0, -1.0, 0.0], [-3.0, -2.0, -1.0, 0.0]]
        ),
    }


def test_repeat_current_preserves_padding_mask_and_time() -> None:
    history = _history()
    controlled = apply_short_memory_history_ablation(history, "repeat_current")

    torch.testing.assert_close(
        controlled["history_main_images"].squeeze(-1),
        torch.tensor([[0.0, 3.0, 3.0, 3.0], [7.0, 7.0, 7.0, 7.0]]),
    )
    assert torch.equal(controlled["history_frame_mask"], history["history_frame_mask"])
    assert torch.equal(
        controlled["history_time_offsets"], history["history_time_offsets"]
    )


def test_shuffle_past_reverses_only_valid_history() -> None:
    history = _history()
    controlled = apply_short_memory_history_ablation(history, "shuffle_past")

    torch.testing.assert_close(
        controlled["history_main_images"].squeeze(-1),
        torch.tensor([[0.0, 2.0, 1.0, 3.0], [6.0, 5.0, 4.0, 7.0]]),
    )
    torch.testing.assert_close(
        controlled["history_states"].squeeze(-1),
        torch.tensor([[10.0, 12.0, 11.0, 13.0], [16.0, 15.0, 14.0, 17.0]]),
    )


def test_unknown_history_ablation_is_rejected() -> None:
    with pytest.raises(ValueError, match="history_ablation"):
        apply_short_memory_history_ablation(_history(), "random")


def test_intermediate_video_observation_does_not_advance_policy_history() -> None:
    env = object.__new__(BehaviorEnv)
    env.history_length = 4
    env.history_decision_stride = 1
    env.history_ablation = "none"
    env.num_envs = 1
    env._history_step = 1
    env._action_frequency = 60.0
    env._observation_history = []
    env.prompt_controller = _PromptController()
    env._extract_obs_image = lambda obs: obs
    raw_obs = [
        {
            "main_images": torch.zeros(2, 2, 3, dtype=torch.uint8),
            "wrist_images": torch.zeros(2, 2, 2, 3, dtype=torch.uint8),
            "state": torch.zeros(23),
        }
    ]

    video_obs = env._wrap_obs(raw_obs, [{}], record_history=False)
    assert len(env._observation_history) == 0
    assert "history_main_images" not in video_obs

    policy_obs = env._wrap_obs(raw_obs, [{}], record_history=True)
    assert len(env._observation_history) == 1
    assert policy_obs["history_frame_mask"].tolist() == [[False, False, False, True]]


def test_history_decision_stride_subsamples_raw_policy_buffer() -> None:
    env = object.__new__(BehaviorEnv)
    env.history_length = 3
    env.history_decision_stride = 2
    env.history_ablation = "none"
    env.num_envs = 1
    env._action_frequency = 60.0
    env._observation_history = []

    for decision_index in range(5):
        env._history_step = decision_index * 32
        value = torch.tensor([[float(decision_index)]])
        env._append_history(
            {
                "main_images": value,
                "wrist_images": value[:, None],
                "states": value,
            }
        )

    history = env._build_history_observation()
    torch.testing.assert_close(
        history["history_main_images"].squeeze(-1),
        torch.tensor([[0.0, 2.0, 4.0]]),
    )
    torch.testing.assert_close(
        history["history_time_offsets"],
        torch.tensor([[-128.0 / 60.0, -64.0 / 60.0, 0.0]]),
    )


def test_chunk_records_only_the_final_observation_as_policy_history() -> None:
    env = object.__new__(BehaviorEnv)
    env._history_step = 0
    env.ignore_terminations = False
    env.auto_reset = False
    env.prompt_controller = _PromptController()
    raw_obs = [[{"frame": step_index}] for step_index in range(3)]
    zeros = [torch.zeros(1) for _ in raw_obs]
    infos = [[{}] for _ in raw_obs]
    env.env_chunk_step = lambda _actions: (raw_obs, zeros, zeros, zeros, infos)
    env._calc_step_reward = lambda reward: reward
    env._record_metrics = lambda _reward, step_infos: step_infos
    env._extract_info_done = lambda _info: False
    record_history_calls = []

    def wrap_obs(_obs, _infos, *, record_history):
        record_history_calls.append(record_history)
        return {"main_images": torch.zeros(1, 2, 2, 3, dtype=torch.uint8)}

    env._wrap_obs = wrap_obs
    env.chunk_step(torch.zeros(1, 3, 23))

    assert record_history_calls == [False, False, True]
    assert env._history_step == 3


def test_decision_trace_summarizes_pickup_and_gripper_state(tmp_path) -> None:
    env = object.__new__(BehaviorEnv)
    env._decision_trace_dir = str(tmp_path)
    env._decision_trace_object_name = "radio_89"
    env._decision_trace_gripper_indices = (14, 22)
    env._decision_trace_records = []
    env._decision_index = 3
    env._history_step = 128
    env.num_envs = 1
    env.seed = 42
    env.worker_info = SimpleNamespace(rank=2)

    def info(distance, *, in_hand=False, on_support=True, left_support=False):
        return {
            "activity_instance_id": 242,
            "reward": {
                "task_specific": {
                    "current_stage_name": "pickup_from_support",
                    "completed_stage_count": 1,
                    "stage_infos": {
                        "pickup_from_support": {
                            "eef_to_obj_distance": distance,
                            "in_hand": in_hand,
                            "on_support": on_support,
                            "has_left_support": left_support,
                            "has_picked_up": False,
                        }
                    },
                }
            },
        }

    first = info(0.2)
    final = info(0.1, left_support=True)
    final["decision_trace_state"] = {
        "object_position": [1.0, 2.0, 3.0],
        "object_orientation": [0.0, 0.0, 0.0, 1.0],
    }
    actions = torch.zeros(1, 2, 23)
    actions[0, :, 14] = torch.tensor([-1.0, 1.0])
    actions[0, :, 22] = 0.5
    states = torch.zeros(1, 256)
    states[0, 193:195] = 0.02
    states[0, 232:234] = 0.03

    env._record_decision_trace(
        chunk_actions=actions,
        raw_infos_list=[[first], [final]],
        decision_prompts=["pick up radio from coffee table"],
        final_obs={"states": states},
    )
    record = env._decision_trace_records[0]
    assert record["decision_index"] == 3
    assert record["eef_to_obj_distance_min"] == pytest.approx(0.1)
    assert record["has_left_support_any"]
    assert not record["in_hand_any"]
    assert record["left_gripper_command"] == [-1.0, 1.0]
    assert record["right_gripper_width"] == pytest.approx(0.06)
    assert record["object_position"] == [1.0, 2.0, 3.0]

    env.flush_decision_trace()
    output = json.loads((tmp_path / "rank_2_seed_42.json").read_text())
    assert output["gripper_action_indices"] == [14, 22]
    assert output["records"] == env._decision_trace_records

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    PrimitivePromptInterval,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
    OpenPiPytorchSFTActionModel,
    control_observation_history,
)


def _observation(*, critical: bool = True) -> Observation:
    history = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
    return Observation(
        images={key: history.clone() for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        image_masks={key: torch.ones(1, 3, dtype=torch.bool) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        state=torch.zeros(1, 2),
        tokenized_prompt=torch.zeros(1, 2, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
        history_states=torch.arange(6, dtype=torch.float32).reshape(1, 3, 2),
        history_frame_mask=torch.ones(1, 3, dtype=torch.bool),
        history_time_offsets=torch.tensor([[-2.0, -1.0, 0.0]]),
        history_contrastive_mask=torch.tensor([critical]),
    )


class _HistoryLossModel(nn.Module):
    action_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def compute_loss(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        noise: torch.Tensor,
        time: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        self.calls.append((noise.detach().clone(), time.detach().clone()))
        first_frame = observation.images["base_0_rgb"][:, 0].flatten(1).mean(1)
        loss = self.weight.square() + 2.0 - first_frame
        return loss[:, None].expand(-1, actions.shape[1])


def test_memory_critical_sampling_filters_primitive_and_gripper_action() -> None:
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.memory_critical_enabled = True
    dataset.memory_critical_primitive_prompts = frozenset(
        {"pick up radio from coffee table"}
    )
    dataset.memory_critical_gripper_indices = (14, 22)
    dataset.memory_critical_close_threshold = 0.0
    dataset.memory_critical_require_full_history = False
    dataset.meta = SimpleNamespace(fps=30)
    dataset.prompt_action_horizon = 32
    dataset.primitive_prompt_intervals = {
        10: [
            PrimitivePromptInterval(
                start_frame=100,
                end_frame=200,
                primitive_index=0,
                subtask="pick up radio from coffee table",
                target_source="primitive",
            )
        ]
    }
    actions = torch.ones(32, 23)
    item = {"timestamp": torch.tensor(120 / 30), "action": actions}

    assert not dataset._is_memory_critical_sample(item, 10)
    actions[-1, 22] = -1.0
    assert dataset._is_memory_critical_sample(item, 10)
    dataset.memory_critical_primitive_prompts = frozenset({"press radio"})
    assert not dataset._is_memory_critical_sample(item, 10)


def test_history_controls_preserve_current_frame_and_change_past() -> None:
    observation = _observation()
    repeated = control_observation_history(observation, "repeat_current")
    shuffled = control_observation_history(observation, "shuffle_past")

    assert repeated.images["base_0_rgb"].flatten().tolist() == [2.0, 2.0, 2.0]
    assert shuffled.images["base_0_rgb"].flatten().tolist() == [1.0, 0.0, 2.0]
    torch.testing.assert_close(
        repeated.history_time_offsets, observation.history_time_offsets
    )
    torch.testing.assert_close(
        shuffled.history_frame_mask, observation.history_frame_mask
    )


def test_paired_history_margin_reuses_noise_and_skips_unmarked_samples() -> None:
    model = _HistoryLossModel()
    wrapper = OpenPiPytorchSFTActionModel(
        model,
        num_steps=5,
        action_env_dim=2,
        history_contrastive_weight=0.25,
        history_contrastive_margin=0.1,
        history_contrastive_min_valid_frames=3,
    )
    actions = torch.zeros(1, 2, 2)

    output = wrapper.sft_forward((_observation(), actions))

    assert len(model.calls) == 3
    for noise, time in model.calls[1:]:
        torch.testing.assert_close(noise, model.calls[0][0])
        torch.testing.assert_close(time, model.calls[0][1])
    torch.testing.assert_close(output["vla_loss"], torch.tensor(3.0))
    torch.testing.assert_close(
        output["history_contrastive_loss"], torch.tensor(1.6)
    )
    torch.testing.assert_close(output["loss"], torch.tensor(3.4))

    model.calls.clear()
    unmarked = wrapper.sft_forward((_observation(critical=False), actions))
    assert len(model.calls) == 1
    assert unmarked["history_contrastive_loss"] == 0


def test_rank_without_critical_still_matches_global_control_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _HistoryLossModel()
    wrapper = OpenPiPytorchSFTActionModel(
        model,
        num_steps=5,
        action_env_dim=2,
        history_contrastive_weight=0.25,
        history_contrastive_min_valid_frames=3,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def _global_rank_has_critical(flag: torch.Tensor, **_kwargs) -> None:
        flag.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", _global_rank_has_critical)

    output = wrapper.sft_forward(
        (_observation(critical=False), torch.zeros(1, 2, 2))
    )

    assert len(model.calls) == 3
    assert output["history_contrastive_fraction"] == 0
    assert output["history_contrastive_loss"] == 0

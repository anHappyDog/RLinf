from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from toolkits.mem.short_memory_offline_eval import (
    ablate_observation_history,
    summarize_losses,
)


def _observation() -> Observation:
    values = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.tensor([[False, True, True, True]])
    return Observation(
        images={"base_0_rgb": values[:, :, None]},
        image_masks={"base_0_rgb": mask},
        state=torch.zeros(1, 3),
        history_states=values[:, :, None] + 10.0,
        history_frame_mask=mask,
        history_time_offsets=torch.tensor([[0.0, -2.0, -1.0, 0.0]]),
    )


def test_offline_controls_keep_metadata_and_change_only_valid_history() -> None:
    observation = _observation()
    repeated = ablate_observation_history(observation, "repeat_current")
    shuffled = ablate_observation_history(observation, "shuffle_past")

    torch.testing.assert_close(
        repeated.images["base_0_rgb"].squeeze(-1),
        torch.tensor([[0.0, 3.0, 3.0, 3.0]]),
    )
    torch.testing.assert_close(
        shuffled.images["base_0_rgb"].squeeze(-1),
        torch.tensor([[0.0, 2.0, 1.0, 3.0]]),
    )
    assert repeated.history_frame_mask is observation.history_frame_mask
    assert repeated.history_time_offsets is observation.history_time_offsets


def test_offline_summary_requires_correct_history_to_beat_both_controls() -> None:
    summary = summarize_losses(
        {
            "correct": [0.1, 0.2],
            "repeat_current": [0.3, 0.4],
            "shuffle_past": [0.2, 0.5],
        }
    )

    assert summary["directional_gate"] is True
    assert summary["correct_win_count"] == {
        "repeat_current": 2,
        "shuffle_past": 2,
    }

from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation


def test_eval_snapshot_saves_only_first_normalized_input(tmp_path, monkeypatch) -> None:
    wrapper = object.__new__(OpenPiPytorchEvalActionModel)
    wrapper._eval_snapshot_dir = tmp_path
    wrapper._eval_snapshot_saved = False
    monkeypatch.setenv("RANK", "2")
    observation = Observation(
        images={"base": torch.ones(1, 2, 2, 3)},
        image_masks={"base": torch.ones(1, dtype=torch.bool)},
        state=torch.tensor([[1.0, 2.0]]),
        history_states=torch.ones(1, 2, 2),
        history_frame_mask=torch.ones(1, 2, dtype=torch.bool),
        history_time_offsets=torch.tensor([[-1.0, 0.0]]),
    )

    wrapper._save_eval_snapshot(
        observation,
        model_actions=torch.full((1, 2, 3), 4.0),
        env_actions=torch.full((1, 2, 2), 5.0),
    )
    wrapper._save_eval_snapshot(
        observation,
        model_actions=torch.zeros(1, 2, 3),
        env_actions=torch.zeros(1, 2, 2),
    )

    snapshot = torch.load(tmp_path / "rank_2.pt", weights_only=False)
    assert snapshot["observation"]["state"].tolist() == [[1.0, 2.0]]
    assert snapshot["model_actions"].unique().tolist() == [4.0]
    assert snapshot["env_actions"].unique().tolist() == [5.0]


def test_eval_observation_override_is_ranked_and_one_shot(
    tmp_path, monkeypatch
) -> None:
    wrapper = object.__new__(OpenPiPytorchEvalActionModel)
    wrapper._eval_observation_override_path = tmp_path / "override.pt"
    wrapper._eval_observation_override_consumed = False
    wrapper._observation_dict_to_device = Observation.from_dict
    monkeypatch.setenv("RANK", "1")
    live = Observation(
        images={"base": torch.zeros(1, 2, 2, 3)},
        image_masks={"base": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros(1, 2),
    )
    exact = Observation(
        images={"base": torch.ones(1, 2, 2, 3)},
        image_masks={"base": torch.ones(1, dtype=torch.bool)},
        state=torch.tensor([[3.0, 4.0]]),
    )
    torch.save({"rank_1": exact.to_dict()}, wrapper._eval_observation_override_path)

    first = wrapper._maybe_override_first_eval_observation(live)
    second = wrapper._maybe_override_first_eval_observation(live)

    assert first.state.tolist() == [[3.0, 4.0]]
    assert second is live


def test_eval_model_action_override_is_ranked_and_one_shot(
    tmp_path, monkeypatch
) -> None:
    wrapper = object.__new__(OpenPiPytorchEvalActionModel)
    wrapper._eval_model_action_override_path = tmp_path / "actions.pt"
    wrapper._eval_model_action_override_consumed = False
    monkeypatch.setenv("RANK", "3")
    sampled = torch.zeros(1, 2, 3)
    torch.save(
        {"rank_3": torch.full((2, 3), 4.0)},
        wrapper._eval_model_action_override_path,
    )

    first = wrapper._maybe_override_first_eval_model_actions(sampled)
    second = wrapper._maybe_override_first_eval_model_actions(sampled)

    assert first.shape == sampled.shape
    assert first.unique().tolist() == [4.0]
    assert second is sampled

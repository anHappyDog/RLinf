from __future__ import annotations

import pytest
import torch

from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from toolkits.mem.grasp_event_types import GraspEventSample
from toolkits.mem.midstage_pickup_eval import (
    build_command,
    write_model_action_overrides,
    write_observation_overrides,
)


def test_midstage_command_pairs_raw_states_and_instances(tmp_path) -> None:
    command = build_command(
        python="python",
        repo_root=tmp_path,
        checkpoint="/checkpoint",
        norm_stats_path="/norm.json",
        instance_dir="/instances",
        instance_ids=[0, 0],
        demonstration_paths=[
            "/raw/episode_00000010.hdf5",
            "/raw/episode_00000020.hdf5",
        ],
        frame_indices=[1064, 1133],
        output_dir=tmp_path / "output",
        seed=42,
        max_episode_steps=512,
        valid_history_lengths=[3, 5],
        observation_override_path=tmp_path / "override.pt",
        model_action_override_path=tmp_path / "actions.pt",
    )

    assert "env.eval.activity_instance_ids=[0,0]" in command
    assert (
        "env.eval.demonstration_reset_paths=[/raw/episode_00000010.hdf5,"
        "/raw/episode_00000020.hdf5]" in command
    )
    assert "env.eval.demonstration_reset_frame_indices=[1064,1133]" in command
    assert "env.eval.demonstration_reset_stage=pickup_from_support" in command
    assert (
        "env.eval.demonstration_reset_valid_history_lengths=[3,5]" in command
    )
    assert "env.eval.prompt_mode=oracle_stage" in command
    assert "env.eval.oracle_initial_stage=pickup_from_support" in command
    assert "env.eval.max_episode_steps=512" in command
    assert any(
        item.startswith("rollout.model.openpi.eval_snapshot_dir=") for item in command
    )
    assert (
        "rollout.model.openpi.eval_observation_override_path="
        f"{tmp_path / 'override.pt'}" in command
    )
    assert (
        "rollout.model.openpi.eval_model_action_override_path="
        f"{tmp_path / 'actions.pt'}" in command
    )


def test_midstage_command_rejects_unpaired_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="must align"):
        build_command(
            python="python",
            repo_root=tmp_path,
            checkpoint="/checkpoint",
            norm_stats_path="/norm.json",
            instance_dir="/instances",
            instance_ids=[1, 2],
            demonstration_paths=["/raw/episode_00000010.hdf5"],
            frame_indices=[1064, 1133],
            output_dir=tmp_path / "output",
            seed=42,
            max_episode_steps=512,
        )


def test_midstage_exports_ranked_tensor_only_observations(tmp_path) -> None:
    observation = Observation(
        images={"base": torch.ones(1, 2, 2, 3)},
        image_masks={"base": torch.ones(1, dtype=torch.bool)},
        state=torch.tensor([[1.0, 2.0]]),
    )
    samples = [
        GraspEventSample(
            phase="close_onset",
            episode_index=10,
            frame_index=20,
            valid_history_frames=1,
            observation=observation,
            actions=torch.zeros(2, 3),
        )
    ]
    cache = tmp_path / "selection.pt"
    output = tmp_path / "override.pt"
    torch.save(samples, cache)

    write_observation_overrides(
        selection_cache=cache,
        episode_indices=[10],
        frame_indices=[20],
        output_path=output,
    )

    payload = torch.load(output, weights_only=True)
    assert payload["rank_0"]["state"].tolist() == [[1.0, 2.0]]


def test_midstage_exports_ranked_tensor_only_model_actions(tmp_path) -> None:
    observation = Observation(
        images={"base": torch.ones(1, 2, 2, 3)},
        image_masks={"base": torch.ones(1, dtype=torch.bool)},
        state=torch.tensor([[1.0, 2.0]]),
    )
    samples = [
        GraspEventSample(
            phase="close_onset",
            episode_index=10,
            frame_index=20,
            valid_history_frames=1,
            observation=observation,
            actions=torch.full((2, 3), 4.0),
        )
    ]
    cache = tmp_path / "selection.pt"
    output = tmp_path / "actions.pt"
    torch.save(samples, cache)

    write_model_action_overrides(
        selection_cache=cache,
        episode_indices=[10],
        frame_indices=[20],
        output_path=output,
    )

    payload = torch.load(output, weights_only=True)
    assert payload["rank_0"].shape == (2, 3)
    assert payload["rank_0"].unique().tolist() == [4.0]

import torch

from rlinf.models.embodiment.openpi.forward_inputs import OpenPILiberoForwardInputs
from rlinf.workers.trajectory.records import (
    boundary_request,
    env_result,
    policy_input,
    rollout_results,
)


def _forward_inputs() -> OpenPILiberoForwardInputs:
    return OpenPILiberoForwardInputs.from_model_inputs(
        {
            "chains": torch.zeros(2, 2, 1, 2),
            "denoise_inds": torch.zeros(2, 1, dtype=torch.int64),
            "tokenized_prompt": torch.zeros(2, 3, dtype=torch.int64),
            "tokenized_prompt_mask": torch.ones(2, 3, dtype=torch.bool),
            "action": torch.zeros(2, 2),
            "model_action": torch.zeros(2, 2),
            "observation/image": torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
            "observation/wrist_image": torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
            "observation/state": torch.zeros(2, 2),
        }
    )


def test_critical_path_and_storage_results_have_separate_ownership() -> None:
    request = policy_input(
        global_step=3,
        rollout_epoch=1,
        chunk_step=2,
        slot_ids=(4, 5),
        observations={"state": torch.zeros(2, 3)},
    )
    output, rollout = rollout_results(
        request,
        actions=torch.zeros(2, 1, 2),
        forward_inputs=_forward_inputs(),
        prev_logprobs=torch.zeros(2, 1),
        state_values=torch.zeros(2, 1),
        versions=torch.zeros(2, 1),
    )
    result = env_result(
        request,
        rewards=torch.ones(2, 1),
        dones=torch.zeros(2, 1, dtype=torch.bool),
        terminations=torch.zeros(2, 1, dtype=torch.bool),
        truncations=torch.zeros(2, 1, dtype=torch.bool),
    )

    assert set(vars(output)) == {
        "global_step",
        "rollout_epoch",
        "chunk_step",
        "slot_ids",
        "actions",
    }
    assert rollout.forward_inputs is not None
    assert result.rewards.sum() == 2


def test_boundary_request_selects_only_required_slots_and_observations() -> None:
    request = policy_input(
        global_step=0,
        rollout_epoch=0,
        chunk_step=1,
        slot_ids=(10, 11, 12),
        observations={
            "state": torch.arange(6).reshape(3, 2),
            "extra_view_images": None,
        },
    )
    selected = boundary_request(
        request,
        kind="timeout",
        observations=request.observations,
        mask=torch.tensor([False, True, False]),
    )

    assert selected is not None
    assert selected.slot_ids == (11,)
    assert selected.observations["state"].tolist() == [[2, 3]]
    assert selected.observations["extra_view_images"] is None

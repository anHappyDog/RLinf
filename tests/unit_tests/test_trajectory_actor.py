from dataclasses import replace

import torch

from rlinf.models.embodiment.openpi.forward_inputs import OpenPILiberoForwardInputs
from rlinf.workers.trajectory.actor import prepare_actor_batch, shuffle_actor_batch
from rlinf.workers.trajectory.storage import TrajectoryBatch


def _forward_inputs(batch_size: int) -> OpenPILiberoForwardInputs:
    return OpenPILiberoForwardInputs.from_model_inputs(
        {
            "chains": torch.zeros(batch_size, 2, 1, 2),
            "denoise_inds": torch.zeros(batch_size, 1, dtype=torch.int64),
            "tokenized_prompt": torch.zeros(batch_size, 3, dtype=torch.int64),
            "tokenized_prompt_mask": torch.ones(batch_size, 3, dtype=torch.bool),
            "action": torch.zeros(batch_size, 2),
            "model_action": torch.zeros(batch_size, 2),
            "observation/image": torch.zeros(batch_size, 3, 2, 2, dtype=torch.uint8),
            "observation/wrist_image": torch.zeros(
                batch_size, 3, 2, 2, dtype=torch.uint8
            ),
            "observation/state": torch.zeros(batch_size, 2),
        }
    )


def _trajectory() -> TrajectoryBatch:
    # E=2, S=2, B=2, A=1. Flattened ForwardInputs are in E,S,B order.
    dones = torch.zeros(2, 2, 2, 1, dtype=torch.bool)
    terminations = dones.clone()
    truncations = dones.clone()
    truncations[0, 0, 0] = True
    dones[0, 0, 0] = True
    timeout_mask = truncations.any(dim=-1)
    tail_mask = ~dones[:, -1].any(dim=-1)
    return TrajectoryBatch(
        global_step=0,
        slot_ids=(0, 1),
        env_rewards=torch.ones(2, 2, 2, 1),
        dones=dones,
        terminations=terminations,
        truncations=truncations,
        actions=torch.zeros(2, 2, 2, 1, 2),
        forward_inputs=_forward_inputs(8),
        prev_logprobs=torch.zeros(2, 2, 2, 1),
        state_values=torch.zeros(2, 2, 2, 1),
        timeout_values=torch.zeros(2, 2, 2, 1),
        timeout_mask=timeout_mask,
        tail_values=torch.zeros(2, 2, 1),
        tail_mask=tail_mask,
    )


def test_prepare_actor_batch_converts_time_order_and_computes_gae() -> None:
    batch = prepare_actor_batch(
        _trajectory(),
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
    )

    assert batch["rewards"].shape == (2, 4, 1)
    assert batch["prev_logprobs"].shape == (2, 4, 1)
    assert batch["forward_inputs"]["chains"].shape == (2, 4, 2, 1, 2)
    # epoch 0 / slot 0 ends after step 0, so step 1 is excluded from training.
    assert not batch["loss_mask"][1, 0, 0]
    assert batch["advantages"][0, 0, 0] == 1
    assert batch["advantages"][0, 1, 0] == 2


def test_prepare_actor_batch_requires_boundary_values() -> None:
    trajectory = replace(_trajectory(), tail_values=None)

    try:
        prepare_actor_batch(
            trajectory,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
        )
    except ValueError as error:
        assert "tail_values" in str(error)
    else:
        raise AssertionError("missing boundary values must fail")


def test_shuffle_actor_batch_does_not_apply_legacy_t_plus_one_slice() -> None:
    batch = prepare_actor_batch(
        _trajectory(),
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
    )
    shuffled = shuffle_actor_batch(batch, torch.arange(8))

    assert shuffled["prev_logprobs"].shape[0] == 8
    assert shuffled["prev_values"].shape[0] == 8
    assert shuffled["dones"].shape[0] == 8
    assert shuffled["forward_inputs"]["chains"].shape[0] == 8

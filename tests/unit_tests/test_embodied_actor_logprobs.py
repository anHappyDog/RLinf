from contextlib import contextmanager, nullcontext

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.actor.embodied_fsdp_actor_worker import (
    EmbodiedFSDPActor,
    _masked_reference_kl,
)


class _LogprobModel(torch.nn.Module):
    def forward(self, *, forward_inputs, **kwargs):
        del kwargs
        return {"logprobs": forward_inputs["expected_logprobs"]}


class _ScaledLogprobModel(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, *, forward_inputs, **kwargs):
        del kwargs
        return {"logprobs": forward_inputs["expected_logprobs"] * self.scale}


def test_recompute_prev_logprobs_replaces_rollout_values_and_restores_mode():
    actor = object.__new__(EmbodiedFSDPActor)
    actor.model = _LogprobModel().train()
    actor.device = torch.device("cpu")
    actor.amp_context = nullcontext()
    actor.cfg = OmegaConf.create({"actor": {"micro_batch_size": 2}})
    actor.kl_beta = 0.0

    expected = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    actor.rollout_batch = {
        "prev_logprobs": torch.zeros_like(expected),
        "forward_inputs": {"expected_logprobs": expected.clone()},
    }

    recompute_impl = EmbodiedFSDPActor.recompute_prev_logprobs
    while hasattr(recompute_impl, "__wrapped__"):
        recompute_impl = recompute_impl.__wrapped__
    metrics = recompute_impl(actor)

    assert actor.model.training
    assert torch.equal(actor.rollout_batch["prev_logprobs"], expected)
    assert metrics["actor/rollout_logprob_abs_diff"] == pytest.approx(2.5)
    assert metrics["actor/rollout_logprob_diff"] == pytest.approx(2.5)
    assert metrics["actor/rollout_logprob_abs_diff_max"] == pytest.approx(4.0)


def test_recompute_prev_logprobs_caches_frozen_reference_and_restores_actor():
    actor = object.__new__(EmbodiedFSDPActor)
    actor.model = _ScaledLogprobModel(2.0).train()
    actor.device = torch.device("cpu")
    actor.amp_context = nullcontext()
    actor.cfg = OmegaConf.create({"actor": {"micro_batch_size": 1}})
    actor.kl_beta = 0.01
    actor.ref_policy_state_dict = {"scale": torch.tensor(1.0)}

    @contextmanager
    def swap_reference(state_dict):
        active_scale = actor.model.scale.detach().clone()
        actor.model.load_state_dict(state_dict)
        try:
            yield
        finally:
            actor.model.load_state_dict({"scale": active_scale})

    actor.swap_sharded_model_state_dict = swap_reference
    actor.rollout_batch = {
        "prev_logprobs": torch.zeros(2, 1, 1),
        "forward_inputs": {"expected_logprobs": torch.ones(2, 1, 1)},
    }

    recompute_impl = EmbodiedFSDPActor.recompute_prev_logprobs
    while hasattr(recompute_impl, "__wrapped__"):
        recompute_impl = recompute_impl.__wrapped__
    metrics = recompute_impl(actor)

    torch.testing.assert_close(
        actor.rollout_batch["prev_logprobs"], torch.full((2, 1, 1), 2.0)
    )
    torch.testing.assert_close(actor.rollout_batch["ref_logprobs"], torch.ones(2, 1, 1))
    assert actor.model.scale.item() == pytest.approx(2.0)
    assert actor.model.training
    assert metrics["actor/reference_logprob_abs_diff"] == pytest.approx(1.0)


def test_masked_reference_kl_uses_only_executed_action_coordinates():
    logprobs = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]], requires_grad=True
    )
    ref_logprobs = torch.zeros_like(logprobs)
    loss = _masked_reference_kl(
        logprobs,
        ref_logprobs,
        penalty_type="mse",
        executed_action_mask=torch.tensor([[True, True, False]]),
        loss_mask=torch.tensor([[True]]),
        sample_weights=None,
    )

    assert loss.item() == pytest.approx(1.25)
    loss.backward()
    assert torch.equal(logprobs.grad[0, 2], torch.zeros(2))


def test_masked_reference_kl_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        _masked_reference_kl(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 2),
            penalty_type="mse",
            executed_action_mask=None,
            loss_mask=None,
            sample_weights=None,
        )

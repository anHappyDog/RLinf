from contextlib import nullcontext

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class _LogprobModel(torch.nn.Module):
    def forward(self, *, forward_inputs, **kwargs):
        del kwargs
        return {"logprobs": forward_inputs["expected_logprobs"]}


def test_recompute_prev_logprobs_replaces_rollout_values_and_restores_mode():
    actor = object.__new__(EmbodiedFSDPActor)
    actor.model = _LogprobModel().train()
    actor.device = torch.device("cpu")
    actor.amp_context = nullcontext()
    actor.cfg = OmegaConf.create({"actor": {"micro_batch_size": 2}})

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

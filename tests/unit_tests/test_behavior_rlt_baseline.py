"""BEHAVIOR RLT data routing, action units, warmup and checkpoint contracts."""

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import DirectGaussianActor
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    load_full_wrapper_weights,
)


def test_behavior_rlt_retains_grounded_loader(monkeypatch):
    import rlinf.data.datasets.openpi_rlinf as loaders

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"openpi": {"use_rlt": True, "config_name": "pi05_behavior"}}
            }
        }
    )
    marker = object()
    monkeypatch.setitem(
        loaders._SFT_DATALOADER_BUILDERS, "behavior", lambda: lambda *a: marker
    )
    assert loaders.build_openpi_rlinf_sft_dataloader(cfg, 1, 0, "data") is marker


def test_physical_action_affine_does_not_clip_pose_commands():
    actor = DirectGaussianActor(
        3,
        4,
        hidden_dim=8,
        num_hidden_layers=1,
        action_offset=[2.0, -3.0],
        action_scale=[0.5, 2.0],
        action_clip=None,
    )
    for parameter in actor.parameters():
        parameter.data.zero_()
    actor.mlp.net[-1].bias.data.fill_(2.0)
    out = actor(torch.zeros(2, 3), torch.zeros(2, 4), deterministic=True)
    torch.testing.assert_close(out, torch.tensor([[3.0, 1.0, 3.0, 1.0]]).expand(2, -1))


def test_full_task_route_collects_base_before_actor_warmup():
    route = SimulatorRLTRoute(use_schedule=True, warmup_updates=100, full_task=True)
    actor = torch.ones(2, 3, 4)
    ref = torch.full_like(actor, 2.0)
    for version, expected in [(0, ref), (100, actor)]:
        ctx = RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": ref},
            student_actions=actor,
            result={"forward_inputs": {}},
            mode="train",
            version=version,
            rlt_switch_flags=torch.zeros(2, dtype=torch.bool),
        )
        out = route.route(ctx)
        torch.testing.assert_close(out.actions, expected)
        assert out.result["forward_inputs"]["record_transition"].all()
        assert out.result["forward_inputs"]["actor_switch"].all() == (version == 100)


def test_stage1_base_init_but_stage2_requires_complete_token(tmp_path):
    wrapper = torch.nn.Module()
    wrapper.model = torch.nn.Linear(2, 2)
    wrapper.rlt_module = torch.nn.Linear(2, 2)
    weights = wrapper.state_dict()
    base = {k: v for k, v in weights.items() if k.startswith("model.")}
    path = tmp_path / "weights.pt"
    torch.save(base, path)
    load_full_wrapper_weights(
        wrapper, path, expect_rlt=False, require_complete_base=True
    )
    with pytest.raises(ValueError, match="no rlt_module"):
        load_full_wrapper_weights(wrapper, path, expect_rlt=True)
    base["rlt_module.weight"] = weights["rlt_module.weight"]
    torch.save(base, path)
    with pytest.raises(RuntimeError, match="all rlt_module"):
        load_full_wrapper_weights(wrapper, path, expect_rlt=False)


def test_execution_aware_critic_bootstraps_at_actual_prefix():
    from rlinf.algorithms.residual import chunk_target
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin

    # Check the actual critic entry uses the primitive target even for a short chunk.
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "rlt_td3_mlp_policy"}},
            "algorithm": {
                "rlt_execution_aware": True,
                "gamma": 0.5,
                "bootstrap_truncation": True,
            },
        }
    )

    class Worker(RLTACLossMixin):
        pass

    w = Worker()
    w.cfg = cfg
    w.torch_dtype = torch.float32
    w._next_actions_for_critic_target = lambda obs: (torch.zeros(1, 4), None, None)
    w.target_model = lambda **kw: torch.full((1, 2), 8.0)
    w.model = lambda **kw: torch.zeros(1, 2, requires_grad=True)
    batch = {
        "curr_obs": {},
        "next_obs": {},
        "actions": torch.zeros(1, 4),
        "rewards": torch.tensor([[1.0, 2.0, 100.0, 100.0]]),
        "executed_action_mask": torch.tensor([[True, True, False, False]]),
        "terminations": torch.zeros(1, 4, dtype=torch.bool),
        "truncations": torch.tensor([[False, True, False, False]]),
        "dones": torch.tensor([[False, True, False, False]]),
    }
    loss, metrics = RLTACLossMixin.forward_critic.__wrapped__(w, batch)
    target = chunk_target(
        batch["rewards"],
        batch["executed_action_mask"],
        batch["terminations"],
        batch["truncations"],
        torch.tensor([[8.0]]),
        0.5,
        True,
    )
    torch.testing.assert_close(loss, target.square().mean())


@pytest.mark.parametrize("residual,expected_actor_updates", [(False, 2), (True, 1)])
def test_schedule_does_not_confuse_replay_semantics_with_actor_warmup(
    monkeypatch, residual, expected_actor_updates
):
    from types import SimpleNamespace

    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy

    cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 4,
                "micro_batch_size": 2,
                "model": {
                    "model_type": "residual_mlp_policy"
                    if residual
                    else "rlt_td3_mlp_policy"
                },
            },
            "algorithm": {"rlt_execution_aware": not residual},
        }
    )
    if residual:
        cfg.algorithm.residual = {"critic_warmup_updates": 2}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    updates = []
    worker = SimpleNamespace(
        cfg=cfg,
        use_rlt_schedule=True,
        _world_size=1,
        update_step=0,
        critic_actor_ratio=2,
        pending_update_budget=4,
        model=SimpleNamespace(train=lambda: None),
        _rlt_updates_to_run=lambda: (4, {}),
        update_one_epoch=lambda **kw: updates.append(kw) or {},
        process_train_metrics=lambda metrics: metrics,
    )
    metrics = RLTACFSDPPolicy.run_training(worker)
    assert len(updates) == 4 and worker.update_step == 4
    assert metrics["rlt/critic_updates_run"] == [4.0]
    assert metrics["rlt/actor_updates_run"] == [float(expected_actor_updates)]

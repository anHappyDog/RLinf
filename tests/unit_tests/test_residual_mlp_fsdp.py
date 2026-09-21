# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-GPU regression for TD3 critic-only FSDP warmup and target EMA."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.models import get_model
from rlinf.workers.actor.fsdp_residual_td3_policy_worker import ResidualTD3FSDPPolicy
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_critic_only_fsdp_warmup_then_actor_and_target_updates(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    cfg = OmegaConf.load(root / "behavior_residual_mlp_a.yaml")
    cfg.actor.fsdp_config = OmegaConf.merge(
        OmegaConf.load(root / "hybrid_engines/fsdp.yaml"), cfg.actor.fsdp_config
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.distributed.init_process_group(
        "nccl", init_method=(tmp_path / "dist_init").as_uri(), rank=0, world_size=1
    )
    try:
        strategy = FSDPStrategyBase.create(cfg.actor, 1)
        mesh = init_device_mesh("cuda", (1,))
        model = strategy.wrap_model(get_model(cfg.actor.model).cuda(), mesh)
        target = strategy.wrap_model(get_model(cfg.actor.model).cuda(), mesh)
        target.requires_grad_(False)
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(model.q_head.parameters(), lr=1e-4)
        worker = SimpleNamespace(
            cfg=cfg,
            model=model,
            target_model=target,
            target_model_initialized=True,
            target_update_type="all",
            qf_optimizer=critic_optimizer,
        )
        EmbodiedSACFSDPPolicy.soft_update_target_model(worker, tau=1.0)
        initial_actor = {
            k: p.detach().clone() for k, p in model.actor.named_parameters()
        }
        initial_target = {k: p.detach().clone() for k, p in target.named_parameters()}
        obs = {
            "z_rl": torch.randn(256, 2048, device="cuda"),
            "proprio": torch.randn(256, 32, device="cuda"),
            "ref_chunk": torch.randn(256, 32, 23, device="cuda"),
        }
        batch = {
            "curr_obs": obs,
            "next_obs": obs,
            "actions": obs["ref_chunk"].flatten(1),
            "rewards": torch.randn(256, 32, device="cuda"),
            "executed_action_mask": torch.ones(
                256, 32, device="cuda", dtype=torch.bool
            ),
            "terminations": torch.zeros(256, 32, device="cuda", dtype=torch.bool),
            "truncations": torch.zeros(256, 32, device="cuda", dtype=torch.bool),
        }
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(120):
            torch.cuda.synchronize()
            start = time.perf_counter()
            critic_optimizer.zero_grad(set_to_none=True)
            loss, _ = ResidualTD3FSDPPolicy.forward_critic(worker, batch)
            assert torch.isfinite(loss)
            loss.backward()
            model.clip_grad_norm_(10)
            critic_optimizer.step()
            if step >= 100 and step % 2 == 0:
                actor_optimizer.zero_grad(set_to_none=True)
                loss, _, _ = ResidualTD3FSDPPolicy.forward_actor(worker, batch)
                assert torch.isfinite(loss)
                loss.backward()
                model.clip_grad_norm_(10)
                actor_optimizer.step()
            # The previous one-unit configuration loses online actor parameter
            # registrations after critic-only backward and fails this EMA.
            assert (
                dict(model.named_parameters()).keys()
                == dict(target.named_parameters()).keys()
            )
            EmbodiedSACFSDPPolicy.soft_update_target_model(worker)
            if step == 99:
                for name, param in model.actor.named_parameters():
                    torch.testing.assert_close(
                        param, initial_actor[name], rtol=0, atol=0
                    )
            torch.cuda.synchronize()
            if step >= 102:
                times.append(time.perf_counter() - start)
        assert any(
            not torch.equal(p, initial_actor[k])
            for k, p in model.actor.named_parameters()
        )
        assert any(
            not torch.equal(p, initial_target[k]) for k, p in target.named_parameters()
        )
        print(
            json.dumps(
                {
                    "critic_updates": 120,
                    "actor_updates": 10,
                    "batch": 256,
                    "median_update_seconds": float(np.median(times)),
                    "p90_update_seconds": float(np.percentile(times, 90)),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        )
    finally:
        torch.distributed.destroy_process_group()
        torch.set_num_threads(previous_threads)

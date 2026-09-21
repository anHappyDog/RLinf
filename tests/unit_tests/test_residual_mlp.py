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

"""CPU contracts for residual A; no simulator, Ray cluster or VLA allocation."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.residual import chunk_target
from rlinf.models.embodiment.mlp_policy.residual_mlp_policy import ResidualMLPPolicy


def make_policy(dim=23, chunk=32):
    torch.manual_seed(17)
    return ResidualMLPPolicy(4, 3, dim, chunk, [0.02] * dim, mlp_hidden_dim=16)


def observations(dim=23, chunk=32):
    return {
        "z_rl": torch.randn(3, 4),
        "proprio": torch.randn(3, 3),
        "ref_chunk": torch.randn(3, chunk, dim) * 3,
    }


@pytest.mark.parametrize("dim,chunk", [(23, 32), (7, 10), (4, 1)])
def test_zero_residual_and_exploration_bounds(dim, chunk):
    model = make_policy(dim, chunk)
    obs = observations(dim, chunk)
    action, result = model.predict_action_batch(obs, mode="eval")
    assert torch.equal(action, obs["ref_chunk"])
    action, _, _ = model.sac_forward(obs, apply_action_noise=True, noise_sigma=100)
    delta = action - obs["ref_chunk"].flatten(1)
    assert (delta.abs() <= 0.020002).all()
    assert (delta.reshape(3, chunk, dim).abs().sum((0, 1)) > 0).all()
    assert torch.equal(result["forward_inputs"]["action"], obs["ref_chunk"].flatten(1))


def test_actual_chunk_discount_and_terminal_semantics():
    rewards = torch.tensor([[1.0, 2.0, 99.0], [1.0, 2.0, 99.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    term = torch.tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
    trunc = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=torch.bool)
    q = torch.full((3, 1), 8.0)
    target = chunk_target(rewards, mask, term, trunc, q, 0.5, True)
    torch.testing.assert_close(target[:, 0], torch.tensor([2.0, 4.0, 3.75]))
    target = chunk_target(rewards, mask, term, trunc, q, 0.5, False)
    torch.testing.assert_close(target[:, 0], torch.tensor([2.0, 2.0, 3.75]))
    with pytest.raises(ValueError):
        chunk_target(rewards, torch.zeros_like(mask), term, trunc, q, 0.5)


def test_worker_losses_update_actor_and_critics():
    from types import SimpleNamespace

    from rlinf.workers.actor.fsdp_residual_td3_policy_worker import (
        ResidualTD3FSDPPolicy,
    )

    model = make_policy()
    target = copy.deepcopy(model)
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"epsilon": [0.02] * 23, "action_dim": 23}},
            "algorithm": {
                "gamma": 0.99,
                "residual": {
                    "target_noise_sigma": 0.1,
                    "target_noise_clip": 0.25,
                    "bootstrap_truncation": True,
                    "penalty": 0.01,
                },
            },
        }
    )
    actor_optim = torch.optim.Adam(model.actor.parameters(), lr=0.001)
    critic_optim = torch.optim.Adam(model.q_head.parameters(), lr=0.001)
    worker = SimpleNamespace(
        model=model, target_model=target, cfg=cfg, qf_optimizer=critic_optim
    )
    obs = observations()
    actions, _, _ = model.sac_forward(obs, apply_action_noise=True)
    batch = {
        "curr_obs": obs,
        "next_obs": observations(),
        "actions": actions.detach(),
        "rewards": torch.ones(3, 32),
        "executed_action_mask": torch.ones(3, 32, dtype=torch.bool),
        "terminations": torch.zeros(3, 32, dtype=torch.bool),
        "truncations": torch.zeros(3, 32, dtype=torch.bool),
    }
    oldq = [p.clone() for p in model.q_head.parameters()]
    loss, _ = ResidualTD3FSDPPolicy.forward_critic(worker, batch)
    loss.backward()
    critic_optim.step()
    critic_optim.zero_grad()
    assert any(not torch.equal(a, b) for a, b in zip(oldq, model.q_head.parameters()))
    loss, _, _ = ResidualTD3FSDPPolicy.forward_actor(worker, batch)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()
    assert model.actor.mlp.net[-1].weight.abs().sum() > 0
    assert all(
        torch.equal(a, b)
        for a, b in zip(target.parameters(), make_policy().parameters())
    )


def test_replay_preserves_timeout_final_observation_and_mask():
    from types import SimpleNamespace

    from rlinf.data.schema.embodied_types import Trajectory
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin

    class ReplayHarness(RLTACReplayMixin):
        pass

    harness = ReplayHarness()
    harness.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "residual_mlp_policy"}},
            "env": {"train": {"auto_reset": True}},
        }
    )
    harness.replay_buffer = SimpleNamespace(
        _flatten_trajectory=lambda t: TrajectoryReplayBuffer._flatten_trajectory(
            None, t
        )
    )
    t = Trajectory(max_episode_length=2)
    # The final policy output only supplies next_obs, and has no reward/action execution.
    t.actions = torch.zeros(3, 1, 4)
    t.rewards = torch.tensor([[[1.0, 0.0]], [[2.0, 3.0]]])
    t.executed_action_mask = torch.tensor([[[True, False]], [[True, True]]])
    t.dones = torch.tensor([[[False, False]], [[True, True]], [[False, False]]])
    t.terminations = torch.zeros_like(t.dones)
    t.truncations = t.dones.clone()
    t.forward_inputs = {"record_transition": torch.ones(3, 1, 1, dtype=torch.bool)}
    t.curr_obs = {k: torch.zeros(2, 1, 4) for k in ["z_rl", "proprio", "ref_chunk"]}
    t.next_obs = {k: torch.full((2, 1, 4), 7.0) for k in t.curr_obs}
    rows, count = harness._transition_replay_trajectories(t)
    assert count == 1 and len(rows) == 2
    assert (rows[0].next_obs["z_rl"] == 7).all()
    assert rows[0].executed_action_mask.flatten().tolist() == [True, False]
    assert rows[0].truncations.any() and not rows[1].truncations.any()


def test_frozen_feature_reference_matches_eval_sampler(monkeypatch):
    from types import SimpleNamespace

    from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
        OpenPiPytorchEvalActionModel,
    )
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0

    monkeypatch.setattr(model_module, "preprocess_observation", lambda obs, train: obs)

    class Core:
        action_horizon = 2
        action_dim = 3

        def build_prefix_cache(self, observation):
            hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]])
            return hidden, torch.tensor([[True, True, False]]), ()

        def run_suffix(self, observation, actions, time, cache, mask):
            return actions * 0.25

        def velocity_from_suffix(self, suffix):
            return suffix

        sample_actions = Pi0.sample_actions

    wrapper = SimpleNamespace(model=Core(), num_steps=10, device=torch.device("cpu"))
    obs = SimpleNamespace(state=torch.ones(1, 3))
    wrapper._repack_env_obs = lambda env: env
    wrapper.input_transform = lambda env, transpose: env
    wrapper._observation_dict_to_device = lambda env: obs
    wrapper.output_transform = lambda data: {
        "actions": data["actions"].double() * 2 + 3
    }
    wrapper._sample_actions_from_prefix_cache = (
        lambda *args,
        **kw: OpenPiPytorchEvalActionModel._sample_actions_from_prefix_cache(
            wrapper, *args, **kw
        )
    )
    noise = torch.randn(1, 2, 3)
    features = OpenPiPytorchEvalActionModel.extract_residual_obs(
        wrapper, {}, noise=noise.clone()
    )
    reference, _ = OpenPiPytorchEvalActionModel._predict_eval(
        wrapper, obs, noise=noise.clone(), rng=None
    )
    assert torch.equal(features["ref_chunk"], reference)
    torch.testing.assert_close(features["z_rl"], torch.tensor([[2.0, 4.0]]))
    assert not any(value.requires_grad for value in features.values())


def test_residual_route_records_all_phases_and_final_features():
    from types import SimpleNamespace

    from rlinf.algorithms.rlt.rollout import predict_rlt_actions
    from rlinf.algorithms.rlt.route import ResidualRoute

    model = make_policy()
    obs = observations()
    final = observations()
    features = SimpleNamespace(extract_residual_obs=lambda value: value)
    actions, result = predict_rlt_actions(
        policy_model=model,
        feature_model=features,
        rlt_route=ResidualRoute(1),
        env_obs=obs,
        final_obs=final,
        mode="train",
        version=0,
    )
    assert torch.equal(actions, obs["ref_chunk"])
    assert result["forward_inputs"]["record_transition"].all()
    assert torch.equal(
        result["forward_inputs"]["rlt_transition_ref_chunk"], final["ref_chunk"]
    )
    actions, result = predict_rlt_actions(
        policy_model=model,
        feature_model=features,
        rlt_route=ResidualRoute(1),
        env_obs=obs,
        final_obs=None,
        mode="train",
        version=1,
        rlt_switch_flags=torch.zeros(3, dtype=torch.bool),
    )
    assert not torch.equal(actions, obs["ref_chunk"])
    assert result["forward_inputs"]["record_transition"].all()


def test_training_config_and_factory_without_ray(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from hydra import compose, initialize_config_dir

    import rlinf.config as config
    from rlinf.models import get_model

    root = Path(__file__).resolve().parents[2]
    for key in [
        "B1K_SUBPOOL_MANIFEST",
        "B1K_EVAL_MANIFEST",
        "B1K_SUBPOOL_RESULT_DIR",
        "B1K_GROUNDED_TOKEN_MAPPING",
        "B1K_ASSET_FINGERPRINT",
    ]:
        monkeypatch.setenv(key, "/tmp/residual-test-placeholder")
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setattr(config, "Cluster", lambda *args, **kw: SimpleNamespace())
    monkeypatch.setattr(
        config,
        "HybridComponentPlacement",
        lambda *args, **kw: SimpleNamespace(get_world_size=lambda name: 1),
    )
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="behavior_residual_mlp_a")
    cfg = config.validate_cfg(cfg)
    model = get_model(cfg.actor.model)
    assert isinstance(model, ResidualMLPPolicy)
    assert cfg.runner.resume_dir is None
    assert cfg.rollout.rlt_feature_model.num_steps == 10
    assert cfg.actor.model.num_action_chunks == 32
    assert not cfg.env.train.subpool.reward_overrides


def test_replay_storage_sampling(tmp_path):
    from rlinf.data.schema.embodied_types import Trajectory
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer

    buffer = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=10,
        sample_window_size=10,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    trajectory = Trajectory(max_episode_length=1)
    trajectory.actions = torch.randn(1, 1, 6)
    trajectory.rewards = torch.tensor([[[1.0, 2.0]]])
    trajectory.dones = torch.tensor([[[False, True]]])
    trajectory.terminations = trajectory.dones.clone()
    trajectory.truncations = torch.zeros_like(trajectory.dones)
    trajectory.executed_action_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    trajectory.curr_obs = {
        "z_rl": torch.randn(1, 1, 4),
        "proprio": torch.randn(1, 1, 3),
        "ref_chunk": torch.randn(1, 1, 2, 3),
    }
    trajectory.next_obs = {k: v + 1 for k, v in trajectory.curr_obs.items()}
    buffer.add_trajectories([trajectory, copy.deepcopy(trajectory)])
    batch = buffer.sample(2)
    assert batch["actions"].shape == (2, 6)
    assert batch["executed_action_mask"].shape == (2, 2)
    buffer.close()
    torch.testing.assert_close(
        batch["next_obs"]["z_rl"] - batch["curr_obs"]["z_rl"], torch.ones(2, 4)
    )


def test_incomplete_frozen_checkpoint_rejected(tmp_path):
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        load_full_wrapper_weights,
    )

    wrapper = torch.nn.Module()
    wrapper.model = torch.nn.Linear(3, 2)
    path = tmp_path / "full_weights.pt"
    torch.save({"model.weight": wrapper.model.weight.detach()}, path)
    with pytest.raises(RuntimeError, match="incomplete"):
        load_full_wrapper_weights(
            wrapper, path, expect_rlt=False, require_complete_base=True
        )

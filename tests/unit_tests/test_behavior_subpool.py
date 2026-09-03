import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior.behavior_env import (
    BehaviorEnv,
    BehaviorProcess,
    BehaviorProcessPool,
    BehaviorSubpoolEnv,
    _repeat_terminal_subpool_chunk,
    _support_surface_distance,
)
from rlinf.envs.behavior.subpool import (
    SubpoolCatalog,
    SubpoolSnapshot,
    SubpoolStore,
    full_state_sha256,
    validate_round_robin_coverage,
    validate_subpool_env_config,
    validate_subpool_export_request,
    validate_subpool_rollout_horizons,
)
from rlinf.envs.behavior.utils import (
    apply_runtime_renderer_settings,
    sync_robot_after_pose_override,
)


def _state(offset=0):
    return {
        "sim": torch.arange(4, dtype=torch.float32) + offset,
        "robot": {
            "ag_obj_constraint_params": {
                "right": {"object": "radio", "contact": torch.tensor([1.0])}
            }
        },
    }


def _record(state, snapshot_id="state-0", **overrides):
    values = {
        "snapshot_id": snapshot_id,
        "state_path": f"states/{snapshot_id}.pt",
        "state_sha256": full_state_sha256(state),
        "activity_name": "turning_on_radio",
        "scene_model": "Rs_int",
        "asset_fingerprint": "behavior-assets-v1",
        "subtask_id": 1,
        "skill": "pick up from",
        "pool_type": "canonical",
        "task_description": "pick up the radio",
        "control_json": json.dumps({"skill": "pick up from"}),
        "metadata": {"reward": {"potential_terms": []}, "instance_id": 1},
    }
    values.update(overrides)
    if "control_json" not in overrides:
        values["control_json"] = json.dumps({"skill": values["skill"]})
    return SubpoolSnapshot(**values)


def test_store_round_trip_and_checksum_validation(tmp_path):
    state = _state()
    manifest = tmp_path / "manifest.jsonl"
    record = _record(state)
    SubpoolStore(manifest).append(record, state)

    catalog = SubpoolCatalog.from_jsonl(manifest)
    loaded = catalog.load_state(record)
    assert torch.equal(loaded["sim"], state["sim"])
    assert loaded["robot"]["ag_obj_constraint_params"]["right"]["object"] == "radio"

    state_path = tmp_path / record.state_path
    state_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        catalog.load_state(record)


def test_sampling_balances_subtasks_before_snapshot_count(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    for index in range(20):
        state = _state(index)
        subtask_id = 0 if index < 19 else 1
        record = _record(
            state,
            snapshot_id=f"state-{index}",
            subtask_id=subtask_id,
            skill=f"skill-{subtask_id}",
        )
        store.append(record, state)

    catalog = SubpoolCatalog.from_jsonl(manifest)
    rng = np.random.default_rng(7)
    sampled = [catalog.sample(rng).subtask_id for _ in range(2000)]
    fraction_subtask_one = sum(value == 1 for value in sampled) / len(sampled)
    assert 0.45 < fraction_subtask_one < 0.55


def test_catalog_rejects_mixed_runtime_instances(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    state = _state()
    store.append(_record(state, snapshot_id="instance-1"), state)
    second_state = _state(1)
    store.append(
        _record(
            second_state,
            snapshot_id="instance-2",
            metadata={
                "reward": {"potential_terms": []},
                "instance_id": 2,
            },
        ),
        second_state,
    )

    catalog = SubpoolCatalog.from_jsonl(manifest)
    with pytest.raises(ValueError, match="persistent BEHAVIOR simulator"):
        _ = catalog.runtime_signature


def test_catalog_rejects_cross_subtask_reward_scale_drift(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    state = _state()
    store.append(_record(state, snapshot_id="task-0", subtask_id=0), state)
    second_state = _state(1)
    store.append(
        _record(
            second_state,
            snapshot_id="task-1",
            subtask_id=1,
            metadata={
                "reward": {
                    "potential_terms": [],
                    "success_bonus": 20.0,
                },
                "instance_id": 1,
            },
        ),
        second_state,
    )

    with pytest.raises(ValueError, match="same cumulative step-penalty budget"):
        SubpoolCatalog.from_jsonl(manifest)


def test_catalog_rejects_path_escape(tmp_path):
    state = _state()
    record = replace(_record(state), state_path="../outside.pt")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    catalog = SubpoolCatalog.from_jsonl(manifest, verify_states=False)
    with pytest.raises(ValueError, match="escapes catalog root"):
        catalog.state_path(record)


def test_snapshot_rejects_flat_state_that_loses_assisted_grasp():
    state = _state()
    with pytest.raises(ValueError, match="assisted-grasp constraints"):
        replace(_record(state), state_path="states/state-0.npy")


def test_correctness_config_rejects_unsafe_optimizations():
    safe = OmegaConf.create(
        {
            "num_env_subprocess": 1,
            "skip_intermediate_obs_in_chunk": False,
            "auto_reset": False,
            "enable_offload": False,
            "renderer_mode": "official",
        }
    )
    validate_subpool_env_config(safe, num_envs=1, pipeline_stage_num=1)

    unsafe = OmegaConf.merge(safe, {"skip_intermediate_obs_in_chunk": True})
    with pytest.raises(ValueError, match="skip_intermediate_obs_in_chunk"):
        validate_subpool_env_config(unsafe, num_envs=1, pipeline_stage_num=1)

    non_parity = OmegaConf.merge(safe, {"renderer_mode": "rlinf"})
    with pytest.raises(ValueError, match="renderer_mode must be official"):
        validate_subpool_env_config(non_parity, num_envs=1, pipeline_stage_num=1)


def test_official_renderer_mode_does_not_import_or_override_omnigibson():
    apply_runtime_renderer_settings("official")

    with pytest.raises(ValueError, match="Unsupported BEHAVIOR renderer_mode"):
        apply_runtime_renderer_settings("unknown")


def test_sync_robot_after_pose_override_resets_controller_targets():
    calls = []
    positions = torch.tensor([1.0, 2.0])

    class FakeRobot:
        n_joints = 2

        def keep_still(self):
            calls.append("keep_still")

        def get_joint_positions(self):
            calls.append("get_joint_positions")
            return positions

        def set_joint_positions(self, *, positions, drive):
            calls.append(("set_joint_positions", positions.clone(), drive))

        def set_joint_velocities(self, *, velocities, drive):
            calls.append(("set_joint_velocities", velocities.clone(), drive))

    sync_robot_after_pose_override(FakeRobot())

    assert calls[0:2] == ["keep_still", "get_joint_positions"]
    assert calls[2][0] == "set_joint_positions"
    assert torch.equal(calls[2][1], positions)
    assert calls[2][2] is False
    assert calls[3][0] == "set_joint_velocities"
    assert torch.equal(calls[3][1], torch.zeros_like(positions))
    assert calls[3][2] is False
    assert calls[4] == "keep_still"


def test_terminal_subpool_chunk_freezes_state_and_executes_nothing():
    terminal_obs = {"camera": torch.tensor([1.0])}
    terminal_info = {"subpool": {"success": True}}
    observations, rewards, terms, truncs, infos, executed = (
        _repeat_terminal_subpool_chunk(terminal_obs, terminal_info, chunk_size=3)
    )

    assert len(observations) == 3
    assert all(step_obs[0] is terminal_obs for step_obs in observations)
    assert all(step_info[0] is terminal_info for step_info in infos)
    assert not torch.stack(rewards).any()
    assert not torch.stack(terms).any()
    assert not torch.stack(truncs).any()
    assert not torch.stack(executed).any()


def test_support_surface_distance_tracks_vertical_and_footprint_error():
    support = (torch.tensor([-1.0, -1.0, 0.0]), torch.tensor([1.0, 1.0, 0.8]))
    centered_object = (
        torch.tensor([-0.1, -0.1, 1.0]),
        torch.tensor([0.1, 0.1, 1.2]),
    )
    outside_object = (
        torch.tensor([1.9, -0.1, 1.0]),
        torch.tensor([2.1, 0.1, 1.2]),
    )

    assert _support_surface_distance(centered_object, support) == pytest.approx(0.2)
    assert _support_surface_distance(outside_object, support) == pytest.approx(
        np.hypot(1.0, 0.2)
    )


def test_round_robin_requires_one_subtask_per_env_rank():
    validate_round_robin_coverage((0, 1, 2, 3), env_world_size=4, fixed_subtask_id=None)
    with pytest.raises(ValueError, match="one subtask per env rank"):
        validate_round_robin_coverage(
            (0, 1, 2), env_world_size=4, fixed_subtask_id=None
        )
    validate_round_robin_coverage((0, 1, 2, 3), env_world_size=4, fixed_subtask_id=2)


def test_export_requires_task_reward_and_explicit_episode_ids():
    validate_subpool_export_request(
        instance_reward_mode="task",
        run_episode_idx=None,
        run_episode_indices=[10],
    )
    with pytest.raises(ValueError, match="instance_reward_mode=task"):
        validate_subpool_export_request(
            instance_reward_mode="potential",
            run_episode_idx=None,
            run_episode_indices=[10],
        )
    with pytest.raises(ValueError, match="positional run_episode_idx"):
        validate_subpool_export_request(
            instance_reward_mode="task",
            run_episode_idx=10,
            run_episode_indices=None,
        )
    with pytest.raises(ValueError, match="non-empty explicit"):
        validate_subpool_export_request(
            instance_reward_mode="task",
            run_episode_idx=None,
            run_episode_indices=None,
        )


def test_rollout_horizon_must_cover_every_subtask_timeout():
    validate_subpool_rollout_horizons(
        (384, 1152, 384, 512),
        episode_horizon=1152,
        rollout_horizon=1152,
    )
    with pytest.raises(ValueError, match="max_episode_steps"):
        validate_subpool_rollout_horizons(
            (384, 1152),
            episode_horizon=1024,
            rollout_horizon=1152,
        )
    with pytest.raises(ValueError, match="max_steps_per_rollout_epoch"):
        validate_subpool_rollout_horizons(
            (384, 1152),
            episode_horizon=1152,
            rollout_horizon=1024,
        )


def test_subpool_env_keeps_bootstrap_template_separate_from_task_instance(
    tmp_path, monkeypatch
):
    state = _state()
    manifest = tmp_path / "manifest.jsonl"
    record = _record(
        state,
        metadata={
            "reward": {
                "potential_terms": [],
                "max_steps": 16,
                "success_bonus": 10.0,
                "timeout_penalty": -2.0,
                "step_penalty": -0.0625,
                "progress_clip": 1.0,
            },
            "instance_id": 7,
        },
    )
    SubpoolStore(manifest).append(record, state)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "num_env_subprocess": 1,
            "skip_intermediate_obs_in_chunk": False,
            "auto_reset": False,
            "enable_offload": False,
            "renderer_mode": "official",
            "max_episode_steps": 16,
            "max_steps_per_rollout_epoch": 16,
            "subpool": {
                "manifest_path": str(manifest),
                "bootstrap_instance_id": 0,
                "fixed_subtask_id": 1,
                "subtask_sampling": "round_robin",
                "pool_weights": {"canonical": 1.0},
                "dynamic_updates": False,
            },
            "omni_config": {
                "task": {
                    "activity_name": "placeholder",
                    "activity_instance_id": 99,
                    "instance_resample_mode": "disabled",
                    "online_object_sampling": False,
                },
                "scene": {"scene_model": "placeholder"},
            },
        }
    )
    captured = {}

    def fake_behavior_init(self, parent_cfg, *args, **kwargs):
        captured["cfg"] = parent_cfg

    monkeypatch.setattr(BehaviorEnv, "__init__", fake_behavior_init)
    BehaviorSubpoolEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=SimpleNamespace(group_world_size=1),
    )

    assert captured["cfg"].omni_config.task.activity_instance_id == 0
    assert captured["cfg"].omni_config.task.activity_name == "turning_on_radio"
    assert captured["cfg"].omni_config.scene.scene_model == "Rs_int"
    assert record.metadata["instance_id"] == 7


def test_behavior_process_is_pinned_to_parent_env_node_and_gpu(monkeypatch):
    node_id = "a" * 56
    captured = {}

    class FakeMethod:
        def remote(self):
            return object()

    class FakeProcess:
        get_activity_name = FakeMethod()

    class FakeActorOptions:
        def remote(self, *_args):
            return FakeProcess()

    def fake_options(**kwargs):
        captured.update(kwargs)
        return FakeActorOptions()

    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: node_id),
    )
    monkeypatch.setattr(ray, "get", lambda _refs: ["turning_on_radio"])
    monkeypatch.setattr(BehaviorProcess, "options", fake_options)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    cfg = OmegaConf.create(
        {
            "total_num_envs": 1,
            "num_env_subprocess": 1,
            "behavior": {"init_retry_count": 1},
        }
    )
    pool = BehaviorProcessPool(
        cfg,
        total_num_envs=1,
        num_env_subprocess=1,
        pipeline_stage_num=1,
    )

    strategy = captured["scheduling_strategy"]
    assert strategy.node_id == node_id
    assert strategy.soft is False
    assert captured["runtime_env"] == {
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "3",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
    }
    assert pool.activity_name == "turning_on_radio"

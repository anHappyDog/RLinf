import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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
    _compact_policy_observation,
    _isolated_appdata_path,
    _rebase_scene_state,
    _repeat_terminal_subpool_chunk,
    _SubpoolSlotRuntime,
    _support_surface_distance,
    _translate_proprio_position_to_scene,
    _validate_vector_environment_contract,
)
from rlinf.envs.behavior.subpool import (
    FailureStateStore,
    SubpoolCatalog,
    SubpoolSnapshot,
    SubpoolStore,
    analyze_subtask_failure,
    full_state_sha256,
    merge_subpool_manifests,
    relative_tilt_angle_deg,
    validate_round_robin_coverage,
    validate_subpool_env_config,
    validate_subpool_export_request,
    validate_subpool_rollout_horizons,
)
from rlinf.envs.behavior.subpool_reward import apply_reward_overrides
from rlinf.envs.behavior.utils import (
    apply_runtime_renderer_settings,
    sync_robot_after_pose_override,
)
from toolkits.b1k_grounded.build_failure_recovery_catalog import (
    build_recovery_catalog,
)
from toolkits.b1k_grounded.prepare_canonical_pool import (
    assemble_canonical_pool,
    collect_snapshot_scores,
    load_snapshot_scores,
    write_catalog_partitions,
    write_stratified_split,
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


def test_vector_subpool_rejects_unsafe_subset_step_implementation():
    class UnsafeVectorEnvironment:
        pass

    with pytest.raises(RuntimeError, match="global scene registry"):
        _validate_vector_environment_contract(
            UnsafeVectorEnvironment,
            required=True,
        )

    UnsafeVectorEnvironment.preserves_global_scene_registry_on_subset_step = True
    _validate_vector_environment_contract(
        UnsafeVectorEnvironment,
        required=True,
    )


def test_reward_overrides_do_not_mutate_manifest_spec():
    manifest_spec = {
        "potential_terms": [],
        "step_penalty": -0.01,
        "success_bonus": 10.0,
        "timeout_penalty": -2.0,
        "max_steps": 1280,
    }

    resolved = apply_reward_overrides(manifest_spec, {"step_penalty": 0.0})

    assert resolved["step_penalty"] == 0.0
    assert manifest_spec["step_penalty"] == -0.01


def test_compact_policy_observation_drops_segmentation_payload():
    head_rgb = torch.zeros(8, 8, 4)
    left_rgb = torch.ones(4, 4, 4)
    right_rgb = torch.full((4, 4, 4), 2.0)
    segmentation = torch.arange(64).reshape(8, 8)
    state = torch.arange(32)
    raw_obs = {
        "robot": {
            "zed_link:Camera:0": {
                "rgb": head_rgb,
                "seg_instance_id": segmentation,
            },
            "left_realsense_link:Camera:0": {
                "rgb": left_rgb,
                "seg_instance_id": segmentation[:4, :4],
            },
            "right_realsense_link:Camera:0": {
                "rgb": right_rgb,
                "seg_instance_id": segmentation[:4, :4],
            },
            "proprio": state,
        },
        "_subpool": {"task_description": "<subgoal>pick up the radio"},
    }

    compact = _compact_policy_observation(raw_obs)

    assert set(compact) == {
        "main_images",
        "wrist_images",
        "state",
        "task_description",
    }
    assert compact["main_images"].shape == (8, 8, 3)
    assert compact["main_images"].dtype == torch.uint8
    assert compact["wrist_images"].shape == (2, 4, 4, 3)
    assert torch.equal(compact["state"], state)
    assert compact["task_description"] == "<subgoal>pick up the radio"
    assert "seg_instance_id" not in str(compact.keys())


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


def test_catalog_state_cache_reuses_validated_cpu_state(tmp_path):
    state = _state()
    manifest = tmp_path / "manifest.jsonl"
    record = _record(state)
    SubpoolStore(manifest).append(record, state)

    catalog = SubpoolCatalog.from_jsonl(
        manifest,
        verify_states=False,
        state_cache_size=1,
    )
    first = catalog.load_state(record)
    state_path = tmp_path / record.state_path
    state_path.write_bytes(b"corrupt after validated cache fill")
    second = catalog.load_state(record)

    assert second is first
    assert catalog.state_cache_info == {
        "size": 1,
        "capacity": 1,
        "hits": 1,
        "misses": 1,
    }


def test_catalog_state_cache_evicts_least_recently_used_state(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    first_state = _state()
    second_state = _state(10)
    first_record = _record(first_state, snapshot_id="state-0")
    second_record = _record(second_state, snapshot_id="state-1")
    store = SubpoolStore(manifest)
    store.append(first_record, first_state)
    store.append(second_record, second_state)

    catalog = SubpoolCatalog.from_jsonl(
        manifest,
        verify_states=False,
        state_cache_size=1,
    )
    catalog.load_state(first_record)
    catalog.load_state(second_record)

    assert catalog.state_cache_info == {
        "size": 1,
        "capacity": 1,
        "hits": 0,
        "misses": 2,
    }
    (tmp_path / first_record.state_path).write_bytes(b"corrupt after eviction")
    with pytest.raises(ValueError, match="checksum mismatch"):
        catalog.load_state(first_record)


def test_failure_state_store_writes_restorable_state_and_metadata(tmp_path):
    state = _state()
    source_snapshot = _record(state, episode_index=50).to_dict()
    store = FailureStateStore(tmp_path / "failures", run_id="radio-pickup")

    metadata_path = store.capture(
        state,
        policy_global_step=34,
        collection_index=91,
        sampling_group=3,
        capture_kind="terminal",
        source_snapshot=source_snapshot,
        outcome={
            "failure_reason": "timeout",
            "success": False,
            "timeout": True,
            "elapsed_steps": 1280,
            "return": -2.0,
        },
        analysis=analyze_subtask_failure(
            "pick up from",
            {
                "in_hand": False,
                "on_original_support": True,
                "tilt_angle_deg": 90.0,
                "tipped": True,
            },
            termination_reason="timeout",
        ).to_dict(),
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state_path = metadata_path.parent / metadata["state_path"]
    loaded = torch.load(state_path, weights_only=False)
    assert metadata_path.parent.name == "global_step_000034"
    assert metadata["run_id"] == "radio-pickup"
    assert metadata["collection_index"] == 91
    assert metadata["sampling_group"] == 3
    assert metadata["source_snapshot"]["snapshot_id"] == "state-0"
    assert metadata["capture_kind"] == "terminal"
    assert metadata["outcome"]["failure_reason"] == "timeout"
    assert metadata["analysis"]["recovery"]["status"] == "eligible"
    assert (
        hashlib.sha256(state_path.read_bytes()).hexdigest() == metadata["state_sha256"]
    )
    assert torch.equal(loaded["sim"], state["sim"])
    assert not list((tmp_path / "failures").rglob(".*.pt.*"))
    assert not list((tmp_path / "failures").rglob(".*.json.*"))


def test_failure_state_store_rejects_success_outcome(tmp_path):
    state = _state()
    store = FailureStateStore(tmp_path / "failures")

    with pytest.raises(ValueError, match="cannot describe a success"):
        store.capture(
            state,
            policy_global_step=None,
            collection_index=None,
            sampling_group=None,
            capture_kind="terminal",
            source_snapshot=_record(state).to_dict(),
            outcome={"success": True, "timeout": False},
            analysis={
                "failure_tags": ["gripper_empty"],
                "recovery": {"status": "not_needed", "reason": "upright"},
            },
        )


def test_build_recovery_catalog_selects_eligible_terminal_states(tmp_path):
    state = _state()
    canonical_manifest = tmp_path / "canonical" / "manifest.jsonl"
    canonical = _record(state, episode_index=50)
    SubpoolStore(canonical_manifest).append(canonical, state)
    failure_store = FailureStateStore(tmp_path / "failures")
    reference_orientation = [0.0, 0.0, 0.0, 1.0]
    analysis = analyze_subtask_failure(
        "pick up from",
        {
            "in_hand": False,
            "on_original_support": True,
            "reference_orientation_xyzw": reference_orientation,
            "tilt_angle_deg": 90.0,
            "tipped": True,
        },
        termination_reason="timeout",
    )
    terminal_metadata_path = failure_store.capture(
        state,
        policy_global_step=20,
        collection_index=None,
        sampling_group=0,
        capture_kind="terminal",
        source_snapshot=canonical.to_dict(),
        outcome={"success": False, "timeout": True},
        analysis=analysis.to_dict(),
    )
    failure_store.capture(
        state,
        policy_global_step=20,
        collection_index=None,
        sampling_group=0,
        capture_kind="stable_recovery_event",
        source_snapshot=canonical.to_dict(),
        outcome={"success": False, "timeout": False},
        analysis=analysis.to_dict(),
    )

    output_manifest = tmp_path / "recovery" / "manifest.jsonl"
    terminal_failure_id = json.loads(
        terminal_metadata_path.read_text(encoding="utf-8")
    )["failure_id"]
    summary = build_recovery_catalog(
        tmp_path / "failures",
        canonical_manifest,
        output_manifest,
        {terminal_failure_id},
    )

    catalog = SubpoolCatalog.from_jsonl(output_manifest)
    recovery = next(
        record for record in catalog.records if record.pool_type == "recovery"
    )
    assert summary == {
        "canonical_snapshots": 1,
        "recovery_snapshots": 1,
        "total_snapshots": 2,
    }
    assert recovery.frame_index is None
    assert (
        recovery.metadata["recovery_provenance"]["reference_orientation_xyzw"]
        == reference_orientation
    )
    assert torch.equal(catalog.load_state(recovery)["sim"], state["sim"])


@pytest.mark.parametrize(
    ("facts", "expected_status", "expected_tag"),
    [
        (
            {
                "in_hand": False,
                "on_original_support": True,
                "tilt_angle_deg": 2.0,
                "tipped": False,
            },
            "not_needed",
            "target_upright",
        ),
        (
            {
                "in_hand": False,
                "on_original_support": True,
                "tilt_angle_deg": 82.0,
                "tipped": True,
            },
            "eligible",
            "target_tipped",
        ),
        (
            {
                "in_hand": False,
                "on_original_support": False,
                "tilt_angle_deg": 90.0,
                "tipped": True,
            },
            "ineligible",
            "target_off_original_support",
        ),
    ],
)
def test_pickup_failure_analysis_uses_simulator_facts(
    facts, expected_status, expected_tag
):
    analysis = analyze_subtask_failure(
        "pick up from",
        facts,
        termination_reason="timeout",
    )

    assert analysis.recovery_status == expected_status
    assert expected_tag in analysis.failure_tags


def test_unknown_skill_failure_is_not_admitted_to_recovery():
    analysis = analyze_subtask_failure(
        "press",
        {"completed": False},
        termination_reason="timeout",
    )

    assert analysis.recovery_status == "unknown"
    assert analysis.analyzer == "generic-v1"


def test_relative_tilt_ignores_yaw_but_detects_fall():
    half_sqrt = np.sqrt(0.5)
    identity = [0.0, 0.0, 0.0, 1.0]
    yaw_90 = [0.0, 0.0, half_sqrt, half_sqrt]
    roll_90 = [half_sqrt, 0.0, 0.0, half_sqrt]

    assert relative_tilt_angle_deg(identity, yaw_90) == pytest.approx(0.0)
    assert relative_tilt_angle_deg(identity, roll_90) == pytest.approx(90.0)


def test_behavior_process_extracts_pickup_failure_facts():
    class Target:
        name = "radio_89"

        @staticmethod
        def get_position_orientation(frame="world"):
            assert frame == "scene"
            return torch.tensor([1.0, 2.0, 3.0]), torch.tensor(
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
            )

        @staticmethod
        def get_linear_velocity():
            return torch.tensor([0.01, 0.0, 0.0])

        @staticmethod
        def get_angular_velocity():
            return torch.tensor([0.0, 0.02, 0.0])

    process_type = BehaviorProcess.__ray_metadata__.modified_class
    process = object.__new__(process_type)
    slot = _SubpoolSlotRuntime(
        control=SimpleNamespace(skill="pick up from"),
        subtask_id=1,
        task_reward=SimpleNamespace(
            _stage_defs=(
                {},
                {"objects": (Target(), SimpleNamespace(name="coffee_table"))},
            )
        ),
        failure_reference_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    process.failure_tipped_angle_deg = 45.0
    process.failure_max_linear_speed = 0.05
    process.failure_max_angular_speed = 0.2

    facts = process._failure_facts(
        slot, {"completed": False, "in_hand": False, "on_support": True}
    )

    assert facts["target_name"] == "radio_89"
    assert facts["support_name"] == "coffee_table"
    assert facts["tilt_angle_deg"] == pytest.approx(90.0)
    assert facts["tipped"] is True
    assert facts["stable"] is True


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


def test_catalog_samples_initial_states_across_episodes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    state = _state()
    store.append(
        _record(state, snapshot_id="episode-10", episode_index=10),
        state,
    )
    second_state = _state(1)
    store.append(
        _record(
            second_state,
            snapshot_id="episode-20",
            episode_index=20,
            metadata={
                "reward": {"potential_terms": []},
                "instance_id": 2,
            },
        ),
        second_state,
    )

    catalog = SubpoolCatalog.from_jsonl(manifest)
    assert catalog.runtime_signature == ("turning_on_radio", "Rs_int")
    rng = np.random.default_rng(7)
    sampled_episodes = {
        catalog.sample(rng, subtask_id=1).episode_index for _ in range(100)
    }
    assert sampled_episodes == {10, 20}


def test_catalog_shuffled_round_robin_covers_states_and_keeps_retries(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    for episode_index in (10, 20, 30, 40):
        state = _state(episode_index)
        store.append(
            _record(
                state,
                snapshot_id=f"episode-{episode_index}",
                episode_index=episode_index,
            ),
            state,
        )
    catalog = SubpoolCatalog.from_jsonl(manifest)

    first_update = [
        catalog.shuffled_round_robin_snapshot(
            seed=123,
            update_index=7,
            logical_group_index=group_index,
            subtask_id=1,
            pool_weights={"canonical": 1.0, "recovery": 0.0},
        )
        for group_index in range(4)
    ]
    retry = catalog.shuffled_round_robin_snapshot(
        seed=123,
        update_index=7,
        logical_group_index=2,
        subtask_id=1,
        pool_weights={"canonical": 1.0, "recovery": 0.0},
    )

    assert {record.episode_index for record in first_update} == {10, 20, 30, 40}
    assert retry.snapshot_id == first_update[2].snapshot_id


def test_prepare_canonical_pool_filters_gt_and_writes_matched_split(tmp_path):
    source_manifest = tmp_path / "source" / "manifest.jsonl"
    source_store = SubpoolStore(source_manifest)
    for episode_index in range(8):
        state = _state(episode_index)
        control = {"skill": "pick up from", "subgoal": "pick up the radio"}
        if episode_index == 5:
            control["subgoal"] = None
        end_frame = 201 + episode_index
        if episode_index == 6:
            end_frame = 1481
        source_store.append(
            _record(
                state,
                snapshot_id=f"episode-{episode_index}",
                episode_index=episode_index,
                control_json=json.dumps(control),
                metadata={
                    "instance_id": episode_index,
                    "gt_validation": {
                        "success": episode_index != 7,
                        "start_frame": 200,
                        "end_frame": end_frame,
                    },
                    "reward": {
                        "max_steps": 3000,
                        "potential_terms": [{"key": "distance"}],
                        "step_penalty": -1 / 3000,
                    },
                },
            ),
            state,
        )

    assembled_manifest = tmp_path / "assembled" / "manifest.jsonl"
    records = assemble_canonical_pool(
        [source_manifest],
        assembled_manifest,
        subtask_id=1,
        horizon=1280,
    )
    assert len(records) == 5
    assert {record.episode_index for record in records} == set(range(5))
    assert all(record.metadata["reward"]["max_steps"] == 1280 for record in records)
    assert all(not record.metadata["reward"]["potential_terms"] for record in records)

    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "snapshots": {
                    f"episode-{episode_index}": {
                        "successes": episode_index * 2,
                        "attempts": 10,
                    }
                    for episode_index in range(5)
                }
            }
        ),
        encoding="utf-8",
    )
    scores = load_snapshot_scores(scores_path)
    split_dir = tmp_path / "split"
    summary = write_stratified_split(
        assembled_manifest,
        scores_path,
        split_dir,
        split_size=2,
        seed=9,
    )

    train = SubpoolCatalog.from_jsonl(split_dir / "train" / "manifest.jsonl")
    heldout = SubpoolCatalog.from_jsonl(split_dir / "heldout_eval" / "manifest.jsonl")
    assert len(train.records) == len(heldout.records) == 2
    assert {record.snapshot_id for record in train.records}.isdisjoint(
        record.snapshot_id for record in heldout.records
    )
    assert summary["train"]["count"] == summary["heldout_eval"]["count"] == 2
    assert scores["episode-4"].success_rate == pytest.approx(0.8)

    partitions = write_catalog_partitions(
        assembled_manifest,
        tmp_path / "partitions",
        partition_size=2,
    )
    assert [len(SubpoolCatalog.from_jsonl(path).records) for path in partitions] == [
        2,
        2,
        1,
    ]
    metrics_path = tmp_path / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                f"eval/snapshot/episode_{episode_index}/attempts": 10
                for episode_index in range(5)
            }
            | {
                f"eval/snapshot/episode_{episode_index}/success": episode_index / 10
                for episode_index in range(5)
            }
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline = collect_snapshot_scores(
        assembled_manifest,
        [metrics_path],
        baseline_path,
        expected_attempts=10,
    )
    assert baseline["snapshots"]["episode-4"]["successes"] == 4


def test_outcome_group_reset_ignores_desynchronized_auto_reset_rng():
    class FakeCatalog:
        subtask_ids = (1,)

        @staticmethod
        def sample(rng, **_kwargs):
            return SimpleNamespace(snapshot_id=str(rng.integers(1_000_000)))

    def make_env():
        env = BehaviorSubpoolEnv.__new__(BehaviorSubpoolEnv)
        env.num_envs = 1
        env.catalog = FakeCatalog()
        env._sampling_seed = 123
        env._sampling_groups = [0]
        env._rngs = [np.random.default_rng(123)]
        env._fixed_subtask_id = 1
        env._pool_weights = {"canonical": 1.0}
        env._subtask_cursors = [0]
        env._pending_outcome_collection_index = None
        env._pending_outcome_logical_group_indices = None
        env._pending_outcome_update_index = None
        env._outcome_snapshot_schedule = "random"
        env._sticky_outcome_snapshot = False
        env._active_outcome_snapshots = [None]
        return env

    early_env = make_env()
    late_env = make_env()
    late_env._rngs[0].integers(1_000_000, size=17)
    early_env.prepare_outcome_group_reset(9)
    late_env.prepare_outcome_group_reset(9)

    assert (
        early_env._sample_reset_snapshot(0).snapshot_id
        == late_env._sample_reset_snapshot(0).snapshot_id
    )


def test_fixed_snapshot_per_env_is_stable_across_evaluation_resets(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    for episode_index in (10, 20):
        state = _state(episode_index)
        store.append(
            _record(
                state,
                snapshot_id=f"episode-{episode_index}",
                episode_index=episode_index,
            ),
            state,
        )
    env = BehaviorSubpoolEnv.__new__(BehaviorSubpoolEnv)
    env.num_envs = 1
    env.catalog = SubpoolCatalog.from_jsonl(manifest)
    env._sampling_seed = 123
    env._sampling_groups = [1]
    env._rngs = [np.random.default_rng(124)]
    env._fixed_subtask_id = 1
    env._pool_weights = {"canonical": 1.0, "recovery": 0.0}
    env._subtask_cursors = [1]
    env._pending_outcome_collection_index = None
    env._pending_outcome_logical_group_indices = None
    env._pending_outcome_update_index = None
    env._outcome_snapshot_schedule = "random"
    env._sticky_outcome_snapshot = False
    env._active_outcome_snapshots = [None]
    env._fixed_snapshot_per_env = True

    first = env._sample_reset_snapshot(0)
    second = env._sample_reset_snapshot(0)

    assert first.snapshot_id == second.snapshot_id


def test_rebase_scene_state_preserves_object_local_pose_and_velocity():
    state = {
        "pos": torch.tensor([1.0, 2.0, 0.0]),
        "ori": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "registry": {
            "system_registry": {},
            "object_registry": {
                "radio": {
                    "root_link": {
                        "pos": torch.tensor([2.0, 4.0, 0.5]),
                        "ori": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                        "lin_vel": torch.tensor([1.0, 0.0, 0.0]),
                        "ang_vel": torch.tensor([0.0, 0.0, 1.0]),
                    }
                }
            },
        },
    }

    rebased = _rebase_scene_state(
        state,
        target_position=torch.tensor([10.0, -1.0, 0.0]),
        target_orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    root = rebased["registry"]["object_registry"]["radio"]["root_link"]
    assert torch.allclose(root["pos"], torch.tensor([11.0, 1.0, 0.5]))
    assert torch.allclose(root["lin_vel"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(
        state["registry"]["object_registry"]["radio"]["root_link"]["pos"],
        torch.tensor([2.0, 4.0, 0.5]),
    )


def test_vector_subpool_routes_dynamic_candidates_per_terminal_slot(monkeypatch):
    terminations = torch.tensor([[True, False], [False, False]])
    truncations = torch.tensor([[False, False], [True, False]])
    base_result = (None, None, terminations, truncations, None)
    monkeypatch.setattr(BehaviorEnv, "chunk_step", lambda *_args: base_result)

    env = BehaviorSubpoolEnv.__new__(BehaviorSubpoolEnv)
    env._dynamic_updates = True
    env.pool = MagicMock()
    env.pool.drain_pool_candidates.return_value = [
        {"success_state": "slot-0"},
        {"recovery_state": "slot-1"},
    ]
    env.current_snapshots = [MagicMock(), MagicMock()]
    env._append_online_candidates = MagicMock()

    assert env.chunk_step(torch.zeros(2, 2, 1)) is base_result
    assert env._append_online_candidates.call_args_list == [
        call(
            {"success_state": "slot-0"},
            snapshot=env.current_snapshots[0],
            success=True,
        ),
        call(
            {"recovery_state": "slot-1"},
            snapshot=env.current_snapshots[1],
            success=False,
        ),
    ]


def test_vector_proprio_robot_position_is_scene_relative():
    state = torch.arange(256, dtype=torch.float32)
    raw_obs = {"R1Pro": {"R1Pro::proprio": state}}

    result = _translate_proprio_position_to_scene(
        raw_obs,
        position_indices=np.s_[140:143],
        scene_position=torch.tensor([28.0, -3.0, 0.5]),
    )

    translated = result["R1Pro"]["R1Pro::proprio"]
    assert torch.equal(translated[140:143], torch.tensor([112.0, 144.0, 141.5]))
    assert torch.equal(translated[:140], state[:140])
    assert torch.equal(translated[143:], state[143:])
    assert torch.equal(state, torch.arange(256, dtype=torch.float32))


def test_vector_chunk_never_resumes_a_terminal_slot(monkeypatch):
    class Tracker:
        def __init__(self, outcomes):
            self.outcomes = iter(outcomes)
            self.steps = 0

        def step(self, _stage_info):
            self.steps += 1
            success, timeout = next(self.outcomes)
            return SimpleNamespace(
                success=success,
                timeout=timeout,
                reward=float(success),
                potential=0.0,
                progress=0.0,
                cumulative_progress=0.0,
                cumulative_step_penalty=0.0,
                cumulative_terminal_reward=float(success),
            )

    process_type = BehaviorProcess.__ray_metadata__.modified_class
    process = object.__new__(process_type)
    process.skip_intermediate_obs_in_chunk = True
    process.dynamic_pool_updates = False
    process.failure_state_store = None
    process.state_capture_interval = 1
    process.subpool_slots = [
        _SubpoolSlotRuntime(
            reward_tracker=Tracker([(True, False)]),
            subtask_id=1,
        ),
        _SubpoolSlotRuntime(
            reward_tracker=Tracker([(False, False), (False, False), (False, True)]),
            subtask_id=1,
        ),
    ]
    active_calls = []

    def step_shard(_actions, env_indices, *, need_obs):
        active_calls.append(list(env_indices))
        return (
            None if not need_obs else [{"slot": index} for index in env_indices],
            torch.zeros(len(env_indices)),
            torch.zeros(len(env_indices), dtype=torch.bool),
            torch.zeros(len(env_indices), dtype=torch.bool),
            [{"stage": {}} for _ in env_indices],
        )

    process._step_shard = step_shard
    process._observe_policy = lambda indices: [{"slot": index} for index in indices]
    process._apply_direct_navigation_predicate = lambda *_args: None
    process._attach_arm_specific_distances = lambda *_args: None
    process._maybe_capture_stable_recovery_event = lambda *_args: None
    monkeypatch.setattr(
        "rlinf.envs.behavior.behavior_env.get_stage_info",
        lambda info, _subtask_id: info["stage"],
    )

    observations, _rewards, terms, truncs, _infos, executed = (
        process._chunk_step_until_done(torch.zeros(2, 4, 23), [0, 1])
    )

    assert active_calls == [[0, 1], [1], [1]]
    assert torch.stack(executed, dim=1).tolist() == [
        [True, False, False, False],
        [True, True, True, False],
    ]
    assert torch.stack(terms, dim=1).tolist()[0] == [True, False, False, False]
    assert torch.stack(truncs, dim=1).tolist()[1] == [False, False, True, False]
    assert observations[-1] == [{"slot": 0}, {"slot": 1}]


def test_behavior_process_pool_preserves_slot_alignment_when_merging_shards():
    pool = object.__new__(BehaviorProcessPool)
    pool.num_env_subprocess = 2
    pool.skip_intermediate_obs_in_chunk = False
    plan = pool._slice_plan(global_start=1, num_envs=5)
    shard_results = []
    for _subprocess, positions, _local_rows in plan:
        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []
        executed = []
        for timestep in range(2):
            observations.append(
                [{"slot": position, "timestep": timestep} for position in positions]
            )
            rewards.append(
                torch.tensor([10 * timestep + position for position in positions])
            )
            terminations.append(
                torch.tensor([position == 2 for position in positions])
            )
            truncations.append(
                torch.tensor([position == 4 for position in positions])
            )
            infos.append([{"slot": position} for position in positions])
            executed.append(torch.tensor([position != 3 for position in positions]))
        shard_results.append(
            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
                executed,
            )
        )

    merged = pool._merge_shards(
        shard_results,
        plan,
        slice_num_envs=5,
        chunk_size=2,
    )
    observations, rewards, terminations, truncations, infos, executed = merged

    for timestep in range(2):
        assert observations[timestep] == [
            {"slot": position, "timestep": timestep} for position in range(5)
        ]
        assert rewards[timestep].tolist() == [
            10 * timestep + position for position in range(5)
        ]
        assert terminations[timestep].tolist() == [False, False, True, False, False]
        assert truncations[timestep].tolist() == [False, False, False, False, True]
        assert infos[timestep] == [{"slot": position} for position in range(5)]
        assert executed[timestep].tolist() == [True, True, True, False, True]


def test_catalog_rejects_mixed_runtime_scenes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    store = SubpoolStore(manifest)
    state = _state()
    store.append(_record(state, snapshot_id="scene-a"), state)
    second_state = _state(1)
    store.append(
        _record(second_state, snapshot_id="scene-b", scene_model="Beechwood_0_int"),
        second_state,
    )

    catalog = SubpoolCatalog.from_jsonl(manifest)
    with pytest.raises(ValueError, match="persistent BEHAVIOR simulator"):
        _ = catalog.runtime_signature


def test_merge_subpool_manifests_copies_all_episode_states(tmp_path):
    input_manifests = []
    for episode_index in (10, 20):
        source_root = tmp_path / f"episode-{episode_index}"
        manifest = source_root / "manifest.jsonl"
        state = _state(episode_index)
        record = _record(
            state,
            snapshot_id=f"episode-{episode_index}",
            episode_index=episode_index,
            metadata={
                "reward": {"potential_terms": []},
                "instance_id": episode_index // 10,
            },
        )
        SubpoolStore(manifest).append(record, state)
        input_manifests.append(manifest)

    output_manifest = tmp_path / "merged" / "manifest.jsonl"
    merged = merge_subpool_manifests(input_manifests, output_manifest)

    assert [record.episode_index for record in merged.records] == [10, 20]
    assert torch.equal(merged.load_state(merged.records[1])["sim"], _state(20)["sim"])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        merge_subpool_manifests(input_manifests, output_manifest)


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
    validate_subpool_env_config(
        OmegaConf.merge(safe, {"auto_reset": True}),
        num_envs=1,
        pipeline_stage_num=1,
    )

    boundary_obs = OmegaConf.merge(safe, {"skip_intermediate_obs_in_chunk": True})
    validate_subpool_env_config(boundary_obs, num_envs=1, pipeline_stage_num=1)

    non_parity = OmegaConf.merge(safe, {"renderer_mode": "rlinf"})
    with pytest.raises(ValueError, match="renderer_mode must be official"):
        validate_subpool_env_config(non_parity, num_envs=1, pipeline_stage_num=1)

    termination_without_subpool = OmegaConf.merge(
        safe,
        {"subpool": {"skip_official_task_termination": True}},
    )
    with pytest.raises(
        ValueError,
        match="skip_official_task_termination requires subpool.enabled",
    ):
        validate_subpool_env_config(
            termination_without_subpool,
            num_envs=1,
            pipeline_stage_num=1,
        )

    termination_with_subpool = OmegaConf.merge(
        safe,
        {
            "subpool": {
                "enabled": True,
                "skip_official_task_termination": True,
            }
        },
    )
    validate_subpool_env_config(
        termination_with_subpool,
        num_envs=1,
        pipeline_stage_num=1,
    )

    missing_capture_dir = OmegaConf.merge(
        safe,
        {"subpool": {"failure_state_capture": {"enabled": True}}},
    )
    with pytest.raises(ValueError, match="failure_state_capture.output_dir"):
        validate_subpool_env_config(
            missing_capture_dir,
            num_envs=1,
            pipeline_stage_num=1,
        )


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


def test_terminal_subpool_chunk_can_return_only_boundary_observation():
    terminal_obs = {"camera": torch.tensor([1.0])}
    terminal_info = {"subpool": {"success": True}}
    observations, _rewards, _terms, _truncs, _infos, _executed = (
        _repeat_terminal_subpool_chunk(
            terminal_obs,
            terminal_info,
            chunk_size=3,
            skip_intermediate_obs=True,
        )
    )

    assert observations[:2] == (None, None)
    assert observations[2][0] is terminal_obs


def test_subpool_metrics_report_actual_primitive_steps():
    env = BehaviorEnv.__new__(BehaviorEnv)
    env.num_envs = 1
    env.max_episode_steps = torch.tensor(2848)
    env.ignore_terminations = False
    env.returns = torch.zeros(1)
    env.success_once = torch.zeros(1, dtype=torch.bool)

    metrics = env._record_metrics(
        torch.tensor([2.0]),
        [
            {
                "episode_length": 2847,
                "subpool": {
                    "subtask_id": 1,
                    "pool_type": "canonical",
                    "success": True,
                    "timeout": False,
                    "elapsed_steps": 907,
                },
            }
        ],
    )["episode"]

    assert metrics["episode_length"].item() == 907
    assert metrics["episode_len"].item() == 907
    assert metrics["reward"].item() == pytest.approx(2 / 907)


def test_single_env_observation_batching_owns_sensor_buffers():
    env = BehaviorEnv.__new__(BehaviorEnv)
    env.num_envs = 1
    env.task_description = "pick up the radio"
    main = torch.randint(0, 256, (8, 8, 3), dtype=torch.uint8)
    wrists = torch.randint(0, 256, (2, 4, 4, 3), dtype=torch.uint8)
    state = torch.arange(32, dtype=torch.float32)

    wrapped = env._wrap_obs(
        [
            {
                "main_images": main,
                "wrist_images": wrists,
                "state": state,
                "task_description": "<subgoal>pick up the radio",
            }
        ]
    )

    assert wrapped["main_images"].shape == (1, 8, 8, 3)
    assert wrapped["wrist_images"].shape == (1, 2, 4, 4, 3)
    assert wrapped["states"].shape == (1, 32)
    assert wrapped["main_images"].data_ptr() != main.data_ptr()
    assert wrapped["wrist_images"].data_ptr() != wrists.data_ptr()
    assert wrapped["states"].data_ptr() != state.data_ptr()
    assert wrapped["task_descriptions"] == ["<subgoal>pick up the radio"]

    main.zero_()
    wrists.zero_()
    state.zero_()
    assert wrapped["main_images"].count_nonzero() > 0
    assert wrapped["wrist_images"].count_nonzero() > 0
    assert wrapped["states"].count_nonzero() > 0


def test_behavior_process_dumps_flat_simulator_state(monkeypatch):
    expected = torch.tensor([1.0, 2.0, 3.0])

    class FakeSimulator:
        def dump_state(self, *, serialized):
            assert serialized is True
            return expected

    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(sim=FakeSimulator()))
    process_type = BehaviorProcess.__ray_metadata__.modified_class

    assert process_type.dump_serialized_state() is expected


def test_behavior_process_can_skip_official_task_termination():
    calls = []

    class FakeVectorEnv:
        def step(self, actions, **kwargs):
            calls.append((actions, kwargs))
            return "result"

    process_type = BehaviorProcess.__ray_metadata__.modified_class
    process = process_type.__new__(process_type)
    process.env = FakeVectorEnv()
    process.step_supports_get_obs = True
    process.step_supports_render = True
    process.step_supports_env_indices = True
    process.skip_official_task_termination = True

    result = process._call_step(
        [torch.zeros(2)],
        env_indices=[0],
        get_obs=False,
        render=False,
    )

    assert result == "result"
    assert len(calls) == 1
    assert torch.equal(calls[0][0][0], torch.zeros(2))
    assert calls[0][1] == {
        "get_obs": False,
        "render": False,
        "evaluate_termination": False,
        "env_indices": [0],
    }


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


def test_isolated_appdata_path_uses_stable_node_name(monkeypatch):
    monkeypatch.delenv("RLINF_NODE_RANK", raising=False)
    monkeypatch.setenv("RANK", "17")

    path = _isolated_appdata_path(
        "/shared/omnigibson-appdata",
        node_name="collector-host",
        visible_devices="3",
        process_index=0,
    )

    assert path == (
        "/shared/omnigibson-appdata/node_collector-host/rank_17_gpu_3/process_0"
    )


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
    monkeypatch.setenv("RLINF_NODE_RANK", "2")
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("OMNIGIBSON_APPDATA_PATH", "/shared/omnigibson-appdata")
    monkeypatch.setenv("OMNIGIBSON_DATASET_PATH", "/datasets/behavior-1k-assets")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/collector-inductor-cache")

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
            "OMNIGIBSON_APPDATA_PATH": (
                "/shared/omnigibson-appdata/node_2/rank_17_gpu_3/process_0"
            ),
            "OMNIGIBSON_DATASET_PATH": "/datasets/behavior-1k-assets",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/collector-inductor-cache",
        }
    }
    assert pool.activity_name == "turning_on_radio"

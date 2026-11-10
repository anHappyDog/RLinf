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

"""Datasets, batching, and the on-the-wire shapes they produce."""

import asyncio
import contextlib
import copy
import importlib.util
import inspect
import json
import pickle
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import rlinf.data.datasets.d4rl as d4rl_dataset_module
import rlinf.utils.obs_compression as obs_compression
from rlinf.data.datasets.d4rl import D4RLDataset
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.data.schema.embodied_trajectory import (
    LeRobotEpisodeAccumulator,
    RolloutGeometry,
    TrajectoryCollector,
    TrajectoryMode,
    TrajectoryPlan,
    select_trajectory_collector,
    select_trajectory_dispatcher,
)
from rlinf.data.schema.embodied_types import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    LeRobotFrame,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    Trajectory,
    TrajectoryKey,
    TrajectorySource,
    TrajectoryStep,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_episode_data,
)
from rlinf.data.storage.lerobot import add_frame_to_dataset, episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.channel.hooks import ChannelContext
from rlinf.scheduler.cluster.utils import (
    TensorPlaceholder,
    pack_dataclass_tensors,
    unpack_dataclass_tensors,
)
from rlinf.utils.env_helpers import SmoothInterveneController
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.obs_compression import (
    _CODEC_KEY,
    compress_obs,
    decompress_obs,
    infer_obs_batch_size,
    is_compressed_image,
    is_compression_enabled,
)
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import FSDPRewardWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _write_native_mixture_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    catalog = tmp_path / "collection"
    (catalog / "catalog").mkdir(parents=True)
    records = []
    sources = []
    for schema in ("arm_a", "arm_b"):
        root = catalog / schema
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        (root / "meta/info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": 15.15,
                    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                }
            )
        )
        data, episodes, provenance = [], [], []
        for episode in range(12):
            start = len(data)
            for frame in range(5):
                data.append(
                    {
                        "action": [episode * 100.0 + frame, 7.0],
                        "observation.state": [episode * 10.0 + frame, 2.0],
                        "source.extra.supervision_mask": frame != 2,
                    }
                )
            episodes.append(
                {
                    "episode_index": episode,
                    "length": 5,
                    "dataset_from_index": start,
                    "dataset_to_index": start + 5,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": episode,
                    "videos/observation.images.front/from_timestamp": 2.5,
                }
            )
            provenance.append(
                {
                    "episode_index": episode,
                    "provenance": {
                        "semantic_core_status": "UNKNOWN" if episode == 10 else "known",
                        "supervision_mask_feature": "source.extra.supervision_mask",
                    },
                }
            )
            records.append(
                {
                    "catalog_id": f"{schema}:{episode}",
                    "partition": schema,
                    "provenance_file": f"{schema}/meta/clip_provenance.jsonl",
                    "source_dataset": schema,
                    "domain": "real",
                    "episode_index": episode,
                    "source_group_id": f"{schema}:{episode // 2}",
                    "scene_id": f"scene_{episode // 2}",
                    "split": "development_only" if episode == 9 else "unassigned",
                    "quality_tier": "controller_motion_candidate"
                    if episode == 11
                    else "native_source_success",
                    "core_range_in_export": [1, 4],
                    "frames": 5,
                    "fps": 15.15,
                    "control_mode": f"{schema}_native",
                    "task": "pick up the object",
                }
            )
        pq.write_table(
            pa.Table.from_pylist(data), root / "data/chunk-000/file-000.parquet"
        )
        pq.write_table(
            pa.Table.from_pylist(episodes),
            root / "meta/episodes/chunk-000/file-000.parquet",
        )
        (root / "meta/clip_provenance.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in provenance)
        )
        sources.append(
            {
                "name": schema,
                "source_dataset": schema,
                "weight": 1 if schema == "arm_a" else 3,
                "cameras": {"base_0_rgb": "observation.images.front"},
            }
        )
    (catalog / "catalog/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    return catalog, sources


def _write_adapted_mixture_fixture(tmp_path):
    catalog, sources = _write_native_mixture_fixture(tmp_path)
    path = catalog / "catalog/episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row["episode_index"] < 6]
    for row in rows:
        row["assigned_split"] = ("train", "val", "test")[row["episode_index"] // 2]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for source in sources:
        source["adapter"] = {
            "kind": "joint_relative",
            "joint_action_indices": [0],
            "joint_state_indices": [0],
            "gripper_action_indices": [1],
            "gripper_open": 7.0,
            "gripper_closed": -1.0,
        }
        source["normalization"] = {"transform": "quantile"}
    return catalog, sources


def test_native_control_adapter_shared_anchor_and_reversible_grippers():
    from rlinf.data.datasets.openpi.control_schema import (
        ControlAdapter,
        normalize,
        unnormalize,
    )

    adapter = ControlAdapter.from_config(
        {
            "kind": "joint_relative",
            "joint_action_indices": [0, 2],
            "joint_state_indices": [1, 3],
            "gripper_action_indices": [1],
            "gripper_state_indices": [0],
            "gripper_open": 1.0,
            "gripper_closed": -1.0,
        }
    )
    actions = np.asarray(
        [[[1, 1, 3], [2, 0, 4], [5, -1, 6]], [[8, -1, 3], [4, 0, 4], [3, 1, 6]]],
        dtype=np.float32,
    )
    anchor = np.asarray([[1, 2, 0, 3], [-1, 5, 0, 2]], dtype=np.float32)
    original_actions, original_anchor = actions.copy(), anchor.copy()
    encoded = adapter.encode_actions(actions, anchor)
    np.testing.assert_allclose(encoded[..., 0], actions[..., 0] - anchor[:, None, 1])
    np.testing.assert_allclose(encoded[..., 2], actions[..., 2] - anchor[:, None, 3])
    np.testing.assert_allclose(encoded[..., 1], [[0, 0.5, 1], [1, 0.5, 0]])
    np.testing.assert_allclose(adapter.encode_state(anchor)[..., 0], [0, 1])
    np.testing.assert_allclose(adapter.decode_actions(encoded, anchor), actions)
    np.testing.assert_allclose(
        adapter.decode_actions(adapter.encode_actions(actions, anchor[0]), anchor[0]),
        actions,
    )
    np.testing.assert_array_equal(actions, original_actions)
    np.testing.assert_array_equal(anchor, original_anchor)
    np.testing.assert_array_equal(
        ControlAdapter.from_config(None).encode_actions(actions, anchor), actions
    )
    stats = {
        "transform": "quantile",
        "q01": [-1, 0, 2],
        "q99": [1, 1, 2],
        "passthrough_indices": [1],
    }
    values = np.asarray([[4, 1, 2], [-4, 0, 2]], dtype=np.float32)
    normalized = normalize(values, stats)
    assert normalized[0, 0] > 1 and normalized[1, 0] < -1
    np.testing.assert_array_equal(normalized[:, 1:], values[:, 1:] - [0, 2])
    np.testing.assert_allclose(unnormalize(normalized, stats), values)
    with pytest.raises(ValueError, match="nonoverlapping"):
        ControlAdapter.from_config(
            {
                "kind": "joint_relative",
                "joint_action_indices": [0],
                "joint_state_indices": [0],
                "gripper_action_indices": [0],
            }
        )


def test_native_mixture_adapted_statistics_and_fixed_holdout(tmp_path):
    from rlinf.data.datasets.openpi.mixture import (
        MixtureSftDataset,
        NumericEpisodeCache,
        prepare_mixture,
    )

    catalog, sources = _write_adapted_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    summary = prepare_mixture(
        catalog,
        prepared,
        sources,
        val_fraction=0,
        action_horizon=3,
        action_dim=4,
        sampling_unit="episode",
    )
    splits = {
        split: [
            json.loads(line)
            for line in (prepared / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "val", "test")
    }
    assert all(summary[split]["episodes"] == 4 for split in splits)
    assert all(row["episode_index"] < 2 for row in splits["train"])
    assert all(2 <= row["episode_index"] < 4 for row in splits["val"])
    assert all(row["episode_index"] >= 4 for row in splits["test"])
    stats = json.loads((prepared / "normalization.json").read_text())["schemas"]
    cache = NumericEpisodeCache()
    for schema, schema_stats in stats.items():
        rows = [row for row in splits["train"] if row["schema_id"] == schema]
        expected = []
        for row in rows:
            actions, states, supervised = cache.read(row)
            for frame in row["anchor_indices"]:
                for future in range(frame, min(frame + 3, len(actions))):
                    if supervised[future]:
                        expected.append([actions[future, 0] - states[frame, 0], 0])
        expected = np.asarray(expected)
        np.testing.assert_allclose(schema_stats["action"]["mean"], expected.mean(0))
        np.testing.assert_allclose(
            schema_stats["action"]["q01"], np.quantile(expected, 0.01, axis=0)
        )
        np.testing.assert_allclose(
            schema_stats["action"]["q99"], np.quantile(expected, 0.99, axis=0)
        )
        assert schema_stats["action"]["count"] == len(expected)
        assert schema_stats["action"]["passthrough_indices"] == [1]
        assert set(schema_stats["train_catalog_ids"]) == {
            row["catalog_id"] for row in rows
        }
    dataset = MixtureSftDataset(prepared, action_horizon=3, action_dim=4)
    row, frame, _, normalized_actions, mask = dataset.numeric_item(0)
    from rlinf.data.datasets.openpi.control_schema import unnormalize

    native, state, _ = cache.read(row)
    encoded = unnormalize(normalized_actions[:, :2], stats[row["schema_id"]]["action"])
    assert encoded[0, 0] == pytest.approx(native[frame, 0] - state[frame, 0])
    np.testing.assert_array_equal(normalized_actions[:, 1], 0)
    np.testing.assert_array_equal(mask[:, 2:], False)


def test_native_mixture_episode_sampling_balances_original_groups(tmp_path):
    from rlinf.data.datasets.openpi.mixture import (
        MixtureSampler,
        MixtureSftDataset,
        prepare_mixture,
    )

    catalog, sources = _write_native_mixture_fixture(tmp_path)
    path = catalog / "catalog/episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    duplicate = dict(rows[0], catalog_id="arm_a:extra_slice")
    rows.append(duplicate)
    rows[0]["core_range_in_export"] = [1, 2]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared = tmp_path / "prepared"
    prepare_mixture(catalog, prepared, sources, val_fraction=0, sampling_unit="episode")
    dataset = MixtureSftDataset(prepared, action_horizon=3, action_dim=4)
    sampler = MixtureSampler(dataset, batch_size=2)
    totals, sizes, clips = {}, {}, {}
    offset = 0
    for row in dataset.episodes:
        key = (row["schema_id"], row["source_group_id"])
        count = len(row["anchor_indices"])
        totals[key] = (
            totals.get(key, 0) + sampler.weights[offset : offset + count].sum()
        )
        sizes[key] = sizes.get(key, 0) + count
        clips[key] = clips.get(key, 0) + 1
        offset += count
    assert len(set(sizes.values())) > 1 and len(set(clips.values())) > 1
    for source in sources:
        values = [
            float(value)
            for (name, _), value in totals.items()
            if name == source["name"]
        ]
        np.testing.assert_allclose(values, source["weight"] / len(values))


def test_native_mixture_contract_and_sampler_resume_identity(tmp_path):
    from rlinf.data.datasets.openpi.mixture import (
        MixtureSampler,
        MixtureSftDataset,
        prepare_mixture,
    )

    catalog, sources = _write_adapted_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_mixture(catalog, prepared, sources, action_horizon=3, action_dim=4)
    for kwargs in (
        {"action_horizon": 4, "action_dim": 4},
        {"action_horizon": 3, "action_dim": 5},
    ):
        with pytest.raises(ValueError):
            MixtureSftDataset(prepared, **kwargs)
    dataset = MixtureSftDataset(prepared, action_horizon=3, action_dim=4)
    sampler = MixtureSampler(dataset, batch_size=2, seed=19, samples_per_epoch=20)
    iterator = iter(sampler)
    next(iterator)
    next(iterator)
    saved = sampler.state_dict()
    expected = list(iterator)
    restored = MixtureSampler(dataset, batch_size=2, seed=19, samples_per_epoch=20)
    with pytest.raises(ValueError):
        restored.load_state_dict({"epoch": 0, "position": 2})
    restored.load_state_dict(saved)
    assert list(restored) == expected
    for world_size, rank in ((2, 0), (2, 1)):
        changed = MixtureSampler(
            dataset,
            batch_size=2,
            seed=19,
            samples_per_epoch=20,
            world_size=world_size,
            rank=rank,
        )
        with pytest.raises(ValueError, match="topology"):
            changed.load_state_dict(saved)
    for seed, budget in ((20, 20), (19, 24)):
        changed = MixtureSampler(
            dataset, batch_size=2, seed=seed, samples_per_epoch=budget
        )
        with pytest.raises(ValueError):
            changed.load_state_dict(saved)
    changed_sources = copy.deepcopy(sources)
    changed_sources[0]["weight"] = 2
    other = tmp_path / "other"
    prepare_mixture(catalog, other, changed_sources, action_horizon=3, action_dim=4)
    changed_dataset = MixtureSftDataset(other, action_horizon=3, action_dim=4)
    changed = MixtureSampler(
        changed_dataset, batch_size=2, seed=19, samples_per_epoch=20
    )
    with pytest.raises(ValueError):
        changed.load_state_dict(saved)
    for filename in ("train.jsonl", "normalization.json"):
        tampered = tmp_path / filename.replace(".", "_")
        shutil.copytree(prepared, tampered)
        target = tampered / filename
        target.write_text(target.read_text() + "\n")
        with pytest.raises(ValueError):
            MixtureSftDataset(tampered, action_horizon=3, action_dim=4)


def test_native_mixture_sample_cache_preserves_inputs_without_source_media(tmp_path):
    import gzip
    import pickle

    from rlinf.data.datasets.openpi.mixture import (
        MixtureSampler,
        MixtureSftDataset,
        prepare_mixture,
    )

    catalog, sources = _write_adapted_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_mixture(catalog, prepared, sources, action_horizon=3, action_dim=4)
    dataset = MixtureSftDataset(prepared, action_horizon=3, action_dim=4)
    row, _, state, actions, mask = dataset.numeric_item(0)
    sample = {
        "observation": {
            "state": state,
            "image": np.random.default_rng(0)
            .uniform(-1, 1, (224, 224, 3))
            .astype(np.float32),
        },
        "actions": actions,
        "action_loss_mask": mask,
        "sample_schema": row["schema_id"],
        "sample_group": row["source_group_id"],
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    with gzip.open(cache / "0.pkl.gz", "wb") as stream:
        pickle.dump(sample, stream, protocol=5)
    (cache / "metadata.json").write_text(
        json.dumps({"contract": dataset.sample_cache_contract(), "indices": [0]})
    )
    shutil.rmtree(catalog)
    cached = MixtureSftDataset(
        prepared, action_horizon=3, action_dim=4, sample_cache_dir=cache
    )
    actual = cached[0]
    for key in ("state", "image"):
        np.testing.assert_array_equal(
            actual["observation"][key], sample["observation"][key]
        )
    for key in ("actions", "action_loss_mask"):
        np.testing.assert_array_equal(actual[key], sample[key])
    assert actual["sample_group"] == sample["sample_group"]
    assert list(MixtureSampler(dataset, batch_size=1, samples_per_epoch=20)) == list(
        MixtureSampler(cached, batch_size=1, samples_per_epoch=20)
    )
    with pytest.raises(FileNotFoundError):
        cached[1]
    with pytest.raises(ValueError, match="Sample cache requires identical"):
        MixtureSftDataset(
            prepared,
            action_horizon=3,
            action_dim=4,
            max_token_len=201,
            sample_cache_dir=cache,
        )


def test_native_mixture_group_splits_and_training_only_statistics(tmp_path):
    from rlinf.data.datasets.openpi.mixture import NumericEpisodeCache, prepare_mixture

    catalog, sources = _write_native_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    summary = prepare_mixture(
        catalog, prepared, sources, val_fraction=0.5, seed=3, allow_development=True
    )
    train = [
        json.loads(line) for line in (prepared / "train.jsonl").read_text().splitlines()
    ]
    val = [
        json.loads(line) for line in (prepared / "val.jsonl").read_text().splitlines()
    ]
    assert train and val
    assert not (
        {row["source_group_id"] for row in train}
        & {row["source_group_id"] for row in val}
    )
    assert not (
        {row["split_group"] for row in train} & {row["split_group"] for row in val}
    )
    assert not any(row["development_only"] for row in val)
    assert summary["excluded"]["quality"] == 2
    assert summary["excluded"]["unknown_core"] == 2
    stats = json.loads((prepared / "normalization.json").read_text())["schemas"]
    cache = NumericEpisodeCache()
    for schema in stats:
        rows = [row for row in train if row["schema_id"] == schema]
        values = [cache.read(row) for row in rows]
        action = np.concatenate([a[mask] for a, _, mask in values])
        state = np.concatenate([s[mask] for _, s, mask in values])
        np.testing.assert_allclose(stats[schema]["action"]["mean"], action.mean(axis=0))
        np.testing.assert_allclose(stats[schema]["state"]["mean"], state.mean(axis=0))
        assert stats[schema]["action"]["count"] == len(action)
        assert stats[schema]["action"]["std"][1] == 1
        assert stats[schema]["action"]["transform"] == "mean_std"
        assert stats[schema]["state"]["transform"] == "tanh_mean_std"
        assert set(stats[schema]["train_catalog_ids"]) == {
            row["catalog_id"] for row in rows
        }
    limited = tmp_path / "limited"
    prepare_mixture(
        catalog,
        limited,
        sources,
        val_fraction=0.5,
        seed=3,
        allow_development=True,
        max_train_episodes=1,
        max_val_episodes=1,
    )
    for split, originals in (("train", train), ("val", val)):
        rows = [
            json.loads(line)
            for line in (limited / f"{split}.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
        assert {row["catalog_id"] for row in rows} <= {
            row["catalog_id"] for row in originals
        }


def test_native_mixture_core_tail_dimension_and_fractional_clock(tmp_path):
    from rlinf.data.datasets.openpi.mixture import MixtureSftDataset, prepare_mixture

    catalog, sources = _write_native_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_mixture(catalog, prepared, sources, val_fraction=0, seed=3)
    dataset = MixtureSftDataset(prepared, action_horizon=4, action_dim=4)
    row, frame, state, actions, mask = dataset.numeric_item(0)
    assert frame == 1 and row["fps"] == 15.15
    assert row["videos"]["base_0_rgb"]["from_timestamp"] == 2.5
    assert 2.5 + frame / row["fps"] == pytest.approx(2.56600660066)
    np.testing.assert_array_equal(
        mask, [[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(state[2:], 0)
    assert np.all((-1 <= state) & (state <= 1))
    schema = dataset.schemas[row["schema_id"]]
    raw_state = np.asarray([row["episode_index"] * 10.0 + frame, 2.0])
    np.testing.assert_allclose(
        state[:2],
        np.tanh((raw_state - schema["state"]["mean"]) / schema["state"]["std"]),
        atol=1e-7,
    )
    _, frame, _, actions, mask = dataset.numeric_item(1)
    assert frame == 3
    np.testing.assert_array_equal(
        mask, [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(actions[1], actions[2])
    np.testing.assert_array_equal(actions[2], actions[3])
    with pytest.raises(ValueError, match="exceeds model dimension"):
        MixtureSftDataset(prepared, action_dim=1)
    evaluation = MixtureSftDataset(prepared, max_samples=17, sample_multiple=4)
    assert len(evaluation) == 16
    assert evaluation.alignment_dropped == 1
    assert {row["schema_id"] for row in evaluation.episodes} == {"arm_a", "arm_b"}
    assert len(evaluation.episodes) == 16
    assert (
        sum(
            len(row["anchor_indices"])
            for row in evaluation.episodes
            if row["schema_id"] == "arm_a"
        )
        == 8
    )
    with pytest.raises(ValueError, match="at least 4 anchors"):
        MixtureSftDataset(prepared, max_samples=3, sample_multiple=4)


def test_native_mixture_bounded_state_uses_official_discrete_tokenizer():
    tokenizer_module = pytest.importorskip("openpi.models.tokenizer")
    tokenizer = tokenizer_module.PaligemmaTokenizer.__new__(
        tokenizer_module.PaligemmaTokenizer
    )
    tokenizer._max_len = 200
    encoded = {}

    def encode(text, add_bos):
        encoded["text"] = text
        return [1]

    tokenizer._tokenizer = SimpleNamespace(encode=encode)
    state = np.tanh(np.arange(-2, 3, dtype=np.float32))
    tokenizer.tokenize("pick up", state)
    values = encoded["text"].split("State: ")[1].split(";")[0]
    discrete = np.asarray([int(value) for value in values.split()])
    assert len(np.unique(discrete)) == 5
    assert discrete.min() > 0 and discrete.max() < 255


def test_native_mixture_weighted_sampler_and_loader_resume(tmp_path):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from rlinf.data.datasets.openpi.mixture import (
        MixtureSampler,
        MixtureSftDataset,
        prepare_mixture,
    )

    catalog, sources = _write_native_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_mixture(catalog, prepared, sources, val_fraction=0, seed=3)
    dataset = MixtureSftDataset(prepared, action_horizon=4, action_dim=4)
    sampler = MixtureSampler(dataset, batch_size=2, seed=19, samples_per_epoch=40)
    boundary = sum(
        len(row["anchor_indices"])
        for row in dataset.episodes
        if row["schema_id"] == "arm_a"
    )
    assert sampler.weights[:boundary].sum() == pytest.approx(1)
    assert sampler.weights[boundary:].sum() == pytest.approx(3)
    first = list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
    sampler.load_state_dict({"epoch": 0, "position": 0})
    assert first == list(sampler)
    rank0 = MixtureSampler(
        dataset, batch_size=2, seed=19, world_size=2, rank=0, samples_per_epoch=40
    )
    rank1 = MixtureSampler(
        dataset, batch_size=2, seed=19, world_size=2, rank=1, samples_per_epoch=40
    )
    assert [index for pair in zip(list(rank0), list(rank1)) for index in pair] == first
    sampler.set_epoch(0)
    loader = StatefulDataLoader(
        list(range(len(dataset))), batch_size=2, sampler=sampler, num_workers=0
    )
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    checkpoint = copy.deepcopy(loader.state_dict())
    expected = list(iterator)
    restored_sampler = MixtureSampler(
        dataset, batch_size=2, seed=19, samples_per_epoch=40
    )
    restored = StatefulDataLoader(
        list(range(len(dataset))), batch_size=2, sampler=restored_sampler, num_workers=0
    )
    restored.load_state_dict(checkpoint)
    actual = list(restored)
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    evaluation = MixtureSampler(
        dataset, batch_size=3, shuffle=False, world_size=2, rank=1
    )
    assert list(evaluation) == list(range(1, len(dataset), 2))


def test_native_mixture_sampler_above_multinomial_category_limit():
    from rlinf.data.datasets.openpi.mixture import MixtureSampler

    boundary = 2**24
    dataset = SimpleNamespace(
        episodes=[
            {
                "schema_id": "large",
                "source_group_id": "a",
                "anchor_indices": range(boundary),
            },
            {"schema_id": "tail", "source_group_id": "b", "anchor_indices": range(1)},
        ],
        schemas={"large": {"weight": 1.0}, "tail": {"weight": 1.0}},
        sampling_unit="frame",
        manifest_fingerprint="large-mixture",
        requires_contract=True,
    )
    sampler = MixtureSampler(dataset, batch_size=1, samples_per_epoch=2048, seed=19)
    global_draws = list(sampler)
    assert all(0 <= index <= boundary for index in global_draws)
    assert 0.45 < global_draws.count(boundary) / len(global_draws) < 0.55

    ranks = [copy.copy(sampler) for _ in range(2)]
    for rank, local in enumerate(ranks):
        local.rank = rank
        local.world_size = 2
        local.num_samples = 1024
        local.set_epoch(0)
    assert [index for pair in zip(*map(list, ranks)) for index in pair] == global_draws

    local = ranks[1]
    local.set_epoch(3)
    iterator = iter(local)
    next(iterator)
    next(iterator)
    saved = local.state_dict()
    remainder = list(iterator)
    restored = copy.copy(local)
    restored.load_state_dict(saved)
    assert list(restored) == remainder
    assert global_draws[1::2][2:] != remainder


def test_native_mixture_worker_checkpoint_restores_epoch(tmp_path, monkeypatch):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from rlinf.data.datasets.openpi.mixture import (
        MixtureSampler,
        MixtureSftDataset,
        prepare_mixture,
    )
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    catalog, sources = _write_native_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_mixture(catalog, prepared, sources, val_fraction=0)
    dataset = MixtureSftDataset(prepared, action_horizon=4, action_dim=4)

    def make_loader():
        return StatefulDataLoader(
            list(range(len(dataset))),
            batch_size=2,
            sampler=MixtureSampler(dataset, batch_size=2, samples_per_epoch=20),
            num_workers=0,
        )

    original = make_loader()
    original.sampler.set_epoch(3)
    iterator = iter(original)
    next(iterator)
    torch.save([copy.deepcopy(original.state_dict())], tmp_path / "data.pt")
    expected = list(iterator)
    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create({"data": {"type": "mixture"}})
    worker._rank = 0
    worker._data_epoch = 0
    worker.data_loader = make_loader()
    monkeypatch.setattr(FSDPSftWorker, "load_checkpoint", lambda self, path: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    worker.load_checkpoint(str(tmp_path))
    assert worker._data_epoch == 3
    actual = list(worker.data_iter)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)


def test_native_mixture_evaluation_records_time_and_restores_rng(monkeypatch):
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    class FlowLoss(torch.nn.Module):
        def forward(self, *, forward_type, data, train):
            assert not train and not self.training
            return data["actions"].square().mean() + torch.rand(())

    timer_calls = []

    @contextlib.contextmanager
    def timer():
        timer_calls.append("enter")
        yield
        timer_calls.append("exit")

    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create({"data": {"type": "mixture"}, "actor": {"seed": 0}})
    worker._rank = 0
    worker.device = torch.device("cpu")
    worker.model = FlowLoss()
    worker.amp_context = contextlib.nullcontext()
    worker.worker_timer = timer
    worker.eval_data_loader = torch.utils.data.DataLoader(
        [{"actions": torch.ones(2, 3)}, {"actions": torch.full((2, 3), 2.0)}],
        batch_size=1,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: None)
    rng = torch.get_rng_state().clone()
    first = worker.run_eval()
    assert worker.model.training
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert first == worker.run_eval()
    assert timer_calls == ["enter", "exit", "enter", "exit"]


def _scalar_trajectory(value: float) -> Trajectory:
    """One-sample trajectory so checkpoint tests write only tiny files."""
    return Trajectory(max_episode_length=1, rewards=torch.full((1, 1, 1), float(value)))


def test_memory_checkpoint_indexes_only_cached_trajectories(tmp_path):
    """A memory checkpoint records only the trajectories still held in cache."""
    checkpoint = tmp_path / "checkpoint"
    buffer = TrajectoryReplayBuffer(sample_window_size=2)
    restored = None
    try:
        for value in range(3):
            buffer.add_trajectories([_scalar_trajectory(value)])
        original_stats = buffer.get_stats()
        buffer.save_checkpoint(str(checkpoint))
        assert buffer.get_stats() == original_stats

        index = json.loads((checkpoint / "trajectory_index.json").read_text())
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        assert index["trajectory_id_list"] == [1, 2]
        assert metadata["size"] == 2
        assert metadata["total_samples"] == 2
        assert metadata["trajectory_counter"] == 3
        assert len(list(checkpoint.glob("trajectory_*.pt"))) == 2

        restored = TrajectoryReplayBuffer(sample_window_size=2)
        restored.load_checkpoint(str(checkpoint))
        assert restored.get_stats()["num_trajectories"] == 2
        assert restored.get_stats()["total_samples"] == 2
    finally:
        buffer.close()
        if restored is not None:
            restored.close()
        shutil.rmtree(checkpoint, ignore_errors=True)


def test_auto_save_checkpoint_copies_current_window(tmp_path):
    """An auto-save checkpoint copies the current window, not a stale sample window."""
    source = tmp_path / "buffer"
    checkpoint = tmp_path / "checkpoint"
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(source),
        cache_size=3,
        sample_window_size=2,
    )
    restored = None
    try:
        buffer.add_trajectories([_scalar_trajectory(0)])
        buffer.add_trajectories([_scalar_trajectory(1)])
        buffer.sample(1)
        buffer.add_trajectories([_scalar_trajectory(2)])
        original_stats = buffer.get_stats()
        buffer.save_checkpoint(str(checkpoint))
        assert buffer.get_stats() == original_stats

        index = json.loads((checkpoint / "trajectory_index.json").read_text())
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        assert index["trajectory_id_list"] == [1, 2]
        assert metadata["size"] == 2
        assert metadata["total_samples"] == 2
        assert metadata["trajectory_counter"] == 3
        assert len(list(checkpoint.glob("trajectory_*.pt"))) == 2

        restored = TrajectoryReplayBuffer(sample_window_size=2)
        restored.load_checkpoint(str(checkpoint))
        assert restored.get_stats()["num_trajectories"] == 2
        assert restored.get_stats()["total_samples"] == 2
    finally:
        buffer.close()
        if restored is not None:
            restored.close()
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(checkpoint, ignore_errors=True)


def test_auto_save_checkpoint_resaves_restored_window(tmp_path):
    """A resumed auto-save buffer copies old and newly written trajectories."""
    source = tmp_path / "source"
    first_checkpoint = tmp_path / "first_checkpoint"
    resumed_source = tmp_path / "resumed_source"
    second_checkpoint = tmp_path / "second_checkpoint"
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(source),
        sample_window_size=2,
    )
    resumed = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(resumed_source),
        sample_window_size=2,
    )
    restored = None
    try:
        for value in range(3):
            buffer.add_trajectories([_scalar_trajectory(value)])
        buffer.save_checkpoint(str(first_checkpoint))

        resumed.load_checkpoint(str(first_checkpoint))
        resumed.add_trajectories([_scalar_trajectory(3)])
        resumed.save_checkpoint(str(second_checkpoint))

        index = json.loads((second_checkpoint / "trajectory_index.json").read_text())
        metadata = json.loads((second_checkpoint / "metadata.json").read_text())
        assert index["trajectory_id_list"] == [2, 3]
        assert metadata["size"] == 2
        assert metadata["total_samples"] == 2
        assert metadata["trajectory_counter"] == 4
        assert len(list(second_checkpoint.glob("trajectory_*.pt"))) == 2

        restored = TrajectoryReplayBuffer(sample_window_size=2)
        restored.load_checkpoint(str(second_checkpoint))
        assert restored.get_stats()["num_trajectories"] == 2
        assert restored.get_stats()["total_samples"] == 2
    finally:
        buffer.close()
        resumed.close()
        if restored is not None:
            restored.close()
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(first_checkpoint, ignore_errors=True)
        shutil.rmtree(resumed_source, ignore_errors=True)
        shutil.rmtree(second_checkpoint, ignore_errors=True)


class TestD4RLDataset:
    """Tests for loading D4RL transition datasets."""

    @pytest.mark.parametrize("has_next_observations", [False, True])
    def test_from_path_converts_standard_hdf5_dataset(
        self, tmp_path, monkeypatch, has_next_observations
    ):
        """Standard D4RL files go through ``qlearning_dataset``.

        AntMaze files have no ``next_observations``; MuJoCo v2 files have both
        ``next_observations`` and ``timeouts``.
        """
        dataset_path = tmp_path / "standard-d4rl.hdf5"
        dataset_path.touch()
        raw = {
            "observations": np.array([[0.0], [1.0], [10.0], [11.0], [12.0]]),
            "actions": np.array([[-1.5], [0.25], [0.5], [0.75], [1.5]]),
            "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "terminals": np.array([False, False, False, False, True]),
            "timeouts": np.array([False, True, False, False, False]),
        }
        if has_next_observations:
            raw["next_observations"] = np.array([[1.0], [2.0], [11.0], [12.0], [13.0]])
        converted = {
            "observations": raw["observations"][[0, 2, 3]],
            "actions": raw["actions"][[0, 2, 3]],
            "rewards": raw["rewards"][[0, 2, 3]],
            "terminals": raw["terminals"][[0, 2, 3]],
            "next_observations": raw["observations"][[1, 3, 4]],
        }

        env = mock.Mock()
        env.get_dataset.return_value = raw
        gym_api = mock.Mock()
        gym_api.make.return_value = env

        def qlearning_dataset(actual_env, *, dataset):
            assert actual_env is env
            assert dataset is raw
            assert "timeouts" in dataset
            return converted

        d4rl_api = mock.Mock(qlearning_dataset=qlearning_dataset)
        monkeypatch.setattr(d4rl_dataset_module, "gym", gym_api)
        monkeypatch.setattr(d4rl_dataset_module, "d4rl", d4rl_api)

        dataset = D4RLDataset.from_path(dataset_path, task_name="antmaze-test-v0")

        np.testing.assert_allclose(dataset.observations[:, 0], [0.0, 10.0, 11.0])
        np.testing.assert_allclose(dataset.next_observations[:, 0], [1.0, 11.0, 12.0])
        np.testing.assert_allclose(dataset.actions[:, 0], [-0.99999, 0.5, 0.75])
        np.testing.assert_allclose(dataset.rewards, [0.0, 2.0, 3.0])
        np.testing.assert_allclose(dataset.dones_float, [1.0, 0.0, 1.0])
        env.get_dataset.assert_called_once_with(h5path=str(dataset_path))
        env.close.assert_called_once_with()

    def test_from_path_preserves_materialized_transition_dataset(
        self, tmp_path, monkeypatch
    ):
        """Existing files with next observations remain supported."""
        dataset_path = tmp_path / "materialized-d4rl.hdf5"
        dataset_path.touch()
        raw = {
            "observations": np.array([[1.0], [2.0]]),
            "actions": np.array([[-0.25], [0.25]]),
            "rewards": np.array([3.0, 4.0]),
            "terminals": np.array([False, True]),
            "next_observations": np.array([[2.0], [3.0]]),
        }

        env = mock.Mock()
        env.get_dataset.return_value = raw
        gym_api = mock.Mock()
        gym_api.make.return_value = env
        d4rl_api = mock.Mock()
        monkeypatch.setattr(d4rl_dataset_module, "gym", gym_api)
        monkeypatch.setattr(d4rl_dataset_module, "d4rl", d4rl_api)

        dataset = D4RLDataset.from_path(dataset_path, task_name="custom-test-v0")

        np.testing.assert_array_equal(dataset.observations, raw["observations"])
        np.testing.assert_array_equal(
            dataset.next_observations, raw["next_observations"]
        )
        d4rl_api.qlearning_dataset.assert_not_called()
        env.close.assert_called_once_with()


class TestMathDatasetMultithread:
    """Tests for ReasoningDataset multithread processing consistency."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = mock.Mock()
        tokenizer.is_fast = True
        tokenizer.eos_token_id = 2

        def apply_chat_template_side_effect(
            prompts, tokenize=False, add_generation_prompt=True
        ):
            """Mock apply_chat_template that handles generator input."""
            # Convert generator to list if needed
            prompts_list = list(prompts) if not isinstance(prompts, list) else prompts
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n"
                if isinstance(prompt, str)
                else prompt
                for prompt in prompts_list
            ]

        tokenizer.apply_chat_template = mock.Mock(
            side_effect=apply_chat_template_side_effect
        )
        tokenizer.batch_encode_plus = mock.Mock(
            side_effect=lambda texts: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            }
        )
        tokenizer.encode = mock.Mock(side_effect=lambda text: [1] * len(text.split()))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = DictConfig(
            {
                "data": {
                    "max_prompt_length": 1000,
                    "prompt_key": "question",
                    "answer_key": "answer",
                    "apply_chat_template": True,
                    "filter_prompt_by_length": False,
                    "process_workers": 4,
                    "process_batch_size": 32,
                }
            }
        )
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing (at least 10000 items)."""
        # Generate at least 10000 math problems
        data = []
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a // b if b != 0 else 0),
        ]

        for i in range(10000):
            op_symbol, op_func = random.choice(operations)
            a = random.randint(1, 1000)
            b = random.randint(1, 1000) if op_symbol != "/" else random.randint(1, 100)
            if op_symbol == "/" and b == 0:
                b = 1
            result = op_func(a, b)
            question = f"What is {a} {op_symbol} {b}?"
            data.append({"question": question, "answer": str(result)})

        return data

    def test_multithread_vs_singlethread_consistency(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test that multithread processing produces identical results to single-thread processing.

        This test verifies that:
        1. Results from multi-worker processing match single-worker processing
        2. All keys are the same
        3. All values are the same
        """
        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        # We need to reload the raw data to avoid double processing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        time_start = time.time()
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )
        time_elapse_multithread = time.time() - time_start

        # Test with single thread
        time_start = time.time()
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)
        time_elapse_singlethread = time.time() - time_start

        # Verify lengths are equal
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify all items have the same keys and values
        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            keys_mt, keys_st = item_mt.keys(), item_st.keys()
            assert keys_mt == keys_st, (
                f"Keys mismatch at index {idx}: "
                f"multithread={list(keys_mt)}, singlethread={list(keys_st)}"
            )

            # Check all values are equal
            unequal_keys = [key for key in keys_mt if item_mt[key] != item_st[key]]
            assert len(unequal_keys) == 0, (
                f"Values mismatch at index {idx} for keys: {unequal_keys}"
            )

        # Log timing information (for debugging)
        print(
            f"Data count: {len(data_multithread)}, "
            f"Multithread processing time: {time_elapse_multithread:.2f}s, "
            f"Singlethread processing time: {time_elapse_singlethread:.2f}s"
        )

    def test_multithread_consistency_with_filter(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test multithread processing consistency when filter_prompt_by_length is enabled.
        """
        # Update config to enable filtering
        mock_config.data.filter_prompt_by_length = True
        mock_config.data.max_prompt_length = 50  # Reasonable limit to test filtering

        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )

        # Test with single thread
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)

        # Verify consistency
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify that some data was filtered (not all data passed)
        assert len(data_multithread) <= len(raw_data), (
            f"Filtering should reduce data size, but got {len(data_multithread)} >= {len(raw_data)}"
        )

        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            assert item_mt.keys() == item_st.keys(), f"Keys mismatch at index {idx}"
            for key in item_mt.keys():
                assert item_mt[key] == item_st[key], (
                    f"Mismatch at index {idx}, key {key}"
                )

        print(
            f"Filtered data count: {len(data_multithread)}/{len(raw_data)} "
            f"(max_prompt_length={mock_config.data.max_prompt_length})"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def test_split_dict_to_chunk_keeps_mixed_fields_aligned():
    batch = {
        "values": torch.arange(10),
        "sample_ids": list(range(10)),
        "nested": {"values": torch.arange(10) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3)

    assert [chunk["values"].tolist() for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["sample_ids"] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["nested"]["values"].tolist() for chunk in chunks] == [
        [100, 101, 102, 103],
        [104, 105, 106],
        [107, 108, 109],
    ]


def test_split_dict_to_chunk_returns_requested_number_of_chunks():
    batch = {"values": torch.arange(2), "sample_ids": ["a", "b"]}

    chunks = split_dict_to_chunk(batch, 4)

    assert len(chunks) == 4
    assert [chunk["values"].tolist() for chunk in chunks] == [[0], [1], [], []]
    assert [chunk["sample_ids"] for chunk in chunks] == [["a"], ["b"], [], []]


class _LegacyDataset:
    """Mimics lerobot < 0.2: the task lives inside the frame dict."""

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class _PostRevertDataset(_LegacyDataset):
    """Mimics lerobot >= 0.4: back to ``add_frame(frame)``, but it pops the task.

    0.4 reverted the 0.3.x signature, so a version-number check would dispatch
    this one wrongly. It also mutates the caller's dict.
    """

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append({**frame, "task": frame.pop("task")})


class _CurrentDataset:
    """Mimics lerobot >= 0.3: the task is a separate argument.

    Like the real implementation, a ``task`` key inside *frame* is rejected
    because it is not part of the feature schema.
    """

    def __init__(self):
        self.frames = []
        self.tasks = []
        self.saved_episodes = 0

    def add_frame(self, frame, task, timestamp=None):
        if "task" in frame:
            raise ValueError("Extra features: {'task'}")
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        self.saved_episodes += 1


def _make_writer(dataset):
    # ``create()`` needs a real lerobot install, so attach the dataset the way
    # ``create()`` would.
    writer = LeRobotDatasetWriter()
    writer.dataset = dataset
    return writer


def _episode(n=2):
    return [{"state": i, "actions": i, "task": "pick up the cube"} for i in range(n)]


def test_legacy_dataset_keeps_task_in_frame():
    dataset = _LegacyDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


def test_current_dataset_gets_task_as_argument():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode(_episode())

    assert dataset.tasks == ["pick up the cube"] * 2
    assert all("task" not in f for f in dataset.frames)
    assert dataset.frames[0]["state"] == 0
    assert dataset.saved_episodes == 1


ALL_SHAPES = [_LegacyDataset, _CurrentDataset, _PostRevertDataset]


def test_post_revert_dataset_keeps_task_in_frame():
    # lerobot >= 0.4 took the 0.3.x signature back out again.
    dataset = _PostRevertDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_caller_frames_are_not_mutated(dataset_cls):
    # The DAgger worker shares these dicts with the in-memory training store,
    # and lerobot >= 0.4 pops "task" out of whatever frame it is handed.
    episode = _episode()
    before = [dict(f) for f in episode]
    _make_writer(dataset_cls()).add_episode(episode)

    assert episode == before


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_frame_without_task_is_rejected(dataset_cls):
    with pytest.raises(ValueError, match="missing the required 'task' field"):
        add_frame_to_dataset(dataset_cls(), {"state": 0, "actions": 0})


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_add_frame_to_dataset_is_usable_standalone(dataset_cls):
    # The toolkit collectors drive LeRobotDataset directly, without the writer.
    dataset = dataset_cls()
    add_frame_to_dataset(dataset, {"state": 0, "task": "wipe the table"})

    assert len(dataset.frames) == 1


def test_empty_episode_is_skipped():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode([])

    assert dataset.frames == []
    assert dataset.saved_episodes == 0


# --------------------------------------------------------------------------
# episode_boundaries: dataset format v2.1 vs v3.0
# --------------------------------------------------------------------------


class _V21Dataset:
    """Dataset format v2.1: a dict of two tensors on the dataset itself."""

    def __init__(self, starts, ends):
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }


class _V30Meta:
    def __init__(self, starts, ends):
        self.episodes = {"dataset_from_index": starts, "dataset_to_index": ends}


class _V30Dataset:
    """Dataset format v3.0 (lerobot >= 0.4): columns on ``meta.episodes``."""

    def __init__(self, starts, ends):
        self.episode_data_index = None
        self.meta = _V30Meta(starts, ends)


@pytest.mark.parametrize("dataset_cls", [_V21Dataset, _V30Dataset])
def test_episode_boundaries_agree_across_formats(dataset_cls):
    starts, ends = episode_boundaries(dataset_cls([0, 3, 7], [3, 7, 9]))

    assert starts == [0, 3, 7]
    assert ends == [3, 7, 9]
    assert all(isinstance(x, int) for x in starts + ends)


def test_episode_boundaries_reports_an_unknown_layout():
    class _Alien:
        episode_data_index = None
        meta = None

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Alien())


def test_episode_boundaries_rejects_v30_meta_without_the_columns():
    class _Partial:
        episode_data_index = None
        meta = _V30Meta([0], [1])

    _Partial.meta.episodes = {"length": [1]}

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Partial())


# Skip codec round-trip tests when the optional backends are not installed.
_CODECS = []
try:
    import lz4.frame  # noqa: F401

    _CODECS.append("lz4")
except ImportError:
    pass
try:
    import zstandard  # noqa: F401

    _CODECS.append("zstd")
except ImportError:
    pass

requires_codec = pytest.mark.skipif(
    not _CODECS, reason="no observation compression codec (lz4/zstd) installed"
)


def _make_payload(num_envs: int = 4) -> dict:
    """A payload shaped like EnvWorker._build_rollout_input_data output."""
    obs = {
        "main_images": torch.randint(0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (num_envs, 6, 6, 3), dtype=torch.uint8
        ),
        "states": torch.randn(num_envs, 7, dtype=torch.float32),
        "task_descriptions": ["put carrot on plate"] * num_envs,
    }
    return {
        "obs": obs,
        "final_obs": {
            "main_images": torch.randint(
                0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8
            ),
            "states": torch.randn(num_envs, 7, dtype=torch.float32),
        },
        "rlt_switch_flags": None,
    }


def _assert_payload_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_payload_equal(va, vb)
        elif isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"tensor mismatch for {key!r}"
        else:
            assert va == vb, f"value mismatch for {key!r}"


def _cfg(**overrides):
    # A plain dict is sufficient: the codec only calls ``config.get(...)``,
    # which both ``dict`` and OmegaConf's ``DictConfig`` support identically.
    base = {"enable": True, "codec": "lz4", "level": 1, "xor_delta": True}
    base.update(overrides)
    return base


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
@pytest.mark.parametrize("xor_delta", [True, False])
def test_compress_decompress_is_lossless(codec, xor_delta):
    payload = _make_payload()
    config = _cfg(codec=codec, xor_delta=xor_delta)

    compressed = compress_obs(payload, config)
    # Image tensors are replaced by self-describing marker dicts...
    assert _CODEC_KEY in compressed["obs"]["main_images"]
    assert _CODEC_KEY in compressed["obs"]["extra_view_images"]
    # ...while non-image fields are passed through untouched.
    assert isinstance(compressed["obs"]["states"], torch.Tensor)
    assert compressed["obs"]["task_descriptions"] == payload["obs"]["task_descriptions"]

    restored = decompress_obs(compressed)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_single_env_batch_roundtrip():
    # XOR-delta is skipped when there is only one frame; must still be lossless.
    payload = _make_payload(num_envs=1)
    restored = decompress_obs(compress_obs(payload, _cfg(xor_delta=True)))
    _assert_payload_equal(payload, restored)


def test_disabled_config_is_passthrough():
    payload = _make_payload()
    assert compress_obs(payload, _cfg(enable=False)) is payload
    assert compress_obs(payload, None) is payload
    assert not is_compression_enabled(None)
    assert not is_compression_enabled(_cfg(enable=False))
    assert is_compression_enabled(_cfg(enable=True))


def test_decompress_on_uncompressed_payload_is_noop():
    # The rollout worker always routes received data through decompress_obs, so
    # it must be a no-op on payloads sent without compression.
    payload = _make_payload()
    restored = decompress_obs(payload)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_only_uint8_images_are_compressed():
    # A float image-shaped tensor is not a uint8 observation and must be left
    # untouched, as must low-rank uint8 tensors (e.g. flags).
    payload = {
        "obs": {
            "float_map": torch.randn(4, 8, 8, 3),
            "uint8_flags": torch.ones(4, dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert isinstance(compressed["obs"]["float_map"], torch.Tensor)
    assert isinstance(compressed["obs"]["uint8_flags"], torch.Tensor)


def test_unknown_codec_raises():
    payload = _make_payload()
    with pytest.raises(ValueError, match="Unknown observation compression codec"):
        compress_obs(payload, _cfg(codec="bogus"))


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
def test_routing_split_then_compress_roundtrip(codec):
    """Compression must be compatible with the Env->Rollout channel routing.

    The env worker installs compression as a ``split_fn`` so it runs *after*
    the scheduler splits the batch: ``infer_batch_size`` and ``split_batch``
    see plain tensors, and each shard is compressed independently. This test
    reproduces that flow with the real routing helpers and asserts the payload
    survives split -> compress -> decompress -> merge unchanged.
    """
    routing = pytest.importorskip("rlinf.scheduler.worker.routing")

    payload = _make_payload(num_envs=6)
    # The scheduler infers the batch size from the *uncompressed* payload.
    assert routing.infer_batch_size(payload) == 6

    # split_fn = split_batch first, then compress each shard (env send path).
    split_sizes = [2, 1, 3]
    shards = routing.split_batch(payload, split_sizes)
    compressed_shards = [compress_obs(shard, _cfg(codec=codec)) for shard in shards]

    # Rollout side: decompress each shard, then merge (merge_obs path).
    restored_shards = [decompress_obs(shard) for shard in compressed_shards]
    merged = routing.merge_batches(restored_shards)
    _assert_payload_equal(payload, merged)


def test_infer_obs_batch_size_uncompressed():
    payload = _make_payload(num_envs=5)
    assert infer_obs_batch_size(payload) == 5
    # Also accepts a bare obs dict (no "obs" wrapper).
    assert infer_obs_batch_size(payload["obs"]) == 5


@requires_codec
def test_infer_obs_batch_size_with_compressed_images():
    # The rollout worker infers batch size on the receive path, before
    # decompression, so a compressed image must still report its batch size.
    payload = _make_payload(num_envs=5)
    compressed = compress_obs(payload, _cfg())
    assert is_compressed_image(compressed["obs"]["main_images"])
    assert infer_obs_batch_size(compressed) == 5


@requires_codec
def test_infer_obs_batch_size_images_only():
    # Regression: a batch whose only batched field is a (compressed) image,
    # with no states/task_descriptions, must not break batch-size inference.
    payload = {
        "obs": {
            "main_images": torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert infer_obs_batch_size(compressed) == 3


def test_infer_obs_batch_size_raises_when_unbatched():
    with pytest.raises(ValueError, match="Cannot infer batch size"):
        infer_obs_batch_size({"obs": {}})


def test_env_output_composes_one_transition_object():
    transition = EnvTransition(
        rewards=torch.ones(2, 1),
        dones=torch.zeros(2, 1, dtype=torch.bool),
    )
    output = EnvOutput(obs={"states": torch.zeros(2, 3)}, transition=transition)

    assert output.transition is transition
    assert output.rewards is transition.rewards
    assert output.dones is transition.dones
    assert set(output.__dataclass_fields__) == {
        "obs",
        "transition",
        "final_obs",
        "env_infos",
    }


def test_removed_duplicate_types_are_not_schema_api():
    import rlinf.data.schema as schema

    for name in (
        "ChunkStepResult",
        "DummyPolicyInput",
        "EmbodiedRolloutResult",
        "EnvResult",
        "PolicyCompletion",
    ):
        assert not hasattr(schema, name)


def test_env_part_completion_reuses_the_environment_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    transition = EnvTransition(rewards=torch.ones(2, 1))
    part = EnvPart(
        sources=[TrajectorySource(key, 2)],
        transition=transition,
        next_obs={"states": torch.zeros(2, 3)},
        requires_inference=True,
    )

    completed = part.complete(
        next_obs=part.next_obs,
        next_rlt_obs={"states": torch.ones(2, 3)},
        final_prev_values=torch.tensor([[2.0], [3.0]]),
    )

    assert completed.transition is transition
    assert not completed.requires_inference
    assert torch.equal(completed.bootstrap_values, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(completed.final_prev_values, completed.bootstrap_values)


def test_transport_results_store_contiguous_cpu_tensors():
    non_contiguous = torch.arange(12).reshape(3, 4).T

    env_transition = EnvTransition(rewards=non_contiguous)
    policy_output = PolicyOutput(
        forward_inputs={"states": non_contiguous},
    )

    assert env_transition.rewards.device.type == "cpu"
    assert env_transition.rewards.is_contiguous()
    assert not hasattr(policy_output, "actions")
    assert policy_output.forward_inputs["states"].device.type == "cpu"
    assert policy_output.forward_inputs["states"].is_contiguous()


def test_policy_output_detaches_nested_model_tensors_for_transport():
    source = torch.arange(12.0, requires_grad=True)
    model_output = (source * 2).reshape(3, 4).T
    policy_output = PolicyOutput(
        forward_inputs={
            "nested": {
                "states": model_output,
                "features": [model_output[:, :2]],
            }
        },
        prev_logprobs=model_output,
        prev_values=model_output[:, :1],
        versions=model_output,
    )
    part = PolicyPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 4)],
        obs={},
        output=policy_output,
    )

    _, tensors = pack_dataclass_tensors(part)
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(not tensor.requires_grad for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_nested_dataclass_transport_separates_tensors_from_skeleton():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    shared = torch.arange(4).reshape(2, 2)
    event = PolicyPart(
        sources=[TrajectorySource(key, 2)],
        obs={"states": shared, "task_descriptions": ["a", "b"]},
        output=PolicyOutput(
            forward_inputs={"nested": {"states": shared}},
            prev_values=torch.ones(2, 1),
        ),
    )

    skeleton, tensors = pack_dataclass_tensors(event)
    restored = unpack_dataclass_tensors(skeleton, tensors)

    assert isinstance(skeleton.obs["states"], TensorPlaceholder)
    assert isinstance(
        skeleton.output.forward_inputs["nested"]["states"],
        TensorPlaceholder,
    )
    assert len(tensors) == 2
    assert restored.obs["states"] is restored.output.forward_inputs["nested"]["states"]
    assert torch.equal(restored.output.prev_values, torch.ones(2, 1))


def test_policy_input_split_merge_preserves_sources_and_nested_payloads():
    keys = [TrajectoryKey(0, 0, rank, 0, 2) for rank in range(2)]
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(12).reshape(4, 3),
            "labels": np.arange(4),
        },
        rlt_switch_flags=torch.tensor([False, True, False, True]),
        sources=[TrajectorySource(key, 2) for key in keys],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[0].sources == [TrajectorySource(keys[0], 1)]
    assert shards[1].sources == [
        TrajectorySource(keys[0], 1, offset=1),
        TrajectorySource(keys[1], 2),
    ]
    assert merged.sources == policy_input.sources
    assert torch.equal(merged.obs["states"], policy_input.obs["states"])
    assert np.array_equal(merged.obs["labels"], policy_input.obs["labels"])
    assert torch.equal(merged.rlt_switch_flags, policy_input.rlt_switch_flags)


def test_external_policy_input_split_merge_preserves_actions():
    key = TrajectoryKey(0, 0, 0, 0, 1)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        external_actions=torch.arange(24).reshape(4, 2, 3),
        sources=[TrajectorySource(key, 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert all(not shard.requires_inference for shard in shards)
    assert not merged.requires_inference
    assert torch.equal(merged.external_actions, policy_input.external_actions)
    assert merged.sources == policy_input.sources


def test_online_lerobot_payload_survives_source_routing():
    episode_data = {
        "chunk_actions": torch.arange(12).reshape(3, 2, 2),
        "obs_list": [
            {"images": torch.arange(6).reshape(3, 2)},
            {"images": torch.arange(6, 12).reshape(3, 2)},
        ],
        "terminations": torch.tensor([False, True, False]),
        "truncations": torch.tensor([False, False, True]),
        "infos_list": [
            {"score": np.arange(3)},
            {"score": np.arange(3, 6)},
        ],
    }
    split_data = split_episode_data(episode_data, [1, 2])
    merged_data = merge_episode_data(split_data)

    assert merged_data is not None
    assert torch.equal(merged_data["chunk_actions"], episode_data["chunk_actions"])
    assert torch.equal(
        merged_data["obs_list"][1]["images"],
        episode_data["obs_list"][1]["images"],
    )
    assert np.array_equal(
        merged_data["infos_list"][0]["score"],
        episode_data["infos_list"][0]["score"],
    )


def test_env_part_split_merge_preserves_offsets():
    current_key = TrajectoryKey(1, 2, 3, 0, 4)
    previous_key = TrajectoryKey(1, 2, 3, 0, 3)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        sources=[TrajectorySource(current_key, 4)],
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(previous_key, 4)],
                transition=EnvTransition(rewards=torch.arange(4).reshape(4, 1)),
                next_obs={"states": torch.arange(12).reshape(4, 3)},
                requires_inference=False,
                initial_transition=EnvTransition(
                    dones=torch.zeros(4, 1, dtype=torch.bool)
                ),
            )
        ],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[1].sources == [TrajectorySource(current_key, 3, offset=1)]
    assert shards[1].env_parts[0].sources == [
        TrajectorySource(previous_key, 3, offset=1)
    ]
    assert merged.request_sizes == [1, 3]
    assert torch.equal(
        shards[1].env_parts[0].initial_transition.dones,
        torch.zeros(3, 1, dtype=torch.bool),
    )
    assert torch.equal(
        merged.env_parts[1].next_obs["states"],
        torch.arange(12).reshape(4, 3)[1:],
    )


def test_scalar_batch_leaves_survive_a_split_merge_round_trip():
    """Scalars broadcast by a split must collapse back to the original value.

    Source fragments carry scalar info flags (e.g. ``record_reset``) unchanged,
    so reassembling them must not turn one flag into a list of per-shard copies.
    """
    for value in (True, 3, 1.5, "reset"):
        shards = split_batch_value(value, [2, 2])
        assert shards == [value, value]
        assert merge_batch_values(shards) == value

    nested = {"flag": True, "name": "abc", "states": torch.arange(4).reshape(4, 1)}
    shards = split_batch_value(nested, [3, 1])
    merged = merge_batch_values(shards)
    assert merged["flag"] is True
    assert merged["name"] == "abc"
    assert torch.equal(merged["states"], nested["states"])


def test_merging_conflicting_scalar_batch_values_is_rejected():
    with pytest.raises(ValueError, match="conflicting scalar"):
        merge_batch_values([True, False])


def test_episode_data_round_trip_keeps_scalar_info_flags_intact():
    episode_data = {
        "chunk_actions": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "obs_list": [{"states": torch.arange(4).reshape(4, 1)}],
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
        "truncations": torch.zeros(4, 1, dtype=torch.bool),
        "infos_list": [{"record_reset": True, "segment_advance": False}],
    }

    merged = merge_episode_data(split_episode_data(episode_data, [3, 1]))

    assert merged["infos_list"][0]["record_reset"] is True
    assert merged["infos_list"][0]["segment_advance"] is False
    assert torch.equal(merged["chunk_actions"], episode_data["chunk_actions"])


def test_policy_part_owns_routed_fragment_split_and_merge():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    actions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    part = PolicyPart(
        sources=[TrajectorySource(key, 1), TrajectorySource(key, 3, offset=1)],
        obs={"states": torch.arange(12).reshape(4, 3)},
        output=PolicyOutput(
            forward_inputs={"action": actions},
            prev_values=torch.arange(4).reshape(4, 1),
        ),
    )

    fragments = part.split()
    merged = PolicyPart.merge(fragments)

    assert len(fragments) == 2
    assert fragments[1].sources == [TrajectorySource(key, 3, offset=1)]
    assert torch.equal(fragments[0].output.forward_inputs["action"], actions[:1])
    assert torch.equal(merged.obs["states"], part.obs["states"])


def test_trajectory_step_owns_intervention_and_transition_conversion():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    model_actions = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    policy = PolicyPart(
        sources=[TrajectorySource(key, 1)],
        obs={"states": torch.zeros(1, 2), "task_descriptions": ["pick"]},
        output=PolicyOutput(
            forward_inputs={"action": model_actions, "model_action": model_actions},
            prev_values=torch.ones(1, 1),
        ),
    )
    env = EnvPart(
        sources=[TrajectorySource(key, 1)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            intervene_actions=torch.tensor([[9.0, 8.0, 7.0, 6.0]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
        next_obs={"states": torch.ones(1, 2), "task_descriptions": ["pick"]},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=torch.full((1, 1), 5.0),
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )

    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=True,
        collect_transitions=True,
        enable_rlt=False,
        include_final_value=True,
    )

    assert torch.equal(step.actions, torch.tensor([[1.0, 2.0, 7.0, 6.0]]))
    assert torch.equal(step.forward_inputs["action"], step.actions)
    assert "model_action" not in step.forward_inputs
    assert "task_descriptions" not in step.curr_obs
    assert "task_descriptions" not in step.next_obs
    assert torch.equal(step.final_prev_values, torch.full((1, 1), 5.0))


def test_trajectory_owns_step_materialization_splitting_and_batching():
    steps = [
        TrajectoryStep(
            actions=torch.tensor([[1.0], [2.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            initial_dones=torch.zeros(2, 1, dtype=torch.bool),
            forward_inputs={"action": torch.tensor([[1.0], [2.0]])},
            versions=torch.ones(2, 1),
        ),
        TrajectoryStep(
            actions=torch.tensor([[3.0], [4.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            final_prev_values=torch.zeros(2, 1),
            forward_inputs={"action": torch.tensor([[3.0], [4.0]])},
            versions=torch.ones(2, 1),
        ),
    ]

    trajectory = Trajectory.from_steps(steps, max_episode_length=8)
    shards = trajectory.split(2)
    batch = Trajectory.to_batch(shards)

    assert trajectory.actions.shape == (2, 2, 1)
    assert trajectory.dones.shape == (3, 2, 1)
    assert [shard.actions.shape for shard in shards] == [(2, 1, 1)] * 2
    assert torch.equal(batch["actions"], trajectory.actions)
    assert torch.equal(batch["forward_inputs"]["action"], trajectory.actions)


def test_policy_input_methods_replace_legacy_routing_helpers():
    policy_input = PolicyInput(
        obs={"states": torch.arange(8).reshape(4, 2)},
        sources=[TrajectorySource(TrajectoryKey(1, 0, 0, 0, 0), 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert torch.equal(merged.obs["states"], policy_input.obs["states"])


def test_accumulators_are_not_public_schema_api():
    import rlinf.data.schema as schema

    assert not hasattr(schema, "TrajectoryAccumulator")
    assert not hasattr(schema, "EmbodiedTrajectoryBuilder")
    assert not hasattr(schema, "EmbodiedLerobotTrajectoryBuilder")


def _config(**overrides):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "max_episode_steps": 4,
                    "max_steps_per_rollout_epoch": 1,
                    "rollout_epoch": 1,
                    "total_num_envs": 1,
                }
            },
            "rollout": {
                "collect_prev_infos": True,
                "collect_transitions": False,
                "pipeline_stage_num": 1,
            },
            "actor": {
                "micro_batch_size": 1,
                "seed": 1,
                "model": {"action_dim": 1, "num_action_chunks": 1},
            },
            "algorithm": {
                "adv_type": "gae",
                "dagger": {"online_lerobot": {"enabled": False}},
                "gae_lambda": 1.0,
                "gamma": 0.5,
                "group_size": 1,
                "loss_type": "actor_critic",
                "normalize_advantages": False,
                "reward_type": "chunk_level",
                "shuffle_rollout": False,
            },
            "reward": {"env_reward_weight": 1.0, "reward_weight": 1.0},
            "runner": {
                "enable_decoupled_mode": False,
                "task_type": "embodied",
                "use_training_pipeline": False,
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _source(key: TrajectoryKey, size: int = 1, offset: int = 0):
    return TrajectorySource(key, size, offset)


def _policy(
    key: TrajectoryKey,
    value: float = 1.0,
    *,
    external: bool = False,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    kwargs = (
        {"external_actions": value_tensor.unsqueeze(1)}
        if external
        else {
            "output": PolicyOutput(
                forward_inputs={"action": value_tensor},
                prev_logprobs=torch.zeros(1, 1),
                prev_values=torch.zeros(1, 1),
                versions=torch.full((1, 1), 7.0),
            )
        }
    )
    return PolicyPart(
        sources=[_source(key, **source_kwargs)],
        obs={"states": value_tensor},
        **kwargs,
    )


def _env_part(
    key: TrajectoryKey,
    *,
    episode_data=None,
    initial_transition=None,
    value: float = 2.0,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    return EnvPart(
        sources=[_source(key, **source_kwargs)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.ones(1, 1, dtype=torch.bool),
            truncations=torch.ones(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            episode_data=episode_data,
        ),
        next_obs={"states": value_tensor},
        next_rlt_obs={"states": value_tensor},
        bootstrap_values=torch.tensor([[4.0]]),
        final_prev_values=torch.tensor([[5.0]]),
        initial_transition=initial_transition,
    )


def _episode_data():
    return {
        "chunk_actions": torch.zeros(1, 1, 1),
        "obs_list": [{"states": torch.zeros(1, 1)}],
        "terminations": torch.zeros(1, 1, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, dtype=torch.bool),
        "infos_list": [{}],
    }


def _make(cfg=None, **geometry):
    """Build the public collector with a deterministic cluster geometry."""
    shape = {
        "source_count": 1,
        "chunk_count": 1,
        "shards_per_source": 1,
        "actor_world_size": 1,
    }
    shape.update(geometry)
    collector = TrajectoryCollector()
    with patch.object(
        RolloutGeometry, "from_cfg", return_value=RolloutGeometry(**shape)
    ):
        collector.setup(
            ChannelContext(name="Actor", cfg=cfg if cfg is not None else _config())
        )
    return collector


def _collect(collector, *parts):
    outputs = []
    for part in parts:
        outputs.extend(collector.collect(part, DEFAULT_KEY))
    return outputs


def test_policy_part_requires_exactly_one_policy_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    common = {"sources": [_source(key)], "obs": {"states": torch.zeros(1, 1)}}

    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(**common)
    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(
            **common,
            output=PolicyOutput(forward_inputs={}),
            external_actions=torch.zeros(1, 1),
        )


def test_setup_requires_the_run_config():
    with pytest.raises(ValueError, match="needs the run config"):
        TrajectoryCollector().setup(ChannelContext(name="Actor", cfg=None))


@pytest.mark.parametrize(
    ("overrides", "mode", "dispatcher"),
    [
        ({}, TrajectoryMode.ROLLOUT, "least_loaded"),
        (
            {"runner": {"enable_decoupled_mode": False}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"enable_decoupled_mode": True}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"use_training_pipeline": True}},
            TrajectoryMode.PIPELINE,
            None,
        ),
        (
            {"algorithm": {"dagger": {"online_lerobot": {"enabled": True}}}},
            TrajectoryMode.LEROBOT,
            "least_loaded",
        ),
    ],
    ids=["sync", "async", "decoupled", "pipeline", "lerobot"],
)
def test_plan_is_the_single_mode_and_dispatcher_source(overrides, mode, dispatcher):
    cfg = _config(**overrides)

    assert TrajectoryPlan.mode_from_cfg(cfg) is mode
    assert select_trajectory_collector(cfg) is TrajectoryCollector
    assert select_trajectory_dispatcher(cfg) == dispatcher


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "runner": {
                "enable_decoupled_mode": True,
                "use_training_pipeline": True,
            }
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"dagger": {"online_lerobot": {"enabled": True}}},
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"adv_type": "opd"},
        },
    ],
    ids=["pipeline-decoupled", "pipeline-lerobot", "pipeline-opd"],
)
def test_plan_rejects_unsupported_mode_combinations(overrides):
    with pytest.raises(ValueError, match="does not support"):
        TrajectoryPlan.mode_from_cfg(_config(**overrides))


@pytest.mark.parametrize(
    ("decoupled", "part_order"),
    [
        (False, ("policy", "env")),
        (False, ("env", "policy")),
        (True, ("policy", "env")),
        (True, ("env", "policy")),
    ],
    ids=[
        "sync-policy-first",
        "sync-env-first",
        "decoupled-policy-first",
        "decoupled-env-first",
    ],
)
def test_rollout_output_matches_main_fields_for_sync_and_decoupled(
    decoupled, part_order
):
    """Assert the fields built by main's EnvWorker remain byte-for-byte equal."""
    cfg = _config(runner={"enable_decoupled_mode": decoupled})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    parts = {
        "policy": _policy(key),
        "env": _env_part(key, initial_transition=initial),
    }

    [(_, trajectory)] = _collect(collector, *(parts[name] for name in part_order))

    # These are the exact values main appends in EnvWorker._run_interact_once:
    # one initial boundary, one policy step, and gamma * terminal bootstrap.
    assert torch.equal(trajectory.actions, torch.tensor([[[1.0]]]))
    assert torch.equal(trajectory.prev_logprobs, torch.tensor([[[0.0]]]))
    assert torch.equal(trajectory.prev_values, torch.tensor([[[0.0]], [[5.0]]]))
    assert torch.equal(trajectory.rewards, torch.tensor([[[3.0]]]))
    assert torch.equal(trajectory.dones, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.truncations, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.terminations, torch.tensor([[[False]], [[False]]]))
    assert torch.equal(trajectory.versions, torch.tensor([[[7.0]]]))
    assert torch.equal(trajectory.forward_inputs["action"], torch.tensor([[[1.0]]]))


@pytest.mark.parametrize("decoupled", [False, True], ids=["async", "decoupled"])
def test_rollout_sources_flush_independently_like_main_env_workers(decoupled):
    """A slow env source must not add a global barrier outside pipeline mode."""
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"enable_decoupled_mode": decoupled},
    )
    collector = _make(cfg, source_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    first_key = TrajectoryKey(4, 0, 0, 0, 0)
    first_outputs = _collect(
        collector,
        _policy(first_key),
        _env_part(first_key, initial_transition=initial),
    )
    assert len(first_outputs) == 1

    second_key = TrajectoryKey(4, 0, 1, 0, 0)
    second_outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )
    assert len(second_outputs) == 1


def test_collector_joins_out_of_order_routed_fragments_and_initial_state():
    collector = _make(source_count=1, chunk_count=1, shards_per_source=1)
    collector._joiner._source_batch_size = 2
    key = TrajectoryKey(0, 0, 0, 0, 0)
    initial = EnvTransition(dones=torch.zeros(2, 1, dtype=torch.bool))
    policy_tail = _policy(key, 11.0, size=1, offset=1)
    policy_head = _policy(key, 10.0, size=1, offset=0)
    env = EnvPart(
        sources=[_source(key, size=2)],
        transition=EnvTransition(
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            truncations=torch.zeros(2, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.tensor([[20], [21]])},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=initial,
    )

    assert _collect(collector, policy_tail, policy_head) == []
    [(_, trajectory)] = _collect(collector, env)

    assert torch.equal(trajectory.actions, torch.tensor([[[10.0], [11.0]]]))
    assert torch.equal(trajectory.dones[0], initial.dones)


def test_collector_rejects_initial_state_after_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 1)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="Only chunk zero"):
        _collect(collector, _env_part(key, initial_transition=EnvTransition()))


def test_collector_requires_initial_state_for_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="missing its initial state"):
        _collect(collector, _env_part(key))


def test_collector_keeps_joined_parts_when_output_materialization_fails():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy = _policy(key)
    env = _env_part(key, initial_transition=EnvTransition())
    original_emit = collector._output.emit
    collector._output.emit = Mock(side_effect=RuntimeError("invalid output"))

    _collect(collector, policy)
    with pytest.raises(RuntimeError, match="invalid output"):
        _collect(collector, env)

    collector._output.emit = original_emit
    assert len(_collect(collector, policy)) == 1


def test_rollout_collector_rejects_duplicate_completed_key():
    cfg = _config(env={"train": {"rollout_epoch": 2}})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )
        == []
    )
    with pytest.raises(ValueError, match="duplicate trajectory event"):
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )


def test_rollout_materializes_each_epoch_boundary_and_final_value_in_order():
    collector = _make(_config(env={"train": {"rollout_epoch": 2}}))
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 1, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    [(_, trajectory)] = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert torch.equal(
        trajectory.dones,
        torch.tensor([[[False]], [[True]], [[False]], [[True]]]),
    )
    assert torch.equal(
        trajectory.prev_values,
        torch.tensor([[[0.0]], [[5.0]], [[0.0]], [[5.0]]]),
    )


def test_rollout_omits_policy_statistics_when_collection_is_disabled():
    collector = _make(_config(rollout={"collect_prev_infos": False}))
    key = TrajectoryKey(3, 0, 0, 0, 0)

    [(_, trajectory)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=EnvTransition()),
    )

    assert trajectory.prev_logprobs is None
    assert trajectory.prev_values is None


def test_history_reward_assignment_matches_main_across_chunks():
    cfg = _config(
        env={
            "train": {
                "auto_reset": False,
                "max_steps_per_rollout_epoch": 2,
            }
        },
        reward={
            "env_reward_weight": 1.0,
            "history_reward_assign": True,
            "reward_mode": "history_buffer",
            "reward_weight": 2.0,
        },
    )
    collector = _make(cfg, chunk_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 0, 0, 0, 1)
    first_env = _env_part(first_key, initial_transition=initial)
    first_env.transition.dones = torch.zeros(1, 1, dtype=torch.bool)
    first_env.transition.truncations = torch.zeros(1, 1, dtype=torch.bool)
    second_env = _env_part(second_key)
    second_env.transition.reward_model_output = torch.tensor([[3.0]])
    second_env.transition.reward_assign_lengths = [2]

    assert _collect(collector, _policy(first_key), first_env) == []
    [(_, trajectory)] = _collect(
        collector,
        second_env,
        _policy(second_key),
    )

    # Current reward: 1 + 2 * 3. History assignment adds 2 * 3 to step 0.
    assert torch.equal(trajectory.rewards, torch.tensor([[[7.0]], [[7.0]]]))


@pytest.mark.parametrize("loss_type", ["rlt_ac", "rlt_td3"])
def test_rlt_output_matches_main_transition_and_intervention_behavior(loss_type):
    collector = _make(_config(algorithm={"loss_type": loss_type}))
    key = TrajectoryKey(3, 0, 0, 0, 0)
    current_ref_chunk = torch.tensor([[1.0, 2.0]])
    intervened_chunk = torch.tensor([[9.0, 8.0]])
    policy = PolicyPart(
        sources=[_source(key)],
        obs={"states": torch.zeros(1, 1)},
        output=PolicyOutput(
            forward_inputs={
                "action": current_ref_chunk,
                "z_rl": torch.tensor([[3.0, 4.0]]),
                "proprio": torch.tensor([[5.0, 6.0, 7.0]]),
                "ref_chunk": current_ref_chunk,
            },
            prev_logprobs=torch.zeros(1, 1),
            prev_values=torch.zeros(1, 1),
        ),
    )
    env = EnvPart(
        sources=[_source(key)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
            intervene_actions=intervened_chunk,
            intervene_flags=torch.ones(1, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.ones(1, 1)},
        next_rlt_obs={
            "z_rl": torch.tensor([[13.0, 14.0]]),
            "proprio": torch.tensor([[15.0, 16.0, 17.0]]),
            "ref_chunk": torch.tensor([[11.0, 12.0]]),
        },
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=EnvTransition(),
    )

    [(_, trajectory)] = _collect(collector, env, policy)

    assert set(trajectory.curr_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert set(trajectory.next_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert torch.equal(trajectory.curr_obs["ref_chunk"][0], intervened_chunk)
    assert torch.equal(
        trajectory.next_obs["ref_chunk"][0], torch.tensor([[11.0, 12.0]])
    )


def test_pipeline_output_contains_main_actor_training_fields():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    key = TrajectoryKey(2, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    [(queue_key, batch)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=initial),
    )

    assert queue_key == "0_0_pipeline_actor"
    assert {
        "actions",
        "advantages",
        "dones",
        "prev_logprobs",
        "prev_values",
        "returns",
        "rewards",
    } <= batch.keys()
    assert batch["actions"].shape[0] == 1
    assert batch["advantages"].shape == batch["returns"].shape


def test_pipeline_waits_for_all_sources_and_routes_each_actor_like_main():
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"use_training_pipeline": True},
    )
    collector = _make(
        cfg,
        source_count=2,
        actor_world_size=2,
    )
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(2, 0, 0, 0, 0)
    second_key = TrajectoryKey(2, 0, 1, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert {queue_key for queue_key, _ in outputs} == {
        "0_0_pipeline_actor",
        "1_1_pipeline_actor",
    }
    assert all("advantages" in batch for _, batch in outputs)


def test_pipeline_flushes_each_epoch_to_the_actor_specific_key():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    collector._output._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    collector._output._pipeline_micro_batches = lambda batch, actor_rank: [batch]

    for epoch_id in (0, 1):
        key = TrajectoryKey(2, epoch_id, 0, 0, 0)
        outputs = _collect(
            collector,
            _env_part(key, initial_transition=EnvTransition()),
            _policy(key),
        )
        assert outputs[0][0] == "0_0_pipeline_actor"
        assert (2, epoch_id) not in collector._output._accumulators


def test_online_lerobot_accepts_external_policy_actions_and_emits_shards():
    cfg = _config(
        algorithm={"dagger": {"online_lerobot": {"enabled": True}}},
    )
    collector = _make(cfg, shards_per_source=2)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    outputs = _collect(
        collector,
        _policy(key, external=True),
        _env_part(
            key,
            episode_data=_episode_data(),
            initial_transition=EnvTransition(),
        ),
    )

    assert len(outputs) == 2
    assert all(key == DEFAULT_KEY for key, _ in outputs)


def test_online_lerobot_accumulator_records_external_intervened_action():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=2,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0, 2.0]]]),
        obs_list=[
            {
                "states": torch.tensor([[0.0, 0.0]]),
                "task_descriptions": ["pick"],
            }
        ],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "intervene_action": torch.tensor([[9.0, 8.0]]),
                "intervene_flag": torch.tensor([True]),
            }
        ],
    )

    [episode] = accumulator.drain_episodes()
    assert torch.equal(
        torch.from_numpy(episode[0]["actions"]), torch.tensor([9.0, 8.0])
    )


def test_online_lerobot_accumulator_preserves_auto_reset_observation():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0]]]),
        obs_list=[{"states": torch.tensor([[100.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "final_observation": {"states": torch.tensor([[10.0]])},
                "final_info": {"success_once": torch.tensor([True])},
            }
        ],
    )
    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[2.0]]]),
        obs_list=[{"states": torch.tensor([[20.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[{}],
    )

    first, second = accumulator.drain_episodes()
    assert first[0]["state"].item() == 10.0
    assert first[0]["is_success"].item()
    assert second[0]["state"].item() == 100.0


def test_online_lerobot_accumulator_filters_unsuccessful_episodes():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        only_success=True,
        num_action_chunks=1,
        action_dim=1,
    )

    for terminated, truncated, success in (
        (True, False, False),
        (False, True, True),
        (True, False, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[1.0]]]),
            obs_list=[{"states": torch.tensor([[1.0]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[truncated]]),
            infos_list=[{"success_once": torch.tensor([success])}],
        )

    [episode] = accumulator.drain_episodes()
    assert episode[-1]["is_success"].item()
    assert episode[-1]["done"].item()


def test_online_lerobot_accumulator_expands_vectorized_action_chunks():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=2,
        num_action_chunks=2,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        obs_list=[
            {"states": torch.tensor([[10.0], [30.0]])},
            {"states": torch.tensor([[20.0], [40.0]])},
        ],
        terminations=torch.tensor([[False, True], [False, True]]),
        truncations=torch.zeros(2, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    episodes = accumulator.drain_episodes()
    assert len(episodes) == 2
    assert [frame["actions"].item() for frame in episodes[0]] == [1.0, 2.0]
    assert [frame["actions"].item() for frame in episodes[1]] == [3.0, 4.0]


def test_online_lerobot_accumulator_uses_policy_intervention_metadata():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=2,
        action_dim=1,
    )
    policy_output = PolicyOutput(
        forward_inputs={"action": torch.tensor([[[9.0], [8.0]]])},
        intervene_flags=torch.tensor([[False, True]]),
    )
    chunk_actions = torch.tensor([[[1.0], [2.0]]])

    accumulator.append_chunk_episode_data(
        policy_output=policy_output,
        chunk_actions=chunk_actions,
        obs_list=[
            {"states": torch.tensor([[10.0]])},
            {"states": torch.tensor([[20.0]])},
        ],
        terminations=torch.tensor([[False, True]]),
        truncations=torch.zeros(1, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    [episode] = accumulator.drain_episodes()
    assert [frame["actions"].item() for frame in episode] == [1.0, 8.0]
    assert [frame["intervene_flag"].item() for frame in episode] == [False, True]


def test_online_lerobot_accumulator_applies_recording_controls():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    for state, action, info, terminated in (
        (10.0, 1.0, {"record_reset": True}, False),
        (20.0, 2.0, {"pre_record": True}, False),
        (30.0, 3.0, {"segment_advance": True}, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[action]]]),
            obs_list=[{"states": torch.tensor([[state]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[False]]),
            infos_list=[info],
        )

    [episode] = accumulator.drain_episodes()
    assert len(episode) == 1
    assert episode[0]["state"].item() == 10.0
    assert episode[0]["actions"].item() == 3.0
    assert episode[0]["segment_id"].item() == 1


def test_lerobot_frame_owns_observation_and_image_conversion():
    frame = LeRobotFrame.from_step(
        observation={
            "main_images": torch.full((2, 2, 3), 0.5),
            "wrist_images": torch.zeros(2, 2, 2, 3),
            "states": torch.tensor([1.0, 2.0]),
            "task_descriptions": ["pick"],
        },
        action=torch.tensor([1.0, 2.0]).numpy(),
        info={
            "intervene_action": torch.tensor([9.0, 8.0]),
            "intervene_flag": torch.tensor(True),
            "success_once": torch.tensor(True),
        },
        segment_id=3,
        action_dim=2,
    )

    assert frame is not None
    output = frame.to_dict(episode_success=True, done=True)
    assert output["actions"].tolist() == [9.0, 8.0]
    assert output["image"].dtype.name == "uint8"
    assert {"wrist_image-0", "wrist_image-1"} <= output.keys()
    assert output["task"] == "pick"
    assert output["segment_id"].item() == 3


def test_offline_lerobot_export_reuses_canonical_frame_conversion():
    collector = object.__new__(CollectEpisode)
    collector.num_envs = 1
    buffer = {
        "observations": [
            {
                "main_images": torch.full((2, 2, 3), 0.5),
                "states": torch.tensor([1.0, 2.0]),
                "task_descriptions": ["pick"],
            }
        ],
        "actions": [torch.tensor([1.0, 2.0])],
        "terminated": [True],
        "infos": [
            {},
            {
                "intervene_action": torch.tensor([9.0, 8.0]),
                "intervene_flag": torch.tensor([True]),
            },
        ],
        "segment_ids": [4],
    }

    [frame] = collector._buffer_to_lerobot_ep(buffer, env_idx=0, is_success=True)
    assert frame["actions"].tolist() == [9.0, 8.0]
    assert frame["image"].dtype.name == "uint8"
    assert frame["task"] == "pick"
    assert frame["segment_id"].item() == 4
    assert frame["done"].item()


def _load_real_data_collector(monkeypatch):
    """Load ``collect_real_data.DataCollector`` by its repository path.

    The installed i2rt SDK also ships an ``examples`` package. Loading this
    entrypoint by path avoids that namespace shadowing.
    """
    import importlib.util
    import sys
    from pathlib import Path

    import psutil

    monkeypatch.setattr(psutil, "process_iter", lambda: ())
    path = (
        Path(__file__).resolve().parents[2] / "examples/embodiment/collect_real_data.py"
    )
    spec = importlib.util.spec_from_file_location("_real_data_collection_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_replay_collector_skips_start_preview_and_aborted_transitions(
    monkeypatch, tmp_path
):
    module = _load_real_data_collector(monkeypatch)

    class Accumulator:
        def __init__(self, **kwargs):
            self.samples = []

        def append(self, step):
            self.samples.append(
                (step.curr_obs["states"].item(), step.next_obs["states"].item())
            )

        def to_trajectory(self):
            return SimpleNamespace(
                intervene_flags=torch.zeros(1), samples=self.samples.copy()
            )

    class Env:
        index = 0

        def reset(self, **kwargs):
            return {"states": torch.tensor([[float(self.index)]])}, {}

        def step(self, action):
            self.index += 1
            start = self.index in (2, 6)
            abort = self.index == 4
            invalid_done = self.index == 8
            success = self.index == 10
            done = invalid_done or success
            pre = self.index in (1, 4, 5)
            info = {
                "keyboard_event": ["start" if start else "abort" if abort else None],
                "keyboard_phase": ["pre" if pre else "rec"],
                "record_reset": [start or abort],
                "pre_record": [pre],
                "manual_done": [done],
                "recording_invalid": [invalid_done],
            }
            return (
                {"states": torch.tensor([[float(self.index)]])},
                torch.tensor([float(done)]),
                torch.tensor([done]),
                torch.tensor([False]),
                info,
            )

        def close(self):
            pass

    monkeypatch.setattr(module, "TrajectoryAccumulator", Accumulator)
    collector = object.__new__(module.DataCollector)
    collector.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            record_task_description=True, logger=SimpleNamespace(log_path=str(tmp_path))
        ),
        env=SimpleNamespace(eval=SimpleNamespace(max_episode_steps=100)),
    )
    saved = []
    collector.replay_buffer = SimpleNamespace(
        add_trajectories=saved.extend,
        close=lambda: None,
        auto_save_path=str(tmp_path / "demo_buffer"),
    )
    env = Env()
    collector.env = env
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector._target_step_period = None
    collector.action_dim = 14
    collector.total_cnt = 0
    collector.manual_episode_control_only = True
    logs = []
    collector.log_info = logs.append
    collector.run()
    assert env.index == 10
    assert collector.total_cnt == 2
    assert len(saved) == 1
    assert saved[0].samples == [(8.0, 9.0), (9.0, 10.0)]
    assert any("Recording overflow" in message for message in logs)
    assert sum("Total: 1/1" in message for message in logs) == 1


def test_lerobot_only_collector_does_not_build_replay_trajectories(monkeypatch):
    module = _load_real_data_collector(monkeypatch)

    class Accumulator:
        def __init__(self, **kwargs):
            raise AssertionError("replay accumulator should stay unused")

    class Env:
        resets = 0

        def reset(self, **kwargs):
            self.resets += 1
            return {"states": torch.tensor([[0.0]])}, {}

        def step(self, action):
            return (
                {"states": torch.tensor([[1.0]])},
                torch.tensor([1.0]),
                torch.tensor([True]),
                torch.tensor([False]),
                {"manual_done": [True]},
            )

        def close(self):
            pass

    monkeypatch.setattr(module, "TrajectoryAccumulator", Accumulator)
    collector = object.__new__(module.DataCollector)
    collector.replay_buffer = None
    collector.env = Env()
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector._target_step_period = None
    collector.action_dim = 14
    collector.total_cnt = 0
    collector.manual_episode_control_only = True
    logs = []
    collector.log_info = logs.append
    collector.run()
    assert collector.total_cnt == 1
    assert all("Replay buffer saved" not in message for message in logs)


def test_lerobot_frame_keeps_last_substep_of_flattened_intervene_chunk():
    action_dim = 20
    chunk = 20
    policy = np.arange(action_dim, dtype=np.float32)
    expert = np.arange(chunk * action_dim, dtype=np.float32) + 100
    collector = object.__new__(CollectEpisode)
    collector.num_envs = 1
    buffer = {
        "observations": [{"state": np.zeros(32, dtype=np.float32)}],
        "actions": [policy],
        "terminated": [True],
        "infos": [
            {},
            {
                "intervene_flag": np.ones(chunk, dtype=bool),
                "intervene_action": expert,
            },
        ],
        "segment_ids": [0],
    }
    frames = collector._buffer_to_lerobot_ep(buffer, env_idx=0, is_success=True)
    assert frames is not None and len(frames) == 1
    assert frames[0]["actions"].shape == (action_dim,)
    assert np.array_equal(frames[0]["actions"], expert[-action_dim:])
    assert bool(frames[0]["intervene_flag"][0]) is True


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_export_format_keeps_older_names_and_refuses_two_episode_stores():
    from rlinf.envs.wrappers.collect_episode import resolve_export_formats

    assert resolve_export_formats("pickle") == ("pickle_episode",)
    assert resolve_export_formats(["replay_buffer", "lerobot"]) == (
        "replay_buffer",
        "lerobot_dataset",
    )
    with pytest.raises(ValueError, match="one of"):
        resolve_export_formats(["pickle_episode", "lerobot_dataset"])
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_export_formats("parquet")


class _OperatorTakes:
    """An env publishing the episode edges an operator's buttons produce.

    Each step pops one edge: ``pre`` waits to record, ``start`` begins a take,
    ``rec`` records a frame, ``discard`` drops the take, ``abort`` ends it
    without success, and ``end`` ends it successfully.
    """

    def __init__(self, edges):
        import gymnasium as gym

        self.edges = list(edges)
        self.tick = 0
        self.observation_space = gym.spaces.Box(0, 1, (1,))
        self.action_space = gym.spaces.Box(-1, 1, (1, 2))

    def _observation(self):
        return {
            "states": np.full((1, 3), float(self.tick), dtype=np.float32),
            "main_images": np.full((1, 8, 8, 3), self.tick % 255, dtype=np.uint8),
            "task_descriptions": ["take"],
        }

    def reset(self, *, seed=None, options=None):
        return self._observation(), {}

    def step(self, action):
        self.tick += 1
        edge = self.edges.pop(0)
        recording = edge in ("rec", "end")
        info = {
            "pre_record": np.array([not recording and edge != "start"]),
            "record_reset": np.array([edge in ("start", "discard", "abort")]),
            "episode_discarded": np.array([edge == "discard"]),
            "success": np.array([edge == "end"]),
        }
        return (
            self._observation(),
            np.zeros(1),
            np.array([edge == "end"]),
            np.array([False]),
            info,
        )

    def close(self):
        return None


def test_streaming_keeps_aborted_and_unfinished_takes_but_not_discards(tmp_path):
    """Only an operator's discard drops recorded frames.

    A take ended by a fault, or left open when collection stops, is still
    saved, flagged unsuccessful, so the merge step can leave it out.
    """
    pytest.importorskip("lerobot")
    pytest.importorskip("cv2")
    parquet = pytest.importorskip("pyarrow.parquet")

    edges = ["pre", "start", "rec", "rec", "discard", "pre"]
    edges += ["start", "rec", "rec", "abort"]
    edges += ["start", "rec", "end"]
    edges += ["start", "rec", "rec"]
    collector = CollectEpisode(
        _OperatorTakes(edges),
        str(tmp_path),
        export_format="lerobot_dataset",
        streaming=True,
        fps=30,
    )
    try:
        collector.reset()
        for edge in edges:
            collector.step(np.zeros((1, 2)))
            if edge == "end":
                collector.reset()
    finally:
        collector.close()

    takes = []
    for shard in sorted((tmp_path / "lerobot_dataset" / "rank_0").iterdir()):
        for data in sorted(shard.glob("data/**/*.parquet")):
            table = parquet.read_table(data)
            success = np.asarray(table["is_success"][0].as_py()).reshape(-1)[0]
            takes.append((table.num_rows, bool(success)))
    assert takes == [(2, False), (2, True), (2, False)]


def test_policy_input_compression_preserves_attached_env_parts(monkeypatch):
    """Trajectory requests remain lossless across split/compress/merge routing."""
    monkeypatch.setattr(
        obs_compression,
        "_get_backend",
        lambda _codec: (
            lambda raw, _level: raw,
            lambda raw: raw,
        ),
    )
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key=key, size=4)
    current_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    next_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    policy_input = PolicyInput(
        obs={"main_images": current_images},
        sources=[source],
        env_parts=[
            EnvPart(
                sources=[source],
                transition=EnvTransition(rewards=torch.ones(4, 2)),
                next_obs={"main_images": next_images},
            )
        ],
    )
    env_worker = object.__new__(EnvWorker)
    env_worker.obs_compression_cfg = {
        "enable": True,
        "codec": "lz4",
        "level": 1,
        "xor_delta": True,
    }

    shards = env_worker._split_and_compress_policy_input(policy_input, [1, 3])

    assert is_compressed_image(shards[0].obs["main_images"])
    assert is_compressed_image(shards[0].env_parts[0].next_obs["main_images"])
    merged = MultiStepRolloutWorker._merge_policy_inputs(shards)
    assert torch.equal(merged.obs["main_images"], current_images)
    assert torch.equal(
        torch.cat([part.next_obs["main_images"] for part in merged.env_parts], dim=0),
        next_images,
    )


def test_sync_and_async_workers_share_the_part_channel_contract():
    """Only Rollout publishes parts; Env communicates exclusively with Rollout."""
    assert "actor_channel" not in inspect.signature(EnvWorker.interact).parameters
    assert "actor_channel" not in inspect.signature(AsyncEnvWorker.interact).parameters
    assert (
        "actor_channel" in inspect.signature(MultiStepRolloutWorker.generate).parameters
    )
    assert (
        "actor_channel"
        in inspect.signature(AsyncMultiStepRolloutWorker.generate).parameters
    )


def test_decoupled_evaluation_service_is_reused_and_cancelled_on_stop():
    async def run():
        worker = object.__new__(AsyncMultiStepRolloutWorker)
        worker.env_decoupled_mode = True
        worker._generate_task = None
        worker._evaluate_task = None
        blocker = asyncio.Event()
        calls = 0

        async def serve(_input_channel, _output_channel):
            nonlocal calls
            calls += 1
            await blocker.wait()

        worker._run_evaluate_service = serve
        await worker.ensure_evaluate_service(Mock(), Mock())
        first_task = worker._evaluate_task
        await worker.ensure_evaluate_service(Mock(), Mock())

        assert worker._evaluate_task is first_task
        assert calls == 1

        worker.stop()
        await asyncio.sleep(0)
        assert first_task.cancelled()

    asyncio.run(run())


def test_async_runner_reuses_decoupled_evaluation_service_across_validations():
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"enable_decoupled_mode": True}})
    runner.env_channel = Mock()
    runner.rollout_channel = Mock()
    runner.rollout = Mock()
    runner.rollout.ensure_evaluate_service.return_value = Mock(
        wait=Mock(return_value=None)
    )
    runner.env = Mock()
    runner.env.evaluate.return_value = Mock(wait=Mock(return_value=[{}]))

    with patch(
        "rlinf.runners.async_embodied_runner.compute_evaluate_metrics",
        return_value={},
    ):
        runner.evaluate()
        runner.evaluate()

    assert runner.rollout.ensure_evaluate_service.call_count == 2
    runner.rollout.evaluate.assert_not_called()


def test_decoupled_policy_route_round_trip():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"policy": ["stale"]}
    rollout.recv_from_and_record_batch_routes_with_timeout = AsyncMock(
        return_value=(_policy_input(), [8, 8])
    )
    rollout.send_to_recorded_batch_routes = Mock()

    policy_input, split_sizes = asyncio.run(
        rollout._receive_policy_input(None, "policy", 0)
    )
    source = torch.zeros(16, 4, requires_grad=True)
    output = source * 2
    rollout._send_actions(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=rollout._merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    sent_output = rollout.send_to_recorded_batch_routes.call_args.kwargs["data"]
    assert torch.equal(sent_output, output)
    assert not sent_output.requires_grad
    assert sent_output.grad_fn is None
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=sent_output,
        tag="policy",
        split_fn=rollout._split_actions,
        split_sizes=[8, 8],
    )


class _HangingGet:
    def __init__(self, future: asyncio.Future):
        self._future = future

    async def async_wait(self):
        return await self._future


class _PartialGatherChannel:
    """First get is ready. Later gets stay pending until the test releases them."""

    def __init__(self):
        self.gets = 0
        self.futures: list[asyncio.Future] = []

    def get(self, key, async_op=True):
        del key, async_op
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.futures.append(future)
        self.gets += 1
        if self.gets == 1:
            future.set_result({"batch_index": "0_0_train_policy", "batch": {"n": 1}})
        return _HangingGet(future)


def test_decoupled_gather_returns_a_partial_batch_at_the_deadline():
    """A shard still in flight at the deadline is the next gather's first shard."""
    route_key = ("route",)
    plan = SimpleNamespace(entries=[SimpleNamespace(key=route_key)] * 3)
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.batch_router = {"policy": []}
    rollout._world_size = 2
    rollout._worker_address = SimpleNamespace(root_group_name="RolloutGroup")
    rollout._manager_proxy = None
    channel = _PartialGatherChannel()

    async def _gather():
        return await asyncio.wait_for(
            rollout.recv_from_and_record_batch_routes_with_timeout(
                "EnvGroup",
                channel,
                tag="policy",
                batch_size=8,
                merge_fn=lambda items: items,
                infer_batch_size_fn=lambda item: 1,
                timeout_time=0.05,
                recv_queue_size=3,
            ),
            timeout=1.0,
        )

    async def run():
        started = time.monotonic()
        merged, sizes = await _gather()
        elapsed = time.monotonic() - started

        assert 0.03 <= elapsed < 1.0
        assert merged == [{"n": 1}]
        assert sizes == [1]
        assert channel.gets == 2
        assert rollout.batch_router["policy"] == ["0_0_train_policy"]

        async def _release_late_shard():
            await asyncio.sleep(0.01)
            channel.futures[1].set_result(
                {"batch_index": "1_0_train_policy", "batch": {"n": 2}}
            )

        release = asyncio.create_task(_release_late_shard())
        merged, sizes = await _gather()
        await release

        assert merged == [{"n": 2}]
        assert sizes == [1]
        assert channel.gets == 3
        assert rollout.batch_router["policy"] == [
            "0_0_train_policy",
            "1_0_train_policy",
        ]
        for future in channel.futures:
            if not future.done():
                future.set_result(
                    {"batch_index": "2_0_train_policy", "batch": {"n": 3}}
                )
        await asyncio.sleep(0)

    with (
        patch("rlinf.scheduler.decoupled_build_recv_plan", return_value=plan),
        patch("rlinf.scheduler.get_group_world_size", return_value=2),
    ):
        asyncio.run(run())


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=torch.zeros(8, 4))

    env._recv_actions(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_action_batch_size,
        decoupled_mode=True,
    )


def test_smooth_intervention_builds_external_input_without_policy_history():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    env = SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0, 3.0]])

    policy_input = controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    assert isinstance(policy_input, PolicyInput)
    assert not policy_input.requires_inference
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )


def test_smooth_intervention_requires_online_lerobot_dagger():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 2, "action_dim": 3}},
            "algorithm": {
                "loss_type": "ppo",
                "dagger": {"online_lerobot": {"enabled": True}},
            },
            "env": {
                "train": {
                    "smooth_intervene": True,
                    "env_type": "realworld",
                    "use_pico": True,
                    "use_spacemouse": False,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="loss_type=embodied_dagger"):
        SmoothInterveneController.from_cfg(
            cfg,
            stage_num=1,
            enable_train=True,
            train_num_envs_per_stage=1,
        )


def test_env_sends_external_request_after_intervention_continues():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = True
    env.env_decoupled_mode = False
    env.env_list = [
        SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0]])
    ]
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 2, enabled=True)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(
        obs={"states": torch.zeros(1, 4)},
        transition=EnvTransition(
            dones=torch.tensor([[False, False]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, {}, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert not policy_input.requires_inference
    assert policy_input.env_parts[0] is not None
    assert policy_input.env_parts[0].next_obs is None
    assert torch.equal(policy_input.obs["states"], env_output.obs["states"])
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
    )


def _publish_step_env(**overrides):
    """Build the minimal EnvWorker used by _publish_step tests."""
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = False
    env.env_decoupled_mode = False
    env.collect_final_values = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    for name, value in overrides.items():
        setattr(env, name, value)
    return env


def _terminal_env_output(reset_obs, terminal_obs, **transition):
    flags = {"dones": torch.ones(1, 1, dtype=torch.bool), **transition}
    return EnvOutput(
        obs={"states": reset_obs},
        final_obs={"states": terminal_obs},
        transition=EnvTransition(**flags),
    )


def test_env_sends_terminal_observation_only_as_a_boundary_override():
    env = _publish_step_env()
    reset_obs = torch.zeros(1, 4)
    terminal_obs = torch.ones(1, 4)
    env_output = _terminal_env_output(
        reset_obs,
        terminal_obs,
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, None, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert torch.equal(policy_input.obs["states"], reset_obs)
    assert torch.equal(policy_input.env_parts[0].next_obs["states"], terminal_obs)
    assert policy_input.env_parts[0].requires_inference


def test_rollout_completes_each_epoch_through_policy_inputs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        side_effect=[
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[1]])},
            ),
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[2]])},
            ),
        ]
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[2]])},
        requires_inference=False,
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[3]])},
        requires_inference=True,
    )
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.tensor([[1]])},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[2]])},
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[3]])},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        side_effect=[
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[1]])}}),
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[2]])}}),
            (
                torch.zeros(1, 1),
                {
                    "forward_inputs": {"states": torch.tensor([[3]])},
                    "prev_values": torch.tensor([[4.0, 5.0]]),
                },
            ),
        ]
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(
        generate_one_epoch(
            rollout,
            input_channel=Mock(),
            output_channel=Mock(),
            actor_channel=actor_channel,
        )
    )

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    assert sum(isinstance(part, PolicyPart) for part in parts) == 2
    env_parts = [part for part in parts if isinstance(part, EnvPart)]
    assert len(env_parts) == 2
    assert env_parts[0].initial_transition is completion0.initial_transition
    assert env_parts[0].next_rlt_obs is None
    assert env_parts[0].next_obs is None
    assert torch.equal(env_parts[1].bootstrap_values, torch.tensor([[4.0]]))
    assert torch.equal(env_parts[1].final_prev_values, torch.tensor([[4.0, 5.0]]))
    assert rollout._predict_rollout_actions.call_count == 3


def test_rollout_routes_external_input_without_model_inference():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        return_value=PolicyOutput(
            forward_inputs={"action": torch.ones(1, 6)},
        )
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    dummy_actions = torch.full((1, 2, 3), 2.0)
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.zeros(1, 4)},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    external_actions=dummy_actions,
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.ones(1, 2, 3),
            {"forward_inputs": {"action": torch.ones(1, 6)}},
        )
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(generate_one_epoch(rollout, Mock(), Mock(), actor_channel))

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    inferred_parts = [
        part for part in parts if isinstance(part, PolicyPart) and part.inferred
    ]
    assert len(inferred_parts) == 1
    external_part = next(
        part for part in parts if isinstance(part, PolicyPart) and not part.inferred
    )
    assert torch.equal(external_part.external_actions, dummy_actions)
    assert rollout._predict_rollout_actions.call_count == 1
    assert torch.equal(rollout._send_actions.call_args_list[1].args[1], dummy_actions)


def test_decoupled_final_completion_uses_next_bootstrap():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.stage_num = 1
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 1
    env.enable_online_lerobot = False
    env.env_decoupled_mode = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(obs={"states": torch.zeros(1, 4)})

    completion = env._publish_step(
        rollout_channel=Mock(),
        env_output=env_output,
        initial_transition=EnvTransition(),
        reward_model_output=None,
        chunk_step_data=None,
        epoch_id=0,
        chunk_id=0,
        stage_id=0,
    )

    assert completion is not None
    assert completion.initial_transition is not None
    assert torch.equal(completion.next_obs["states"], env_output.obs["states"])
    env._send_policy_input.assert_not_called()

    env._send_train_bootstrap(
        rollout_channel=Mock(),
        env_outputs=[env_output],
        step_id=1,
        epoch_id=0,
        previous_env_parts={0: completion},
    )

    policy_input = env._send_policy_input.call_args.args[1]
    assert policy_input.env_parts == [completion]


def test_env_closes_reward_stream_with_final_chunk():
    env = object.__new__(EnvWorker)
    env.rollout_epoch = 2
    env.stage_num = 1
    env.n_train_chunk_steps = 2
    env._prefetched_train_bootstrap = None
    env._trajectory_step = 0
    env._rank = 0
    env.env_list = [SimpleNamespace()]
    env.use_training_pipeline = False
    env.env_decoupled_mode = False
    env.enable_online_lerobot = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_actions = Mock(return_value=torch.zeros(1, 4))
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 4)
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                transition=EnvTransition(rewards=torch.zeros(1, 1)),
            ),
            {},
            {},
        )
    )
    env.get_reward_model_output = Mock(return_value=torch.zeros(1, 1))
    env._publish_step = Mock()
    env.record_env_metrics = Mock()
    env.store_last_obs_and_intervened_info = Mock()
    env.finish_rollout = Mock()
    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]


def test_smooth_intervention_holds_at_the_last_commanded_step():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    seen = []

    def get_hold_actions(fallback=None):
        seen.append(fallback)
        return [[1.0, 2.0, 3.0]]

    env = SimpleNamespace(get_hold_actions=get_hold_actions)
    controller.remember_actions(0, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))

    controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    # The wrapper receives the final step of the chunk that was just executed,
    # so it can hold that pose instead of snapping back to its own default.
    assert len(seen) == 1
    assert seen[0].tolist() == [[4.0, 5.0, 6.0]]


def test_rollout_runs_requested_terminal_inference_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "prev_values": torch.zeros(1, 1),
                "forward_inputs": {"states": torch.ones(1, 1)},
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once_with(completion.next_obs)
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.bootstrap_values, torch.zeros(1, 1))
    assert torch.equal(part.final_prev_values, torch.zeros(1, 1))
    assert part.next_rlt_obs is None
    assert part.next_obs is None


def test_rollout_reuses_policy_obs_for_transition_next_obs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = True
    actor_channel = Mock()
    policy_obs = {"states": torch.ones(1, 4)}
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
    )

    rollout._publish_env_part(
        completion,
        policy_obs,
        {"unused": torch.ones(1, 1)},
        actor_channel,
    )

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is policy_obs
    assert part.next_rlt_obs is None


def test_rollout_runs_terminal_inference_for_rlt_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    rollout.hf_model = SimpleNamespace()
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "forward_inputs": {
                    "rlt_transition_z_rl": torch.ones(1, 2),
                    "rlt_transition_proprio": torch.ones(1, 3),
                    "rlt_transition_ref_chunk": torch.ones(1, 4),
                }
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once()
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.next_rlt_obs["z_rl"], torch.ones(1, 2))


def test_rollout_keeps_next_forward_inputs_only_for_rlt():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    actor_channel = Mock()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy_input = PolicyInput(
        obs={"states": torch.zeros(1, 4)},
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(key, 1)],
                transition=EnvTransition(),
            )
        ],
        request_sizes=[1],
    )
    forward_inputs = {
        "rlt_transition_z_rl": torch.ones(1, 2),
        "rlt_transition_proprio": torch.ones(1, 3),
        "rlt_transition_ref_chunk": torch.ones(1, 4),
        "unrelated": torch.ones(1, 8),
    }

    rollout._publish_env_parts(policy_input, forward_inputs, actor_channel)

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is None
    assert torch.equal(
        part.next_rlt_obs["z_rl"],
        forward_inputs["rlt_transition_z_rl"],
    )
    assert set(part.next_rlt_obs) == {"z_rl", "proprio", "ref_chunk"}


def test_vlm_trend_batch_video_metadata_stays_nested_per_sample():
    pytest.importorskip("transformers.video_utils")
    from transformers.video_utils import VideoMetadata

    from rlinf.data.datasets.vlm.vlm_trend_reward import VLMTrendRewardSFTDataset

    captured = {}

    class _Processor:
        video_token = "<|video_pad|>"

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured["videos_kwargs"] = kwargs["videos_kwargs"]
            batch = len(kwargs["text"])
            return {
                "input_ids": torch.zeros(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    VLMTrendRewardSFTDataset.process_inputs(
        processor=_Processor(),
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[["task a"], ["task b"]],
        videos=[
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        ],
        answer_text=None,
        video_fps=24.0,
    )

    metadata = captured["videos_kwargs"]["video_metadata"]
    assert len(metadata) == 2
    assert all(len(sample) == 2 for sample in metadata)
    assert all(
        isinstance(item, VideoMetadata) for sample in metadata for item in sample
    )
    assert metadata[0][0].total_num_frames == 5
    assert metadata[0][0].frames_indices == [0, 1, 2, 3, 4]
    assert metadata[1][1].total_num_frames == 5


# --------------------------------------------------------------------------
# VLM SFT collate: transformers >= 5 emits mm_token_type_ids
# --------------------------------------------------------------------------


def _vlm_sft_sample(idx, seq_len, mm_token_type_ids):
    """A sample shaped like ``Robo2VLMSFTDataset.encode_prompt`` output."""
    return SimpleNamespace(
        idx=idx,
        length=seq_len,
        prompt=torch.arange(1, seq_len + 1, dtype=torch.long),
        answer="a",
        solution="s",
        image_data=None,
        prompt_text="p",
        meta={},
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        label_mask=torch.zeros(seq_len, dtype=torch.bool),
        multi_modal_inputs={
            "pixel_values": torch.randn(4, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": mm_token_type_ids,
        },
    )


def test_vlm_collate_left_pads_mm_token_type_ids_like_prompts():
    from rlinf.data.datasets.vlm.collate_fn import collate_fn

    # The three processor output shapes seen in the wild: (1, L) tensor,
    # plain list, and (L,) tensor.
    samples = [
        _vlm_sft_sample(0, 9, torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 0]])),
        _vlm_sft_sample(1, 5, [0, 1, 1, 0, 0]),
        _vlm_sft_sample(2, 7, torch.tensor([0, 1, 1, 1, 1, 0, 0])),
    ]

    batch = collate_fn(samples)

    mm_ids = batch["multi_modal_inputs"]["mm_token_type_ids"]
    assert mm_ids.shape == (3, 9)
    assert mm_ids.dtype == torch.long

    for i, sample in enumerate(samples):
        pad_len = 9 - sample.length
        # Left padding is 0 (text token) and masked by attention_mask.
        assert (mm_ids[i, :pad_len] == 0).all()
        assert (batch["attention_mask"][i, :pad_len] == 0).all()
        # Valid part matches the processor output bit-for-bit.
        expected = torch.as_tensor(
            sample.multi_modal_inputs["mm_token_type_ids"], dtype=torch.long
        ).flatten()
        assert torch.equal(mm_ids[i, pad_len:], expected)
        # Padding positions line up with the prompt's padding positions.
        assert (batch["prompt"][i, :pad_len] == 0).all()

    assert isinstance(batch["multi_modal_inputs"]["pixel_values"], list)
    assert batch["multi_modal_inputs"]["image_grid_thw"].shape == (3, 3)


def test_vlm_collate_stacks_mm_token_type_ids_without_padding():
    from rlinf.data.datasets.vlm.collate_fn import collate_fn

    samples = [
        _vlm_sft_sample(0, 5, torch.tensor([0, 1, 1, 0, 0])),
        _vlm_sft_sample(1, 5, torch.tensor([[0, 0, 1, 1, 0]])),
    ]

    batch = collate_fn(samples)

    mm_ids = batch["multi_modal_inputs"]["mm_token_type_ids"]
    assert mm_ids.shape == (2, 5)
    assert torch.equal(mm_ids[0], torch.tensor([0, 1, 1, 0, 0]))
    assert torch.equal(mm_ids[1], torch.tensor([0, 0, 1, 1, 0]))


def test_vlm_collate_still_rejects_unknown_mm_keys():
    from rlinf.data.datasets.vlm.collate_fn import collate_fn

    sample = _vlm_sft_sample(0, 5, torch.tensor([0, 1, 1, 0, 0]))
    sample.multi_modal_inputs["some_future_field"] = torch.zeros(5)

    with pytest.raises(ValueError, match="Unsupported multi_modal_input key"):
        collate_fn([sample])


# --------------------------------------------------------------------------
# Reward model splits: preprocessing and FSDPRewardWorker loaders
# --------------------------------------------------------------------------

_PREPROCESS_REWARD_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples/reward/preprocess_reward_dataset.py"
)


@pytest.fixture(scope="module")
def preprocess_reward():
    spec = importlib.util.spec_from_file_location(
        "preprocess_reward_dataset_under_test", _PREPROCESS_REWARD_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_reward_episodes(raw_dir: Path, successes: list[bool], num_frames: int = 4):
    """Write episode pickles in the layout produced by ``CollectEpisode``.

    The first entry is the reset frame with an empty info dict. A successful
    episode reports ``success`` on its last frame only.
    """
    raw_dir.mkdir()
    for ep_idx, success in enumerate(successes):
        observations = [
            {"main_images": torch.full((3, 8, 8), ep_idx, dtype=torch.uint8)}
            for _ in range(num_frames)
        ]
        infos = [{}] + [
            {"success": success and step == num_frames - 1}
            for step in range(1, num_frames)
        ]
        label = "success" if success else "fail"
        with open(raw_dir / f"episode_{ep_idx}_{label}.pkl", "wb") as f:
            pickle.dump({"observations": observations, "infos": infos}, f)


def _save_reward_split(path: Path, num_samples: int) -> None:
    RewardDatasetPayload(
        images=[torch.zeros(3, 8, 8, dtype=torch.uint8)] * num_samples,
        labels=[i % 2 for i in range(num_samples)],
    ).save(str(path))


def _build_reward_loaders(
    train_path: Path,
    val_path: Path,
    micro_batch_size: int,
    world_size: int,
    val_check_interval: int,
):
    """Run ``FSDPRewardWorker.build_dataloader`` as rank 0 without launching it."""
    worker = object.__new__(FSDPRewardWorker)
    worker.cfg = OmegaConf.create(
        {
            "data": {
                "train_data_paths": str(train_path),
                "val_data_paths": str(val_path),
                "num_workers": 0,
            },
            "actor": {"micro_batch_size": micro_batch_size},
            "runner": {"val_check_interval": val_check_interval},
        }
    )
    worker._world_size = world_size
    worker._rank = 0
    return worker.build_dataloader()


@pytest.mark.parametrize(
    ("successes", "empty_split"),
    [
        # No successful episode: both splits are empty, train is reported first.
        ([False] * 5, "train"),
        # A single episode always goes to val, leaving train with no episode.
        ([True], "train"),
        # The only successful episode lands in train under seed 0.
        ([True] + [False] * 9, "val"),
    ],
)
def test_reward_preprocess_rejects_split_without_success_frames(
    preprocess_reward, tmp_path, successes, empty_split
):
    raw_dir = tmp_path / "raw"
    _write_reward_episodes(raw_dir, successes)
    train_path = tmp_path / "out" / "train.pt"
    val_path = tmp_path / "out" / "val.pt"

    with pytest.raises(ValueError, match=f"The {empty_split} split is empty"):
        preprocess_reward.preprocess_and_save_reward_datasets(
            raw_data_path=str(raw_dir),
            train_output_path=str(train_path),
            val_output_path=str(val_path),
            random_seed=0,
        )

    assert not train_path.exists()
    assert not val_path.exists()


def test_reward_preprocess_output_feeds_the_worker_loaders(preprocess_reward, tmp_path):
    raw_dir = tmp_path / "raw"
    _write_reward_episodes(raw_dir, [True] * 5)
    train_path = tmp_path / "out" / "train.pt"
    val_path = tmp_path / "out" / "val.pt"

    metadata = preprocess_reward.preprocess_and_save_reward_datasets(
        raw_data_path=str(raw_dir),
        train_output_path=str(train_path),
        val_output_path=str(val_path),
        random_seed=0,
    )
    train_loader, val_loader = _build_reward_loaders(
        train_path, val_path, micro_batch_size=1, world_size=1, val_check_interval=1
    )

    assert len(train_loader.dataset) == metadata["num_train_samples"] > 0
    assert len(val_loader.dataset) == metadata["num_val_samples"] > 0


@pytest.mark.parametrize("num_train", [0, 6])
def test_reward_loaders_reject_train_split_without_full_batch(tmp_path, num_train):
    # 6 samples over 2 ranks leave 3 per rank, fewer than a micro batch of 4.
    _save_reward_split(tmp_path / "train.pt", num_train)
    _save_reward_split(tmp_path / "val.pt", 8)

    with pytest.raises(ValueError, match=f"has {num_train} samples"):
        _build_reward_loaders(
            tmp_path / "train.pt",
            tmp_path / "val.pt",
            micro_batch_size=4,
            world_size=2,
            val_check_interval=1,
        )


def test_reward_loaders_reject_empty_val_split_when_validating(tmp_path):
    _save_reward_split(tmp_path / "train.pt", 8)
    _save_reward_split(tmp_path / "val.pt", 0)

    with pytest.raises(ValueError, match="validation split is empty"):
        _build_reward_loaders(
            tmp_path / "train.pt",
            tmp_path / "val.pt",
            micro_batch_size=4,
            world_size=2,
            val_check_interval=1,
        )


def test_reward_loaders_allow_empty_val_split_without_validation(tmp_path):
    # runner.val_check_interval <= 0 disables validation, so val data is unused.
    _save_reward_split(tmp_path / "train.pt", 8)
    _save_reward_split(tmp_path / "val.pt", 0)

    train_loader, _ = _build_reward_loaders(
        tmp_path / "train.pt",
        tmp_path / "val.pt",
        micro_batch_size=4,
        world_size=2,
        val_check_interval=-1,
    )

    assert len(train_loader) == 1


def test_reward_loaders_accept_one_batch_per_rank(tmp_path):
    _save_reward_split(tmp_path / "train.pt", 8)
    _save_reward_split(tmp_path / "val.pt", 1)

    train_loader, val_loader = _build_reward_loaders(
        tmp_path / "train.pt",
        tmp_path / "val.pt",
        micro_batch_size=4,
        world_size=2,
        val_check_interval=1,
    )

    assert len(train_loader) == 1
    assert len(val_loader) == 1


def _write_official_mixture_fixture(
    tmp_path,
    name,
    schema,
    offset=0.0,
    *,
    control_mode="native_molmobot_arm7_gripper1_command_unscaled",
    state_dim=9,
):
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / name
    root.mkdir()
    assets = root / "checkpoint/assets/molmobot"
    assets.mkdir(parents=True)
    (assets / "norm_stats.json").write_text("{}")
    (assets.parent / "train_config.pkl").write_bytes(b"numeric fixture config")
    actions = np.arange(56, dtype=np.float32).reshape(7, 8) / 10 + offset
    actions[:, 7] = np.linspace(0, 1, 7) if state_dim == 8 else np.arange(7) * 20
    states = np.arange(7 * state_dim, dtype=np.float32).reshape(7, state_dim) / 100
    supervised = np.array([True, True, True, False, True, True, True])
    data = root / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "action": actions.tolist(),
                "observation.state": states.tolist(),
                "supervised": supervised,
            }
        ),
        data,
    )
    row = {
        "catalog_id": name,
        "schema_id": schema,
        "data_file": str(data),
        "row_offset": 0,
        "frames": 7,
        "core_range": [0, 7],
        "supervision_mask_feature": "supervised",
        "anchor_indices": list(range(7)),
        "source_group_id": f"group_{name}",
        "task": "Pick up the egg",
    }
    for split in ("train", "val", "test"):
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    (root / "normalization.json").write_text(
        json.dumps(
            {
                "schemas": {
                    schema: {
                        "weight": 1.0,
                        "native_control_mode": control_mode,
                    }
                }
            }
        )
    )
    (root / "contract.json").write_text(
        json.dumps(
            {
                "files": {
                    file: hashlib.sha256((root / file).read_bytes()).hexdigest()
                    for file in (
                        "train.jsonl",
                        "val.jsonl",
                        "test.jsonl",
                        "normalization.json",
                    )
                }
            }
        )
    )
    return root, actions, states


def test_official_mixture_preserves_native_targets_and_excludes_partial_chunks(
    tmp_path,
):
    from rlinf.data.datasets.openpi_official import OfficialMixtureDataset

    root, actions, states = _write_official_mixture_fixture(tmp_path, "native", "molmo")
    dataset = OfficialMixtureDataset(
        root, model_path=root / "checkpoint", action_horizon=3
    )
    assert len(dataset) == 2
    assert dataset.available_samples == 7
    assert dataset.incomplete_chunk_dropped == 5
    for index, expected_frame in enumerate((0, 4)):
        _, frame, sample = dataset.numeric_item(index)
        assert frame == expected_frame
        np.testing.assert_array_equal(sample["actions"], actions[frame : frame + 3])
        np.testing.assert_array_equal(
            sample["observation/joint_position"], states[frame, :7]
        )
        np.testing.assert_array_equal(
            sample["observation/gripper_position"], states[frame, 7:]
        )
        assert "action_loss_mask" not in sample


def test_official_mixture_preserves_droid_native8_and_rejects_incorrect_state_dim(
    tmp_path,
):
    from rlinf.data.datasets.openpi_official import OfficialMixtureDataset

    control = "droid_absolute_joint_position_and_normalized_closure"
    root, actions, states = _write_official_mixture_fixture(
        tmp_path, "droid", "droid", control_mode=control, state_dim=8
    )
    dataset = OfficialMixtureDataset(
        root, model_path=root / "checkpoint", action_horizon=3
    )
    assert len(dataset) == 2
    _, frame, raw = dataset.numeric_item(1)
    np.testing.assert_array_equal(raw["actions"], actions[frame : frame + 3])
    np.testing.assert_array_equal(raw["observation/joint_position"], states[frame, :7])
    np.testing.assert_array_equal(
        raw["observation/gripper_position"], states[frame, 7:]
    )
    invalid, _, _ = _write_official_mixture_fixture(
        tmp_path, "invalid_droid", "droid", control_mode=control
    )
    with pytest.raises(ValueError, match="requires native action8/state8"):
        OfficialMixtureDataset(
            invalid, model_path=invalid / "checkpoint", action_horizon=3
        )


@pytest.mark.parametrize("first_source", ["molmo", "droid"])
def test_official_mixture_keeps_closure_units_across_source_order(
    tmp_path, first_source
):
    transforms = pytest.importorskip("openpi.transforms")
    dataset_openpi = pytest.importorskip("molmobot_pi0.dataset_openpi")
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import TrainConfig

    from rlinf.data.datasets.openpi_official import OfficialMixtureDataset

    second_source = "droid" if first_source == "molmo" else "molmo"
    fixtures = {
        "molmo": _write_official_mixture_fixture(tmp_path, "molmo", "molmo"),
        "droid": _write_official_mixture_fixture(
            tmp_path,
            "droid",
            "droid",
            control_mode="droid_absolute_joint_position_and_normalized_closure",
            state_dim=8,
        ),
    }
    model_path = fixtures["molmo"][0] / "checkpoint"
    config = TrainConfig(
        name="native_units",
        model=Pi0Config(action_dim=8, action_horizon=3),
        data=dataset_openpi.MlSpacesDatasetConfigFactory(repo_id=str(tmp_path)),
    )
    (model_path / "assets/train_config.pkl").write_bytes(pickle.dumps(config))
    stats = {
        "mean": np.arange(8, dtype=np.float32) / 10,
        "std": np.linspace(0.5, 1.2, 8, dtype=np.float32),
    }
    norm_stats = {key: transforms.NormStats(**stats) for key in ("state", "actions")}
    import openpi.shared.normalize as normalize

    normalize.save(model_path / "assets/molmobot", norm_stats)
    for root, _, _ in fixtures.values():
        for split in ("train", "val", "test"):
            path = root / f"{split}.jsonl"
            row = json.loads(path.read_text())
            row.update(
                fps=15,
                videos={
                    slot: {"path": f"{slot}.mp4", "from_timestamp": 1.0}
                    for slot in ("base_0_rgb", "left_wrist_0_rgb")
                },
            )
            path.write_text(json.dumps(row) + "\n")
        contract_path = root / "contract.json"
        contract = json.loads(contract_path.read_text())
        import hashlib

        for filename in contract["files"]:
            contract["files"][filename] = hashlib.sha256(
                (root / filename).read_bytes()
            ).hexdigest()
        contract_path.write_text(json.dumps(contract))
    dataset = OfficialMixtureDataset(
        {name: str(fixtures[name][0]) for name in (first_source, second_source)},
        model_path=model_path,
        action_horizon=3,
    )
    # External tokenization and video decoding need neither downloads nor assets.
    with (
        mock.patch("openpi.models.tokenizer.PaligemmaTokenizer") as tokenizer,
        mock.patch(
            "lerobot.common.datasets.video_utils.decode_video_frames",
            return_value=torch.zeros(1, 3, 16, 16),
        ),
    ):
        tokenizer.return_value.tokenize.return_value = (
            np.zeros(4, dtype=np.int32),
            np.ones(4, dtype=bool),
        )
        for index in (1, 3, 0, 2, 1, 3):
            source = first_source if index < 2 else second_source
            _, actions, states = fixtures[source]
            frame = 4 if index % 2 else 0
            expected_state = states[frame, :8].copy()
            expected_actions = actions[frame : frame + 3].copy()
            if source == "molmo":
                expected_state[7] = np.clip(expected_state[7] / 0.824033, 0, 1)
                expected_actions[:, 7] /= 255
            sample = dataset[index]
            np.testing.assert_allclose(
                sample["observation"]["state"],
                (expected_state - stats["mean"]) / (stats["std"] + 1e-6),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                sample["actions"],
                (expected_actions - stats["mean"]) / (stats["std"] + 1e-6),
                rtol=1e-6,
                atol=1e-6,
            )
            assert sample["observation"]["image_mask"] == {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            }
        restored = pickle.loads(pickle.dumps(dataset))
        for index in (0, 2):
            expected, actual = dataset[index], restored[index]
            np.testing.assert_array_equal(actual["actions"], expected["actions"])
            np.testing.assert_array_equal(
                actual["observation"]["state"], expected["observation"]["state"]
            )


def test_official_mixture_target_exposure_preserves_stream_and_resume(tmp_path):
    from rlinf.data.datasets.openpi_official import (
        ExposureMatchedSampler,
        OfficialMixtureDataset,
    )

    target, _, _ = _write_official_mixture_fixture(tmp_path, "target", "molmo")
    auxiliary, _, _ = _write_official_mixture_fixture(
        tmp_path, "auxiliary", "molmo", 1.0
    )
    single = OfficialMixtureDataset(
        {"target": target}, model_path=target / "checkpoint", action_horizon=3
    )
    mixture = OfficialMixtureDataset(
        {"target": target, "aux": auxiliary},
        model_path=target / "checkpoint",
        action_horizon=3,
    )
    base = {
        "target_schemas": ["target/molmo"],
        "target_samples_per_epoch": 16,
        "global_batch_size": 4,
        "seed": 31,
    }
    for rank in (0, 1):
        pure = ExposureMatchedSampler(
            single, auxiliary_samples_per_epoch=0, world_size=2, rank=rank, **base
        )
        mixed = ExposureMatchedSampler(
            mixture, auxiliary_samples_per_epoch=12, world_size=2, rank=rank, **base
        )
        pure_sequence = [single.numeric_item(i)[1] for i in pure]
        mixed_indices = list(mixed)
        target_sequence = [
            mixture.numeric_item(i)[1]
            for i in mixed_indices
            if mixture.numeric_item(i)[0]["schema_id"] == "target/molmo"
        ]
        assert target_sequence == pure_sequence
        assert len(target_sequence) == 8
        assert len(mixed_indices) == 14
        mixed.set_epoch(1)
        iterator = iter(mixed)
        next(iterator)
        next(iterator)
        saved = mixed.state_dict()
        remaining = list(iterator)
        restored = ExposureMatchedSampler(
            mixture, auxiliary_samples_per_epoch=12, world_size=2, rank=rank, **base
        )
        restored.load_state_dict(saved)
        assert list(restored) == remaining
        changed = ExposureMatchedSampler(
            mixture, auxiliary_samples_per_epoch=8, world_size=2, rank=rank, **base
        )
        with pytest.raises(ValueError, match="exposure plan"):
            changed.load_state_dict(saved)


def test_native_mixture_final_export_relocation_and_original_episode_groups(tmp_path):
    from rlinf.data.datasets.openpi.mixture import MixtureSftDataset, prepare_mixture

    catalog, sources = _write_native_mixture_fixture(tmp_path)
    source = {**sources[0], "arms": 1}
    root = catalog / source["name"]
    records = [
        row
        for row in map(
            json.loads, (catalog / "catalog/episodes.jsonl").read_text().splitlines()
        )
        if row["source_dataset"] == source["name"]
    ][:4]
    (root / "meta/partition.json").write_text(
        json.dumps({"metadata": {"arms": 1, "domain": "real"}})
    )
    provenance = []
    final_rows = []
    for row in records:
        index = row["episode_index"]
        provenance.append(
            {
                "episode_index": index,
                "provenance": {
                    "core_range_in_export": [1, 4],
                    "supervision_mask_feature": "source.extra.supervision_mask",
                },
            }
        )
        final_rows.append(
            {
                **row,
                "clip_id": f"clip_{index}",
                "partition": "/removed/export",
                "source_revision": "fixed",
                "source_episode_id": str(index // 2),
                "source_group_id": "publisher_shard",
                "quality_tier": "unannotated",
                "visual_review_status": "accepted",
                "visual_pickup_confirmed": True,
            }
        )
    for row in final_rows:
        row.pop("scene_id")
    (root / "meta/collection_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in final_rows)
    )
    (root / "meta/clip_provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance)
    )
    output = tmp_path / "prepared_final"
    summary = prepare_mixture(
        root, output, [source], val_fraction=0, sampling_unit="episode"
    )
    dataset = MixtureSftDataset(output, action_horizon=3, action_dim=4)
    assert summary["train"]["episodes"] == 4
    assert {row["split_group"] for row in dataset.episodes} == {"arm_a:publisher_shard"}
    assert len({row["source_group_id"] for row in dataset.episodes}) == 2
    assert all(row["data_file"].startswith(str(root)) for row in dataset.episodes)
    row, frame, _, _, mask = dataset.numeric_item(0)
    assert row["row_offset"] == row["episode_index"] * 5
    assert frame == 1
    assert mask[:, 0].tolist() == [True, False, True]


def test_realized_joint_trajectory_uses_future_measured_states_and_time_masks(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from rlinf.data.datasets.openpi.mixture import NumericEpisodeCache
    from rlinf.data.datasets.openpi.realized_trajectory import (
        RECIPE,
        RealizedTrajectoryCache,
    )

    times = np.array([3.0, 3.1, 3.3, 3.5, 3.8])
    rows = [
        {
            "observation.state": [float(t - 3)] * 7 + [0.0],
            "action": [999.0] * 7 + [g],
            "source.timestamp": float(t),
            "mask": True,
        }
        for t, g in zip(times, [0.0, 255.0, 0.0, 255.0, 255.0])
    ]
    path = tmp_path / "packed.parquet"
    pq.write_table(pa.Table.from_pylist(rows + rows), path)
    row = {
        "catalog_id": "second_episode",
        "data_file": str(path),
        "row_offset": 5,
        "frames": 5,
        "core_range": [0, 5],
        "supervision_mask_feature": "mask",
        "target_recipe": {
            **RECIPE,
            "gripper_target": {
                "feature": "action",
                "index": 7,
                "open": 0.0,
                "closed": 255.0,
            },
            "gripper_state": None,
        },
    }
    trajectory_cache = RealizedTrajectoryCache(NumericEpisodeCache())
    sample = trajectory_cache.sample(row, 0)
    assert trajectory_cache.valid_anchors(row).tolist() == [
        bool(trajectory_cache.sample(row, frame)["action_mask"].any())
        for frame in range(row["frames"])
    ]
    np.testing.assert_allclose(sample["target_timestamps"], 3 + np.arange(1, 17) / 15)
    np.testing.assert_allclose(sample["actions"][:12, 0], np.arange(1, 13) / 15)
    assert sample["action_mask"][:12].all()
    assert not sample["action_mask"][12:].any()
    assert sample["actions"][:3, 7].tolist() == [0.0, 1.0, 1.0]
    assert sample["state_mask"].tolist() == [True] * 7 + [False]
    assert (
        not RealizedTrajectoryCache(NumericEpisodeCache())
        .sample(row, 4)["action_mask"]
        .any()
    )


def test_realized_joint_trajectory_shared_statistics_exclude_fixed_validation(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from rlinf.data.datasets.openpi.mixture import MixtureSftDataset, prepare_mixture

    root = tmp_path / "collection"
    partition = root / "robot"
    (root / "catalog").mkdir(parents=True)
    (partition / "meta/episodes").mkdir(parents=True)
    (partition / "data").mkdir()
    (partition / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "data_path": "data/{file_index}.parquet",
                "video_path": "{video_key}/{file_index}.mp4",
            }
        )
    )
    records, episodes, data, provenance = [], [], [], []
    for episode, split in enumerate(["train", "val"]):
        for frame in range(8):
            data.append(
                {
                    "observation.state": [episode * 1000.0 + frame / 5] * 7 + [0.0],
                    "action": [9999.0] * 8,
                    "source.timestamp": frame / 5,
                }
            )
        episodes.append(
            {
                "episode_index": episode,
                "length": 8,
                "dataset_from_index": episode * 8,
                "dataset_to_index": (episode + 1) * 8,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "videos/front/chunk_index": 0,
                "videos/front/file_index": 0,
                "videos/front/from_timestamp": episode * 2.0,
            }
        )
        records.append(
            {
                "catalog_id": str(episode),
                "partition": "robot",
                "provenance_file": "robot/meta/clip_provenance.jsonl",
                "source_dataset": "robot",
                "arms": 1,
                "episode_index": episode,
                "source_group_id": str(episode),
                "split": "unassigned",
                "assigned_split": split,
                "quality_tier": "native_source_success",
                "frames": 8,
                "core_range_in_export": [0, 8],
                "fps": 5,
                "control_mode": "unrelated_native",
                "task": "pick",
            }
        )
        provenance.append({"episode_index": episode, "provenance": {}})
    pq.write_table(pa.Table.from_pylist(data), partition / "data/0.parquet")
    pq.write_table(
        pa.Table.from_pylist(episodes), partition / "meta/episodes/0.parquet"
    )
    (root / "catalog/episodes.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in records)
    )
    (partition / "meta/clip_provenance.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in provenance)
    )
    source = {
        "name": "robot",
        "source_dataset": "robot",
        "arms": 1,
        "weight": 1,
        "cameras": {"base_0_rgb": "front"},
        "gripper_identity": "unknown",
        "trajectory": {"gripper_target": None, "gripper_state": None},
    }
    output = tmp_path / "prepared"
    prepare_mixture(
        root,
        output,
        [source],
        action_horizon=16,
        action_dim=8,
        target_recipe="realized_joint_trajectory",
    )
    dataset = MixtureSftDataset(output, action_horizon=16, action_dim=8)
    stats = dataset.schemas["robot"]
    assert max(stats["action"]["mean"]) < 2
    assert stats["action"]["count"][7] == 0
    _, _, state, actions, mask = dataset.numeric_item(0)
    assert state[7] == 0
    assert not mask[:, 7].any()
    assert np.isfinite(actions).all()
    assert (
        len(
            json.loads((output / "normalization.json").read_text())["target_recipe"][
                "joint_state_indices"
            ]
        )
        == 7
    )


def test_realized_sampler_balances_original_episodes_and_resumes_global_stream(
    tmp_path,
):
    from collections import Counter

    from rlinf.data.datasets.openpi.mixture import MixtureSftDataset, prepare_mixture
    from rlinf.data.datasets.openpi.realized_trajectory import RealizedTrajectorySampler

    root, sources = _write_native_mixture_fixture(tmp_path)
    output = tmp_path / "prepared"
    prepare_mixture(root, output, sources, val_fraction=0, sampling_unit="episode")
    dataset = MixtureSftDataset(output, action_horizon=3, action_dim=4)

    def sampler(rank=0, world_size=1):
        return RealizedTrajectorySampler(
            dataset,
            samples_per_epoch=12000,
            batch_size=1,
            rank=rank,
            world_size=world_size,
            seed=17,
        )

    import copy
    import pickle

    legacy = copy.copy(dataset)
    legacy.episodes = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    for index, row in enumerate(legacy.episodes):
        indexed = dataset.episodes[index]
        np.testing.assert_array_equal(
            indexed.pop("anchor_indices"), row["anchor_indices"]
        )
        assert indexed == {
            key: value for key, value in row.items() if key != "anchor_indices"
        }
    legacy_sampler = RealizedTrajectorySampler(
        legacy, samples_per_epoch=12000, batch_size=1, seed=17
    )
    assert list(legacy_sampler) == list(sampler())
    serialized = pickle.dumps(dataset)
    assert len(serialized) < 32_768
    reopened = pickle.loads(serialized)
    np.testing.assert_array_equal(reopened.ends, dataset.ends)
    assert list(
        RealizedTrajectorySampler(
            reopened, samples_per_epoch=12000, batch_size=1, seed=17
        )
    ) == list(sampler())

    reference = list(sampler())
    rank0, rank1 = list(sampler(0, 2)), list(sampler(1, 2))
    assert [value for pair in zip(rank0, rank1) for value in pair] == reference
    positions = [
        int(np.searchsorted(dataset.ends, index, side="right")) for index in reference
    ]
    counts = Counter(dataset.episodes[index]["schema_id"] for index in positions)
    assert abs(counts["arm_a"] / len(reference) - 0.25) < 0.02
    for schema in counts:
        groups = Counter(
            dataset.episodes[index]["source_group_id"]
            for index in positions
            if dataset.episodes[index]["schema_id"] == schema
        )
        assert all(
            abs(count / counts[schema] - 1 / len(groups)) < 0.035
            for count in groups.values()
        )
    assert len(set(reference)) < len(reference)
    consumed = sampler()
    iterator = iter(consumed)
    assert [next(iterator) for _ in range(79)] == reference[:79]
    restored = sampler()
    restored.load_state_dict(consumed.state_dict())
    assert list(restored) == reference[79:]
    consumed.set_epoch(1)
    assert list(consumed) != reference


def test_realized_sampler_world_size_migration_preserves_global_prefix(tmp_path):
    from itertools import islice

    from rlinf.data.datasets.openpi.mixture import MixtureSftDataset, prepare_mixture
    from rlinf.data.datasets.openpi.realized_trajectory import RealizedTrajectorySampler

    root, sources = _write_native_mixture_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_mixture(root, prepared, sources, val_fraction=0, sampling_unit="episode")
    dataset = MixtureSftDataset(prepared, action_horizon=3, action_dim=4)

    def make_sampler(rank, world_size):
        return RealizedTrajectorySampler(
            dataset,
            samples_per_epoch=224000,
            batch_size=2,
            world_size=world_size,
            rank=rank,
            seed=17,
        )

    states, old_next = [], []
    for rank in range(14):
        sampler = make_sampler(rank, 14)
        sampler.set_epoch(2)
        iterator = iter(sampler)
        list(islice(iterator, 256))
        states.append(sampler.state_dict())
        old_next.append(list(islice(iterator, 16)))
    expected = [index for row in zip(*old_next, strict=True) for index in row]
    new_next = []
    for rank in range(16):
        sampler = make_sampler(rank, 16)
        with pytest.raises(ValueError, match="resume contract differs"):
            sampler.load_state_dict(states[0])
        migration = sampler.migrate_world_size(states)
        assert migration["global_samples_consumed_in_epoch"] == 3584
        assert sampler.position == 224
        assert sampler.num_samples == 14000
        assert sampler.epoch == 2
        new_next.append(list(islice(iter(sampler), 14)))
    assert [index for row in zip(*new_next, strict=True) for index in row] == expected

    for field, value in (("seed", 18), ("position", 255), ("num_samples", 16002)):
        changed = copy.deepcopy(states)
        changed[0][field] = value
        with pytest.raises(ValueError, match="migration"):
            make_sampler(0, 16).migrate_world_size(changed)

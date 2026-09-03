from __future__ import annotations

import numpy as np
import torch

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_data_loader import (
    _IMAGE_KEYS,
    _LEROBOT_IMAGE_KEY,
    _LEROBOT_LEFT_WRIST_KEY,
    _LEROBOT_RIGHT_WRIST_KEY,
    _LEROBOT_STATE_KEY,
    _Repack,
    _sft_collate,
    _TransformedStreamingDataset,
)
from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
    _group_episode_chunks,
)


def _raw_frame(value: float) -> dict:
    image = np.full((3, 2, 2), value, dtype=np.float32)
    return {
        _LEROBOT_IMAGE_KEY: image,
        _LEROBOT_LEFT_WRIST_KEY: image + 1,
        _LEROBOT_RIGHT_WRIST_KEY: image + 2,
        _LEROBOT_STATE_KEY: np.arange(32, dtype=np.float32) + value,
        "timestamp": torch.tensor(value),
    }


def _fake_openpi_transform(raw: dict) -> dict:
    repacked = _Repack()(raw)
    base_value = float(repacked["observation/state"][0])
    result = {
        "image": {
            key: np.full((2, 2, 3), base_value + index, dtype=np.float32)
            for index, key in enumerate(_IMAGE_KEYS)
        },
        "image_mask": {key: np.asarray(True) for key in _IMAGE_KEYS},
        "state": repacked["observation/state"],
        "tokenized_prompt": np.arange(4, dtype=np.int64),
        "tokenized_prompt_mask": np.ones(4, dtype=np.bool_),
    }
    if "actions" in repacked:
        result["actions"] = repacked["actions"]
    return result


class _OneItemDataset:
    def __init__(self, item: dict) -> None:
        self.item = item
        self.hf_dataset = [item]

    def __getitem__(self, _index: int) -> dict:
        return dict(self.item)


def test_short_memory_sft_transform_and_collate() -> None:
    past = _raw_frame(1.0)
    current_history = _raw_frame(2.0)
    current = {
        **current_history,
        "task": "turn on the radio",
        "prompt": "press radio",
        "action": np.zeros((32, 32), dtype=np.float32),
        "_history_frames": [None, past, current_history],
        "_history_frame_mask": torch.tensor([False, True, True]),
        "_history_time_offsets": torch.tensor([0.0, -1.0, 0.0]),
        "_history_contrastive_mask": True,
    }
    transformed = _TransformedStreamingDataset(
        _OneItemDataset(current),
        _fake_openpi_transform,
        history_state_dim=23,
    )[0]

    assert transformed["image"][_IMAGE_KEYS[0]].shape == (3, 2, 2, 3)
    assert transformed["history_states"].shape == (3, 23)
    assert np.count_nonzero(transformed["history_states"][0]) == 0
    np.testing.assert_array_equal(
        transformed["history_frame_mask"], [False, True, True]
    )

    observation, actions = _sft_collate([transformed, transformed])
    assert observation.images[_IMAGE_KEYS[0]].shape == (2, 3, 2, 2, 3)
    assert observation.history_states.shape == (2, 3, 23)
    assert observation.history_frame_mask.shape == (2, 3)
    assert observation.history_time_offsets.shape == (2, 3)
    assert observation.history_contrastive_mask.tolist() == [True, True]
    assert actions.shape == (2, 32, 32)


def test_short_memory_stream_partitions_complete_episodes() -> None:
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset._dist_rank = 0
    dataset._dist_world_size = 1
    dataset.chunks = [
        (0, 250, 0),
        (250, 500, 250),
        (500, 750, 0),
        (750, 900, 250),
    ]
    dataset.history_length = 6
    dataset.seed = 42
    dataset.shuffle = True

    dataset._select_streaming_chunk()

    active_episode_chunks = _group_episode_chunks(dataset._active_chunks)
    assert sorted(dataset._active_chunks) == sorted(dataset.chunks)
    assert all(group[0][2] == 0 for group in active_episode_chunks)
    assert all(
        group[index][0] == group[index - 1][1]
        for group in active_episode_chunks
        for index in range(1, len(group))
    )
    assert dataset.current_streaming_chunk_idx == 0
    assert dataset.current_streaming_frame_idx in {0, 500}


def test_short_memory_stream_retains_history_only_within_episode() -> None:
    dataset = BehaviorSftDataset.__new__(BehaviorSftDataset)
    dataset.history_length = 6
    dataset._active_chunks = [(0, 2, 0), (2, 4, 2), (4, 6, 0)]
    dataset.current_streaming_chunk_idx = 0
    dataset.current_streaming_frame_idx = 2
    dataset._should_obs_loaders_reload = False
    sentinel = object()
    dataset._stream_history = [sentinel]

    dataset._advance_streaming_chunk_if_needed()

    assert dataset.current_streaming_chunk_idx == 1
    assert dataset.current_streaming_frame_idx == 2
    assert dataset._should_obs_loaders_reload is False
    assert dataset._stream_history == [sentinel]

    dataset.current_streaming_frame_idx = 4
    dataset._advance_streaming_chunk_if_needed()

    assert dataset.current_streaming_chunk_idx == 2
    assert dataset.current_streaming_frame_idx == 4
    assert dataset._should_obs_loaders_reload is True

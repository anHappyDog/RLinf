# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_data_loader import (
    _sft_collate,
    _TransformedMapDataset,
)
from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
    _mean_valid_action_loss,
)


def test_mean_valid_action_loss_ignores_padded_skill_tail():
    losses = torch.tensor([[1.0, 2.0, 100.0], [3.0, 100.0, 100.0]])
    action_is_pad = torch.tensor(
        [[False, False, True], [False, True, True]],
        dtype=torch.bool,
    )

    loss = _mean_valid_action_loss(losses, action_is_pad)

    assert loss.item() == pytest.approx(2.0)


def test_mean_valid_action_loss_preserves_legacy_unmasked_batches():
    losses = torch.tensor([[1.0, 3.0]])

    assert _mean_valid_action_loss(losses, None).item() == pytest.approx(2.0)


def test_mean_valid_action_loss_rejects_fully_padded_batch():
    with pytest.raises(ValueError, match="no valid action timesteps"):
        _mean_valid_action_loss(
            torch.ones((1, 2)),
            torch.ones((1, 2), dtype=torch.bool),
        )


def test_transformed_map_dataset_preserves_action_padding_outside_transform():
    source = [{"value": 3, "action_is_pad": torch.tensor([False, True])}]

    dataset = _TransformedMapDataset(source, lambda item: {"value": item["value"] + 1})

    assert dataset[0]["value"] == 4
    assert dataset[0]["action_is_pad"].tolist() == [False, True]


def test_sft_collate_returns_action_padding_mask():
    item = {
        "image": {
            key: torch.zeros((4, 4, 3), dtype=torch.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "image_mask": dict.fromkeys(
            ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), True
        ),
        "state": torch.zeros(32),
        "tokenized_prompt": torch.zeros(8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(8, dtype=torch.bool),
        "actions": torch.zeros((3, 32)),
        "action_is_pad": torch.tensor([False, False, True]),
    }

    _, actions, action_is_pad = _sft_collate([item])

    assert actions.shape == (1, 3, 32)
    assert action_is_pad.dtype == torch.bool
    assert action_is_pad.tolist() == [[False, False, True]]

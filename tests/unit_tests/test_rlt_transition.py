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

import torch

from rlinf.algorithms.rlt.transition import apply_rlt_interventions


def test_apply_rlt_interventions_updates_selected_reference_actions():
    obs = {"ref_chunk": torch.zeros(2, 6)}
    actions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    flags = torch.tensor([[True, False], [False, True]])

    apply_rlt_interventions(obs, actions, flags)

    expected = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 8.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(obs["ref_chunk"], expected)

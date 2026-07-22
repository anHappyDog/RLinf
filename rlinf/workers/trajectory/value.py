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

import torch

from rlinf.data.trajectory import ValueRequest, ValueResult


def infer_value_request(model, request: ValueRequest) -> ValueResult:
    """Run the model's real value-head inference path for selected slots."""
    if not (hasattr(model, "value_head") or hasattr(model, "q_head")):
        raise RuntimeError("model has no value_head or q_head")
    with torch.no_grad():
        _, output = model.predict_action_batch(
            env_obs=request.observations,
            mode="train",
            compute_values=True,
        )
    values = output.get("prev_values")
    if values is None:
        raise RuntimeError("model value-head path did not return prev_values")
    return ValueResult(
        global_step=request.global_step,
        rollout_epoch=request.rollout_epoch,
        chunk_step=request.chunk_step,
        slot_ids=request.slot_ids,
        kind=request.kind,
        values=values[:, :1].detach().cpu().contiguous(),
    )

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

"""Agentic rollout request and result schemas."""

from rlinf.data.schema.agentic.requests import (
    FinishReasonEnum,
    RolloutRequest,
    SeqGroupInfo,
    build_rollout_requests_from_batch,
    get_batch_size,
    get_seq_length,
)
from rlinf.data.schema.agentic.types import (
    BatchResizingIterator,
    DynamicRolloutResult,
    RolloutResult,
)

__all__ = [
    "BatchResizingIterator",
    "DynamicRolloutResult",
    "FinishReasonEnum",
    "RolloutRequest",
    "RolloutResult",
    "SeqGroupInfo",
    "build_rollout_requests_from_batch",
    "get_batch_size",
    "get_seq_length",
]

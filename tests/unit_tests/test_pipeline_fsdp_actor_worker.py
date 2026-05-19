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

import asyncio
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from rlinf.workers.actor.pipeline_fsdp_actor_worker import (
    GlobalBatchState,
    PipelineEmbodiedFSDPActor,
)


class _FakeChannel:
    def __init__(self, items):
        self._items = deque(items)

    def get_nowait(self):
        if not self._items:
            raise asyncio.QueueEmpty
        return self._items.popleft()


def _make_micro_batch(batch_id: int) -> dict[str, torch.Tensor]:
    value = torch.tensor([float(batch_id)], dtype=torch.float32)
    return {
        "mb_id": torch.tensor([batch_id], dtype=torch.int64),
        "rewards": value.clone(),
        "advantages": value.clone(),
        "returns": value.clone(),
    }


def _build_actor() -> PipelineEmbodiedFSDPActor:
    actor = object.__new__(PipelineEmbodiedFSDPActor)
    actor._timer_metrics = {}
    actor.is_weight_offloaded = False
    actor.is_optimizer_offloaded = False
    actor.model = SimpleNamespace(train=MagicMock())
    actor.lr_scheduler = SimpleNamespace(step=MagicMock())
    actor.micro_batches_per_step = 4
    actor.gradient_accumulation = 2
    actor.global_batches_per_step = 2
    actor.update_epoch = 2
    return actor


def test_select_global_batch_never_reuses_done_bucket():
    actor = object.__new__(PipelineEmbodiedFSDPActor)
    actor.update_epoch = 2

    ready_batch = GlobalBatchState(micro_batches=[_make_micro_batch(0)], train_count=1)
    done_batch = GlobalBatchState(micro_batches=[_make_micro_batch(1)], train_count=2)
    global_batches = defaultdict(deque)
    global_batches[1].append(ready_batch)
    global_batches[2].append(done_batch)

    selected = actor.select_global_batch(global_batches)

    assert selected is ready_batch
    assert list(global_batches[2]) == [done_batch]

    selected = actor.select_global_batch(global_batches)
    assert selected is None


def test_compute_micro_batches_requires_exact_divisibility():
    actor = object.__new__(PipelineEmbodiedFSDPActor)

    assert (
        actor.compute_micro_batches(
            total_num_envs=80,
            actor_world_size=2,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )
        == 20
    )

    with pytest.raises(
        AssertionError,
        match="Total flattened rollout samples",
    ):
        actor.compute_micro_batches(
            total_num_envs=81,
            actor_world_size=2,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )


def test_run_training_replays_each_global_batch_exactly_update_epoch_times():
    actor = _build_actor()
    channel = _FakeChannel([_make_micro_batch(i) for i in range(4)])

    train_order = []
    finish_count = 0

    def train_micro_batch(*, micro_batch, metrics, is_last):
        train_order.append((int(micro_batch["mb_id"].item()), is_last))
        metrics.setdefault("actor/loss", []).append(float(micro_batch["mb_id"].item()))

    def finish_global_batch(metrics):
        nonlocal finish_count
        finish_count += 1

    actor.train_micro_batch = train_micro_batch
    actor.finish_global_batch = finish_global_batch

    with (
        patch(
            "rlinf.workers.actor.pipeline_fsdp_actor_worker.compute_rollout_metrics",
            side_effect=lambda batch: {"received_rows": int(batch["rewards"].shape[0])},
        ),
        patch(
            "rlinf.workers.actor.pipeline_fsdp_actor_worker.all_reduce_dict",
            side_effect=lambda metric_dict, op=None: metric_dict,
        ),
    ):
        result = actor.run_training(channel)

    assert train_order == [
        (0, False),
        (1, True),
        (2, False),
        (3, True),
        (0, False),
        (1, True),
        (2, False),
        (3, True),
    ]
    assert finish_count == 4
    actor.model.train.assert_called_once_with()
    actor.lr_scheduler.step.assert_called_once_with()
    assert result["rollout_metrics"] == {"received_rows": 4}
    assert result["training_metrics"] == {"actor/loss": 1.5}

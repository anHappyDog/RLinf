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

import random
from dataclasses import replace

import pytest
import torch

from rlinf.data.trajectory import EnvResult, RewardResult, RolloutResult, ValueResult
from rlinf.models.embodiment.openpi.forward_inputs import OpenPILiberoForwardInputs
from rlinf.workers.trajectory.storage import StorageConfig, TrajectoryStorage

SLOTS = (4, 5, 6, 7)
PARTS = ((6, 4), (7, 5))


def _values(epoch: int, step: int, slot_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(
        [[epoch * 100 + step * 10 + slot_id] for slot_id in slot_ids],
        dtype=torch.float32,
    )


def _forward_inputs(
    epoch: int, step: int, slot_ids: tuple[int, ...]
) -> OpenPILiberoForwardInputs:
    values = _values(epoch, step, slot_ids)
    batch_size = len(slot_ids)
    return OpenPILiberoForwardInputs(
        chains=values[:, None, None, :].expand(batch_size, 5, 2, 1).clone(),
        denoise_inds=torch.arange(4).expand(batch_size, 4).clone(),
        tokenized_prompt=torch.arange(3).expand(batch_size, 3).clone(),
        tokenized_prompt_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        action=values.expand(batch_size, 7).clone(),
        model_action=(values + 1).expand(batch_size, 7).clone(),
        image=torch.full((batch_size, 2, 2, 3), step, dtype=torch.uint8),
        wrist_image=torch.full((batch_size, 2, 2, 3), epoch, dtype=torch.uint8),
        state=values.expand(batch_size, 8).clone(),
    )


def _env_result(
    epoch: int,
    step: int,
    slot_ids: tuple[int, ...],
) -> EnvResult:
    truncations = torch.tensor(
        [
            [step == 0 and slot_id == 5 or step == 1 and slot_id == 7]
            for slot_id in slot_ids
        ]
    )
    terminations = torch.tensor([[step == 1 and slot_id == 6] for slot_id in slot_ids])
    return EnvResult(
        global_step=3,
        rollout_epoch=epoch,
        chunk_step=step,
        slot_ids=slot_ids,
        rewards=_values(epoch, step, slot_ids),
        dones=truncations | terminations,
        terminations=terminations,
        truncations=truncations,
    )


def _rollout_result(
    epoch: int,
    step: int,
    slot_ids: tuple[int, ...],
) -> RolloutResult:
    values = _values(epoch, step, slot_ids)
    return RolloutResult(
        global_step=3,
        rollout_epoch=epoch,
        chunk_step=step,
        slot_ids=slot_ids,
        actions=values.expand(len(slot_ids), 7).clone(),
        forward_inputs=_forward_inputs(epoch, step, slot_ids),
        prev_logprobs=values / 10,
        state_values=values / 100,
    )


def _value_result(
    epoch: int,
    step: int,
    slot_ids: tuple[int, ...],
    kind: str,
) -> ValueResult:
    return ValueResult(
        global_step=3,
        rollout_epoch=epoch,
        chunk_step=step,
        slot_ids=slot_ids,
        kind=kind,
        values=_values(epoch, step, slot_ids) / 1000,
    )


def _config(**changes: object) -> StorageConfig:
    values: dict[str, object] = {
        "global_step": 3,
        "rollout_epochs": 2,
        "chunk_steps": 2,
        "slot_ids": SLOTS,
        "rollout_fields": frozenset(
            {"forward_inputs", "prev_logprobs", "state_values"}
        ),
        "boundary_values": True,
    }
    values.update(changes)
    return StorageConfig(**values)


def _complete_results() -> list[object]:
    results: list[object] = []
    for epoch in range(2):
        for step in range(2):
            for slot_ids in PARTS:
                results.append(_env_result(epoch, step, slot_ids))
                results.append(_rollout_result(epoch, step, slot_ids))
        results.extend(
            [
                _value_result(epoch, 0, (5,), "timeout"),
                _value_result(epoch, 1, (7,), "timeout"),
                _value_result(epoch, 2, (5, 4), "tail"),
            ]
        )
    return results


def _write_all(storage: TrajectoryStorage, results: list[object]) -> None:
    for result in results:
        assert storage.write(result)


def test_arbitrary_arrival_order_exports_the_same_actor_batch() -> None:
    ordered = TrajectoryStorage(_config())
    shuffled = TrajectoryStorage(_config())
    results = _complete_results()
    _write_all(ordered, results)
    random.Random(7).shuffle(results)
    _write_all(shuffled, results)

    assert ordered.ready
    assert shuffled.ready
    expected = ordered.export()
    actual = shuffled.export()
    for name in (
        "env_rewards",
        "dones",
        "terminations",
        "truncations",
        "actions",
        "prev_logprobs",
        "state_values",
        "timeout_values",
        "timeout_mask",
        "tail_values",
        "tail_mask",
    ):
        assert torch.equal(getattr(expected, name), getattr(actual, name))
    assert expected.forward_inputs is not None
    assert actual.forward_inputs is not None
    for (_, expected_value), (_, actual_value) in zip(
        expected.forward_inputs.tensor_fields(),
        actual.forward_inputs.tensor_fields(),
        strict=True,
    ):
        assert torch.equal(expected_value, actual_value)

    assert expected.actions.shape[:3] == (2, 2, 4)
    assert expected.actions[1, 0, :, 0].tolist() == [104, 105, 106, 107]
    assert expected.forward_inputs.batch_size == 2 * 2 * 4
    assert expected.forward_inputs.action[:, 0].tolist() == [
        4,
        5,
        6,
        7,
        14,
        15,
        16,
        17,
        104,
        105,
        106,
        107,
        114,
        115,
        116,
        117,
    ]


def test_exact_duplicate_is_idempotent_but_changed_content_conflicts() -> None:
    storage = TrajectoryStorage(_config())
    result = _env_result(0, 0, PARTS[0])
    assert storage.write(result)
    duplicate = replace(result, rewards=result.rewards.clone())
    assert not storage.write(duplicate)

    changed = replace(result, rewards=result.rewards + 1)
    with pytest.raises(ValueError, match="Conflicting content"):
        storage.write(changed)


def test_different_slot_batch_cannot_overlap_existing_coverage() -> None:
    storage = TrajectoryStorage(_config())
    storage.write(_env_result(0, 0, (4, 5)))
    with pytest.raises(ValueError, match="Overlapping env coverage"):
        storage.write(_env_result(0, 0, (5, 6)))


def test_non_overlapping_results_must_share_one_tensor_schema() -> None:
    storage = TrajectoryStorage(_config())
    storage.write(_env_result(0, 0, (4, 5)))
    incompatible = _env_result(0, 0, (6, 7))
    incompatible = replace(
        incompatible,
        rewards=incompatible.rewards.expand(2, 2).clone(),
        dones=incompatible.dones.expand(2, 2).clone(),
        terminations=incompatible.terminations.expand(2, 2).clone(),
        truncations=incompatible.truncations.expand(2, 2).clone(),
    )
    with pytest.raises(ValueError, match="Inconsistent tensor schema"):
        storage.write(incompatible)


def test_missing_result_never_becomes_ready() -> None:
    storage = TrajectoryStorage(_config())
    results = _complete_results()
    _write_all(storage, results[:-1])

    assert not storage.ready
    assert any("tail value" in problem for problem in storage.missing())
    with pytest.raises(RuntimeError, match="Trajectory is not ready"):
        storage.export()


def test_required_and_unconfigured_optional_fields_fail_immediately() -> None:
    storage = TrajectoryStorage(_config())
    missing_forward_inputs = replace(
        _rollout_result(0, 0, PARTS[0]), forward_inputs=None
    )
    with pytest.raises(ValueError, match="forward_inputs is required"):
        storage.write(missing_forward_inputs)

    observations = {"state": torch.ones(2, 3)}
    unexpected_observations = replace(
        _env_result(0, 0, PARTS[0]), observations=observations
    )
    with pytest.raises(ValueError, match="observations is not configured"):
        storage.write(unexpected_observations)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"global_step": 4}, "global_step"),
        ({"rollout_epoch": 2}, "rollout_epoch"),
        ({"chunk_step": 2}, "chunk_step"),
        ({"slot_ids": (8,)}, "unowned slots"),
    ],
)
def test_coordinates_are_bounded_by_storage_config(
    change: dict[str, object], message: str
) -> None:
    storage = TrajectoryStorage(_config())
    result = replace(_env_result(0, 0, (4,)), **change)
    with pytest.raises(ValueError, match=message):
        storage.write(result)


def test_boundary_values_follow_truncation_and_final_alive_masks() -> None:
    storage = TrajectoryStorage(_config())
    _write_all(storage, _complete_results())
    batch = storage.export()

    assert batch.timeout_mask[0].tolist() == [
        [False, True, False, False],
        [False, False, False, True],
    ]
    assert batch.tail_mask[0].tolist() == [True, True, False, False]
    assert batch.timeout_values[0, 0, 1, 0].item() == pytest.approx(0.005)
    assert batch.tail_values[0, 1, 0].item() == pytest.approx(0.025)
    assert torch.equal(batch.env_rewards, storage.export().env_rewards)


def test_unexpected_boundary_value_prevents_readiness() -> None:
    storage = TrajectoryStorage(_config())
    results = _complete_results()
    results.append(_value_result(0, 0, (4,), "timeout"))
    _write_all(storage, results)

    assert not storage.ready
    assert any("unexpected timeout value" in problem for problem in storage.missing())


def test_external_reward_is_aligned_without_modifying_env_reward() -> None:
    config = _config(
        rollout_epochs=1,
        chunk_steps=1,
        boundary_values=False,
        reward_mode="per_step",
        reward_steps=(0,),
    )
    storage = TrajectoryStorage(config)
    for slot_ids in PARTS:
        env_result = _env_result(0, 0, slot_ids)
        storage.write(env_result)
        storage.write(_rollout_result(0, 0, slot_ids))
        storage.write(
            RewardResult(
                global_step=3,
                rollout_epoch=0,
                chunk_step=0,
                slot_ids=slot_ids,
                rewards=_values(0, 0, slot_ids) + 1000,
            )
        )

    batch = storage.export()
    assert batch.env_rewards[0, 0, :, 0].tolist() == [4, 5, 6, 7]
    assert batch.external_rewards[0, 0, :, 0].tolist() == [1004, 1005, 1006, 1007]
    assert batch.reward_mask.all()

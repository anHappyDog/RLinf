import pytest

from rlinf.config import _validate_embodied_rollout_batch_alignment


def test_embodied_rollout_batch_alignment_accepts_full_batches():
    _validate_embodied_rollout_batch_alignment(
        max_steps_per_rollout_epoch=2880,
        num_action_chunks=8,
        rollout_epoch=1,
        total_num_envs=4,
        global_batch_size=32,
    )


def test_embodied_rollout_batch_alignment_rejects_partial_batches():
    with pytest.raises(ValueError, match="multiple of 64 primitive steps"):
        _validate_embodied_rollout_batch_alignment(
            max_steps_per_rollout_epoch=2848,
            num_action_chunks=8,
            rollout_epoch=1,
            total_num_envs=4,
            global_batch_size=32,
        )


def test_embodied_rollout_batch_alignment_counts_groups_per_update():
    _validate_embodied_rollout_batch_alignment(
        max_steps_per_rollout_epoch=2848,
        num_action_chunks=8,
        rollout_epoch=1,
        total_num_envs=4,
        global_batch_size=32,
        groups_per_update=2,
    )

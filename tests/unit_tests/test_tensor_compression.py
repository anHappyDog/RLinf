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

import pytest
import torch

from rlinf.scheduler.collective.tensor_compression import (
    TensorCompressionOptions,
    TensorWorkspacePool,
)


def test_workspace_pool_prefers_a_reused_slot_and_its_buffer():
    """A released slot outranks an unused slot and retains its workspace."""
    pool = TensorWorkspacePool(TensorCompressionOptions(max_inflight=2))

    first_lease = pool.try_acquire()
    assert first_lease is not None
    first_slot = first_lease.slot
    first_buffer = first_slot.get_buffer(0, 128)
    first_lease.release()

    reused_lease = pool.try_acquire()
    assert reused_lease is not None
    assert reused_lease.slot is first_slot
    assert reused_lease.slot.get_buffer(0, 64).data_ptr() == first_buffer.data_ptr()
    reused_lease.release()


def test_workspace_pool_never_waits_for_a_busy_slot():
    """Saturated pools report no slot instead of blocking a sender."""
    pool = TensorWorkspacePool(TensorCompressionOptions(max_inflight=1))

    lease = pool.try_acquire()
    assert lease is not None
    assert pool.try_acquire() is None

    lease.release()


@pytest.mark.parametrize("codec", ["lz4", "zstd"])
def test_workspace_slot_compresses_and_restores_a_tensor(codec):
    """A slot's codec writes into its reusable buffer without altering bytes."""
    pool = TensorWorkspacePool(TensorCompressionOptions(codec=codec))
    lease = pool.try_acquire()
    assert lease is not None

    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    try:
        destination = lease.slot.get_buffer(
            0, lease.slot.codec.compress_bound(source.numel())
        )
        compressed_numel = lease.slot.codec.compress_into(source, destination)
        assert compressed_numel < source.numel()

        restored = torch.empty_like(source)
        lease.slot.codec.decompress_into(
            destination[:compressed_numel], compressed_numel, restored
        )
        assert torch.equal(restored, source)
    finally:
        lease.release()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"codec": "invalid"}, "Unsupported"),
        ({"level": 0}, "level"),
        ({"min_bytes": 0}, "Minimum"),
        ({"max_inflight": 0}, "inflight"),
    ],
)
def test_compression_options_validate_positive_limits(kwargs, message):
    """Invalid compression limits fail before a collective starts."""
    with pytest.raises(ValueError, match=message):
        TensorCompressionOptions(**kwargs)

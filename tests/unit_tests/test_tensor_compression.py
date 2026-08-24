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

import json
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from rlinf.scheduler.cluster.cluster import Cluster, ClusterEnvVar
from rlinf.scheduler.collective.collective_group import (
    CollectiveGroup,
    CollectiveGroupOptions,
)
from rlinf.scheduler.collective.tensor_buffer_pool import (
    TensorBufferPool,
    TensorBufferPoolOptions,
)
from rlinf.scheduler.collective.tensor_compression import (
    TensorCodecPool,
    TensorCompressionOptions,
    TensorCompressionWireMetadata,
)
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.tensor_codec import LZ4TensorCodec


def test_codec_and_buffer_pools_reuse_resources_independently():
    """Released codecs and buffers are independently preferred for reuse."""
    codec_pool = TensorCodecPool(TensorCompressionOptions(max_inflight=2))
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=512))

    first_lease = codec_pool.try_acquire_compressor()
    assert first_lease is not None
    first_codec = first_lease.codec
    first_buffer_lease = buffer_pool.try_acquire(128)
    assert first_buffer_lease is not None
    first_buffer = first_buffer_lease.tensor
    first_lease.release()
    first_buffer_lease.release()

    reused_lease = codec_pool.try_acquire_compressor()
    assert reused_lease is not None
    assert reused_lease.codec is first_codec
    reused_buffer = buffer_pool.try_acquire(64)
    assert reused_buffer is not None
    assert reused_buffer.tensor.data_ptr() == first_buffer.data_ptr()
    reused_lease.release()
    reused_buffer.release()


def test_buffer_pool_uses_best_fit_buffers_independent_of_tensor_order():
    """The Worker-wide pool chooses the smallest tensor buffer that fits."""
    pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=1024))
    large_lease = pool.try_acquire(512)
    small_lease = pool.try_acquire(128)
    assert large_lease is not None
    assert small_lease is not None
    large_buffer = large_lease.tensor
    small_buffer = small_lease.tensor
    large_lease.release()
    small_lease.release()

    small_reuse = pool.try_acquire(100)
    large_reuse = pool.try_acquire(300)
    assert small_reuse is not None
    assert large_reuse is not None
    assert small_reuse.tensor.data_ptr() == small_buffer.data_ptr()
    assert large_reuse.tensor.data_ptr() == large_buffer.data_ptr()
    small_reuse.release()
    large_reuse.release()


def test_buffer_pool_reuses_same_size_bucket_and_tracks_cached_bytes():
    """Equal-sized buffers share a bucket and update cached accounting eagerly."""
    pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=1024))
    leases = [pool.try_acquire(128), pool.try_acquire(128), pool.try_acquire(256)]
    assert all(lease is not None for lease in leases)
    pointers = {lease.tensor.data_ptr() for lease in leases}

    for lease in leases:
        lease.release()
    assert pool.cached_bytes == 512

    first = pool.try_acquire(100)
    second = pool.try_acquire(200)
    assert first is not None
    assert second is not None
    assert first.tensor.data_ptr() in pointers
    assert second.tensor.data_ptr() in pointers
    assert pool.cached_bytes == 128
    first.release(cache=False)
    second.release(cache=False)


def test_buffer_pool_never_exceeds_its_worker_budget():
    """Active buffers make later acquisitions fall back without overallocating."""
    pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=512))
    buffer = pool.try_acquire(400)
    assert buffer is not None
    assert pool.try_acquire(200) is None
    assert pool.allocated_bytes == 400

    buffer.release(cache=False)
    assert pool.allocated_bytes == 0


def test_buffer_pool_evicts_idle_buffers_to_fit_a_new_shape():
    """Historical shapes cannot make the bounded cache grow indefinitely."""
    pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=512))
    old_buffer = pool.try_acquire(128)
    assert old_buffer is not None
    old_buffer.release()

    replacement = pool.try_acquire(512)
    assert replacement is not None
    assert pool.allocated_bytes == 512
    replacement.release()


def test_buffer_pool_evicts_an_entire_size_bucket_in_one_step():
    """A new large allocation can replace many equal-sized idle buffers."""
    pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=512))
    leases = [pool.try_acquire(128) for _ in range(4)]
    assert all(lease is not None for lease in leases)
    for lease in leases:
        lease.release()

    replacement = pool.try_acquire(512)
    assert replacement is not None
    assert pool.allocated_bytes == 512
    assert pool.cached_bytes == 0
    replacement.release()


def test_buffer_pool_preserves_a_large_buffer_when_a_small_one_can_fit():
    """A speculative small compression cannot consume a much larger buffer."""
    pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=256))
    large_lease = pool.try_acquire(128)
    assert large_lease is not None
    large_buffer = large_lease.tensor
    large_lease.release()

    small_lease = pool.try_acquire(1)
    assert small_lease is not None
    assert small_lease.tensor.data_ptr() != large_buffer.data_ptr()
    small_lease.release(cache=False)
    reused_large = pool.try_acquire(100)
    assert reused_large is not None
    assert reused_large.tensor.data_ptr() == large_buffer.data_ptr()
    reused_large.release()


def test_codec_pool_never_waits_for_a_busy_compressor():
    """Saturated pools report no slot instead of blocking a sender."""
    pool = TensorCodecPool(TensorCompressionOptions(max_inflight=1))

    lease = pool.try_acquire_compressor()
    assert lease is not None
    assert pool.try_acquire_compressor() is None

    lease.release()


def test_codec_pool_defaults_to_four_slots_per_direction():
    """The default pool supports several concurrent CollectiveGroups."""
    pool = TensorCodecPool(TensorCompressionOptions())

    assert pool._fresh_compressors.qsize() == 4
    assert pool._decompressors.qsize() == 4


@pytest.mark.parametrize("codec", ["lz4", "zstd"])
def test_codec_pool_compresses_and_restores_a_tensor(codec):
    """A slot's codec writes into its reusable buffer without altering bytes."""
    options = TensorCompressionOptions(codec=codec)
    codec_pool = TensorCodecPool(options)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions())
    lease = codec_pool.try_acquire_compressor()
    assert lease is not None

    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor_codec = lease.codec
    buffer = buffer_pool.try_acquire(compressor_codec.compress_bound(source.numel()))
    assert buffer is not None
    try:
        compressed_numel = compressor_codec.compress_into(source, buffer.tensor)
        assert compressed_numel < source.numel()
    finally:
        lease.release()
    try:
        restored = torch.empty_like(source)
        decompressor = codec_pool.acquire_decompressor(
            TensorCompressionWireMetadata(
                codec=codec,
                level=options.level,
                compressed_numel=(compressed_numel,),
            )
        )
        try:
            assert decompressor.codec is not compressor_codec
            decompressor.codec.decompress_into(
                buffer.tensor[:compressed_numel], compressed_numel, restored
            )
        finally:
            decompressor.release()
        assert torch.equal(restored, source)
    finally:
        buffer.release()


def test_codec_pool_rejects_mismatched_decompression_metadata():
    """A collective has one codec configuration for both directions."""
    pool = TensorCodecPool(TensorCompressionOptions(codec="lz4", level=1))

    with pytest.raises(ValueError, match="does not match"):
        pool.acquire_decompressor(
            TensorCompressionWireMetadata(codec="zstd", level=1, compressed_numel=(1,))
        )


def test_tensor_over_codec_bound_skips_compression(monkeypatch):
    """A codec input-limit error leaves that tensor on the raw wire path."""
    group = object.__new__(CollectiveGroup)
    options = TensorCompressionOptions(min_bytes=1, max_inflight=1)
    pool = TensorCodecPool(options)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions())
    group._worker = SimpleNamespace(
        _get_tensor_codec_pool=lambda: pool,
        _get_tensor_buffer_pool=lambda: buffer_pool,
    )
    group._tensor_compression = options
    tensor = torch.zeros(128, dtype=torch.uint8)

    monkeypatch.setattr(
        pool._fresh_compressors.queue[0],
        "compress_bound",
        lambda _source_bytes: None,
    )
    wire_tensors, metadata, buffers = group._compress_cpu_tensors(
        [tensor],
        [tensor],
    )

    assert wire_tensors[0] is tensor
    assert metadata is None
    assert buffers == []


def test_buffer_budget_exhaustion_preserves_the_raw_tensor():
    """A tensor that cannot fit the Worker budget follows the baseline wire path."""
    options = TensorCompressionOptions(min_bytes=1)
    pool = TensorCodecPool(options)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=64))
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _get_tensor_codec_pool=lambda: pool,
        _get_tensor_buffer_pool=lambda: buffer_pool,
    )
    group._tensor_compression = options
    tensor = torch.zeros(128, dtype=torch.uint8)

    wire_tensors, metadata, buffers = group._compress_cpu_tensors([tensor], [tensor])

    assert wire_tensors[0] is tensor
    assert metadata is None
    assert buffers == []
    assert buffer_pool.allocated_bytes == 0


def test_buffer_budget_can_compress_part_of_a_tensor_list():
    """One exhausted list entry falls back without discarding earlier compression."""
    options = TensorCompressionOptions(min_bytes=1)
    pool = TensorCodecPool(options)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=160))
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _get_tensor_codec_pool=lambda: pool,
        _get_tensor_buffer_pool=lambda: buffer_pool,
    )
    group._tensor_compression = options
    tensors = [
        torch.zeros(128, dtype=torch.uint8),
        torch.zeros(128, dtype=torch.uint8),
    ]

    wire_tensors, metadata, buffers = group._compress_cpu_tensors(tensors, tensors)

    assert metadata is not None
    assert metadata.compressed_numel[0] is not None
    assert metadata.compressed_numel[1] is None
    assert wire_tensors[0] is not tensors[0]
    assert wire_tensors[1] is tensors[1]
    assert len(buffers) == 1
    buffers[0].release()


def test_compressor_is_released_before_payload_buffers():
    """A slow wire transfer retains buffers without occupying a codec slot."""
    options = TensorCompressionOptions(min_bytes=1, max_inflight=1)
    codec_pool = TensorCodecPool(options)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions())
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _get_tensor_codec_pool=lambda: codec_pool,
        _get_tensor_buffer_pool=lambda: buffer_pool,
    )
    group._tensor_compression = options
    tensor = torch.zeros(128, dtype=torch.uint8)

    _, metadata, payload_buffers = group._compress_cpu_tensors([tensor], [tensor])

    assert metadata is not None
    assert payload_buffers
    next_compressor = codec_pool.try_acquire_compressor()
    assert next_compressor is not None
    next_compressor.release()
    for buffer in payload_buffers:
        buffer.release()


def test_decompressor_is_acquired_after_receiving_wire_payload():
    """A slow GLOO receive does not occupy a Worker-wide decoder slot."""
    events = []

    class Decompressor:
        codec = SimpleNamespace(
            decompress_into=lambda *_args: events.append("decompress")
        )

        def release(self):
            events.append("release")

    codec_pool = SimpleNamespace(
        acquire_decompressor=lambda _metadata: (
            events.append("acquire") or Decompressor()
        )
    )
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=128))
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _get_tensor_codec_pool=lambda: codec_pool,
        _get_tensor_buffer_pool=lambda: buffer_pool,
    )
    group._tensor_compression = TensorCompressionOptions(min_bytes=1)
    group._recv = lambda *_args, **_kwargs: events.append("recv")
    metadata = TensorCompressionWireMetadata(
        codec="lz4", level=1, compressed_numel=(32,)
    )

    group._recv_cpu_tensor_payloads(
        [(torch.empty(128, dtype=torch.uint8), 32)], metadata, comm_id=0
    )

    assert events == ["recv", "acquire", "decompress", "release"]


def test_unhelpful_compression_does_not_cache_its_buffer(monkeypatch):
    """A buffer that does not reduce wire bytes is discarded immediately."""
    options = TensorCompressionOptions(min_bytes=1, max_inflight=1)
    pool = TensorCodecPool(options)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions(max_bytes=1024))
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _get_tensor_codec_pool=lambda: pool,
        _get_tensor_buffer_pool=lambda: buffer_pool,
    )
    group._tensor_compression = options
    tensor = torch.zeros(128, dtype=torch.uint8)
    codec = pool._fresh_compressors.queue[0]
    monkeypatch.setattr(codec, "compress_into", lambda source, destination: 128)

    wire_tensors, metadata, buffers = group._compress_cpu_tensors([tensor], [tensor])

    assert wire_tensors[0] is tensor
    assert metadata is None
    assert buffers == []
    assert buffer_pool.allocated_bytes == 0


def test_worker_lazily_shares_independent_codec_and_buffer_pools():
    """CollectiveGroups share Worker-wide codec and buffer pools."""
    worker = object.__new__(Worker)
    worker._tensor_buffer_pool_options = TensorBufferPoolOptions()
    worker._tensor_compression = TensorCompressionOptions()
    worker._tensor_buffer_pool = None
    worker._tensor_buffer_pool_lock = threading.Lock()
    worker._tensor_codec_pool = None
    worker._tensor_codec_pool_lock = threading.Lock()

    first_codec_pool = worker._get_tensor_codec_pool()
    second_codec_pool = worker._get_tensor_codec_pool()
    first_buffer_pool = worker._get_tensor_buffer_pool()
    second_buffer_pool = worker._get_tensor_buffer_pool()

    assert first_codec_pool is not None
    assert second_codec_pool is first_codec_pool
    assert first_buffer_pool is not None
    assert second_buffer_pool is first_buffer_pool
    assert not hasattr(first_codec_pool, "buffer_pool")


def test_worker_does_not_create_a_codec_pool_when_compression_is_disabled():
    """Disabled compression does not load codec libraries or allocate contexts."""
    worker = object.__new__(Worker)
    worker._tensor_compression = TensorCompressionOptions(enabled=False)
    worker._tensor_codec_pool = None
    worker._tensor_codec_pool_lock = threading.Lock()

    assert worker._get_tensor_codec_pool() is None
    assert worker._tensor_codec_pool is None


def test_lz4_compress_bound_returns_none_for_an_unsupported_input_size():
    """An input-size limit is a normal no-compression outcome."""
    codec = LZ4TensorCodec()

    assert codec.compress_bound(LZ4TensorCodec._MAX_INPUT_SIZE + 1) is None


def test_collective_group_options_exclude_tensor_compression():
    """Tensor compression is not a per-call collective option."""
    with pytest.raises(TypeError, match="tensor_compression"):
        CollectiveGroupOptions(tensor_compression=TensorCompressionOptions())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enabled": "yes"}, "boolean"),
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


def test_compression_options_reject_unknown_cluster_yaml_key():
    """Typos in the public YAML interface fail clearly."""
    with pytest.raises(ValueError, match="Unsupported"):
        TensorCompressionOptions.from_dict({"min_byte": 1024})


@pytest.mark.parametrize(
    ("config", "message"),
    [({"max_bytes": 0}, "buffer pool"), ({"max_byte": 1024}, "Unsupported")],
)
def test_tensor_buffer_pool_options_validate_public_config(config, message):
    """Invalid tensor buffer settings fail before Workers start."""
    with pytest.raises(ValueError, match=message):
        TensorBufferPoolOptions.from_dict(config)


def test_disabled_compression_preserves_the_original_cpu_tensors():
    """``enabled=False`` takes the raw path before acquiring any buffer."""
    group = object.__new__(CollectiveGroup)
    group._tensor_compression = TensorCompressionOptions(enabled=False, min_bytes=1)
    tensors = [torch.zeros(128 * 1024, dtype=torch.uint8)]
    wire_tensors, metadata, buffers = group._compress_cpu_tensors(
        tensors,
        tensors,
    )

    assert wire_tensors is tensors
    assert metadata is None
    assert buffers == []


def test_cluster_serializes_validated_tensor_pool_and_compression_config(monkeypatch):
    """The public YAML config becomes validated Worker configuration."""
    compression_env = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    buffer_pool_env = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_BUFFER_POOL
    )
    monkeypatch.delenv(compression_env, raising=False)
    monkeypatch.delenv(buffer_pool_env, raising=False)

    Cluster._configure_collective_tensor_envs(
        {
            "collective": {
                "tensor_buffer_pool": {"max_bytes": 4096},
                "tensor_compression": {
                    "enabled": False,
                    "codec": "zstd",
                    "level": 3,
                    "min_bytes": 1024,
                    "max_inflight": 2,
                },
            }
        }
    )

    assert json.loads(os.environ[compression_env]) == {
        "enabled": False,
        "codec": "zstd",
        "level": 3,
        "min_bytes": 1024,
        "max_inflight": 2,
    }
    assert json.loads(os.environ[buffer_pool_env]) == {"max_bytes": 4096}


def test_compression_uses_the_default_tensor_buffer_pool_config(monkeypatch):
    """Compression gets a bounded buffer pool without extra YAML."""
    buffer_pool_env = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_BUFFER_POOL
    )
    monkeypatch.delenv(buffer_pool_env, raising=False)

    Cluster._configure_collective_tensor_envs(
        {"collective": {"tensor_compression": {"enabled": True}}}
    )

    assert json.loads(os.environ[buffer_pool_env]) == {"max_bytes": 2 * 1024**3}


def test_tensor_buffer_pool_can_be_configured_without_compression(monkeypatch):
    """The generic buffer pool does not depend on compression being enabled."""
    compression_env = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    buffer_pool_env = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_BUFFER_POOL
    )
    monkeypatch.setenv(compression_env, '{"enabled": true}')

    Cluster._configure_collective_tensor_envs(
        {"collective": {"tensor_buffer_pool": {"max_bytes": 4096}}}
    )

    assert compression_env not in os.environ
    assert json.loads(os.environ[buffer_pool_env]) == {"max_bytes": 4096}


def test_worker_loads_the_job_wide_tensor_compression_config(monkeypatch):
    """Workers load the shared configuration propagated by Cluster."""
    worker = object.__new__(Worker)
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    monkeypatch.setenv(env_var_name, '{"codec": "zstd", "min_bytes": 1024}')

    assert worker._load_tensor_compression_options() == TensorCompressionOptions(
        codec="zstd", min_bytes=1024
    )


def test_worker_loads_the_job_wide_tensor_buffer_pool_config(monkeypatch):
    """Workers load the independent shared tensor-buffer configuration."""
    worker = object.__new__(Worker)
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_BUFFER_POOL
    )
    monkeypatch.setenv(env_var_name, '{"max_bytes": 4096}')

    assert worker._load_tensor_buffer_pool_options() == TensorBufferPoolOptions(
        max_bytes=4096
    )


def test_cluster_yaml_overrides_the_internal_worker_environment(monkeypatch):
    """Workers receive the YAML setting rather than a manually supplied value."""
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    cluster = object.__new__(Cluster)
    node = SimpleNamespace(env_vars={env_var_name: "manual-value"})
    cluster._nodes = [node]

    monkeypatch.setenv(env_var_name, '{"enabled": true}')
    cluster._set_scheduler_env_vars()
    assert node.env_vars[env_var_name] == '{"enabled": true}'

    monkeypatch.delenv(env_var_name)
    cluster._set_scheduler_env_vars()
    assert env_var_name not in node.env_vars


@pytest.mark.parametrize(
    ("env_var", "scheduler_value", "worker_value"),
    [
        (
            ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION,
            '{"codec": "lz4"}',
            '{"codec": "zstd"}',
        ),
        (
            ClusterEnvVar.COLLECTIVE_TENSOR_BUFFER_POOL,
            '{"max_bytes": 4096}',
            '{"max_bytes": 8192}',
        ),
        (ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION, None, '{"codec": "zstd"}'),
    ],
)
def test_worker_env_cannot_override_collective_tensor_config(
    env_var, scheduler_value, worker_value
):
    """User environment merging cannot replace scheduler-owned tensor settings."""
    env_var_name = Cluster.get_full_env_var_name(env_var)
    scheduler_env_vars = (
        {env_var_name: scheduler_value} if scheduler_value is not None else {}
    )
    worker_env_vars = {
        env_var_name: worker_value,
        "USER_SETTING": "preserved",
    }

    merged_env_vars = Cluster._enforce_scheduler_owned_env_vars(
        worker_env_vars,
        scheduler_env_vars,
    )

    assert merged_env_vars.get(env_var_name) == scheduler_value
    assert merged_env_vars["USER_SETTING"] == "preserved"

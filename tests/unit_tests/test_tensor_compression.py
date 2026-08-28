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

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.scheduler.cluster.cluster import Cluster, ClusterEnvVar
from rlinf.scheduler.collective.collective_group import (
    CollectiveGroup,
    CollectiveGroupOptions,
    TensorData,
)
from rlinf.scheduler.collective.tensor_buffer_pool import (
    TensorBufferPool,
    TensorBufferPoolOptions,
)
from rlinf.scheduler.collective.tensor_compression import (
    LZ4CodecProvider,
    LZ4CodecProviderOptions,
    TensorCompressionOptions,
    TensorCompressionWireMetadata,
    ZstdCodecProvider,
    ZstdCodecProviderOptions,
    create_tensor_codec_provider,
)
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.tensor_codec import LZ4TensorCodec


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


def test_zstd_provider_never_waits_for_a_busy_compressor():
    """Saturated Zstd context pools do not block a sender."""
    provider = ZstdCodecProvider(ZstdCodecProviderOptions(max_inflight=1))

    codec = provider.try_acquire_compressor()
    assert codec is not None
    assert provider.try_acquire_compressor() is None
    provider.release(codec)

    reused_codec = provider.try_acquire_compressor()
    assert reused_codec is not None
    provider.release(reused_codec)


def test_lz4_provider_supports_concurrent_round_trips():
    """The shared stateless LZ4 instance is safe across Worker threads."""
    provider = LZ4CodecProvider(LZ4CodecProviderOptions())

    def round_trip(value: int) -> bool:
        source = torch.full((128 * 1024,), value, dtype=torch.uint8)
        compressor = provider.try_acquire_compressor()
        assert compressor is not None
        try:
            capacity = compressor.compress_bound(source.numel())
            assert capacity is not None
            compressed = torch.empty(capacity, dtype=torch.uint8)
            compressed_numel = compressor.compress_into(source, compressed)
        finally:
            provider.release(compressor)

        restored = torch.empty_like(source)
        decompressor = provider.acquire_decompressor()
        try:
            decompressor.decompress_into(compressed, compressed_numel, restored)
        finally:
            provider.release(decompressor)
        return torch.equal(restored, source)

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(round_trip, range(16)))


@pytest.mark.parametrize("codec", ["lz4", "zstd"])
def test_codec_provider_compresses_and_restores_a_tensor(codec):
    """A provider's codec writes and restores tensor bytes."""
    provider_options = (
        LZ4CodecProviderOptions() if codec == "lz4" else ZstdCodecProviderOptions()
    )
    codec_provider = create_tensor_codec_provider(provider_options)
    assert codec_provider.codec_name == codec
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions())

    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor = codec_provider.try_acquire_compressor()
    assert compressor is not None
    try:
        buffer = buffer_pool.try_acquire(compressor.compress_bound(source.numel()))
        assert buffer is not None
        compressed_numel = compressor.compress_into(source, buffer.tensor)
        assert compressed_numel < source.numel()
    finally:
        codec_provider.release(compressor)
    try:
        restored = torch.empty_like(source)
        decompressor = codec_provider.acquire_decompressor()
        try:
            decompressor.decompress_into(
                buffer.tensor[:compressed_numel], compressed_numel, restored
            )
        finally:
            codec_provider.release(decompressor)
        assert torch.equal(restored, source)
    finally:
        buffer.release()


def test_collective_group_prepares_compressed_cpu_tensors():
    """Prepared tensor data keeps raw entries and replaces compressed entries."""
    options = TensorCompressionOptions()
    codec_provider = create_tensor_codec_provider(options.provider)
    buffer_pool = TensorBufferPool(TensorBufferPoolOptions())
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_options=options,
        _tensor_buffer_pool=buffer_pool,
        _get_tensor_codec_provider=lambda: codec_provider,
    )
    fp32_tensor = torch.zeros(4096, dtype=torch.float32)
    uint8_tensor = torch.zeros(16 * 1024, dtype=torch.uint8)
    tensor_data = TensorData(
        cpu_tensor_mask=[True, True],
        cpu_tensors=[fp32_tensor, uint8_tensor],
        accel_tensors=[],
    )

    wire_data, buffers = group._compress_tensor_data(tensor_data)

    assert wire_data.compression is not None
    assert wire_data.compression.compressed_numel[0] is None
    assert wire_data.compression.compressed_numel[1] is not None
    assert wire_data.cpu_tensors[0] is fp32_tensor
    assert wire_data.cpu_tensors[1] is not uint8_tensor
    assert tensor_data.cpu_tensors[0] is fp32_tensor
    assert tensor_data.cpu_tensors[1] is uint8_tensor
    for buffer in buffers:
        buffer.release()


def test_collective_group_restores_a_compressed_cpu_tensor():
    """Tensor-list metadata restores a compressed CPU payload in place."""
    options = TensorCompressionOptions(min_bytes=1)
    codec_provider = create_tensor_codec_provider(options.provider)
    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor = codec_provider.try_acquire_compressor()
    assert compressor is not None
    capacity = compressor.compress_bound(source.numel())
    assert capacity is not None
    wire_tensor = torch.empty(capacity, dtype=torch.uint8)
    try:
        wire_numel = compressor.compress_into(source, wire_tensor)
    finally:
        codec_provider.release(compressor)

    metadata = {
        "meta": [(source.shape, source.dtype)],
        "pb": "payload",
        "cpu_tensor_mask": [True],
        "compression": TensorCompressionWireMetadata(
            codec=options.codec,
            compressed_numel=(wire_numel,),
        ),
    }
    incoming = iter(
        [
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, dtype=torch.uint8),
            wire_tensor[:wire_numel],
        ]
    )

    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_options=options,
        _tensor_buffer_pool=TensorBufferPool(TensorBufferPoolOptions()),
        _get_tensor_codec_provider=lambda: codec_provider,
    )
    group._peer_rank = 0
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)
    group._tensor_to_object = lambda *_args: metadata
    group._recv = lambda tensor, *_args, **_kwargs: tensor.copy_(next(incoming))

    tensors, piggyback_payload = group._recv_tensor_list(comm_id=0)

    assert piggyback_payload == "payload"
    assert torch.equal(tensors[0], source)


def test_float32_compression_can_be_explicitly_enabled():
    """An empty exclusion list restores dtype-agnostic compression."""
    options = TensorCompressionOptions(min_bytes=1, excluded_dtypes=())

    assert options.should_compress(torch.zeros(1, dtype=torch.float32))


def test_worker_lazily_shares_one_codec_provider():
    """Concurrent CollectiveGroups share one Worker-wide codec provider."""
    worker = object.__new__(Worker)
    worker._tensor_compression_options = TensorCompressionOptions()
    worker._tensor_buffer_pool = TensorBufferPool(TensorBufferPoolOptions())
    worker._tensor_codec_provider = None
    worker._lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as executor:
        codec_providers = list(
            executor.map(lambda _: worker._get_tensor_codec_provider(), range(16))
        )

    assert all(provider is codec_providers[0] for provider in codec_providers)
    assert worker._tensor_codec_provider is codec_providers[0]


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(torch.zeros(1, dtype=torch.uint8), id="below-min-bytes"),
        pytest.param(
            torch.zeros(16 * 1024 // 4, dtype=torch.float32),
            id="excluded-dtype",
        ),
    ],
)
def test_ineligible_tensors_do_not_initialize_the_codec_provider(tensor):
    """Raw CPU transfers do not require the configured codec library."""
    options = TensorCompressionOptions()
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_options=options,
        _get_tensor_codec_provider=lambda: pytest.fail(
            "ineligible tensors initialized the codec provider"
        ),
    )
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )

    wire_data, buffers = group._compress_tensor_data(tensor_data)

    assert wire_data is tensor_data
    assert buffers == []


def test_lz4_compress_bound_returns_none_for_an_unsupported_input_size():
    """An input-size limit is a normal no-compression outcome."""
    codec = LZ4TensorCodec()

    assert codec.compress_bound(LZ4TensorCodec._MAX_INPUT_SIZE + 1) is None


def test_collective_group_options_exclude_tensor_compression():
    """Tensor compression is not a per-call collective option."""
    with pytest.raises(TypeError, match="tensor_compression"):
        CollectiveGroupOptions(tensor_compression=TensorCompressionOptions())


def test_tensor_container_helpers_keep_async_send_without_unused_options():
    """Private send helpers retain their baseline async contract only."""
    send_helpers = [
        CollectiveGroup._send_tensor_list,
        CollectiveGroup._send_tensor_dict,
        CollectiveGroup._send_tensor_dataclass,
    ]
    recv_helpers = [
        CollectiveGroup._recv_tensor_list,
        CollectiveGroup._recv_tensor_dict,
        CollectiveGroup._recv_tensor_dataclass,
    ]

    for helper in send_helpers:
        parameters = inspect.signature(helper).parameters
        assert "async_op" in parameters
        assert "options" not in parameters
    for helper in recv_helpers:
        assert "options" not in inspect.signature(helper).parameters


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"enabled": "yes"}, "boolean"),
        ({"codec": "invalid"}, "Unsupported"),
        ({"codec": "lz4", "params": {"acceleration": 0}}, "acceleration"),
        ({"codec": "zstd", "params": {"level": 0}}, "level"),
        ({"min_bytes": 0}, "Minimum"),
        ({"codec": "zstd", "params": {"max_inflight": 0}}, "inflight"),
        ({"excluded_dtypes": "float32"}, "must be a list"),
        ({"excluded_dtypes": ["not_a_dtype"]}, "Unsupported excluded"),
        ({"excluded_dtypes": ["float32", "float32"]}, "duplicates"),
        ({"level": 1}, "Unsupported"),
        ({"max_inflight": 4}, "Unsupported"),
        ({"codec": "lz4", "params": {"max_inflight": 4}}, "LZ4"),
        ({"codec": "zstd", "params": {"acceleration": 1}}, "Zstd"),
        ({"codec": "lz4", "params": 1}, "mapping"),
        ({"min_bytes": 1.5}, "integer"),
        ({"min_bytes": True}, "integer"),
        ({"codec": "lz4", "params": {"acceleration": 1.5}}, "integer"),
        ({"codec": "lz4", "params": {"acceleration": True}}, "integer"),
        ({"codec": "zstd", "params": {"level": 1.5}}, "integer"),
        ({"codec": "zstd", "params": {"level": True}}, "integer"),
        ({"codec": "zstd", "params": {"max_inflight": 4.0}}, "integer"),
        ({"codec": "zstd", "params": {"max_inflight": True}}, "integer"),
    ],
)
def test_compression_options_validate_provider_params(config, message):
    """Each provider accepts only its own valid parameters."""
    with pytest.raises(ValueError, match=message):
        TensorCompressionOptions.from_dict(config)


def test_compression_options_reject_unknown_cluster_yaml_key():
    """Typos in the public YAML interface fail clearly."""
    with pytest.raises(ValueError, match="Unsupported"):
        TensorCompressionOptions.from_dict({"min_byte": 1024})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"max_bytes": 0}, "buffer pool"),
        ({"max_byte": 1024}, "Unsupported"),
        ({"max_bytes": 1.5}, "integer"),
        ({"max_bytes": True}, "integer"),
    ],
)
def test_tensor_buffer_pool_options_validate_public_config(config, message):
    """Invalid tensor buffer settings fail before Workers start."""
    with pytest.raises(ValueError, match=message):
        TensorBufferPoolOptions.from_dict(config)


def test_cluster_serializes_validated_collective_config():
    """The public YAML config becomes validated Worker configuration."""
    cluster = object.__new__(Cluster)
    cluster._set_collective_env_vars(
        OmegaConf.create(
            {
                "collective": {
                    "tensor_buffer_pool": {"max_bytes": 4096},
                    "tensor_compression": {
                        "enabled": False,
                        "codec": "zstd",
                        "min_bytes": 1024,
                        "excluded_dtypes": ["float32", "float64"],
                        "params": {"level": 3, "max_inflight": 2},
                    },
                }
            }
        )
    )

    env_name = Cluster.get_full_env_var_name(ClusterEnvVar.COLLECTIVE_CONFIG)
    assert json.loads(cluster._collective_env_vars[env_name]) == {
        "tensor_buffer_pool": {"max_bytes": 4096},
        "tensor_compression": {
            "enabled": False,
            "codec": "zstd",
            "min_bytes": 1024,
            "excluded_dtypes": ["float32", "float64"],
            "params": {"level": 3, "max_inflight": 2},
        },
    }


def test_cluster_stores_collective_config_in_node_metadata(monkeypatch):
    """NodeManager metadata retains scheduler-owned collective configuration."""
    cluster = object.__new__(Cluster)
    cluster._nodes = [SimpleNamespace(env_vars={})]
    cluster._set_collective_env_vars(
        OmegaConf.create({"collective": {"tensor_buffer_pool": {"max_bytes": 4096}}})
    )
    env_name = Cluster.get_full_env_var_name(ClusterEnvVar.COLLECTIVE_CONFIG)
    monkeypatch.setenv(env_name, "stale-worker-value")

    cluster._set_scheduler_env_vars()

    assert (
        cluster._nodes[0].env_vars[env_name] == cluster._collective_env_vars[env_name]
    )


def test_attached_cluster_restores_collective_config_from_node_metadata(monkeypatch):
    """A Cluster attached inside a Worker can allocate child Workers."""
    from rlinf.scheduler.manager.node_manager import NodeManager

    env_name = Cluster.get_full_env_var_name(ClusterEnvVar.COLLECTIVE_CONFIG)
    serialized_config = json.dumps(
        {"tensor_buffer_pool": {"max_bytes": 4096}}, sort_keys=True
    )
    node = SimpleNamespace(env_vars={env_name: serialized_config})
    manager = SimpleNamespace(get_nodes=lambda: ([node], [], None))
    monkeypatch.setattr("ray.is_initialized", lambda: True)
    monkeypatch.setattr(NodeManager, "get_proxy", lambda no_wait: manager)
    cluster = object.__new__(Cluster)

    cluster._init_from_existing_managers()

    assert cluster._collective_env_vars == {env_name: serialized_config}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"collective": {"tensor_compresssion": {"enabled": True}}},
            "Unsupported cluster.collective",
        ),
        ({"collective": []}, "cluster.collective must be a mapping"),
        (
            {"collective": {"tensor_compression": True}},
            "cluster.collective.tensor_compression must be a mapping",
        ),
    ],
)
def test_cluster_rejects_invalid_collective_config(config, message):
    """Invalid collective configuration fails before Workers start."""
    cluster = object.__new__(Cluster)
    with pytest.raises(ValueError, match=message):
        cluster._set_collective_env_vars(OmegaConf.create(config))


def test_worker_loads_and_probes_collective_resources(monkeypatch):
    """Workers load shared resources and probe an enabled codec library."""
    worker = object.__new__(Worker)
    worker._lock = threading.Lock()
    env_var_name = Cluster.get_full_env_var_name(ClusterEnvVar.COLLECTIVE_CONFIG)
    monkeypatch.setenv(
        env_var_name,
        json.dumps(
            {
                "tensor_buffer_pool": {"max_bytes": 4096},
                "tensor_compression": {
                    "codec": "zstd",
                    "min_bytes": 1024,
                    "params": {"level": 3},
                },
            }
        ),
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.utils.tensor_codec.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert worker._tensor_compression_options == TensorCompressionOptions(
        min_bytes=1024,
        provider=ZstdCodecProviderOptions(level=3),
    )
    assert worker._tensor_buffer_pool.options == TensorBufferPoolOptions(max_bytes=4096)
    assert probes == ["zstd"]
    assert worker._tensor_codec_provider is None


def test_worker_skips_codec_resources_when_compression_is_disabled(monkeypatch):
    """Disabled compression neither probes nor creates codec resources."""
    worker = object.__new__(Worker)
    worker._lock = threading.Lock()
    env_var_name = Cluster.get_full_env_var_name(ClusterEnvVar.COLLECTIVE_CONFIG)
    monkeypatch.setenv(
        env_var_name,
        json.dumps(
            {
                "tensor_compression": {
                    "enabled": False,
                    "codec": "zstd",
                }
            }
        ),
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.utils.tensor_codec.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert probes == []
    with pytest.raises(ValueError, match="not enabled"):
        worker._get_tensor_codec_provider()
    assert worker._tensor_codec_provider is None


def test_net_emulation_uses_the_compressed_wire_size():
    """Compression finishes before a point-to-point bandwidth reservation."""
    group = object.__new__(CollectiveGroup)
    tensor = torch.zeros(1024, dtype=torch.uint8)
    wire_tensor = torch.zeros(64, dtype=torch.uint8)
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )
    metadata = TensorCompressionWireMetadata(codec="lz4", compressed_numel=(64,))
    wire_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[wire_tensor],
        accel_tensors=[],
        compression=metadata,
    )
    events = []

    group._init_process_group = lambda **_kwargs: None
    group._compress_tensor_data = lambda _tensor_data: (
        events.append("compress") or wire_data,
        [],
    )
    group._wait_for_net_emulation = lambda *_payloads, size_bytes=None: events.append(
        ("reserve", size_bytes)
    )
    group._send = lambda *_args, **_kwargs: None
    group._send_tensor_list = lambda *_args, **_kwargs: events.append("send")
    group._cur_worker_address = SimpleNamespace(get_name=lambda: "Src:0")
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)
    group._net_emu_manager = object()

    group._atomic_send(
        work=None,
        object=tensor,
        comm_id=0,
        object_type=CollectiveGroup.TENSOR,
        tensor_data=tensor_data,
    )

    raw_size = group._estimate_payload_size((tensor, None))
    metadata_size = group._estimate_payload_size((metadata,))
    assert events == [
        "compress",
        ("reserve", raw_size - tensor.numel() + wire_tensor.numel() + metadata_size),
        "send",
    ]


def test_compressed_send_skips_size_estimation_without_net_emulation():
    """Disabled network emulation adds no payload-estimation overhead."""
    group = object.__new__(CollectiveGroup)
    tensor = torch.zeros(1024, dtype=torch.uint8)
    metadata = TensorCompressionWireMetadata(codec="lz4", compressed_numel=(64,))
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )
    wire_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[torch.zeros(64, dtype=torch.uint8)],
        accel_tensors=[],
        compression=metadata,
    )

    group._net_emu_manager = None
    group._init_process_group = lambda **_kwargs: None
    group._compress_tensor_data = lambda _tensor_data: (wire_data, [])
    group._estimate_payload_size = lambda *_args: pytest.fail(
        "payload size was estimated with network emulation disabled"
    )
    group._send = lambda *_args, **_kwargs: None
    group._send_tensor_list = lambda *_args, **_kwargs: None
    group._cur_worker_address = SimpleNamespace(get_name=lambda: "Src:0")
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)

    group._atomic_send(
        work=None,
        object=tensor,
        comm_id=0,
        object_type=CollectiveGroup.TENSOR,
        tensor_data=tensor_data,
    )

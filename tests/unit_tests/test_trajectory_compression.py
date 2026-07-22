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

import rlinf.workers.trajectory.live as live
from rlinf.workers.trajectory.compression import (
    CompressionConfig,
    CompressionPipeline,
    compress_tensors,
    decompress_tensors,
)
from rlinf.workers.trajectory.tensor_codec import create_tensor_codec


def _config(codec: str = "lz4") -> CompressionConfig:
    return CompressionConfig(
        enabled=True,
        codec=codec,
        min_bytes=1,
        block_bytes=4096,
    )


def test_block_compression_is_bitwise_for_mixed_dtypes_and_shapes() -> None:
    tensors = {
        "image": torch.zeros((4, 64, 64, 3), dtype=torch.uint8),
        "values": torch.arange(4096, dtype=torch.float32).reshape(512, 8),
        "indices": torch.arange(2048, dtype=torch.int64),
        "mask": torch.arange(4096).remainder(3).bool(),
    }
    config = _config()
    codec = create_tensor_codec(config.codec, level=config.level)

    wire, metadata, stats = compress_tensors(tensors, config, codec)
    restored = decompress_tensors(wire, metadata, codec)

    assert stats["compressed_blocks"] > 0
    assert stats["wire_bytes"] <= stats["raw_bytes"]
    assert restored.keys() == tensors.keys()
    for key in tensors:
        assert restored[key].dtype == tensors[key].dtype
        assert restored[key].shape == tensors[key].shape
        assert torch.equal(restored[key], tensors[key])


def test_incompressible_tensor_uses_single_raw_fallback_without_growth() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randint(
        0,
        256,
        (2 * 1024 * 1024,),
        dtype=torch.uint8,
        generator=generator,
    )
    config = _config()
    codec = create_tensor_codec(config.codec, level=config.level)

    wire, metadata, stats = compress_tensors({"random": source}, config, codec)

    assert metadata == {}
    assert tuple(wire) == ("random",)
    assert wire["random"] is source
    assert stats["wire_bytes"] == stats["raw_bytes"] == source.nbytes


def test_zstd_uses_the_same_transport_contract() -> None:
    source = torch.zeros(256 * 1024, dtype=torch.uint8)
    config = _config("zstd")
    codec = create_tensor_codec(config.codec, level=config.level)

    wire, metadata, stats = compress_tensors({"zeros": source}, config, codec)
    restored = decompress_tensors(wire, metadata, codec)

    assert stats["wire_bytes"] < stats["raw_bytes"]
    assert torch.equal(restored["zeros"], source)


def test_pipeline_reuses_direct_packed_buffers_after_round_trip() -> None:
    tensors = {
        "main_images": torch.zeros((4, 64, 64, 3), dtype=torch.uint8),
        "wrist_images": torch.ones((4, 64, 64, 3), dtype=torch.uint8),
    }
    pipeline = CompressionPipeline(
        CompressionConfig(
            enabled=True,
            codec="lz4",
            min_bytes=1,
            block_bytes=4096,
            num_threads=2,
        )
    )

    first_wire, first_metadata, first_stats = pipeline.compress(tensors)
    first_pointers = {key: tensor.data_ptr() for key, tensor in first_wire.items()}
    restored = pipeline.decompress(first_wire, first_metadata)
    second_wire, second_metadata, second_stats = pipeline.compress(tensors)

    assert first_metadata == second_metadata
    assert first_stats["workspace_allocations"] == 2
    assert second_stats["workspace_allocations"] == 0
    assert first_pointers == {
        key: tensor.data_ptr() for key, tensor in second_wire.items()
    }
    for key, tensor in tensors.items():
        assert torch.equal(restored[key], tensor)
    pipeline.close()


def test_pipeline_parallel_raw_fallback_preserves_original_tensor() -> None:
    generator = torch.Generator().manual_seed(11)
    random = torch.randint(
        0, 256, (256 * 1024,), generator=generator, dtype=torch.uint8
    )
    zeros = torch.zeros(256 * 1024, dtype=torch.uint8)
    pipeline = CompressionPipeline(
        CompressionConfig(enabled=True, min_bytes=1, num_threads=2)
    )

    wire, metadata, stats = pipeline.compress({"random": random, "zeros": zeros})
    restored = pipeline.decompress(wire, metadata)

    assert wire["random"] is random
    assert stats["wire_bytes"] <= stats["raw_bytes"]
    assert torch.equal(restored["random"], random)
    assert torch.equal(restored["zeros"], zeros)
    pipeline.close()


def test_live_channel_does_not_reuse_actor_transfer_compression_pipeline() -> None:
    assert "compress_tensors" not in live.__dict__
    assert "decompress_tensors" not in live.__dict__
    assert "CompressionPipeline" not in live.__dict__
    assert "create_tensor_codec" in live.__dict__

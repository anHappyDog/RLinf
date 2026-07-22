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

import pytest
import torch

from rlinf.workers.trajectory.tensor_codec import (
    NvCompLZ4TensorCodec,
    create_tensor_codec,
)


@pytest.mark.parametrize("codec_name", ["lz4", "zstd"])
@pytest.mark.parametrize(
    "source",
    [
        torch.arange(1024, dtype=torch.float32).reshape(128, 8),
        torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.uint8),
        torch.empty(0, dtype=torch.uint8),
    ],
)
def test_tensor_codec_round_trip(codec_name: str, source: torch.Tensor) -> None:
    codec = create_tensor_codec(codec_name)
    compressed = torch.empty(codec.compress_bound(source.nbytes), dtype=torch.uint8)
    compressed_bytes = codec.compress_into(source, compressed)
    restored = torch.empty_like(source)

    codec.decompress_into(compressed, compressed_bytes, restored)

    assert torch.equal(restored, source)


@pytest.mark.parametrize("codec_name", ["lz4", "zstd"])
def test_tensor_codec_requires_contiguous_cpu_tensors(codec_name: str) -> None:
    codec = create_tensor_codec(codec_name)
    source = torch.zeros((4, 4), dtype=torch.uint8).T
    destination = torch.empty(codec.compress_bound(source.nbytes), dtype=torch.uint8)

    with pytest.raises(ValueError, match="source must be contiguous"):
        codec.compress_into(source, destination)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            lambda device: torch.arange(
                3 * 64 * 1024 + 17, dtype=torch.int32, device=device
            ).reshape(-1, 1),
            id="multi_chunk_int32",
        ),
        pytest.param(
            lambda device: torch.empty(0, dtype=torch.uint8, device=device),
            id="empty",
        ),
    ],
)
def test_nvcomp_lz4_tensor_codec_round_trip(source) -> None:
    pytest.importorskip("nvidia.libnvcomp")
    if not torch.cuda.is_available():
        pytest.skip("nvCOMP requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = source(device)
    codec = NvCompLZ4TensorCodec(chunk_bytes=64 * 1024)
    workspace = codec.allocate_workspace(tensor.nbytes, device)

    compressed = codec.compress_into(tensor, workspace)
    codec.check_statuses(workspace, "compression")
    restored = torch.empty_like(tensor)
    codec.decompress_into(compressed, restored, workspace)
    codec.check_statuses(workspace, "decompression")

    assert torch.equal(restored, tensor)


def test_nvcomp_lz4_tensor_codec_reuses_workspace() -> None:
    pytest.importorskip("nvidia.libnvcomp")
    if not torch.cuda.is_available():
        pytest.skip("nvCOMP requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.zeros(256 * 1024, dtype=torch.uint8, device=device)
    codec = NvCompLZ4TensorCodec()
    workspace = codec.allocate_workspace(source.nbytes, device)
    pointers = (workspace.output.data_ptr(), workspace.temp.data_ptr())

    for value in (0, 1):
        source.fill_(value)
        compressed = codec.compress_into(source, workspace)
        restored = torch.empty_like(source)
        codec.decompress_into(compressed, restored, workspace)
        codec.check_statuses(workspace, "round trip")
        assert torch.equal(restored, source)

    assert pointers == (workspace.output.data_ptr(), workspace.temp.data_ptr())


def test_nvcomp_lz4_tensor_codec_rejects_misaligned_source() -> None:
    pytest.importorskip("nvidia.libnvcomp")
    if not torch.cuda.is_available():
        pytest.skip("nvCOMP requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.empty(1025, dtype=torch.uint8, device=device)[1:]
    codec = NvCompLZ4TensorCodec()
    workspace = codec.allocate_workspace(source.nbytes, device)

    with pytest.raises(ValueError, match="aligned to 4 bytes"):
        codec.compress_into(source, workspace)

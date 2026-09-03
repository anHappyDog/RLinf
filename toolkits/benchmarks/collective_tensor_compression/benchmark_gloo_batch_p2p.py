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

"""Compare Gloo payload batching and small CPU tensor packing."""

import argparse
import json
import pickle
import statistics
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Case:
    """One representative collective payload."""

    name: str
    container: str
    specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]


PROFILE_SPECS = (
    ((16, 48), torch.bool),
    ((16, 1), torch.float32),
    ((16, 5, 7), torch.float32),
    ((16, 5, 7), torch.float32),
    ((16, 5, 50, 32), torch.float32),
    ((16, 8), torch.float32),
    ((16, 1600), torch.float32),
    ((16, 5, 7), torch.float64),
    ((16, 35), torch.float64),
    ((16, 4), torch.int64),
    ((16, 48), torch.int64),
    ((16, 256, 256, 3), torch.uint8),
    ((16, 256, 256, 3), torch.uint8),
)
SMALL_TENSOR_BYTES = 16 * 1024
MODES = ("broadcast", "batch_p2p", "packed_broadcast", "packed_batch_p2p")


def build_cases() -> tuple[Case, ...]:
    """Build payloads from the observed LIBERO Spatial CPU tensor profile."""
    small_specs = tuple(
        spec
        for spec in PROFILE_SPECS
        if torch.empty(spec[0], dtype=spec[1]).numel()
        * torch.empty((), dtype=spec[1]).element_size()
        <= SMALL_TENSOR_BYTES
    )
    return (
        Case("small_control_list", "list", small_specs),
        Case("dense_small_tensor_dict", "dict", small_specs * 8),
        Case("mixed_profile_tensor_list", "list", PROFILE_SPECS),
    )


def make_tensors(case: Case) -> list[torch.Tensor]:
    """Allocate deterministic tensors for a benchmark case."""
    tensors = []
    for index, (shape, dtype) in enumerate(case.specs):
        value = index % 2 if dtype == torch.bool else index % 251
        tensors.append(torch.full(shape, value, dtype=dtype))
    return tensors


def packed_indices(tensors: list[torch.Tensor]) -> tuple[int, ...]:
    """Return the tensors eligible for the small-payload buffer."""
    return tuple(
        index
        for index, tensor in enumerate(tensors)
        if tensor.numel() * tensor.element_size() <= SMALL_TENSOR_BYTES
    )


def make_metadata(case: Case, packed: bool) -> list[torch.Tensor]:
    """Create the metadata tensors sent by the current list/dict protocol."""
    object_type = torch.tensor([1 if case.container == "list" else 2], dtype=torch.int)
    metadata = {"specs": case.specs}
    if packed:
        metadata["packed_indices"] = tuple(
            index
            for index, (shape, dtype) in enumerate(case.specs)
            if torch.empty(shape, dtype=dtype).numel()
            * torch.empty((), dtype=dtype).element_size()
            <= SMALL_TENSOR_BYTES
        )
    tensor_metadata = pickle.dumps(metadata)
    payloads = [object_type]
    if case.container == "dict":
        keys = pickle.dumps([f"tensor_{index}" for index in range(len(case.specs))])
        payloads.extend(
            [
                torch.tensor([len(keys)], dtype=torch.long),
                torch.tensor(bytearray(keys), dtype=torch.uint8),
            ]
        )
    payloads.extend(
        [
            torch.tensor([len(tensor_metadata)], dtype=torch.long),
            torch.tensor(bytearray(tensor_metadata), dtype=torch.uint8),
        ]
    )
    return payloads


def transfer_broadcast(tensors: list[torch.Tensor]) -> None:
    """Transfer every payload tensor with a blocking broadcast."""
    for tensor in tensors:
        dist.broadcast(tensor, src=0)


def transfer_batch_p2p(tensors: list[torch.Tensor], rank: int) -> None:
    """Submit all payload tensors as P2P operations before waiting."""
    op = dist.isend if rank == 0 else dist.irecv
    peer = 1 if rank == 0 else 0
    works = dist.batch_isend_irecv(
        [dist.P2POp(op, tensor, peer=peer) for tensor in tensors]
    )
    for work in works:
        work.wait()


def tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Expose a contiguous tensor as a flat byte view."""
    return tensor.view(torch.uint8).reshape(-1)


def copy_packed_tensors(
    tensors: list[torch.Tensor], indices: tuple[int, ...], packed: torch.Tensor
) -> None:
    """Copy selected tensors into or out of one reusable byte buffer."""
    offset = 0
    for index in indices:
        source = tensor_bytes(tensors[index])
        packed[offset : offset + source.numel()].copy_(source)
        offset += source.numel()


def restore_packed_tensors(
    packed: torch.Tensor, tensors: list[torch.Tensor], indices: tuple[int, ...]
) -> None:
    """Restore packed bytes into the independently allocated result tensors."""
    offset = 0
    for index in indices:
        destination = tensor_bytes(tensors[index])
        destination.copy_(packed[offset : offset + destination.numel()])
        offset += destination.numel()


def timed_transfer(
    metadata: list[torch.Tensor],
    tensors: list[torch.Tensor],
    packed: torch.Tensor,
    indices: tuple[int, ...],
    rank: int,
    mode: str,
) -> float:
    """Measure when both ranks have completed one full protocol transfer."""
    dist.barrier()
    start = time.perf_counter()
    for tensor in metadata:
        dist.broadcast(tensor, src=0)
    if mode.startswith("packed_"):
        if rank == 0:
            copy_packed_tensors(tensors, indices, packed)
        index_set = set(indices)
        wire_tensors = [packed] + [
            tensor for index, tensor in enumerate(tensors) if index not in index_set
        ]
        if mode == "packed_broadcast":
            transfer_broadcast(wire_tensors)
        else:
            transfer_batch_p2p(wire_tensors, rank)
        if rank == 1:
            restore_packed_tensors(packed, tensors, indices)
    elif mode == "broadcast":
        transfer_broadcast(tensors)
    else:
        transfer_batch_p2p(tensors, rank)
    elapsed = torch.tensor([time.perf_counter() - start], dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def percentile(samples: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile from sorted samples."""
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def validate_tensors(
    expected: list[torch.Tensor], actual: list[torch.Tensor], rank: int
) -> None:
    """Verify transfer contents on both ranks outside the timed region."""
    valid = rank == 0 or all(
        torch.equal(expected_tensor, actual_tensor)
        for expected_tensor, actual_tensor in zip(expected, actual)
    )
    status = torch.tensor([valid], dtype=torch.uint8)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if not status.item():
        raise RuntimeError("Received tensors do not match the source payload.")


def run(args: argparse.Namespace) -> None:
    """Run all benchmark cases and print JSON results on rank zero."""
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=2,
        timeout=timedelta(minutes=5),
    )
    results = []
    for case in build_cases():
        source_tensors = make_tensors(case)
        tensors = (
            source_tensors
            if args.rank == 0
            else [torch.empty_like(tensor) for tensor in source_tensors]
        )
        indices = packed_indices(tensors)
        packed = torch.empty(
            sum(tensor_bytes(tensors[index]).numel() for index in indices),
            dtype=torch.uint8,
        )
        metadata_by_mode = {}
        for mode in MODES:
            metadata = make_metadata(case, mode.startswith("packed_"))
            metadata_by_mode[mode] = (
                metadata
                if args.rank == 0
                else [torch.empty_like(tensor) for tensor in metadata]
            )

        for mode in MODES:
            for _ in range(args.warmup):
                timed_transfer(
                    metadata_by_mode[mode], tensors, packed, indices, args.rank, mode
                )
        validate_tensors(source_tensors, tensors, args.rank)

        samples = {mode: [] for mode in MODES}
        for iteration in range(args.repeats):
            modes = MODES if iteration % 2 == 0 else tuple(reversed(MODES))
            for mode in modes:
                samples[mode].append(
                    timed_transfer(
                        metadata_by_mode[mode],
                        tensors,
                        packed,
                        indices,
                        args.rank,
                        mode,
                    )
                )

        if args.rank == 0:
            broadcast_median = statistics.median(samples["broadcast"])
            results.append(
                {
                    "case": case.name,
                    "container": case.container,
                    "tensor_count": len(tensors),
                    "payload_bytes": sum(
                        tensor.numel() * tensor.element_size() for tensor in tensors
                    ),
                    "packed_tensor_count": len(indices),
                    "packed_bytes": packed.numel(),
                    "wire_operations": {
                        "broadcast": len(tensors),
                        "batch_p2p": len(tensors),
                        "packed_broadcast": len(tensors) - len(indices) + 1,
                        "packed_batch_p2p": len(tensors) - len(indices) + 1,
                    },
                    "modes": {
                        mode: {
                            "median_ms": statistics.median(samples[mode]) * 1000,
                            "p95_ms": percentile(samples[mode], 0.95) * 1000,
                            "improvement_percent": (
                                broadcast_median - statistics.median(samples[mode])
                            )
                            / broadcast_median
                            * 100,
                        }
                        for mode in MODES
                    },
                }
            )

    if args.rank == 0:
        print(json.dumps(results, indent=2))
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29671)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

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

"""Benchmark batched CPU P2P through RLinf's complete Worker send/recv path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker, WorkerAddress
from rlinf.scheduler.collective.collective_group import CollectiveGroup
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup

SENDER_GROUP = "worker_p2p_benchmark_sender"
RECEIVER_GROUP = "worker_p2p_benchmark_receiver"
MODES = ("broadcast", "batch_p2p")

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


@dataclass(frozen=True)
class Case:
    """One representative RLinf Worker payload."""

    name: str
    container: str
    specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]


def tensor_num_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Return the storage size of one tensor specification."""
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def build_cases() -> tuple[Case, ...]:
    """Build the same three payloads as the isolated Gloo benchmark."""
    small_specs = tuple(
        spec for spec in PROFILE_SPECS if tensor_num_bytes(*spec) <= SMALL_TENSOR_BYTES
    )
    return (
        Case("small_control_list", "list", small_specs),
        Case("dense_small_tensor_dict", "dict", small_specs * 8),
        Case("mixed_profile_tensor_list", "list", PROFILE_SPECS),
    )


def make_payload(case: Case) -> list[torch.Tensor] | dict[str, torch.Tensor]:
    """Allocate a deterministic payload whose contents can be verified."""
    tensors = []
    for index, (shape, dtype) in enumerate(case.specs):
        value = index % 2 if dtype == torch.bool else index % 251
        tensors.append(torch.full(shape, value, dtype=dtype))
    if case.container == "list":
        return tensors
    return {f"tensor_{index}": tensor for index, tensor in enumerate(tensors)}


def validate_payload(
    payload: list[torch.Tensor] | dict[str, torch.Tensor], case: Case
) -> None:
    """Validate one payload received through the Worker API."""
    tensors = list(payload.values()) if isinstance(payload, dict) else payload
    if len(tensors) != len(case.specs):
        raise RuntimeError(
            f"{case.name} expected {len(case.specs)} tensors, got {len(tensors)}."
        )
    for index, tensor in enumerate(tensors):
        value = index % 2 if tensor.dtype == torch.bool else index % 251
        if not torch.all(tensor == value).item():
            raise RuntimeError(f"{case.name} tensor {index} failed validation.")


class _BenchmarkWorker(Worker):
    """Common benchmark behavior for one side of a Worker pair."""

    @staticmethod
    def _set_sender_mode(group: CollectiveGroup, mode: str) -> None:
        mc_group = group._mc_group
        if mode == "batch_p2p":
            mc_group.send_many = MethodType(
                MultiChannelProcessGroup.send_many, mc_group
            )
            return

        def send_many(
            process_group: MultiChannelProcessGroup,
            tensors: list[torch.Tensor],
            channel_id: int,
        ) -> None:
            for tensor in tensors:
                process_group.send(
                    tensor,
                    device=CollectiveGroup.CPU,
                    channel_id=channel_id,
                )

        mc_group.send_many = MethodType(send_many, mc_group)

    @staticmethod
    def _set_receiver_mode(group: CollectiveGroup, mode: str) -> None:
        mc_group = group._mc_group
        if mode == "batch_p2p":
            mc_group.recv_many = MethodType(
                MultiChannelProcessGroup.recv_many, mc_group
            )
            return

        def recv_many(
            process_group: MultiChannelProcessGroup,
            tensors: list[torch.Tensor],
            channel_id: int,
        ) -> None:
            for tensor in tensors:
                process_group.recv(
                    tensor,
                    device=CollectiveGroup.CPU,
                    channel_id=channel_id,
                )

        mc_group.recv_many = MethodType(recv_many, mc_group)

    @staticmethod
    def _record(
        results: dict[str, dict[str, list[float]]],
        case: Case,
        mode: str,
        elapsed: float,
    ) -> None:
        results.setdefault(case.name, {}).setdefault(mode, []).append(elapsed)


class SenderWorker(_BenchmarkWorker):
    """Send representative payloads from node zero."""

    def run(self, warmup: int, repeats: int) -> dict[str, dict[str, list[float]]]:
        """Run every case and transport mode through ``Worker.send``."""
        peer = WorkerAddress(RECEIVER_GROUP, ranks=0)
        group = self._get_collective_group(peer)
        group._init_process_group()
        results: dict[str, dict[str, list[float]]] = {}
        for case in build_cases():
            payload = make_payload(case)
            for mode in MODES:
                self._set_sender_mode(group, mode)
                for _ in range(warmup):
                    self.send(payload, RECEIVER_GROUP, 0, async_op=True).wait()

            for iteration in range(repeats):
                modes = MODES if iteration % 2 == 0 else tuple(reversed(MODES))
                for mode in modes:
                    self._set_sender_mode(group, mode)
                    start = time.perf_counter()
                    self.send(payload, RECEIVER_GROUP, 0, async_op=True).wait()
                    self._record(results, case, mode, time.perf_counter() - start)
        return results


class ReceiverWorker(_BenchmarkWorker):
    """Receive and validate payloads on node one."""

    def run(self, warmup: int, repeats: int) -> dict[str, dict[str, list[float]]]:
        """Run every case and transport mode through ``Worker.recv``."""
        peer = WorkerAddress(SENDER_GROUP, ranks=0)
        group = self._get_collective_group(peer)
        group._init_process_group()
        results: dict[str, dict[str, list[float]]] = {}
        for case in build_cases():
            for mode in MODES:
                self._set_receiver_mode(group, mode)
                for _ in range(warmup):
                    payload = self.recv(SENDER_GROUP, 0, async_op=True).wait()
                    validate_payload(payload, case)

            for iteration in range(repeats):
                modes = MODES if iteration % 2 == 0 else tuple(reversed(MODES))
                for mode in modes:
                    self._set_receiver_mode(group, mode)
                    start = time.perf_counter()
                    payload = self.recv(SENDER_GROUP, 0, async_op=True).wait()
                    elapsed = time.perf_counter() - start
                    validate_payload(payload, case)
                    self._record(results, case, mode, elapsed)
        return results


def percentile(samples: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(
    sender: dict[str, dict[str, list[float]]],
    receiver: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """Combine sender and receiver completion times for every iteration."""
    summaries = []
    cases = {case.name: case for case in build_cases()}
    for case_name, case in cases.items():
        samples = {
            mode: [
                max(send_time, recv_time)
                for send_time, recv_time in zip(
                    sender[case_name][mode], receiver[case_name][mode]
                )
            ]
            for mode in MODES
        }
        baseline = statistics.median(samples["broadcast"])
        summaries.append(
            {
                "case": case_name,
                "container": case.container,
                "tensor_count": len(case.specs),
                "payload_bytes": sum(tensor_num_bytes(*spec) for spec in case.specs),
                "modes": {
                    mode: {
                        "median_ms": statistics.median(samples[mode]) * 1000,
                        "p95_ms": percentile(samples[mode], 0.95) * 1000,
                        "improvement_percent": (
                            baseline - statistics.median(samples[mode])
                        )
                        / baseline
                        * 100,
                    }
                    for mode in MODES
                },
            }
        )
    return summaries


def run(warmup: int, repeats: int) -> list[dict[str, Any]]:
    """Launch one Worker per node and run the benchmark."""
    cluster = Cluster(num_nodes=0)
    if cluster.num_nodes < 2:
        raise RuntimeError("This benchmark requires a two-node Ray cluster.")

    sender_group = SenderWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=NodePlacementStrategy([0]),
        name=SENDER_GROUP,
    )
    receiver_group = ReceiverWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=NodePlacementStrategy([1]),
        name=RECEIVER_GROUP,
    )
    try:
        receiver_result = receiver_group.run(warmup, repeats)
        sender_result = sender_group.run(warmup, repeats)
        receiver = receiver_result.wait()[0]
        sender = sender_result.wait()[0]
        return summarize(sender, receiver)
    finally:
        sender_group._close()
        receiver_group._close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encoded = json.dumps(run(args.warmup, args.repeats), indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)

#!/usr/bin/env python3
"""Benchmark current trajectory compression, isolation, and cross-host transfer."""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rlinf.workers.trajectory.compression import (
    CompressionConfig,
    CompressionPipeline,
)


@dataclass(frozen=True)
class WorkloadConfig:
    """Tensor dimensions for one Storage shard."""

    profile: str
    slots: int
    chunk_steps: int
    image_height: int
    image_width: int
    image_fields: int
    action_chunks: int = 5
    action_dim: int = 7
    denoise_steps: int = 4
    action_horizon: int = 50
    openpi_action_dim: int = 32
    token_length: int = 48


@dataclass(frozen=True)
class Distribution:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    max_ms: float


def workload_config(profile: str) -> WorkloadConfig:
    """Return the smoke or target OpenPI/LIBERO shard shape."""
    if profile == "target":
        return WorkloadConfig("target", 8, 48, 256, 256, 4)
    if profile == "smoke":
        return WorkloadConfig("smoke", 2, 3, 64, 64, 4)
    raise ValueError("profile must be 'smoke' or 'target'")


def make_workload(config: WorkloadConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build the large tensor leaves present in one Actor-bound shard.

    Images use spatially coherent random 8x8 tiles. This is deterministic and
    compressible without pretending to reproduce an exact camera distribution.
    """
    generator = torch.Generator().manual_seed(seed)
    time_batch = config.chunk_steps * config.slots
    low_height = max(1, config.image_height // 8)
    low_width = max(1, config.image_width // 8)

    def image() -> torch.Tensor:
        low_resolution = torch.randint(
            0,
            256,
            (config.chunk_steps, config.slots, low_height, low_width, 3),
            dtype=torch.uint8,
            generator=generator,
        )
        return (
            low_resolution.repeat_interleave(8, dim=2)
            .repeat_interleave(8, dim=3)[
                :, :, : config.image_height, : config.image_width
            ]
            .contiguous()
        )

    tensors = {f"image_{index}": image() for index in range(config.image_fields)}
    tensors.update(
        chains=torch.randn(
            (
                time_batch,
                config.denoise_steps + 1,
                config.action_horizon,
                config.openpi_action_dim,
            ),
            generator=generator,
        ),
        denoise_inds=torch.zeros((time_batch, config.denoise_steps), dtype=torch.int64),
        tokenized_prompt=torch.randint(
            0,
            32000,
            (time_batch, config.token_length),
            dtype=torch.int64,
            generator=generator,
        ),
        tokenized_prompt_mask=torch.ones(
            (time_batch, config.token_length), dtype=torch.bool
        ),
        actions=torch.randn(
            (
                config.chunk_steps,
                config.slots,
                config.action_chunks,
                config.action_dim,
            ),
            generator=generator,
        ),
        state_values=torch.randn(
            (config.chunk_steps, config.slots, 1), generator=generator
        ),
    )
    return tensors


def tensor_schema(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Return the manifest representation of a workload."""
    return {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.nbytes,
        }
        for key, tensor in tensors.items()
    }


def total_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Return tensor payload bytes without Python metadata."""
    return sum(tensor.nbytes for tensor in tensors.values())


def distribution(samples_s: list[float]) -> Distribution:
    """Summarize duration samples with interpolated percentiles."""
    if not samples_s:
        raise ValueError("at least one timing sample is required")
    values = sorted(sample * 1000 for sample in samples_s)

    def percentile(value: float) -> float:
        position = (len(values) - 1) * value
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction

    return Distribution(
        count=len(values),
        mean_ms=statistics.fmean(values),
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        p999_ms=percentile(0.999),
        max_ms=values[-1],
    )


class PeakRss:
    """Poll process RSS while one benchmark operation is active."""

    def __init__(self) -> None:
        self.peak_bytes = _rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "PeakRss":
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, _rss_bytes())

    def _sample(self) -> None:
        while not self._stop.wait(0.005):
            self.peak_bytes = max(self.peak_bytes, _rss_bytes())


def _rss_bytes() -> int:
    with open("/proc/self/statm") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _compression_config(codec_name: str, args: argparse.Namespace) -> CompressionConfig:
    return CompressionConfig(
        enabled=True,
        codec=codec_name,
        level=args.level,
        min_bytes=args.min_bytes,
        block_bytes=args.block_bytes,
        num_threads=args.compression_threads,
    )


def _raw_round_trip(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.clone() for key, tensor in tensors.items()}


def _compressed_round_trip(
    tensors: dict[str, torch.Tensor],
    pipeline: CompressionPipeline,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    wire, metadata, stats = pipeline.compress(tensors)
    return pipeline.decompress(wire, metadata), stats


def benchmark_local(args: argparse.Namespace) -> dict[str, Any]:
    """Run local raw/codec and event-loop isolation measurements."""
    config = workload_config(args.profile)
    tensors = make_workload(config, args.seed)
    raw_bytes = total_bytes(tensors)
    codec_results = []
    for mode in args.codecs.split(","):
        mode = mode.strip()
        samples = []
        cpu_samples = []
        rss_before = _rss_bytes()
        peak_rss = rss_before
        wire_bytes = raw_bytes
        compression = None if mode == "raw" else _compression_config(mode, args)
        pipeline = None if compression is None else CompressionPipeline(compression)
        for index in range(args.warmup + args.repeats):
            cpu_started = time.process_time()
            started = time.perf_counter()
            with PeakRss() as memory:
                if mode == "raw":
                    restored = _raw_round_trip(tensors)
                    stats = {"wire_bytes": raw_bytes}
                else:
                    assert pipeline is not None
                    restored, stats = _compressed_round_trip(tensors, pipeline)
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.process_time() - cpu_started
            if index == 0:
                _assert_equal(tensors, restored)
            if index >= args.warmup:
                samples.append(elapsed)
                cpu_samples.append(cpu_elapsed)
                peak_rss = max(peak_rss, memory.peak_bytes)
                wire_bytes = stats["wire_bytes"]
            del restored
        if pipeline is not None:
            pipeline.close()
        codec_results.append(
            {
                "mode": mode,
                "latency": asdict(distribution(samples)),
                "cpu_time": asdict(distribution(cpu_samples)),
                "peak_rss_bytes": peak_rss,
                "peak_rss_delta_bytes": max(0, peak_rss - rss_before),
                "raw_bytes": raw_bytes,
                "wire_bytes": wire_bytes,
                "compression_ratio": raw_bytes / wire_bytes,
                "throughput_raw_gib_s": (
                    raw_bytes / (1024**3) / statistics.median(samples)
                ),
            }
        )

    isolation = asyncio.run(_benchmark_isolation(args, config))
    return {
        "kind": "local",
        "host": platform.node(),
        "environment": environment(),
        "workload": asdict(config),
        "tensor_schema": tensor_schema(tensors),
        "raw_bytes": raw_bytes,
        "settings": _settings(args),
        "codec_results": codec_results,
        "isolation_results": isolation,
    }


async def _benchmark_isolation(
    args: argparse.Namespace,
    config: WorkloadConfig,
) -> list[dict[str, Any]]:
    results = []
    tensors = make_workload(config, args.seed)
    for mode in args.isolation_modes.split(","):
        mode = mode.strip()
        process = None
        ready_queue = None
        start_event = None
        result_queue = None
        if mode == "process":
            context = mp.get_context("spawn")
            ready_queue = context.Queue(1)
            result_queue = context.Queue(1)
            start_event = context.Event()
            process = context.Process(
                target=_process_pressure,
                args=(config, vars(args), ready_queue, start_event, result_queue),
            )
            process.start()
            await asyncio.to_thread(ready_queue.get)
        probe = asyncio.create_task(
            _probe_loop(args.isolation_seconds, args.probe_interval_ms / 1000)
        )
        await asyncio.sleep(0)
        started = time.perf_counter()
        if process is not None:
            assert start_event is not None and result_queue is not None
            start_event.set()
            operations = await asyncio.to_thread(result_queue.get)
            await asyncio.to_thread(process.join)
            if process.exitcode != 0:
                raise RuntimeError(f"pressure process exited with {process.exitcode}")
        else:
            operations = await _run_pressure(mode, args, tensors)
        pressure_seconds = time.perf_counter() - started
        samples = await probe
        results.append(
            {
                "mode": mode,
                "probe_lag": asdict(distribution(samples)),
                "pressure_seconds": pressure_seconds,
                "operations": operations,
            }
        )
    return results


async def _probe_loop(duration_s: float, interval_s: float) -> list[float]:
    loop = asyncio.get_running_loop()
    start = loop.time()
    deadline = start + interval_s
    samples = []
    while deadline <= start + duration_s:
        await asyncio.sleep(max(0.0, deadline - loop.time()))
        samples.append(max(0.0, loop.time() - deadline))
        deadline += interval_s
    return samples


async def _run_pressure(
    mode: str,
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
) -> int:
    if mode == "baseline":
        await asyncio.sleep(args.isolation_seconds)
        return 0
    compression = _compression_config("lz4", args)
    pipeline = CompressionPipeline(compression)

    def operation() -> None:
        restored, _stats = _compressed_round_trip(tensors, pipeline)
        del restored

    deadline = time.monotonic() + args.isolation_seconds
    operations = 0
    if mode == "inline":
        while time.monotonic() < deadline:
            operation()
            operations += 1
        return operations
    if mode == "async":
        while time.monotonic() < deadline:
            operation()
            operations += 1
            await asyncio.sleep(0)
        return operations
    if mode == "thread":
        while time.monotonic() < deadline:
            await asyncio.to_thread(operation)
            operations += 1
        return operations
    raise ValueError(f"unknown isolation mode {mode!r}")


def _process_pressure(
    config: WorkloadConfig,
    argument_values: dict[str, Any],
    ready_queue: mp.Queue,
    start_event: mp.Event,
    result_queue: mp.Queue,
) -> None:
    args = argparse.Namespace(**argument_values)
    tensors = make_workload(config, args.seed)
    compression = _compression_config("lz4", args)
    pipeline = CompressionPipeline(compression)
    ready_queue.put(True)
    start_event.wait()
    deadline = time.monotonic() + args.isolation_seconds
    operations = 0
    while time.monotonic() < deadline:
        restored, _stats = _compressed_round_trip(tensors, pipeline)
        del restored
        operations += 1
    result_queue.put(operations)


def benchmark_distributed(args: argparse.Namespace) -> dict[str, Any] | None:
    """Run rank-paired Gloo transfers with the production tensor/block format."""
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=2,
    )
    try:
        config = workload_config(args.profile)
        tensors = make_workload(config, args.seed)
        results = []
        for mode in args.codecs.split(","):
            results.append(_distributed_mode(mode.strip(), tensors, args))
        environments: list[dict[str, Any] | None] = [None, None]
        dist.all_gather_object(environments, environment())
        if args.rank != 0:
            return None
        return {
            "kind": "cross_host",
            "host": platform.node(),
            "peer": args.peer_name,
            "environments": environments,
            "workload": asdict(config),
            "tensor_schema": tensor_schema(tensors),
            "raw_bytes": total_bytes(tensors),
            "settings": _settings(args),
            "results": results,
        }
    finally:
        dist.destroy_process_group()


def _distributed_mode(
    mode: str,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    original_keys = tuple(tensors)
    compression = None if mode == "raw" else _compression_config(mode, args)
    pipeline = None if compression is None else CompressionPipeline(compression)
    if args.rank == 0:
        if compression is None:
            initial_wire, metadata = tensors, {}
        else:
            assert pipeline is not None
            initial_wire, metadata, _stats = pipeline.compress(tensors)
        transfer = [tuple(initial_wire), metadata]
    else:
        transfer = [None, None]
    dist.broadcast_object_list(transfer, src=0)
    wire_keys, metadata = transfer
    wire_keys = tuple(wire_keys)
    block_keys = {tensor_metadata["wire_key"] for tensor_metadata in metadata.values()}
    samples = []
    cpu_samples = []
    wire_bytes = total_bytes(tensors)
    rss_before = _rss_bytes()
    peak_rss = rss_before
    iterations = args.warmup + args.repeats
    for iteration in range(iterations):
        dist.barrier()
        cpu_started = time.process_time()
        started = time.perf_counter()
        if args.rank == 0:
            if compression is None:
                wire = tensors
                stats = {"wire_bytes": total_bytes(tensors)}
            else:
                assert pipeline is not None
                wire, metadata_now, stats = pipeline.compress(tensors)
                if metadata_now != metadata or tuple(wire) != wire_keys:
                    raise RuntimeError("compression layout changed between iterations")
            sizes = torch.tensor(
                [wire[key].numel() for key in wire_keys], dtype=torch.int64
            )
            dist.send(sizes, dst=1)
            for key in wire_keys:
                dist.send(wire[key], dst=1)
            acknowledgement = torch.empty(1, dtype=torch.int64)
            dist.recv(acknowledgement, src=1)
            if acknowledgement.item() != iteration:
                raise RuntimeError("cross-host acknowledgement mismatch")
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.process_time() - cpu_started
            if iteration >= args.warmup:
                samples.append(elapsed)
                cpu_samples.append(cpu_elapsed)
                wire_bytes = stats["wire_bytes"]
                peak_rss = max(peak_rss, _rss_bytes())
        else:
            sizes = torch.empty(len(wire_keys), dtype=torch.int64)
            dist.recv(sizes, src=0)
            received = {}
            for key, size in zip(wire_keys, sizes.tolist(), strict=True):
                if key in original_keys and key not in metadata:
                    target = torch.empty_like(tensors[key])
                    if target.numel() != size:
                        raise RuntimeError("raw tensor size changed")
                elif key in block_keys:
                    target = torch.empty(size, dtype=torch.uint8)
                else:
                    raise RuntimeError(f"unknown wire tensor {key!r}")
                dist.recv(target, src=0)
                received[key] = target
            if compression is not None:
                assert pipeline is not None
                restored = pipeline.decompress(received, metadata)
            else:
                restored = received
            if iteration == 0:
                _assert_equal(tensors, restored)
            dist.send(torch.tensor([iteration], dtype=torch.int64), dst=0)
            if iteration >= args.warmup:
                cpu_samples.append(time.process_time() - cpu_started)
                peak_rss = max(peak_rss, _rss_bytes())
    if args.rank == 1:
        rank1_cpu = distribution(cpu_samples)
        dist.send(
            torch.tensor(
                [
                    rank1_cpu.mean_ms,
                    rank1_cpu.p95_ms,
                    float(peak_rss),
                    float(max(0, peak_rss - rss_before)),
                ],
                dtype=torch.float64,
            ),
            dst=0,
        )
        return None
    rank1_stats = torch.empty(4, dtype=torch.float64)
    dist.recv(rank1_stats, src=1)
    if args.rank != 0:
        return None
    raw_bytes = total_bytes(tensors)
    return {
        "mode": mode,
        "latency": asdict(distribution(samples)),
        "cpu_time_rank0": asdict(distribution(cpu_samples)),
        "peak_rss_rank0_bytes": peak_rss,
        "peak_rss_rank0_delta_bytes": max(0, peak_rss - rss_before),
        "rank1_cpu_mean_ms": rank1_stats[0].item(),
        "rank1_cpu_p95_ms": rank1_stats[1].item(),
        "peak_rss_rank1_bytes": int(rank1_stats[2].item()),
        "peak_rss_rank1_delta_bytes": int(rank1_stats[3].item()),
        "raw_bytes": raw_bytes,
        "wire_bytes": wire_bytes,
        "compression_ratio": raw_bytes / wire_bytes,
        "throughput_raw_gib_s": raw_bytes / (1024**3) / statistics.median(samples),
    }


def _assert_equal(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> None:
    if expected.keys() != actual.keys():
        raise RuntimeError("round-trip tensor keys changed")
    for key in expected:
        if not torch.equal(expected[key], actual[key]):
            raise RuntimeError(f"round-trip tensor {key!r} changed")


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "profile",
        "seed",
        "warmup",
        "repeats",
        "codecs",
        "level",
        "min_bytes",
        "block_bytes",
        "compression_threads",
        "isolation_modes",
        "isolation_seconds",
        "probe_interval_ms",
    )
    return {name: getattr(args, name) for name in names}


def environment() -> dict[str, Any]:
    """Return portable runtime facts required by the artifact contract."""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "network_interfaces": {
            interface.name: {
                "operstate": _read_optional(interface / "operstate"),
                "speed_mbit": _read_optional(interface / "speed"),
            }
            for interface in Path("/sys/class/net").iterdir()
        },
    }


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "distributed"), default="local")
    parser.add_argument("--profile", choices=("smoke", "target"), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codecs", default="raw,lz4,zstd")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--min-bytes", type=int, default=65536)
    parser.add_argument("--block-bytes", type=int, default=1048576)
    parser.add_argument("--compression-threads", type=int, default=1)
    parser.add_argument(
        "--isolation-modes", default="baseline,inline,async,thread,process"
    )
    parser.add_argument("--isolation-seconds", type=float, default=2.0)
    parser.add_argument("--probe-interval-ms", type=float, default=2.0)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int, default=29615)
    parser.add_argument("--peer-name", default="unknown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats positive")
    if min(args.min_bytes, args.block_bytes) < 1:
        parser.error("min-bytes and block-bytes must be positive")
    if args.compression_threads < 1:
        parser.error("compression-threads must be positive")
    if args.mode == "distributed" and (args.rank is None or args.master_addr is None):
        parser.error("distributed mode requires --rank and --master-addr")
    return args


def main() -> None:
    args = _parse_args()
    report = (
        benchmark_local(args) if args.mode == "local" else benchmark_distributed(args)
    )
    if report is None:
        return
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

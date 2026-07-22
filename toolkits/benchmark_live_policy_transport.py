#!/usr/bin/env python3
"""Compare ChannelWorker and direct fixed-frame PolicyInput transport."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from rlinf.data.trajectory import PolicyInput
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.workers.trajectory import (
    ChannelConfig,
    PolicyInputLayout,
    RoutePlan,
    TrajectoryChannel,
    TrajectoryChannelWorker,
    WorkerLayout,
)


class PolicySender(Worker):
    """Publish a persistent sequence and report call latency."""

    def run(self, channel, requests, warmup: int):
        samples = []
        for index, request in enumerate(requests):
            started = time.perf_counter()
            channel.publish_policy_input(request)
            elapsed = time.perf_counter() - started
            if index >= warmup:
                samples.append(elapsed)
        return samples


class PolicyReceiver(Worker):
    """Consume a persistent sequence and report call latency."""

    def run(self, channel, requests, warmup: int):
        samples = []
        for index, expected in enumerate(requests):
            started = time.perf_counter()
            actual = channel.take_policy_input()
            elapsed = time.perf_counter() - started
            if index in (0, len(requests) - 1):
                _assert_equal(expected, actual)
            if index >= warmup:
                samples.append(elapsed)
        return samples


def _request(chunk_step: int, batch_size: int, image_size: int) -> PolicyInput:
    row = torch.arange(image_size, dtype=torch.uint8).view(1, image_size, 1)
    column = torch.arange(image_size, dtype=torch.uint8).view(image_size, 1, 1)
    image = (row + column + chunk_step).expand(image_size, image_size, 3)
    main = image.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous()
    wrist = (main + 17).contiguous()
    return PolicyInput(
        global_step=0,
        rollout_epoch=0,
        chunk_step=chunk_step,
        slot_ids=tuple(range(batch_size)),
        observations={
            "main_images": main,
            "wrist_images": wrist,
            "extra_view_images": None,
            "states": torch.arange(batch_size * 8, dtype=torch.float32).reshape(
                batch_size, 8
            ),
            "task_descriptions": [
                f"LIBERO task {index}" for index in range(batch_size)
            ],
        },
        rlt_switch_flags=torch.zeros(batch_size, dtype=torch.bool),
    )


def _assert_equal(expected: PolicyInput, actual: PolicyInput) -> None:
    if expected.slot_ids != actual.slot_ids:
        raise RuntimeError("PolicyInput slot IDs changed")
    if (
        expected.observations["task_descriptions"]
        != actual.observations["task_descriptions"]
    ):
        raise RuntimeError("PolicyInput descriptions changed")
    for key in ("main_images", "wrist_images", "states"):
        if not torch.equal(expected.observations[key], actual.observations[key]):
            raise RuntimeError(f"PolicyInput {key} changed")


def _distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(value * 1000 for value in samples)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def _run_mode(
    cluster,
    placement,
    mode: str,
    requests: tuple[PolicyInput, ...],
    warmup: int,
) -> dict:
    channel_group = TrajectoryChannelWorker.create_group(maxsize=8).launch(
        cluster,
        placement,
        name=f"live_benchmark_channel_{mode}",
        max_concurrency=8,
        catch_system_failure=False,
    )
    sender_group = PolicySender.create_group().launch(
        cluster,
        placement,
        name=f"live_benchmark_sender_{mode}",
        catch_system_failure=False,
    )
    receiver_group = PolicyReceiver.create_group().launch(
        cluster,
        placement,
        name=f"live_benchmark_receiver_{mode}",
        catch_system_failure=False,
    )
    worker_layout = WorkerLayout((0,))
    batch_size = requests[0].batch_size
    image_shape = tuple(requests[0].observations["main_images"].shape[1:])
    config = ChannelConfig(
        layout=worker_layout,
        route_plan=RoutePlan(batch_size, {"env": 1, "rollout": 1}),
        env_layout=worker_layout,
        rollout_layout=worker_layout,
        env_group_name=sender_group.worker_group_name,
        rollout_group_name=receiver_group.worker_group_name,
        policy_input_layout=(
            PolicyInputLayout(
                batch_size=batch_size,
                image_shape=image_shape,
                state_shape=(8,),
                max_description_bytes=64,
                compress_images=mode == "direct_lz4",
                pin_memory=mode == "direct_pinned",
            )
            if mode != "channel"
            else None
        ),
    )
    channel_group.configure(config).wait()
    channel = TrajectoryChannel.from_worker_group(channel_group, config)
    pending_receive = receiver_group.run(channel, requests, warmup)
    pending_send = sender_group.run(channel, requests, warmup)
    receiver_samples = pending_receive.wait()[0]
    sender_samples = pending_send.wait()[0]

    channel_group.shutdown().wait()
    channel_group._close()
    sender_group._close()
    receiver_group._close()
    return {
        "mode": mode,
        "sender": _distribution(sender_samples),
        "receiver": _distribution(receiver_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.image_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("sizes/repeats must be positive and warmup non-negative")

    requests = tuple(
        _request(index, args.batch_size, args.image_size)
        for index in range(args.warmup + args.repeats)
    )
    cluster = Cluster(num_nodes=1)
    placement = NodePlacementStrategy([0])
    results = tuple(
        _run_mode(cluster, placement, mode, requests, args.warmup)
        for mode in ("channel", "direct", "direct_pinned", "direct_lz4")
    )
    raw_bytes = sum(
        value.nbytes
        for value in requests[0].observations.values()
        if isinstance(value, torch.Tensor)
    )
    report = {
        "host": platform.node(),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "raw_tensor_bytes": raw_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

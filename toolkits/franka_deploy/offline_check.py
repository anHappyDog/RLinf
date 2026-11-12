# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check real checkpoint inference and RPC using saved observations, without SDKs."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .canonical_tick_server import CanonicalTickPolicy
from .molmobot_f2_server import serve
from .pi05_policy import Pi05QtargetPolicy


async def check(args):
    import msgpack_numpy
    from websockets.asyncio.client import connect

    records = [
        json.loads(line) for line in (args.run / "ticks.jsonl").read_text().splitlines()
    ]
    first = next(record for record in records if record["type"] == "tick")
    request = {
        "qpos": np.array(first["qpos_rad"], dtype=np.float32),
        "gripper_width_m": first["gripper_width_m"],
        "prompt": "Pick up the orange cube",
        "images": {
            role: np.array(
                Image.open(
                    args.run / "observations" / f"tick_000001_{role}.jpg"
                ).convert("RGB")
            )
            for role in ("external", "wrist")
        },
    }
    policy = Pi05QtargetPolicy(args.checkpoint, action_dt=0.066)
    prepared = policy.prepare_observation(request)
    narrow = policy.prepare_observation({**request, "gripper_width_m": 0.01})
    assert prepared["state"].shape == (32,)
    assert prepared["tokenized_prompt"].shape == (200,)
    assert not np.array_equal(prepared["tokenized_prompt"], narrow["tokenized_prompt"])
    wrapper = CanonicalTickPolicy(policy, 8)
    task = asyncio.create_task(serve(wrapper, "127.0.0.1", args.port))
    await asyncio.sleep(0.1)
    results = []
    try:
        async with connect(
            f"ws://127.0.0.1:{args.port}", max_size=32 * 1024 * 1024, compression=None
        ) as ws:
            for tick in range(1, 10):
                start = time.monotonic()
                await ws.send(msgpack_numpy.packb(request))
                reply = msgpack_numpy.unpackb(await ws.recv())
                assert reply["qtarget"].shape == (1, 7)
                assert np.isfinite(reply["qtarget"]).all()
                assert reply["metadata"]["episode_step"] == tick
                assert reply["metadata"]["execute_horizon"] == 8
                assert reply["metadata"]["gripper_feedback_used"] is True
                results.append(
                    {
                        "tick": tick,
                        "rpc_s": time.monotonic() - start,
                        "qtarget": reply["qtarget"].tolist(),
                        "closedness": reply["gripper_closedness"].tolist(),
                    }
                )
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    args.output.write_text(
        json.dumps(
            {
                "hardware_connected": False,
                "checkpoint": str(args.checkpoint),
                "metadata": wrapper.metadata,
                "state_shape": list(prepared["state"].shape),
                "token_shape": list(prepared["tokenized_prompt"].shape),
                "width_changes_tokens": True,
                "calls": results,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()

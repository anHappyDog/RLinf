# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Serve canonical qtarget checkpoints as H16 actions dispatched per tick."""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .molmobot_f2_server import serve

if TYPE_CHECKING:
    from .model_adapters import CanonicalQtargetPolicy


class CanonicalTickPolicy:
    """Execute the configured prefix before replanning from a fresh request."""

    def __init__(
        self, policy: "CanonicalQtargetPolicy", execute_horizon: int = 16
    ) -> None:
        if policy.horizon != 16:
            raise ValueError("Expected a canonical action horizon of sixteen.")
        if execute_horizon not in (8, 16):
            raise ValueError("Canonical execute_horizon must be eight or sixteen.")
        self.policy = policy
        self.execute_horizon = execute_horizon
        self.metadata = {
            **policy.metadata,
            "adapter": f"{policy.metadata['adapter']}_per_tick",
            "horizon": 1,
            "model_horizon": 16,
            "execute_horizon": execute_horizon,
            "n_obs_steps": 1,
            "observation_protocol": f"replan from the current measured observation every {execute_horizon} ticks",
        }
        self.reset()

    def reset(self) -> None:
        """Discard buffered actions and reset the episode counter and task."""
        self.chunk = None
        self.action_index = 0
        self.prompt = None
        self.metadata["episode_step"] = 0

    def infer(self, request: dict) -> dict:
        """Return one absolute joint target and continuous gripper closedness."""
        if request.get("reset", False):
            self.reset()
        prompt = request["prompt"]
        if self.prompt is not None and prompt != self.prompt:
            raise ValueError(
                "Task changed within an episode; send reset or reconnect first."
            )
        if self.chunk is None or self.action_index == self.execute_horizon:
            self.chunk = self.policy.infer(request)
            self.action_index = 0
            self.prompt = prompt
        index = self.action_index
        self.action_index += 1
        self.metadata["episode_step"] += 1
        return {
            "qtarget": self.chunk["qtarget"][index : index + 1].copy(),
            "gripper_closedness": self.chunk["gripper_closedness"][
                index : index + 1
            ].copy(),
            "metadata": dict(self.metadata),
        }


def main() -> None:
    """Load the legacy canonical adapter and serve its unchanged action contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/infini/daibo/models/franka_canonical_qtarget_step1390"),
    )
    parser.add_argument(
        "--model-kind",
        choices=("canonical-qtarget", "pi05-qtarget"),
        default="canonical-qtarget",
    )
    parser.add_argument("--action-dt", type=float, default=0.066)
    parser.add_argument("--execute-horizon", type=int, choices=(8, 16), default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.model_kind == "pi05-qtarget":
        from .pi05_policy import Pi05QtargetPolicy

        policy_class = Pi05QtargetPolicy
    else:
        from .model_adapters import CanonicalQtargetPolicy

        policy_class = CanonicalQtargetPolicy
    policy = policy_class(
        args.checkpoint,
        action_dt=args.action_dt,
        device=args.device,
        denoise_steps=args.denoise_steps,
        seed=args.seed,
    )
    asyncio.run(
        serve(CanonicalTickPolicy(policy, args.execute_horizon), args.host, args.port)
    )


if __name__ == "__main__":
    main()

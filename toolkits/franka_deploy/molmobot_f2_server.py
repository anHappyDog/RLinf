# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bridge official MolmoBot-DROID F=2 using one real observation per control tick."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOGGER = logging.getLogger("franka_deploy.molmobot_f2_server")


class MolmoBotF2Policy:
    """Preserve the official policy's history, action buffer, and action clamp.

    Each request is one physical control tick. A reply contains one action;
    predicting H16 and refreshing after eight ticks remain owned by upstream.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        action_dt: float,
        image_input: str = "droid-downsample",
        seed: int = 0,
    ) -> None:
        import torch
        from olmo.eval.configure_real_robot import (
            RealRobotVLAPolicy,
            RealRobotVLAPolicyConfig,
        )

        if not np.isfinite(action_dt) or action_dt <= 0:
            raise ValueError(
                "action_dt must be positive seconds per physical control tick."
            )
        if image_input not in ("native", "droid-downsample"):
            raise ValueError("image_input must be native or droid-downsample.")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = RealRobotVLAPolicyConfig()
        config.checkpoint_path = str(checkpoint.resolve())
        config.action_type = "joint_pos"
        config.action_keys["arm"] = "joint_pos"
        self.policy = RealRobotVLAPolicy(
            config=SimpleNamespace(policy_config=config), task_type="manipulation"
        )
        agent = self.policy.agent
        if (
            agent.action_dim,
            agent.action_horizon,
            self.policy.input_window_size,
            self.policy.obs_step_delta,
        ) != (8, 16, 2, 8):
            raise ValueError(
                "Expected official MolmoBot-DROID F=2: action8/H16, two frames eight ticks apart."
            )
        if agent.state_preprocessor is None or agent.action_postprocessor is None:
            raise ValueError(
                "Expected checkpoint state normalization and action unnormalization."
            )
        for processor in (
            agent.model_config.robot_preprocessor,
            agent.model_config.robot_postprocessor,
        ):
            if "synthmanip" not in processor.stats_by_repo:
                raise ValueError(
                    "Expected checkpoint synthmanip normalization statistics."
                )
        if config.execute_horizon != 8 or config.gripper_representation_count != 1:
            raise ValueError(
                "Expected official execute_horizon=8 and one gripper state value."
            )
        self.image_input = image_input
        self.prompt = None
        self.metadata = {
            "adapter": "official_molmobot_droid_f2_per_tick",
            "checkpoint": config.checkpoint_path,
            "horizon": 1,
            "model_horizon": agent.action_horizon,
            "execute_horizon": config.execute_horizon,
            "action_dt": float(action_dt),
            "official_droid_action_dt": 0.066,
            "observation_protocol": "one fresh measured observation per physical control tick",
            "n_obs_steps": self.policy.input_window_size,
            "obs_step_delta": self.policy.obs_step_delta,
            "history_initialization": "official valid frames only; no duplicated padding",
            "denoise_steps": agent.num_flow_steps,
            "state_semantics": "qpos7 rad plus (1-clip(width_m/0.085,0,1))*0.824033",
            "gripper_feedback_used": True,
            "gripper_max_width_m": 0.085,
            "action_semantics": "absolute joint targets rad; binary closedness 0=open,1=closed",
            "relative_max_joint_delta_rad": list(config.relative_max_joint_delta),
            "clamp_gripper": config.clamp_gripper,
            "normalization": "official checkpoint synthmanip processors",
            "camera_order": list(config.camera_names),
            "image_input": image_input,
            "image_preprocessing": "official checkpoint preprocessor; no bridge crop, rotation or mirroring",
            "seed": seed,
        }

    def reset(self) -> None:
        """Start a new episode with empty official observation/action history."""
        self.policy.reset()
        self.prompt = None

    def prepare_observation(self, request: dict) -> dict:
        """Apply the official DROID width mapping and retain measured joint angles."""
        qpos = np.asarray(request["qpos"], dtype=np.float32)
        width = float(request["gripper_width_m"])
        prompt = request["prompt"]
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError(
                "qpos must contain seven finite measured joint angles in radians."
            )
        if not np.isfinite(width) or width < 0:
            raise ValueError(
                "gripper_width_m must be a finite nonnegative measurement."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty task string.")
        if self.prompt is not None and prompt != self.prompt:
            raise ValueError(
                "Task changed within an episode; send reset or reconnect first."
            )
        observation = {
            "task": prompt,
            "qpos": {
                "arm": qpos,
                "gripper": np.full(2, (1 - np.clip(width / 0.085, 0, 1)) * 0.824033),
            },
        }
        for role, name in (("external", "exo_camera_1"), ("wrist", "wrist_camera")):
            image = np.asarray(request["images"][role])
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError(f"{role} must be a nonempty HxWx3 RGB uint8 image.")
            if self.image_input == "droid-downsample":
                height, width_px = image.shape[:2]
                if height % 360 or width_px % 640:
                    raise ValueError(
                        "Official DROID downsample requires integer multiples of 360x640."
                    )
                image = image[:: height // 360, :: width_px // 640]
            observation[name] = image
        return observation

    def infer(self, request: dict) -> dict:
        """Return one official buffered action without fabricating intermediate observations."""
        if request.get("reset", False):
            self.reset()
        observation = self.prepare_observation(request)
        self.prompt = observation["task"]
        action = self.policy.get_action(observation)
        qtarget = np.asarray(action["arm"], dtype=np.float32)
        gripper = np.asarray(action["gripper"], dtype=np.float32)
        if (
            qtarget.shape != (7,)
            or gripper.shape != (1,)
            or not np.isfinite(qtarget).all()
            or not np.isfinite(gripper).all()
        ):
            raise ValueError(
                "Official policy returned an invalid arm7/gripper1 action."
            )
        return {
            "qtarget": qtarget[None, :],
            "gripper_closedness": gripper / 255.0,
            "metadata": {**self.metadata, "episode_step": self.policy.step_count},
        }


async def serve(policy: MolmoBotF2Policy, host: str, port: int) -> None:
    """Own one episode per connection and send no unsolicited hello message."""
    import msgpack_numpy
    from websockets.asyncio.server import serve as websocket_serve

    episode_lock = asyncio.Lock()

    async def handler(connection):
        async with episode_lock:
            policy.reset()
            try:
                async for message in connection:
                    if not isinstance(message, bytes):
                        await connection.close(
                            code=1003, reason="Expected binary msgpack request"
                        )
                        return
                    request = msgpack_numpy.unpackb(message)
                    if request.get("reset", False) and "qpos" not in request:
                        policy.reset()
                        reply = {"reset": True, "metadata": policy.metadata}
                    else:
                        reply = policy.infer(request)
                    await connection.send(msgpack_numpy.packb(reply))
            except Exception:
                LOGGER.exception("MolmoBot policy request failed")
                await connection.close(
                    code=1011, reason="Policy request failed; see server log"
                )
            finally:
                policy.reset()

    async with websocket_serve(
        handler,
        host,
        port,
        compression=None,
        max_size=32 * 1024 * 1024,
        ping_interval=None,
    ):
        LOGGER.info(
            "Serving %s:%d metadata=%s", host, port, json.dumps(policy.metadata)
        )
        await asyncio.Future()


def main() -> None:
    """Load the official model and preserve the per-tick DROID inference protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--action-dt",
        type=float,
        required=True,
        help="Seconds per physical tick; official DROID uses 0.066.",
    )
    parser.add_argument(
        "--image-input",
        choices=("native", "droid-downsample"),
        default="droid-downsample",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = MolmoBotF2Policy(
        args.checkpoint,
        action_dt=args.action_dt,
        image_input=args.image_input,
        seed=args.seed,
    )
    asyncio.run(serve(policy, args.host, args.port))


if __name__ == "__main__":
    main()

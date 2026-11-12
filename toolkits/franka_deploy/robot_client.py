# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Single-arm Franka client for absolute qtarget7 + closedness policies."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("franka_deploy.robot_client")


def validate_hardware(config: dict, *, execute: bool) -> None:
    """Require explicit device identities before opening any connection."""
    if config.get("hardware_confirmed") is not True:
        raise ValueError(
            "Set hardware_confirmed=true after confirming the physical rig."
        )
    if not config.get("robot_ip") or config.get("backend") != "franky":
        raise ValueError("A confirmed robot_ip and backend=franky are required.")
    gripper = config["gripper"]
    if gripper.get("type") != "robotiq" or not gripper.get("device"):
        raise ValueError("This client requires a Robotiq serial device.")
    if gripper.get("model") not in ("2F-85", "2F-140"):
        raise ValueError("Confirm gripper.model as 2F-85 or 2F-140.")
    if (
        gripper.get("max_width_m") is None
        or not 0 < float(gripper["max_width_m"]) <= 0.14
    ):
        raise ValueError("Confirm Robotiq max_width_m in (0, 0.14].")
    cameras = config["cameras"]
    if set(cameras) != {"external", "wrist"}:
        raise ValueError("Exactly external and wrist camera roles are required.")
    serials = [cameras[role].get("serial") for role in ("external", "wrist")]
    if not all(serials) or len(set(serials)) != 2:
        raise ValueError("Assign two different camera serials to external and wrist.")
    if execute:
        for key in ("action_dt", "max_target_delta_rad", "max_target_velocity_rad_s"):
            value = config.get(key)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Confirm a positive {key} before --execute.")


def decode_actions(reply: dict, max_width_m: float) -> tuple:
    """Validate canonical actions and convert closedness to physical width."""
    qtarget = np.asarray(reply["qtarget"], dtype=np.float64)
    closedness = np.asarray(reply["gripper_closedness"], dtype=np.float64)
    metadata = reply["metadata"]
    horizon = metadata["horizon"]
    action_dt = float(metadata["action_dt"])
    if qtarget.shape != (horizon, 7) or closedness.shape != (horizon,):
        raise ValueError(
            "Expected qtarget[H,7], gripper_closedness[H] and matching horizon."
        )
    if horizon < 1 or not np.isfinite(qtarget).all():
        raise ValueError(
            "Action chunk must be nonempty and contain finite joint targets."
        )
    if not np.isfinite(closedness).all() or np.any((closedness < 0) | (closedness > 1)):
        raise ValueError("gripper_closedness must lie in [0,1]: zero=open, one=closed.")
    if not np.isfinite(action_dt) or action_dt <= 0:
        raise ValueError("metadata.action_dt must be positive seconds.")
    return qtarget, (1.0 - closedness) * max_width_m, action_dt


def build_robot(config: dict):
    """Compose existing RLinf drivers without opening their connections."""
    import robotiq_link  # noqa: F401 - registers the ready-state-preserving driver.
    from fr3_arm import FR3Arm

    from rlinf.robotics import FrankaRobot
    from rlinf.robotics.parts.cameras import CameraInfo

    cameras = {
        role: CameraInfo(
            name=role,
            serial_number=str(camera["serial"]),
            camera_type=camera["type"],
            resolution=tuple(camera["resolution"]),
            fps=camera["fps"],
            enable_depth=False,
        )
        for role, camera in config["cameras"].items()
    }
    gripper = config["gripper"]
    return FrankaRobot(
        arm=FR3Arm(
            config["robot_ip"],
            urdf_path=config["robot_urdf"],
            realtime_config=config["realtime_config"],
        ),
        end_effector=FrankaRobot.declare_end_effector(
            config["robot_ip"],
            node_rank=None,
            name="pickup-gripper",
            backend="franky",
            gripper_type="robotiq_keepalive",
            gripper_connection=gripper["device"],
            end_effector_config={
                "max_width": gripper["max_width_m"],
                "baudrate": gripper["baudrate"],
                "slave_id": gripper["slave_id"],
            },
        ),
        **FrankaRobot.build_cameras(cameras),
    )


def read_width(gripper) -> float:
    """Read actual gPO; reject failed reads instead of upstream's cached fallback."""
    status = gripper._read_status()
    if status is None:
        raise RuntimeError(
            "No live Robotiq position feedback; refusing a cached width."
        )
    if status["fault"] or status["gACT"] != 1 or status["gSTA"] != 3:
        raise RuntimeError(f"Robotiq is not ready: {status}")
    return gripper.max_width * (1.0 - status["position"] / 255.0)


def read_observation(robot, prompt: str) -> dict:
    """Return measured joints/width and role-labelled RGB images without cropping."""
    parts = robot.named_parts
    images = {}
    for role in ("external", "wrist"):
        camera = parts[role]
        bgr = camera.get_observation(max_age=1.0 / camera.camera_info.fps)["frame"]
        if bgr.ndim != 3 or bgr.shape[-1] != 3 or bgr.dtype != np.uint8:
            raise ValueError(f"Camera {role} did not return an HxWx3 uint8 frame.")
        images[role] = np.ascontiguousarray(bgr[..., ::-1])
    qpos = np.asarray(parts["arm"].get_state().arm_joint_position, dtype=np.float64)
    if qpos.shape != (7,) or not np.isfinite(qpos).all():
        raise RuntimeError("Arm did not return seven finite measured joint positions.")
    return {
        "qpos": qpos,
        "gripper_width_m": read_width(parts["end_effector"]),
        "images": images,
        "prompt": prompt,
    }


def preflight_targets(qtarget, measured, model, config, action_dt) -> None:
    """Validate every selected command before the first command is sent."""
    model.validate_targets(qtarget)
    model.validate_targets(measured)


def execute_chunk(robot, reply: dict, config: dict, execute_steps: int) -> None:
    """Execute only the explicitly selected prefix using the FR3 joint tracker."""
    qtarget, widths, action_dt = decode_actions(reply, config["gripper"]["max_width_m"])
    if not np.isclose(action_dt, config["action_dt"], rtol=0, atol=1e-6):
        raise ValueError(
            "Server action_dt differs from the confirmed hardware cadence."
        )
    if not 1 <= execute_steps <= len(qtarget):
        raise ValueError(
            "execute_steps must be explicitly set between 1 and the response horizon."
        )
    qtarget, widths = qtarget[:execute_steps], widths[:execute_steps]
    parts = robot.named_parts
    arm, gripper = parts["arm"], parts["end_effector"]
    measured = np.asarray(arm.get_state().arm_joint_position)
    approach_seconds = config.get("approach_first_target_seconds", 0)
    if approach_seconds:
        if execute_steps != 1 or approach_seconds < 1:
            raise ValueError(
                "Commissioning approach requires one target and at least one second."
            )
        if np.max(np.abs(qtarget[0] - measured)) > config["max_target_delta_rad"]:
            raise ValueError(
                "First model target exceeds the commissioning displacement limit."
            )
        count = int(np.ceil(approach_seconds / action_dt))
        phase = np.arange(1, count + 1) / count
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        width_start = read_width(gripper)
        qtarget = measured + blend[:, None] * (qtarget[0] - measured)
        widths = width_start + blend * (widths[0] - width_start)
        LOGGER.info(
            "Commissioning: approach one model target over %.3fs in %d control updates.",
            count * action_dt,
            count,
        )
    preflight_targets(qtarget, measured, arm.model_contract, config, action_dt)
    read_width(gripper)
    LOGGER.info(
        "Preflight passed for %d selected actions; model=%s",
        execute_steps,
        arm.model_contract.contract(),
    )
    if approach_seconds:
        # Start the tracker at the measured pose before timing target updates.
        arm.send_action({"joint_position": measured})
    velocity_limit = np.minimum(
        arm.model_contract.velocity, config["max_target_velocity_rad_s"]
    )
    for index, (target, width) in enumerate(zip(qtarget, widths, strict=True)):
        if time.monotonic() >= config.get("stop_at_monotonic", float("inf")):
            return
        started = time.monotonic()
        measured_state = arm.get_state()
        measured = np.asarray(measured_state.arm_joint_position)
        commanded = measured + np.clip(
            target - measured, -velocity_limit * action_dt, velocity_limit * action_dt
        )
        if np.max(np.abs(commanded - measured)) > config["max_target_delta_rad"]:
            raise ValueError("Controller command exceeds its displacement limit.")
        width_before = read_width(gripper)
        LOGGER.info(
            "Action before: %s",
            json.dumps(
                {
                    "index": index,
                    "actual_q_rad": measured.tolist(),
                    "qtarget_rad": target.tolist(),
                    "controller_target_rad": commanded.tolist(),
                    "estimated_force_N": np.asarray(measured_state.tcp_force).tolist(),
                    "estimated_torque_Nm": np.asarray(
                        measured_state.tcp_torque
                    ).tolist(),
                    "actual_width_m": width_before,
                    "target_width_m": float(width),
                }
            ),
        )
        arm.send_action({"joint_position": commanded})
        gripper.move(float(width))
        elapsed = time.monotonic() - started
        if elapsed > action_dt:
            raise RuntimeError(f"Command dispatch exceeded action_dt: {elapsed:.4f}s.")
        time.sleep(action_dt - elapsed)
    LOGGER.info(
        "Actions after: %s",
        json.dumps(
            {
                "executed_steps": execute_steps,
                "actual_q_rad": np.asarray(arm.get_state().arm_joint_position).tolist(),
                "actual_width_m": read_width(gripper),
            }
        ),
    )


def main() -> None:
    """Default to configuration-only dry mode; open hardware only on explicit flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware", type=Path, default=Path(__file__).parent / "configs/hardware.yaml"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--observe",
        action="store_true",
        help="Open hardware and read one observation; connection activates the gripper and recovers the arm.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Open hardware and execute policy chunks.",
    )
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8000")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument(
        "--execute-steps",
        type=int,
        help="Required for execution: execute only this many frames per predicted chunk.",
    )
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import yaml

    config = yaml.safe_load(args.hardware.read_text())
    if not args.observe and not args.execute:
        LOGGER.info("Dry mode: no hardware imports, connections or policy requests.")
        LOGGER.info(
            "Hardware confirmation=%s; roles=%s; action=absolute qtarget7 rad + closedness[0,1].",
            config.get("hardware_confirmed"),
            list(config["cameras"]),
        )
        return
    validate_hardware(config, execute=args.execute)
    if args.execute and (
        not args.prompt
        or args.max_chunks < 1
        or args.execute_steps is None
        or args.execute_steps < 1
    ):
        parser.error(
            "--execute requires a prompt, --max-chunks >= 1 and explicit --execute-steps >= 1."
        )
    if args.execute:
        import msgpack
        import msgpack_numpy
        from websockets.sync.client import connect

    robot = build_robot(config)
    LOGGER.warning(
        "Opening hardware: arm recovery/collision settings and Robotiq activation occur on connect."
    )
    try:
        robot.connect()
        if args.observe:
            obs = read_observation(robot, args.prompt)
            LOGGER.info(
                "Observation: %s",
                json.dumps(
                    {
                        "qpos": obs["qpos"].tolist(),
                        "gripper_width_m": obs["gripper_width_m"],
                        "images": {
                            role: list(img.shape) for role, img in obs["images"].items()
                        },
                    }
                ),
            )
            return
        with connect(
            args.endpoint, compression=None, max_size=16 * 1024 * 1024
        ) as websocket:
            config["stop_at_monotonic"] = time.monotonic() + args.max_seconds
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
            for chunk in range(args.max_chunks):
                if time.monotonic() >= config["stop_at_monotonic"]:
                    break
                observation = read_observation(robot, args.prompt)
                if args.output:
                    from PIL import Image

                    for role, frame in observation["images"].items():
                        Image.fromarray(frame).save(
                            args.output / f"chunk_{chunk:03d}_{role}.png"
                        )
                started = time.monotonic()
                websocket.send(
                    msgpack.packb(
                        observation, default=msgpack_numpy.encode, use_bin_type=True
                    )
                )
                reply = msgpack.unpackb(
                    websocket.recv(timeout=60),
                    object_hook=msgpack_numpy.decode,
                    raw=False,
                )
                LOGGER.info(
                    "Chunk %d inference %.3fs; metadata=%s",
                    chunk,
                    time.monotonic() - started,
                    reply["metadata"],
                )
                execute_chunk(robot, reply, config, args.execute_steps)
            final = read_observation(robot, args.prompt)
            LOGGER.info(
                "Final observation: %s",
                json.dumps(
                    {
                        "qpos": final["qpos"].tolist(),
                        "width_m": final["gripper_width_m"],
                        "gripper": robot.named_parts["end_effector"]._read_status(),
                    }
                ),
            )
            if args.output:
                from PIL import Image

                for role, frame in final["images"].items():
                    Image.fromarray(frame).save(args.output / f"final_{role}.png")
    except Exception:
        arm = robot.named_parts["arm"]
        if arm._robot is not None:
            try:
                state = arm._robot.state
                LOGGER.error(
                    "Controller state on abort: %s",
                    json.dumps(
                        {
                            "q": list(state.q),
                            "O_F_ext_hat_K": list(state.O_F_ext_hat_K),
                            "K_F_ext_hat_K": list(state.K_F_ext_hat_K),
                            "cartesian_collision": list(state.cartesian_collision),
                            "current_errors": str(state.current_errors),
                            "last_motion_errors": str(state.last_motion_errors),
                        }
                    ),
                )
            except Exception:
                LOGGER.exception("Failed to read controller state after abort")
        raise
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

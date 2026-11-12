# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run the per-tick MolmoBot-DROID F2 bridge on the existing FR3 backend."""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from robot_client import build_robot, read_observation, validate_hardware

LOGGER = logging.getLogger("franka_deploy.molmobot_f2_client")
ADAPTER = "official_molmobot_droid_f2_per_tick"
ACTION_DT = 0.066


def decode_reply(reply: dict, *, episode_step: int, model) -> tuple:
    """Accept one official buffered absolute action without modifying its targets."""
    metadata = reply["metadata"]
    expected = {
        "adapter": ADAPTER,
        "horizon": 1,
        "model_horizon": 16,
        "execute_horizon": 8,
        "n_obs_steps": 2,
        "obs_step_delta": 8,
        "episode_step": episode_step,
        "clamp_gripper": True,
        "gripper_feedback_used": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Expected F2 metadata {key}={value!r}, got {metadata.get(key)!r}."
            )
    if not np.isclose(float(metadata["action_dt"]), ACTION_DT, rtol=0, atol=1e-6):
        raise ValueError("F2 control tick must be 0.066 seconds.")
    if metadata.get("relative_max_joint_delta_rad") != [0.2] * 7:
        raise ValueError(
            "Expected the server's official whole-vector 0.2rad delta scaling."
        )
    if metadata.get("gripper_max_width_m") != 0.085:
        raise ValueError("Expected the official DROID 0.085m gripper mapping.")
    qtarget = np.asarray(reply["qtarget"], dtype=np.float64)
    closedness = np.asarray(reply["gripper_closedness"], dtype=np.float64)
    if qtarget.shape != (1, 7) or closedness.shape != (1,):
        raise ValueError(
            "Expected exactly one qtarget[1,7] and one gripper_closedness[1]."
        )
    model.validate_targets(qtarget)
    if not np.isfinite(closedness).all() or not np.isin(closedness, [0.0, 1.0]).all():
        raise ValueError(
            "Official F2 gripper action must be binary zero=open or one=closed."
        )
    return qtarget[0], float((1.0 - closedness[0]) * 0.085), metadata


def save_images(output: Path, tick: int, images: dict) -> None:
    """Encode RGB snapshots on a worker thread, outside the control tick."""
    from PIL import Image

    for role, image in images.items():
        Image.fromarray(image).save(output / f"tick_{tick:06d}_{role}.jpg", quality=90)


def write_record(stream, record: dict) -> None:
    """Flush each record so interrupted sessions retain their actual tick history."""
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def calibrated_width(status: dict, gripper_config: dict) -> float:
    """Estimate physical width from this gripper's recorded empty-stroke endpoints."""
    if status["gACT"] != 1 or status["gSTA"] != 3 or status["fault"] != 0:
        raise RuntimeError(f"Gripper is not ready: {status}")
    opened = gripper_config["encoder_open"]
    closed = gripper_config["encoder_closed"]
    if not 0 <= opened < closed <= 255:
        raise ValueError(
            "Expected confirmed open/closed encoder endpoints in ascending order."
        )
    return float(
        np.clip((closed - status["position"]) / (closed - opened), 0, 1)
        * gripper_config["max_width_m"]
    )


def run(args, config: dict) -> None:
    """Open hardware only for an explicit execution request."""
    import msgpack_numpy
    from websockets.sync.client import connect

    validate_hardware(config, execute=False)
    if not np.isclose(float(config["action_dt"]), ACTION_DT, rtol=0, atol=1e-6):
        raise ValueError(
            "Confirm hardware action_dt=0.066 for this official F2 protocol."
        )
    if (
        config["gripper"]["model"] != "2F-85"
        or config["gripper"]["max_width_m"] != 0.085
    ):
        raise ValueError(
            "This DROID bridge requires the confirmed 2F-85/0.085m mapping."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    image_output = args.output / "observations"
    if args.save_images:
        image_output.mkdir(exist_ok=True)
    robot = build_robot(config)
    parts = robot.named_parts
    arm, gripper = parts["arm"], parts["end_effector"]
    controller_metadata = {
        "backend": "FR3Arm native asynchronous Franky JointMotion",
        "trajectory_generator": "vendor Ruckig; zero terminal joint velocity",
        "polymetis_controller_equivalent": False,
        "robot_dynamics_factor": arm.DYNAMICS_FACTOR,
        "motion_dynamics_factor": arm.MOTION_DYNAMICS_FACTOR,
        "effective_dynamics_factor": arm.DYNAMICS_FACTOR * arm.MOTION_DYNAMICS_FACTOR,
        "client_target_slew": False,
        "client_action_scale": 1.0,
        "gripper_command_change_threshold_m": 0.002,
        "gripper_feedback_calibration": {
            key: config["gripper"][key]
            for key in ("encoder_open", "encoder_closed", "calibration_source")
        },
        "model_contract": arm.model_contract.contract(),
    }
    LOGGER.warning(
        "Explicit execution: connecting FR3/Robotiq; no joint homing. Connection may recover controller errors or activate an unready gripper."
    )
    futures = []
    with (
        (args.output / "ticks.jsonl").open("x") as trace,
        ThreadPoolExecutor(max_workers=1) as image_writer,
    ):
        try:
            robot.connect()
            LOGGER.info(
                "Camera warmup %.1fs; no policy actions are sent during warmup.",
                args.camera_warmup_seconds,
            )
            time.sleep(args.camera_warmup_seconds)
            with connect(
                args.endpoint,
                compression=None,
                max_size=32 * 1024 * 1024,
                ping_interval=None,
            ) as websocket:
                started = time.monotonic()
                stop_at = started + args.max_seconds
                previous_start = None
                last_gripper_command = None
                tick = 0
                executed_ticks = 0
                while time.monotonic() < stop_at:
                    tick += 1
                    tick_start = time.monotonic()
                    observation = read_observation(robot, args.prompt)
                    gripper_status = gripper._read_status()
                    observation["gripper_width_m"] = calibrated_width(
                        gripper_status, config["gripper"]
                    )
                    state = arm.get_state()
                    qpos = np.asarray(state.arm_joint_position, dtype=np.float64)
                    dq = np.asarray(state.arm_joint_velocity, dtype=np.float64)
                    if (
                        qpos.shape != (7,)
                        or dq.shape != (7,)
                        or not np.isfinite(qpos).all()
                        or not np.isfinite(dq).all()
                    ):
                        raise RuntimeError(
                            "Expected seven finite measured joint positions and velocities."
                        )
                    observation["qpos"] = qpos
                    if last_gripper_command is None:
                        last_gripper_command = observation["gripper_width_m"]
                    observation_done = time.monotonic()
                    if observation_done >= stop_at:
                        break
                    websocket.send(msgpack_numpy.packb(observation))
                    try:
                        reply = msgpack_numpy.unpackb(
                            websocket.recv(
                                timeout=max(0.001, stop_at - time.monotonic())
                            )
                        )
                    except TimeoutError:
                        if time.monotonic() >= stop_at:
                            write_record(
                                trace,
                                {
                                    "type": "time_limit_during_policy_request",
                                    "tick": tick,
                                },
                            )
                            break
                        raise
                    inference_done = time.monotonic()
                    target, width, metadata = decode_reply(
                        reply, episode_step=tick, model=arm.model_contract
                    )
                    if tick == 1:
                        write_record(
                            trace,
                            {
                                "type": "metadata",
                                "created_utc": datetime.now(timezone.utc).isoformat(),
                                "prompt": args.prompt,
                                "max_seconds": args.max_seconds,
                                "camera_warmup_seconds": args.camera_warmup_seconds,
                                "server": metadata,
                                "controller": controller_metadata,
                            },
                        )
                    if args.save_images and (tick - 1) % 8 == 0:
                        futures.append(
                            image_writer.submit(
                                save_images, image_output, tick, observation["images"]
                            )
                        )
                    executed = time.monotonic() < stop_at
                    gripper_command_sent = False
                    action_started = time.monotonic()
                    if executed:
                        arm.send_action({"joint_position": target})
                        executed_ticks += 1
                        if abs(width - last_gripper_command) > 0.002:
                            gripper.move(width)
                            last_gripper_command = width
                            gripper_command_sent = True
                    action_done = time.monotonic()
                    write_record(
                        trace,
                        {
                            "type": "tick",
                            "tick": tick,
                            "elapsed_s": tick_start - started,
                            "qpos_rad": qpos.tolist(),
                            "dq_rad_s": dq.tolist(),
                            "gripper_width_m": observation["gripper_width_m"],
                            "gripper_status": gripper_status,
                            "qtarget_rad": target.tolist(),
                            "target_width_m": width,
                            "executed": executed,
                            "gripper_command_sent": gripper_command_sent,
                            "model_reinference": (tick - 1) % 8 == 0,
                            "timing": {
                                "observation_s": observation_done - tick_start,
                                "rpc_s": inference_done - observation_done,
                                "action_s": action_done - action_started,
                                "work_before_log_s": action_done - tick_start,
                                "previous_tick_start_interval_s": None
                                if previous_start is None
                                else tick_start - previous_start,
                                "period_target_s": ACTION_DT,
                            },
                        },
                    )
                    previous_start = tick_start
                    if not executed:
                        break
                    remaining = min(
                        ACTION_DT - (time.monotonic() - tick_start),
                        stop_at - time.monotonic(),
                    )
                    if remaining > 0:
                        time.sleep(remaining)
                write_record(
                    trace,
                    {
                        "type": "finished",
                        "ticks_started": tick,
                        "executed_ticks": executed_ticks,
                        "elapsed_s": time.monotonic() - started,
                    },
                )
        except BaseException as error:
            write_record(
                trace,
                {"type": "error", "class": type(error).__name__, "message": str(error)},
            )
            raise
        finally:
            robot.disconnect()
        for future in futures:
            future.result()


def main() -> None:
    """Default to dry mode without constructing SDK objects or opening hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path(__file__).parent / "configs/molmobot_f2.yaml",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8000")
    parser.add_argument("--prompt", default="Pick up the green cube")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Required for --execute; ticks.jsonl must not already exist.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save both RGB views on model reinference ticks 1,9,... using a writer thread.",
    )
    parser.add_argument(
        "--camera-warmup-seconds",
        type=float,
        default=3.0,
        help="Allow camera auto exposure/white balance to settle before the first observation.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.execute:
        LOGGER.info(
            "Dry mode: no SDK imports, robot construction, camera connections or RPC requests."
        )
        LOGGER.info(
            "F2 protocol: 0.066s tick; server model H16/execute8; one untouched target per tick; FR3 native controller differs from Polymetis."
        )
        return
    if (
        args.output is None
        or not args.prompt.strip()
        or not np.isfinite(args.max_seconds)
        or args.max_seconds <= 0
        or not np.isfinite(args.camera_warmup_seconds)
        or args.camera_warmup_seconds < 0
    ):
        parser.error(
            "--execute requires --output, a nonempty prompt and positive --max-seconds."
        )
    import yaml

    config = yaml.safe_load(args.hardware.read_text())
    run(args, config)


if __name__ == "__main__":
    main()

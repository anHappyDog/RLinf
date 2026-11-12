# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run per-tick VLA inference through an isolated Polymetis FR3 server."""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
from molmobot_f2_client import ACTION_DT, decode_reply, save_images, write_record

LOGGER = logging.getLogger("franka_deploy.molmobot_f2_polymetis_client")


def validate_hardware(config: dict, *, execute: bool) -> None:
    """Require an explicit isolated endpoint, real model, and existing peripherals."""
    if (
        config.get("hardware_confirmed") is not True
        or config.get("backend") != "polymetis"
    ):
        raise ValueError("Confirm the physical rig and backend=polymetis.")
    server = config["polymetis"]
    if (
        not server.get("host")
        or not isinstance(server.get("port"), int)
        or not 0 < server["port"] < 65536
    ):
        raise ValueError("Set an explicit polymetis.host and valid polymetis.port.")
    if (
        not config.get("robot_ip")
        or not config.get("robot_urdf")
        or not Path(config["robot_urdf"]).is_file()
    ):
        raise ValueError(
            "A physical robot_ip and saved controller-exported robot_urdf are required."
        )
    gripper = config["gripper"]
    if (
        gripper.get("type") != "robotiq"
        or gripper.get("model") != "2F-85"
        or not gripper.get("device")
        or gripper.get("max_width_m") != 0.085
    ):
        raise ValueError(
            "Require the confirmed serial Robotiq 2F-85 and 0.085m width mapping."
        )
    cameras = config["cameras"]
    if set(cameras) != {"external", "wrist"}:
        raise ValueError("Exactly external and wrist camera roles are required.")
    serials = [cameras[role].get("serial") for role in ("external", "wrist")]
    if not all(serials) or len(set(serials)) != 2:
        raise ValueError(
            "Assign different confirmed camera serials to external and wrist."
        )
    home = config["autohome"]
    if (
        home["enabled"] is not True
        or home["duration"] != 5.0
        or home["home_jointpos"] != [0.0, -0.7, 0.0, -2.35619, 0.0, 1.57079, 0.0]
    ):
        raise ValueError(
            "Require the official DROID enabled 5-second home configuration."
        )
    if (
        gripper["speed"] != 0.085
        or gripper["force"] != 10.0
        or gripper["force_units"] != "percent"
    ):
        raise ValueError(
            "Require official DROID gripper speed=0.085m/s and force=10 percent."
        )
    if config["action_scale"] != 1.0:
        raise ValueError("The official DROID action_scale must remain 1.0.")


def build_robot(config: dict):
    """Compose the new arm with the existing camera and Robotiq drivers, without opening them."""
    from official_robotiq import OfficialRobotiqGripper
    from polymetis_arm import PolymetisArm

    from rlinf.robotics.parts.cameras import Camera, CameraInfo
    from rlinf.robotics.robot import Robot

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
    server = config["polymetis"]
    return Robot(
        arm=PolymetisArm(
            server["host"], server["port"], urdf_path=config["robot_urdf"]
        ),
        end_effector=OfficialRobotiqGripper(
            port=gripper["device"],
            max_width=gripper["max_width_m"],
            baudrate=gripper["baudrate"],
            slave_id=gripper["slave_id"],
        ),
        **Camera.declare(cameras),
    )


def describe_controller(arm, config: dict) -> dict:
    """Record the actual server model and gains after connection."""
    return {
        "backend": "Polymetis RobotInterface",
        "controller_api": "start_joint_impedance() / update_desired_joint_positions()",
        "adaptive": True,
        "gains_source": "actual server metadata defaults; no client overrides",
        "server": arm.server_metadata,
        "model_contract": arm.model_contract.contract(),
        "schedule": "official DROID accumulated sleep_until with overdue-deadline correction",
        "client_target_slew": False,
        "client_action_scale": 1.0,
        "gripper_command_change_threshold_m": 0.002,
        "gripper_feedback_mapping": {
            "source": "official_robotiq.OfficialRobotiqGripper.get_state().width",
            "encoder_open": 3,
            "encoder_closed": 230,
            "max_width_m": 0.085,
        },
        "automatic_homing": True,
        "autohome": config["autohome"],
        "startup_gripper_width_m": 0.085,
        "gripper_speed_m_s": config["gripper"]["speed"],
        "gripper_force": config["gripper"]["force"],
        "gripper_force_units": config["gripper"]["force_units"],
        "operator_enter_before_policy": True,
        "observation_order": ["qpos", "gripper_width", "wrist", "external"],
        "automatic_recovery": False,
    }


def verify_server_configuration(arm, config: dict) -> None:
    """Reject an older running server before opening the gripper or homing."""
    import yaml

    expected = yaml.safe_load(Path(config["controller_profile"]).read_text())
    actual = arm.server_metadata
    for name, value in {"hz": expected["hz"], **expected["controller"]}.items():
        if actual[name] != value:
            raise RuntimeError(
                f"Polymetis server {name}={actual[name]!r}; expected {value!r}. Restart with the configured FR3 profile."
            )


def decode_canonical_reply(
    reply: dict,
    *,
    episode_step: int,
    model,
    execute_horizon: int = 16,
    policy_kind: str = "canonical-qtarget",
) -> tuple:
    """Decode legacy canonical H16 targets and continuous gripper closedness."""
    metadata = reply["metadata"]
    pi05 = policy_kind == "pi05-qtarget"
    expected = {
        "adapter": "official_pi05_canonical_qtarget_per_tick"
        if pi05
        else "official_canonical_qtarget_legacy_molmo_per_tick",
        "horizon": 1,
        "model_horizon": 16,
        "execute_horizon": execute_horizon,
        "n_obs_steps": 1,
        "episode_step": episode_step,
        "gripper_feedback_used": pi05,
        "legacy_state_width_mask0": not pi05,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Expected canonical metadata {key}={value!r}, got {metadata.get(key)!r}."
            )
    if not np.isclose(float(metadata["action_dt"]), ACTION_DT, rtol=0, atol=1e-6):
        raise ValueError("Canonical control tick must be 0.066 seconds.")
    qtarget = np.asarray(reply["qtarget"], dtype=np.float64)
    closedness = np.asarray(reply["gripper_closedness"], dtype=np.float64)
    if qtarget.shape != (1, 7) or closedness.shape != (1,):
        raise ValueError("Expected qtarget[1,7] and gripper_closedness[1].")
    model.validate_targets(qtarget)
    if (
        not np.isfinite(closedness).all()
        or np.any(closedness < 0)
        or np.any(closedness > 1)
    ):
        raise ValueError("Canonical gripper closedness must be finite and in [0,1].")
    return qtarget[0], float((1.0 - closedness[0]) * 0.085), metadata


def read_policy_observation(robot, prompt: str, wrist_rotation: int = 0) -> tuple:
    """Read state once, followed by wrist and external RGB in official DROID order."""
    parts = robot.named_parts
    state = parts["arm"].get_state()
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
    gripper = parts["end_effector"]
    gripper_state = gripper.get_state()
    width = float(gripper_state.width)
    if not np.isfinite(width) or not 0 <= width <= 0.085:
        raise RuntimeError(
            "Expected official Robotiq measured width in [0,0.085] metres."
        )
    gripper_status = gripper_state.raw_status
    images = {}
    for role in ("wrist", "external"):
        camera = parts[role]
        bgr = camera.get_observation(max_age=1.0 / camera.camera_info.fps)["frame"]
        if bgr.ndim != 3 or bgr.shape[-1] != 3 or bgr.dtype != np.uint8:
            raise ValueError(f"Camera {role} did not return an HxWx3 uint8 frame.")
        if np.std(np.mean(bgr, axis=-1)) < 1.0:
            raise RuntimeError(f"Camera {role} returned a constant-color frame.")
        rgb = bgr[..., ::-1]
        if role == "wrist" and wrist_rotation == 180:
            rgb = np.rot90(rgb, 2)
        images[role] = np.ascontiguousarray(rgb)
    return (
        {"qpos": qpos, "gripper_width_m": width, "images": images, "prompt": prompt},
        state,
        gripper_status,
    )


def run(args, config: dict) -> None:
    """Open hardware only for an explicit execution request."""
    import msgpack_numpy
    from websockets.sync.client import connect

    validate_hardware(config, execute=False)
    execute_horizon = args.execute_horizon or (
        8 if args.policy_kind == "molmobot-f2" else 16
    )
    if args.policy_kind == "molmobot-f2" and execute_horizon != 8:
        raise ValueError("Official MolmoBot F2 requires execute_horizon=8.")
    decode_action = (
        decode_reply
        if args.policy_kind == "molmobot-f2"
        else partial(
            decode_canonical_reply,
            execute_horizon=execute_horizon,
            policy_kind=args.policy_kind,
        )
    )
    if not np.isclose(float(config["action_dt"]), ACTION_DT, rtol=0, atol=1e-6):
        raise ValueError("Confirm hardware action_dt=0.066 for this control protocol.")
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
    LOGGER.warning(
        "Explicit execution: wait for Enter before gripper opening and homing; wait again before model actions. No automatic recovery."
    )
    futures = []
    recorder = None
    with (
        (args.output / "ticks.jsonl").open("x") as trace,
        ThreadPoolExecutor(max_workers=1) as image_writer,
    ):
        try:
            robot.connect()
            if args.record_video:
                from video_recorder import CameraVideoRecorder

                recorder = CameraVideoRecorder(
                    {role: parts[role] for role in ("external", "wrist")},
                    args.output / "videos",
                    args.wrist_rotation,
                )
            verify_server_configuration(arm, config)
            controller_metadata = describe_controller(arm, config)
            controller_metadata["wrist_image_rotation_degrees"] = args.wrist_rotation
            LOGGER.info(
                "Wrist image rotation: %d degrees; external image unchanged.",
                args.wrist_rotation,
            )
            gripper_config = config["gripper"]
            gripper_speed = gripper_config["speed"]
            gripper_force = gripper_config["force"]
            home = config["autohome"]
            current_q = arm.get_state().arm_joint_position
            home_q = np.asarray(home["home_jointpos"])
            LOGGER.warning(
                "Home preview: current=%s, target=%s, delta=%s rad, duration=%.1fs",
                np.round(current_q, 4).tolist(),
                home_q.tolist(),
                np.round(home_q - current_q, 4).tolist(),
                home["duration"],
            )
            input(
                "Clear the full arm/camera/cable path. Press Enter to open the gripper and execute home, or Ctrl-C to cancel..."
            )
            gripper.goto(0.085, gripper_speed, gripper_force, blocking=True)
            arm.move_to_joint_positions(
                np.asarray(home["home_jointpos"]), time_to_go=home["duration"]
            )
            LOGGER.info(
                "Camera warmup %.1fs; no policy actions are sent during warmup.",
                args.camera_warmup_seconds,
            )
            time.sleep(args.camera_warmup_seconds)
            if args.save_images:
                preview, _, _ = read_policy_observation(
                    robot, args.prompt, args.wrist_rotation
                )
                preview_output = args.output / "post_home_preview"
                preview_output.mkdir()
                save_images(preview_output, 0, preview["images"])
                LOGGER.info(
                    "Post-home RGB preview saved to %s; no model request sent.",
                    preview_output,
                )
            input(
                "Home complete and cameras ready. Press Enter to start policy actions..."
            )
            with connect(
                args.endpoint,
                compression=None,
                max_size=32 * 1024 * 1024,
                ping_interval=None,
            ) as websocket:
                arm.start_control()
                started = time.monotonic()
                sleep_until = started
                stop_at = started + args.max_seconds
                previous_start = None
                last_gripper_command = 0.085
                tick = 0
                executed_ticks = 0
                while time.monotonic() < stop_at:
                    if recorder is not None:
                        recorder.check()
                    tick += 1
                    tick_start = time.monotonic()
                    observation, state, gripper_status = read_policy_observation(
                        robot, args.prompt, args.wrist_rotation
                    )
                    qpos = np.asarray(state.arm_joint_position, dtype=np.float64)
                    dq = np.asarray(state.arm_joint_velocity, dtype=np.float64)
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
                    target, width, metadata = decode_action(
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
                    if args.save_images and (tick - 1) % execute_horizon == 0:
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
                        if not np.isclose(width, last_gripper_command, atol=0.002):
                            gripper.goto(
                                width, gripper_speed, gripper_force, blocking=False
                            )
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
                            "controller_state": {
                                "timestamp_s": state.timestamp_s,
                                "latency_ms": state.controller_latency_ms,
                                "previous_command_successful": state.previous_command_successful,
                            },
                            "gripper_width_m": observation["gripper_width_m"],
                            "gripper_status": gripper_status,
                            "qtarget_rad": target.tolist(),
                            "target_width_m": width,
                            "executed": executed,
                            "gripper_command_sent": gripper_command_sent,
                            "model_reinference": (tick - 1) % execute_horizon == 0,
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
                    sleep_until += ACTION_DT
                    curr_time = time.monotonic()
                    if curr_time < sleep_until:
                        time.sleep(
                            min(sleep_until - curr_time, max(0.0, stop_at - curr_time))
                        )
                    else:
                        LOGGER.warning("Loop overrun by %.3fs", curr_time - sleep_until)
                        while sleep_until < curr_time:
                            sleep_until += ACTION_DT
                        sleep_until -= ACTION_DT
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
            try:
                arm.stop_control()
            finally:
                try:
                    if recorder is not None:
                        recorder.close()
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
        default=Path(__file__).parent / "configs/molmobot_f2_polymetis.yaml",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wrist-rotation",
        type=int,
        choices=(0, 180),
        default=0,
        help="Rotate wrist RGB before policy preprocessing; saved wrist previews use the same rotation.",
    )
    parser.add_argument(
        "--policy-kind",
        choices=("molmobot-f2", "canonical-qtarget", "pi05-qtarget"),
        default="molmobot-f2",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        choices=(8, 16),
        help="Must match the policy server; canonical defaults to 16, F2 to 8.",
    )
    parser.add_argument("--endpoint", default="ws://127.0.0.1:18081")
    parser.add_argument("--prompt", default="Pick up the green cube")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Required for --execute; ticks.jsonl must not already exist.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Continuously record both cameras at camera FPS, including home/waits; independent of policy inference.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save both RGB views on model reinference ticks using a writer thread.",
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
            "Execution startup: open85mm, official 5-second home, warmup, Enter; then 0.066s accumulated-clock control ticks with adaptive Polymetis impedance. No recovery."
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

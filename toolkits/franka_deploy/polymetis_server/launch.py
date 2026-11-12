# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run an isolated Polymetis service without sudo or automatic recovery."""

import argparse
import json
import logging
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("franka_deploy.polymetis_server")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Connect the physical arm and run torque control.",
    )
    mode.add_argument(
        "--mock",
        action="store_true",
        help="Run the vendor mock client without a robot connection.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("fr3.yaml")
    )
    parser.add_argument("--port", type=int, help="Override the dedicated service port.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/infini/daibo/runs/polymetis_fr3"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.execute and not args.mock:
        LOGGER.info(
            "Dry mode: use --mock for offline service or --execute for the physical arm."
        )
        return

    import yaml
    from omegaconf import OmegaConf
    from polymetis.robot_client.metadata import RobotClientMetadata
    from polymetis.utils.grpc_utils import check_server_exists

    config = yaml.safe_load(args.config.read_text())
    host = config["host"]
    port = args.port if args.port is not None else config["port"]
    if host != "127.0.0.1" or not 0 < port < 65536:
        raise ValueError("Use a valid dedicated localhost port.")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))

    binaries = {
        name: shutil.which(name) for name in ("run_server", "franka_panda_client")
    }
    if not all(binaries.values()):
        raise RuntimeError(
            f"Polymetis binaries missing from the private PATH: {binaries}"
        )
    metadata = RobotClientMetadata(
        **config["controller"],
        hz=config["hz"],
        robot_model_cfg=OmegaConf.create(config["robot_model"]),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run = args.output_root / stamp
    run.mkdir(parents=True)
    metadata_path = run / "metadata.pb"
    metadata_path.write_bytes(metadata.serialize())
    hardware = dict(config["hardware"])
    hardware.update(
        control_ip=host,
        control_port=port,
        hz=config["hz"],
        num_dofs=7,
        mock=args.mock,
        readonly=False,
        robot_client_metadata_path=str(metadata_path),
    )
    hardware_path = run / "hardware.yaml"
    hardware_path.write_text(yaml.safe_dump(hardware, sort_keys=False))
    server_command = [binaries["run_server"], "-s", host, "-p", str(port)]
    if args.execute:
        server_command.append("-r")
    hardware_command = [binaries["franka_panda_client"], str(hardware_path)]
    affinity = config.get("cpu_affinity")
    if affinity is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("Configured CPU affinity requires taskset.")
        server_cpus = ",".join(str(cpu) for cpu in affinity["server"])
        hardware_cpus = ",".join(str(cpu) for cpu in affinity["hardware"])
        server_command = [taskset, "--cpu-list", server_cpus, *server_command]
        hardware_command = [taskset, "--cpu-list", hardware_cpus, *hardware_command]
        LOGGER.info(
            "Dedicated CPU affinity: controller=%s, hardware=%s",
            server_cpus,
            hardware_cpus,
        )
    children = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (
            (run / "server.log").open("w") as server_log,
            (run / "hardware.log").open("w") as hardware_log,
        ):
            server = subprocess.Popen(
                server_command, stdout=server_log, stderr=subprocess.STDOUT
            )
            children.append(server)
            deadline = time.monotonic() + 15
            while not check_server_exists(host, port):
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Polymetis server exited; see {run / 'server.log'}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Polymetis server did not listen; see {run / 'server.log'}"
                    )
                time.sleep(0.1)
            client = subprocess.Popen(
                hardware_command, stdout=hardware_log, stderr=subprocess.STDOUT
            )
            children.append(client)
            (run / "processes.json").write_text(
                json.dumps(
                    {
                        "server_pid": server.pid,
                        "hardware_pid": client.pid,
                        "server_command": server_command,
                        "hardware_command": hardware_command,
                        "hardware_config": str(hardware_path),
                        "cpu_affinity": affinity,
                        "mode": "mock" if args.mock else "physical",
                        "host": host,
                        "port": port,
                        "config": str(args.config),
                    },
                    indent=2,
                )
                + "\n"
            )
            LOGGER.info(
                "Polymetis %s service: %s:%d, logs=%s",
                "mock" if args.mock else "physical",
                host,
                port,
                run,
            )
            while True:
                for child in children:
                    if child.poll() is not None:
                        raise RuntimeError(
                            f"Polymetis process {child.pid} exited with {child.returncode}; see {run}"
                        )
                time.sleep(0.2)
    except KeyboardInterrupt:
        LOGGER.info("Stopping this service's hardware client and server.")
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    main()

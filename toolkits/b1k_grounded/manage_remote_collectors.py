# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Start, inspect, or stop persistent BEHAVIOR collector daemons on one host."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path

_REQUIRED_ENV = (
    "TMPDIR",
    "OMNIGIBSON_DATA_PATH",
    "OMNIGIBSON_DATASET_PATH",
    "OMNIGIBSON_KEY_PATH",
    "OMNIGIBSON_ASSET_PATH",
    "OMNI_KIT_ACCEPT_EULA",
    "RLINF_REMOTE_COLLECTOR_TOKEN",
)


def _parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--gpus", type=_parse_csv_ints, default=[0])
    parser.add_argument("--ports", type=_parse_csv_ints, default=[46100])
    parser.add_argument("--python", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument(
        "--omnigibson-path",
        type=Path,
        help=(
            "Patched OmniGibson source directory containing the omnigibson "
            "package. It is prepended to PYTHONPATH for collector daemons."
        ),
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--session-prefix", default="rlinf_b1k_collector")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def _session_name(prefix: str, gpu: int) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("session-prefix contains unsupported characters.")
    return f"{prefix}_gpu{gpu}"


def _collector_pythonpath(
    repo: Path,
    omnigibson_path: Path | None,
    inherited_pythonpath: str | None,
) -> str:
    """Build an ordered source path for a collector daemon."""
    paths = [str(repo)]
    if omnigibson_path is not None:
        paths.append(str(omnigibson_path))
    if inherited_pythonpath:
        paths.append(inherited_pythonpath)
    return os.pathsep.join(paths)


def _tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _stop(session: str) -> None:
    if _tmux_session_exists(session):
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)


def _start(args: argparse.Namespace, gpu: int, port: int) -> None:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Required environment variables are missing: {missing}.")
    if not 0 <= gpu:
        raise ValueError(f"GPU id must be non-negative, got {gpu}.")
    if not 1 <= port <= 65535:
        raise ValueError(f"Port must be in [1, 65535], got {port}.")
    if args.python is None or args.repo is None or args.log_dir is None:
        raise ValueError("start requires --python, --repo, and --log-dir.")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not (args.repo / "rlinf").is_dir():
        raise FileNotFoundError(f"RLinf package not found below {args.repo}.")
    if (
        args.omnigibson_path is not None
        and not (args.omnigibson_path / "omnigibson").is_dir()
    ):
        raise FileNotFoundError(
            f"OmniGibson package not found below {args.omnigibson_path}."
        )

    session = _session_name(args.session_prefix, gpu)
    if _tmux_session_exists(session):
        if not args.replace:
            raise RuntimeError(
                f"tmux session {session!r} already exists; use --replace to replace it."
            )
        _stop(session)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"collector_gpu{gpu}_port{port}.log"
    appdata_root = Path(os.environ["TMPDIR"]) / args.session_prefix / f"gpu{gpu}"
    appdata_root.mkdir(parents=True, exist_ok=True)
    inductor_cache = appdata_root / "torchinductor"
    triton_cache = appdata_root / "triton"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    command = [
        "env",
        *(f"{name}={os.environ[name]}" for name in _REQUIRED_ENV),
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"OMNIGIBSON_APPDATA_PATH={appdata_root}",
        f"TORCHINDUCTOR_CACHE_DIR={inductor_cache}",
        f"TRITON_CACHE_DIR={triton_cache}",
        "PYTHONPATH="
        + _collector_pythonpath(
            args.repo,
            args.omnigibson_path,
            os.environ.get("PYTHONPATH"),
        ),
        str(args.python),
        "-m",
        "rlinf.envs.behavior.remote_collector",
        "--port",
        str(port),
        "--ray-address",
        args.ray_address,
    ]
    shell_command = (
        f"cd {shlex.quote(str(args.repo))} && exec {shlex.join(command)} "
        f">> {shlex.quote(str(log_path))} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, shell_command], check=True
    )
    print(f"started {session}: gpu={gpu} port={port} log={log_path}")


def main() -> None:
    """Manage a one-daemon-per-GPU collector set on the current machine."""
    args = _parse_args()
    if len(args.gpus) != len(args.ports):
        raise ValueError("--gpus and --ports must contain the same number of entries.")
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus contains duplicates.")
    if len(args.ports) != len(set(args.ports)):
        raise ValueError("--ports contains duplicates.")

    for gpu, port in zip(args.gpus, args.ports):
        session = _session_name(args.session_prefix, gpu)
        if args.action == "start":
            _start(args, gpu, port)
        elif args.action == "stop":
            _stop(session)
            print(f"stopped {session}")
        else:
            state = "running" if _tmux_session_exists(session) else "stopped"
            print(f"{session}: {state} (gpu={gpu}, port={port})")


if __name__ == "__main__":
    main()

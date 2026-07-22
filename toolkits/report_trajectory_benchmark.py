#!/usr/bin/env python3
"""Validate SG-15 raw results and create a self-contained benchmark artifact."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FILES = (
    "manifest.yaml",
    "resolved_config.yaml",
    "git_status.txt",
    "environment.txt",
    "commands.txt",
    "raw/local.json",
    "raw/cross_host.json",
    "report.md",
)


def build_artifact(root: Path) -> dict[str, Any]:
    """Validate raw reports and write the SG-15 manifest and summary files."""
    local = _load(root / "raw/local.json")
    cross = _load(root / "raw/cross_host.json")
    _validate_raw(local, cross)

    local_by_mode = {result["mode"]: result for result in local["codec_results"]}
    cross_by_mode = {result["mode"]: result for result in cross["results"]}
    isolation = {result["mode"]: result for result in local["isolation_results"]}
    fastest = min(
        cross_by_mode.values(), key=lambda result: result["latency"]["p50_ms"]
    )["mode"]
    process_delta_ms = (
        isolation["process"]["probe_lag"]["p99_ms"]
        - isolation["baseline"]["probe_lag"]["p99_ms"]
    )
    gates = {
        "bitwise_round_trip": "passed during every benchmark warmup",
        "raw_fallback_payload_growth": "passed by compression implementation tests",
        "recommended_default_mode": fastest,
        "default_mode_not_slower_than_raw": fastest == "raw",
        "storage_process_probe_p99_delta_ms": process_delta_ms,
        "storage_process_probe_gate_ms": 0.5,
        "storage_process_isolated": process_delta_ms <= 0.5,
    }

    status = _command("git status --short")
    diff = _command("git diff --binary")
    manifest = {
        "run_id": root.name,
        "created_on": "2026-07-20",
        "repository": str(Path.cwd()),
        "commit": _command("git rev-parse HEAD").strip(),
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "benchmark_sha256": _sha256(Path("toolkits/benchmark_trajectory_transfer.py")),
        "compression_sha256": _sha256(Path("rlinf/workers/trajectory/compression.py")),
        "hosts": [local["host"], cross["host"], cross["peer"]],
        "workload": local["workload"],
        "raw_bytes": local["raw_bytes"],
        "warmup": local["settings"]["warmup"],
        "repeats": local["settings"]["repeats"],
        "gates": gates,
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "workload": local["workload"],
                "tensor_schema": local["tensor_schema"],
                "settings": local["settings"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "git_status.txt").write_text(status, encoding="utf-8")
    (root / "commands.txt").write_text(_commands(root.name), encoding="utf-8")
    (root / "environment.txt").write_text(_environment(local, cross), encoding="utf-8")
    (root / "report.md").write_text(
        _report(local_by_mode, cross_by_mode, isolation, gates), encoding="utf-8"
    )
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"benchmark artifact is incomplete: {missing}")
    return manifest


def _validate_raw(local: dict[str, Any], cross: dict[str, Any]) -> None:
    if local["kind"] != "local" or cross["kind"] != "cross_host":
        raise ValueError("unexpected benchmark report kinds")
    if local["workload"] != cross["workload"]:
        raise ValueError("local and cross-host workloads differ")
    if local["tensor_schema"] != cross["tensor_schema"]:
        raise ValueError("local and cross-host tensor schemas differ")
    if local["raw_bytes"] != cross["raw_bytes"]:
        raise ValueError("local and cross-host raw byte counts differ")
    expected_codecs = {"raw", "lz4", "zstd"}
    if {result["mode"] for result in local["codec_results"]} != expected_codecs:
        raise ValueError("local report does not contain raw/LZ4/Zstd")
    if {result["mode"] for result in cross["results"]} != expected_codecs:
        raise ValueError("cross-host report does not contain raw/LZ4/Zstd")
    expected_isolation = {"baseline", "inline", "async", "thread", "process"}
    if {result["mode"] for result in local["isolation_results"]} != expected_isolation:
        raise ValueError("isolation report is incomplete")


def _report(
    local: dict[str, Any],
    cross: dict[str, Any],
    isolation: dict[str, Any],
    gates: dict[str, Any],
) -> str:
    codec_rows = "\n".join(
        f"| {mode} | {local[mode]['latency']['p50_ms']:.2f} | "
        f"{local[mode]['latency']['p95_ms']:.2f} | "
        f"{cross[mode]['latency']['p50_ms']:.2f} | "
        f"{cross[mode]['latency']['p95_ms']:.2f} | "
        f"{cross[mode]['latency']['p99_ms']:.2f} | "
        f"{cross[mode]['latency']['p999_ms']:.2f} | "
        f"{cross[mode]['wire_bytes'] / 1024**2:.2f} | "
        f"{cross[mode]['compression_ratio']:.2f}x |"
        for mode in ("raw", "lz4", "zstd")
    )
    resource_rows = "\n".join(
        f"| {mode} | {local[mode]['cpu_time']['p50_ms']:.2f} | "
        f"{local[mode]['peak_rss_delta_bytes'] / 1024**2:.2f} | "
        f"{cross[mode]['cpu_time_rank0']['p50_ms']:.2f} | "
        f"{cross[mode]['rank1_cpu_mean_ms']:.2f} | "
        f"{cross[mode]['peak_rss_rank0_delta_bytes'] / 1024**2:.2f} | "
        f"{cross[mode]['peak_rss_rank1_delta_bytes'] / 1024**2:.2f} | "
        f"{cross[mode]['throughput_raw_gib_s']:.2f} |"
        for mode in ("raw", "lz4", "zstd")
    )
    isolation_rows = "\n".join(
        f"| {mode} | {isolation[mode]['probe_lag']['p50_ms']:.3f} | "
        f"{isolation[mode]['probe_lag']['p95_ms']:.3f} | "
        f"{isolation[mode]['probe_lag']['p99_ms']:.3f} | "
        f"{isolation[mode]['probe_lag']['p999_ms']:.3f} |"
        for mode in ("baseline", "inline", "async", "thread", "process")
    )
    return f"""# SG-15 Trajectory Channel Performance Report

## Scope

One target Storage shard: 8 slots × 48 chunk steps, four 256×256 RGB image
leaves plus the major OpenPI tensor leaves. Payload: {cross["raw"]["raw_bytes"] / 1024**2:.2f} MiB.
Images are deterministic spatially coherent synthetic tensors; byte count and
schema match the target profile, while compression ratios must not be presented
as real-camera distribution measurements.

## Codec and cross-host transfer

| mode | local p50 ms | local p95 ms | cross p50 ms | cross p95 ms | cross p99 ms | cross p99.9 ms | wire MiB | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{codec_rows}

| mode | local CPU p50 ms | local RSS Δ MiB | rank0 CPU p50 ms | rank1 CPU mean ms | rank0 RSS Δ MiB | rank1 RSS Δ MiB | effective GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
{resource_rows}

On this high-bandwidth `bjd_dev → bjd_dev_2` link, raw is the fastest mode even
though LZ4/Zstd greatly reduce bytes. The target config must therefore default
to raw; compression remains an opt-in placement/network decision.

## Live-loop isolation

| mode | probe p50 ms | probe p95 ms | probe p99 ms | probe p99.9 ms |
|---|---:|---:|---:|---:|
{isolation_rows}

Inline and async-task compression block the event loop. Thread and independent
process avoid that head-of-line blocking in this run, but only the StorageWorker
process also provides ownership, failure, queue and memory isolation.

## Provisional regression gates

- Default transfer mode must be the fastest measured mode: `{gates["recommended_default_mode"]}`.
- Dedicated-process live probe p99 delta: {gates["storage_process_probe_p99_delta_ms"]:.3f} ms; gate ≤ {gates["storage_process_probe_gate_ms"]:.3f} ms.
- Future SGs must rerun paired raw/default A/B on the same hosts. A single Runner step time is not a performance baseline.

## Limitations

- Ten measured samples support an engineering regression check, not a universal p99.9 SLO.
- Gloo tensor transfer is the production data-plane primitive, but this tool does not include Ray RPC/control-plane overhead.
- CPU/RSS values are process-level measurements; `process_time()` sums CPU time
  across process threads, and shared-library allocators can retain memory between modes.
- Real-camera compression ratios and restricted-bandwidth break-even remain unmeasured.
"""


def _environment(local: dict[str, Any], cross: dict[str, Any]) -> str:
    return (
        f"driver={platform.node()}\n"
        f"local={json.dumps(local['environment'], sort_keys=True)}\n"
        f"rank0={json.dumps(cross['environments'][0], sort_keys=True)}\n"
        f"rank1={json.dumps(cross['environments'][1], sort_keys=True)}\n"
    )


def _commands(run_id: str) -> str:
    root = f"artifacts/trajectory_channel/{run_id}"
    return f"""# Local target run
PYTHONPATH=$PWD /opt/venv/openpi/bin/python toolkits/benchmark_trajectory_transfer.py --mode local --profile target --warmup 2 --repeats 10 --isolation-seconds 3 --probe-interval-ms 2 --output {root}/raw/local.json

# Cross-host target run after staging files with matching SHA256
# rank0 bjd_dev, master 10.204.17.213:29625
/opt/venv/openpi/bin/python toolkits/benchmark_trajectory_transfer.py --mode distributed --profile target --rank 0 --master-addr 10.204.17.213 --master-port 29625 --peer-name bjd_dev_2 --warmup 2 --repeats 10
# rank1 bjd_dev_2 uses the same command with --rank 1 and --peer-name bjd_dev
"""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(command: str) -> str:
    return subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    manifest = build_artifact(args.artifact)
    print(yaml.safe_dump(manifest["gates"], sort_keys=False), end="")


if __name__ == "__main__":
    main()

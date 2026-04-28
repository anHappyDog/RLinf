"""Helpers for env worker benchmarking scripts."""

from __future__ import annotations

from typing import Any


def build_same_resource_multi_process_placement(
    *,
    resource_rank: int,
    world_size: int,
) -> str:
    """Build a placement string that packs multiple processes onto one resource."""

    if resource_rank < 0:
        raise ValueError("resource_rank must be non-negative.")
    if world_size <= 0:
        raise ValueError("world_size must be greater than 0.")
    if world_size == 1:
        return str(resource_rank)
    return f"{resource_rank}:0-{world_size - 1}"


def build_env_rollout_benchmark_keys(
    *,
    env_rank: int,
    mode: str = "train",
) -> tuple[str, str]:
    """Build the env->rollout obs key and rollout->env result key for one rank."""

    if env_rank < 0:
        raise ValueError("env_rank must be non-negative.")
    if mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return (
        f"{env_rank}_0_{mode}_obs",
        f"0_{env_rank}_{mode}_rollout_results",
    )


def compute_expected_rollout_response_count(
    *,
    rollout_epoch: int,
    pipeline_stage_num: int,
    n_train_chunk_steps: int,
) -> int:
    """Compute how many fake rollout responses one interact call must return."""

    if rollout_epoch <= 0:
        raise ValueError("rollout_epoch must be greater than 0.")
    if pipeline_stage_num <= 0:
        raise ValueError("pipeline_stage_num must be greater than 0.")
    if n_train_chunk_steps < 0:
        raise ValueError("n_train_chunk_steps must be non-negative.")
    return rollout_epoch * pipeline_stage_num * (n_train_chunk_steps + 1)


def compute_total_env_steps(
    *,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    rollout_epoch: int,
) -> int:
    """Compute the total primitive env steps in one interact call."""

    if total_num_envs <= 0:
        raise ValueError("total_num_envs must be greater than 0.")
    if max_steps_per_rollout_epoch <= 0:
        raise ValueError("max_steps_per_rollout_epoch must be greater than 0.")
    if rollout_epoch <= 0:
        raise ValueError("rollout_epoch must be greater than 0.")
    return total_num_envs * max_steps_per_rollout_epoch * rollout_epoch


def infer_batch_size_from_env_obs(env_batch: dict[str, Any]) -> int:
    """Infer batch size from an env batch emitted by ``EnvWorker.send_env_batch``."""

    obs = env_batch["obs"] if "obs" in env_batch else env_batch
    if not isinstance(obs, dict):
        raise ValueError("env_batch must contain an 'obs' dict or be an obs dict.")

    for key in ("states", "main_images", "wrist_images", "task_descriptions"):
        value = obs.get(key)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        if isinstance(value, list):
            return len(value)

    raise ValueError("Cannot infer batch size from env batch.")


def summarize_scalar_series(values: list[float]) -> dict[str, float]:
    """Summarize a list of scalar values with a compact stats dict."""

    if not values:
        raise ValueError("values must not be empty.")

    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    variance = (
        sum((value - mean) ** 2 for value in sorted_values) / count
        if count > 1
        else 0.0
    )
    return {
        "count": float(count),
        "mean": mean,
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 50.0),
        "p95": _percentile(sorted_values, 95.0),
        "max": sorted_values[-1],
        "stdev": variance**0.5,
    }


def summarize_named_scalar_records(
    records: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Summarize a list of timing/metric dicts key-by-key."""

    merged: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            merged.setdefault(key, []).append(float(value))
    return {
        key: summarize_scalar_series(values)
        for key, values in sorted(merged.items())
        if values
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute a percentile from a sorted list using linear interpolation."""

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight

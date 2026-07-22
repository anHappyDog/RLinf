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

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from rlinf.data.trajectory import EnvResult, RolloutResult, ValueResult
from rlinf.scheduler.placement import PackedPlacementStrategy
from rlinf.workers.trajectory.bypass import TrajectoryWriter, endpoint_schema_id
from rlinf.workers.trajectory.compression import CompressionConfig
from rlinf.workers.trajectory.live import PolicyInputLayout, TrajectoryChannel
from rlinf.workers.trajectory.output import TrajectoryReader
from rlinf.workers.trajectory.route_plan import RoutePlan
from rlinf.workers.trajectory.storage import StorageConfig
from rlinf.workers.trajectory.transport import EndpointSchema, TensorLayout
from rlinf.workers.trajectory.workers import (
    ChannelConfig,
    StorageWorkerConfig,
    TrajectoryChannelWorker,
    TrajectoryStorageWorker,
    WorkerLayout,
)


def validate_trajectory_config(cfg) -> None:
    """Reject configurations not owned by the current trajectory runtime."""
    if not cfg.get("trajectory", {}).get("enabled", False):
        return
    if cfg.runner.get("use_training_pipeline", False):
        raise ValueError("trajectory runtime cannot be combined with training pipeline")
    if int(cfg.rollout.pipeline_stage_num) != 1:
        raise ValueError("trajectory runtime requires rollout.pipeline_stage_num=1")
    if cfg.runner.get("only_eval", False):
        raise ValueError("trajectory runtime does not yet own evaluation-only runs")
    if cfg.runner.get("overlap_env_bootstrap", False):
        raise ValueError(
            "trajectory runtime owns boundary values in Storage and cannot use "
            "legacy overlap_env_bootstrap"
        )
    if cfg.get("reward", {}).get("use_reward_model", False):
        raise ValueError("trajectory runtime does not yet own RewardWorker execution")
    if str(cfg.actor.model.model_type) != "openpi":
        raise ValueError("trajectory runtime currently registers only OpenPI schemas")


def launch_trajectory_workers(
    cfg,
    cluster,
    component_placement,
    *,
    env_group,
    rollout_group,
    actor_group,
) -> dict[str, Any]:
    """Launch and connect the SG-12 trajectory workers for embodied PPO."""
    validate_trajectory_config(cfg)
    env_layout = _layout(env_group)
    rollout_layout = _layout(rollout_group)
    actor_layout = _layout(actor_group)
    storage_placement = component_placement.get_strategy("trajectory_storage")
    storage_group = TrajectoryStorageWorker.create_group().launch(
        cluster,
        name=cfg.trajectory.storage_group_name,
        placement_strategy=storage_placement,
        max_concurrency=16,
        isolate_gpu=False,
    )
    storage_layout = _layout(storage_group)

    channel_group = TrajectoryChannelWorker.create_group(
        maxsize=cfg.trajectory.get("live_queue_size", 0)
    ).launch(
        cluster,
        name=cfg.trajectory.channel_group_name,
        placement_strategy=PackedPlacementStrategy(0, 0),
        max_concurrency=16,
        isolate_gpu=False,
    )
    channel_layout = _layout(channel_group)
    total_slots = int(cfg.env.train.total_num_envs)
    route_plan = RoutePlan(
        total_slots,
        {
            "env": len(env_layout.data_ranks),
            "rollout": len(rollout_layout.data_ranks),
            "actor": len(actor_layout.data_ranks),
            "storage": len(storage_layout.data_ranks),
        },
    )
    channel_config = ChannelConfig(
        layout=channel_layout,
        route_plan=route_plan,
        env_layout=env_layout,
        rollout_layout=rollout_layout,
        env_group_name=env_group.worker_group_name,
        rollout_group_name=rollout_group.worker_group_name,
        policy_input_layout=(
            PolicyInputLayout(
                batch_size=(total_slots + len(env_layout.data_ranks) - 1)
                // len(env_layout.data_ranks),
                image_shape=tuple(cfg.trajectory.live.image_shape),
                state_shape=tuple(cfg.trajectory.live.state_shape),
                extra_view_shape=(
                    tuple(cfg.trajectory.live.extra_view_shape)
                    if cfg.trajectory.live.get("extra_view_shape") is not None
                    else None
                ),
                max_description_bytes=int(cfg.trajectory.live.max_description_bytes),
                compress_images=bool(cfg.trajectory.live.get("compress_images", False)),
                compression_level=int(cfg.trajectory.live.get("compression_level", 1)),
                pin_memory=bool(cfg.trajectory.live.get("pin_memory", False)),
            )
            if cfg.trajectory.live.direct_policy_input
            else None
        ),
    )
    channel_group.configure(channel_config).wait()

    chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // int(
        cfg.actor.model.num_action_chunks
    )
    env_schemas = _schemas_by_rank(route_plan, "env", _env_schemas)
    rollout_schemas = _schemas_by_rank(
        route_plan, "rollout", _openpi_libero_rollout_schemas
    )
    endpoints = tuple(
        schema
        for schemas in (*env_schemas.values(), *rollout_schemas.values())
        for schema in schemas
    )
    storage_configs = []
    compression = CompressionConfig(
        enabled=bool(cfg.trajectory.compression.enabled),
        codec=str(cfg.trajectory.compression.codec),
        level=int(cfg.trajectory.compression.level),
        min_bytes=int(cfg.trajectory.compression.min_bytes),
        block_bytes=int(cfg.trajectory.compression.block_bytes),
        num_threads=int(cfg.trajectory.compression.num_threads),
    )
    for logical_rank, physical_rank in enumerate(storage_layout.data_ranks):
        start, end = route_plan.slot_range("storage", logical_rank)
        config = StorageWorkerConfig(
            layout=storage_layout,
            route_plan=route_plan,
            storage=StorageConfig(
                global_step=0,
                rollout_epochs=int(cfg.env.train.rollout_epoch),
                chunk_steps=chunk_steps,
                slot_ids=tuple(range(start, end)),
                rollout_fields=frozenset(
                    {"forward_inputs", "prev_logprobs", "state_values", "versions"}
                ),
                boundary_values=True,
            ),
            registry_modules=("rlinf.models.embodiment.openpi.forward_inputs",),
            endpoints=endpoints,
            actor_participant="actor",
            ingest_queue_size=int(cfg.trajectory.ingest_queue_size),
            max_inflight_frames=int(cfg.trajectory.max_inflight_frames),
            max_resident_bytes=int(cfg.trajectory.max_resident_bytes),
            backpressure_timeout_s=float(cfg.trajectory.backpressure_timeout_s),
            reservation_timeout_s=float(cfg.trajectory.reservation_timeout_s),
            drain_timeout_s=float(cfg.trajectory.drain_timeout_s),
            compression=compression,
        )
        storage_group.execute_on(physical_rank).configure(config).wait()
        storage_configs.append(config.storage)

    channel = TrajectoryChannel.from_worker_group(channel_group, channel_config)
    writer_args = {
        "route_plan": route_plan,
        "storage_layout": storage_layout,
    }
    return {
        "channel_group": channel_group,
        "storage_group": storage_group,
        "storage_layout": storage_layout,
        "storage_configs": tuple(storage_configs),
        "channel": channel,
        "env_writer": TrajectoryWriter.from_worker_group(
            storage_group,
            source_participant="env",
            source_layout=env_layout,
            schemas_by_rank=env_schemas,
            **writer_args,
        ),
        "rollout_writer": TrajectoryWriter.from_worker_group(
            storage_group,
            source_participant="rollout",
            source_layout=rollout_layout,
            schemas_by_rank=rollout_schemas,
            **writer_args,
        ),
        "reader": TrajectoryReader.from_worker_group(
            storage_group,
            route_plan=route_plan,
            storage_layout=storage_layout,
            actor_layout=actor_layout,
            compression=compression,
        ),
    }


def begin_trajectory_generation(runtime: dict[str, Any], global_step: int) -> None:
    """Start a new bounded Storage generation after the prior one was consumed."""
    group = runtime["storage_group"]
    layout = runtime["storage_layout"]
    for physical_rank, storage in zip(
        layout.data_ranks, runtime["storage_configs"], strict=True
    ):
        group.execute_on(physical_rank).begin_generation(
            replace(storage, global_step=global_step)
        ).wait()


def _layout(group) -> WorkerLayout:
    return WorkerLayout(tuple(worker.rank for worker in group.worker_info_list))


def _schemas_by_rank(
    route_plan: RoutePlan,
    participant: str,
    factory: Callable[[int, int], tuple[EndpointSchema, ...]],
) -> dict[int, tuple[EndpointSchema, ...]]:
    return {
        rank: factory(rank, _source_batch_size(route_plan, participant, rank))
        for rank in range(route_plan.world_sizes[participant])
    }


def _source_batch_size(route_plan: RoutePlan, participant: str, rank: int) -> int:
    start, end = route_plan.slot_range(participant, rank)
    return end - start


def _tensor(
    path: tuple[str, ...], shape: tuple[int, ...], dtype: torch.dtype
) -> TensorLayout:
    return TensorLayout(
        path=path,
        shape=shape,
        dtype=dtype,
        element_size=torch.empty((), dtype=dtype).element_size(),
    )


def _env_schemas(source_rank: int, max_batch_size: int) -> tuple[EndpointSchema, ...]:
    return (
        EndpointSchema(
            schema_id=endpoint_schema_id("EnvResult", "", source_rank),
            max_batch_size=max_batch_size,
            record_type=EnvResult.__name__,
            tensors=(
                _tensor(("rewards",), (5,), torch.float32),
                _tensor(("dones",), (5,), torch.bool),
                _tensor(("terminations",), (5,), torch.bool),
                _tensor(("truncations",), (5,), torch.bool),
            ),
            constants=(
                (("observations",), None),
                (("next_observations",), None),
                (("intervene_actions",), None),
                (("intervene_flags",), None),
                (("rlt_switch_flags",), None),
            ),
        ),
    )


def _openpi_libero_rollout_schemas(
    source_rank: int,
    max_batch_size: int,
) -> tuple[EndpointSchema, ...]:
    rollout = EndpointSchema(
        schema_id=endpoint_schema_id("RolloutResult", "", source_rank),
        max_batch_size=max_batch_size,
        record_type=RolloutResult.__name__,
        tensors=(
            _tensor(("actions",), (5, 7), torch.float64),
            _tensor(("forward_inputs", "chains"), (5, 50, 32), torch.float32),
            _tensor(("forward_inputs", "denoise_inds"), (4,), torch.int64),
            _tensor(("forward_inputs", "tokenized_prompt"), (48,), torch.int64),
            _tensor(("forward_inputs", "tokenized_prompt_mask"), (48,), torch.bool),
            _tensor(("forward_inputs", "action"), (35,), torch.float64),
            _tensor(("forward_inputs", "model_action"), (1600,), torch.float32),
            _tensor(
                ("forward_inputs", "observation/image"),
                (256, 256, 3),
                torch.uint8,
            ),
            _tensor(
                ("forward_inputs", "observation/wrist_image"),
                (256, 256, 3),
                torch.uint8,
            ),
            _tensor(("forward_inputs", "observation/state"), (8,), torch.float32),
            _tensor(("prev_logprobs",), (5, 7), torch.float32),
            _tensor(("state_values",), (1,), torch.float32),
            _tensor(("versions",), (5, 7), torch.float32),
        ),
        constants=((("intervene_flags",), None),),
        forward_schema=("openpi_libero", 1),
    )
    values = tuple(
        EndpointSchema(
            schema_id=endpoint_schema_id("ValueResult", kind, source_rank),
            max_batch_size=max_batch_size,
            record_type=ValueResult.__name__,
            tensors=(_tensor(("values",), (1,), torch.float32),),
            constants=((("kind",), kind), (("versions",), None)),
        )
        for kind in ("timeout", "tail")
    )
    return (rollout, *values)

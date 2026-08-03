# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protocol-level tests for TrajectoryChannel without a Ray cluster."""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from inspect import unwrap
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.actor.fsdp_rlt_ac_policy_worker as rlt_ac_worker
from rlinf.data.embodied_io_struct import (
    EnvOutput,
    EnvResult,
    LeRobotStepResult,
    PolicyInput,
    PolicyOutput,
    RewardRequest,
    RewardResult,
    RolloutResult,
    ValueRequest,
    ValueResult,
)
from rlinf.scheduler.channel.trajectory_channel.channel import (
    AsyncTakeWork,
    TrajectoryChannel,
)
from rlinf.scheduler.channel.trajectory_channel.compression import (
    CompressionConfig,
    TensorCompressor,
)
from rlinf.scheduler.channel.trajectory_channel.data_route import get_data_routes
from rlinf.scheduler.channel.trajectory_channel.owner_key import BatchKey
from rlinf.scheduler.channel.trajectory_channel.storage import (
    LeRobotEpisodeBatch,
    PipelineMicroBatch,
    TrajectoryBatch,
    TrajectoryBatchContext,
    create_embodied_progress,
)
from rlinf.scheduler.channel.trajectory_channel.workers import (
    PackedTrajectoryData,
    QueueKey,
    TrajectoryChannelWorker,
    TrajectoryControllerWorker,
    TrajectoryStorageWorker,
)
from rlinf.scheduler.collective.async_work import AsyncWork
from rlinf.scheduler.manager import WorkerAddress
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.api_reward_worker import EmbodiedAPIRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _RemoteMethod:
    """Expose an async callable through the small ``.remote`` worker surface."""

    def __init__(self, callback: Callable[..., Any]) -> None:
        self._callback = callback

    async def remote(self, *args: Any) -> Any:
        return self._callback(*args)


class _CompletedWork(AsyncWork):
    async def async_wait(self) -> None:
        return None

    def wait(self) -> None:
        return None

    def done(self) -> bool:
        return True


class _TrajectoryTake:
    """Return one trajectory batch through the asynchronous take surface."""

    def __init__(self, batch: Any) -> None:
        self._batch = batch

    async def async_wait(self) -> Any:
        return self._batch


class _TrajectoryInputChannel:
    """Expose the minimal TrajectoryChannel consumer API used by RLT."""

    def __init__(self, batch: Any) -> None:
        self._batch = batch
        self.received_type: type | None = None

    def take(self, data_type: type, *, async_op: bool) -> _TrajectoryTake:
        assert async_op
        self.received_type = data_type
        return _TrajectoryTake(self._batch)


def _controller() -> TrajectoryControllerWorker:
    """Build the controller state that normally exists inside a Ray worker."""
    controller = object.__new__(TrajectoryControllerWorker)
    controller._routes = get_data_routes("ppo")
    controller._availability_queues = defaultdict(asyncio.Queue)
    controller._failure = None
    controller._failure_event = asyncio.Event()
    controller._owners = {}
    controller._active_owners = defaultdict(int)
    controller._owner_cursor = 0
    return controller


def _record_fields(**overrides: Any) -> dict[str, Any]:
    fields = {
        "global_step": 4,
        "actor_rank": 1,
        "pipeline_stage": 0,
        "rollout_epoch": 0,
        "chunk_step": 0,
        "slot_ids": (0,),
    }
    fields.update(overrides)
    return fields


def test_trajectory_records_normalize_tensors() -> None:
    tensor = torch.randn(3, 4, requires_grad=True).transpose(0, 1)
    record = RolloutResult(
        **_record_fields(),
        actions=tensor,
        forward_inputs={"states": tensor},
    )
    policy_input = PolicyInput(
        **_record_fields(),
        env_rank=0,
        observations={"nested": {"states": tensor}},
    )

    for value in (
        record.actions,
        record.forward_inputs["states"],
        policy_input.observations["nested"]["states"],
    ):
        assert value.device.type == "cpu"
        assert value.is_contiguous()
        assert not value.requires_grad


def test_value_requests_use_policy_observation_layout() -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"add_value_head": True}},
            "algorithm": {"bootstrap_type": "standard"},
            "env": {"train": {"auto_reset": True}},
        }
    )
    worker.n_train_chunk_steps = 1
    output = EnvOutput(
        obs={"main_images": torch.ones(2, 1)},
        dones=torch.zeros(2, 1, dtype=torch.bool),
        terminations=torch.zeros(2, 1, dtype=torch.bool),
        truncations=torch.zeros(2, 1, dtype=torch.bool),
    )

    requests = worker._value_requests(
        _record_fields(slot_ids=(0, 1)),
        output,
        slice(None),
        output.obs,
        chunk_step=0,
    )

    assert len(requests) == 1
    assert set(requests[0].observations) == {
        "main_images",
        "wrist_images",
        "extra_view_images",
        "states",
        "task_descriptions",
    }


def test_evaluation_shards_follow_rollout_mapping() -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"env": {"eval": {"total_num_envs": 16}}})
    worker._rank = 0
    worker._world_size = 2
    worker.stage_num = 1
    worker._component_placement = SimpleNamespace(
        get_world_size=lambda component: 4 if component == "rollout" else None
    )

    shards = worker._evaluation_shards(stage_id=0)

    assert [(shard.actor_rank, shard.slot_ids) for shard in shards] == [
        (0, (0, 1, 2, 3)),
        (1, (4, 5, 6, 7)),
    ]


def test_value_consumer_reports_background_failures() -> None:
    class _FailingChannel:
        def __init__(self) -> None:
            self.error: BaseException | None = None

        def take(self, data_type: type, *, async_op: bool) -> _TrajectoryTake:
            assert data_type is ValueRequest
            assert async_op
            return _TrajectoryTake(SimpleNamespace(observations={}))

        async def report_failure(self, error: BaseException) -> None:
            self.error = error

    worker = object.__new__(MultiStepRolloutWorker)
    worker._predict_rollout_actions = lambda observations: (_ for _ in ()).throw(
        KeyError("missing input")
    )
    channel = _FailingChannel()

    with pytest.raises(KeyError, match="missing input"):
        asyncio.run(
            unwrap(MultiStepRolloutWorker._consume_value_requests)(worker, channel)
        )

    assert isinstance(channel.error, KeyError)


@pytest.mark.parametrize(
    ("algorithm", "record_types"),
    (
        (
            "ppo",
            {
                EnvResult,
                RolloutResult,
                ValueRequest,
                ValueResult,
                RewardRequest,
                RewardResult,
            },
        ),
        (
            "nft",
            {
                EnvResult,
                RolloutResult,
                ValueRequest,
                ValueResult,
                RewardRequest,
                RewardResult,
            },
        ),
        (
            "opd",
            {
                EnvResult,
                RolloutResult,
                ValueRequest,
                ValueResult,
                RewardRequest,
                RewardResult,
            },
        ),
        ("sac", {EnvResult, RolloutResult}),
        ("grpo", {EnvResult, RolloutResult, RewardRequest, RewardResult}),
        ("dsrl", {EnvResult, RolloutResult}),
        ("dagger", {EnvResult, RolloutResult, LeRobotStepResult}),
        ("rlt_ac", {EnvResult, RolloutResult}),
    ),
)
def test_routes_keep_policy_traffic_separate_from_training_records(
    algorithm: str, record_types: set[type]
) -> None:
    routes = get_data_routes(algorithm)

    assert routes[PolicyInput].src == "env"
    assert routes[PolicyInput].dst == "rollout"
    assert routes[PolicyInput].via == "channel_worker"
    assert routes[PolicyOutput].src == "rollout"
    assert routes[PolicyOutput].dst == "env"
    assert routes[PolicyOutput].via == "channel_worker"
    assert routes[PolicyOutput].extra_key == "route_key"
    assert routes[TrajectoryBatch].src is None
    assert routes[TrajectoryBatch].dst == "actor"
    assert routes[TrajectoryBatch].via == "storage_worker"

    for data_type in record_types:
        route = routes[data_type]
        assert route.via == "storage_worker"
        if data_type in {ValueRequest, RewardRequest}:
            assert route.dst in {"rollout", "reward"}
        else:
            assert route.dst is None
            assert route.owner_key is not None

    if algorithm == "dagger":
        assert routes[LeRobotEpisodeBatch].src is None
        assert routes[LeRobotEpisodeBatch].dst == "actor"


@pytest.mark.parametrize(
    "loss_type",
    (
        "actor_critic",
        "decoupled_actor_critic",
        "embodied_sac",
        "embodied_dagger",
        "embodied_nft",
    ),
)
def test_loss_types_do_not_select_trajectory_protocols(loss_type: str) -> None:
    with pytest.raises(ValueError, match="algorithm type"):
        get_data_routes(loss_type)


def test_rlt_ingests_trajectory_channel_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RLT consumes TC batches through its transition-replay ingestion path."""
    trajectory = object()
    batch = type("Batch", (), {"to_trajectory": lambda self, cfg: trajectory})()
    channel = _TrajectoryInputChannel(batch)
    policy = object.__new__(RLTACFSDPPolicy)
    policy.cfg = object()
    ingested: list[list[object]] = []
    counters: list[tuple[int, int]] = []
    policy._ingest_rollout_trajectories = lambda trajectories: (
        ingested.append(trajectories) or (3, 1)
    )
    policy._update_rollout_ingest_counters = lambda added, completed: counters.append(
        (added, completed)
    )
    monkeypatch.setattr(rlt_ac_worker, "clear_memory", lambda **_: None)

    asyncio.run(unwrap(RLTACFSDPPolicy.recv_rollout_trajectories)(policy, channel))

    assert channel.received_type is TrajectoryBatch
    assert ingested == [[trajectory]]
    assert counters == [(3, 1)]


class _Placement:
    """Minimal component placement used to test request partitioning."""

    def __init__(self, actor_world_size: int, env_world_size: int):
        self.components = ["actor", "env", "rollout"]
        self._world_sizes = {
            "actor": actor_world_size,
            "env": env_world_size,
            "rollout": 3,
        }

    def get_world_size(self, component: str) -> int:
        return self._world_sizes[component]


def test_eval_requests_are_partitioned_across_rollout_ranks() -> None:
    total_num_envs = 12
    rollout_world_size = 3
    env_world_size = 2
    pipeline_stages = 2
    counts = []
    for rank in range(rollout_world_size):
        worker = object.__new__(MultiStepRolloutWorker)
        worker.placement = _Placement(actor_world_size=4, env_world_size=env_world_size)
        worker.num_pipeline_stages = pipeline_stages
        worker._rank = rank
        counts.append(worker._policy_request_count(total_num_envs, rollout_world_size))

    assert sum(counts) == env_world_size * pipeline_stages
    assert max(counts) - min(counts) <= 1


def test_policy_output_publish_and_take_use_the_same_env_partition() -> None:
    channel = object.__new__(TrajectoryChannel)
    route = get_data_routes("ppo")[PolicyOutput]
    output = PolicyOutput(
        **_record_fields(),
        env_rank=3,
        mode="eval",
        actions=torch.ones(1, 1),
    )

    published_key = channel._get_queue_key(route, PolicyOutput, output)
    consumed_key = channel._get_queue_key(
        route,
        PolicyOutput,
        partition=(3, "eval"),
    )

    assert published_key == consumed_key
    with pytest.raises(ValueError, match="requires an explicit partition"):
        channel._get_queue_key(route, PolicyOutput)


def test_pipeline_routes_use_epoch_scoped_storage_and_actor_partitions() -> None:
    routes = get_data_routes("ppo", use_training_pipeline=True)
    record = EnvResult(
        **_record_fields(rollout_epoch=2),
        rewards=torch.ones(1, 1),
        dones=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
    )

    assert PipelineMicroBatch in routes
    assert TrajectoryBatch not in routes
    assert routes[PipelineMicroBatch].extra_key == "actor_rank"
    owner_key = routes[EnvResult].owner_key(record, WorkerAddress("env", 0))
    assert owner_key.global_step == record.global_step
    assert owner_key.rollout_epoch == 2
    assert owner_key.actor_rank == record.actor_rank


def test_controller_reserves_only_the_matching_partition() -> None:
    async def run() -> None:
        controller = _controller()
        worker = WorkerAddress("trajectory-channel", 2)
        train_key = QueueKey(
            data_type=PolicyOutput,
            src="rollout",
            dst="env",
            extra_key=(0, "train"),
        )
        eval_key = QueueKey(
            data_type=PolicyOutput,
            src="rollout",
            dst="env",
            extra_key=(0, "eval"),
        )

        await controller.notify_available(eval_key, worker)
        train_reserve = asyncio.create_task(controller.reserve(train_key))
        await asyncio.sleep(0)
        assert not train_reserve.done()

        await controller.notify_available(train_key, worker)
        assert await train_reserve == worker
        assert await controller.reserve(eval_key) == worker

    asyncio.run(run())


def test_controller_pins_each_batch_owner_and_rebalances_after_delivery() -> None:
    controller = _controller()
    workers = tuple(WorkerAddress("trajectory-storage", rank) for rank in range(2))
    first_key = BatchKey(global_step=1, actor_rank=0)
    second_key = BatchKey(global_step=2, actor_rank=0)
    third_key = BatchKey(global_step=3, actor_rank=0)

    first_owner = controller.claim_storage_worker(first_key, (), workers)
    second_owner = controller.claim_storage_worker(second_key, (), workers)

    assert first_owner != second_owner
    assert controller.claim_storage_worker(first_key, (), workers) == first_owner

    controller.release_storage_worker(first_key)
    assert controller.claim_storage_worker(third_key, (), workers) == first_owner


def test_controller_failure_unblocks_a_waiting_consumer() -> None:
    async def run() -> None:
        controller = _controller()
        queue_key = QueueKey(data_type=PolicyInput, src="env", dst="rollout")
        reserve = asyncio.create_task(controller.reserve(queue_key))
        await asyncio.sleep(0)

        controller.report_failure(RuntimeError("record failed"))

        with pytest.raises(
            RuntimeError, match="Trajectory channel storage worker failed"
        ):
            await reserve

    asyncio.run(run())


def test_storage_failure_is_propagated_to_waiting_consumers() -> None:
    class FailingStorage:
        async def take(self) -> TrajectoryBatch:
            raise ValueError("record failed")

    async def run() -> None:
        controller = _controller()
        controller_proxy = type("ControllerProxy", (), {})()
        controller_proxy.report_failure = _RemoteMethod(controller.report_failure)
        worker = object.__new__(TrajectoryStorageWorker)
        worker._storage = FailingStorage()
        worker._controller_worker = controller_proxy

        with pytest.raises(ValueError, match="record failed"):
            await worker._publish_completed_outputs()
        with pytest.raises(
            RuntimeError, match="Trajectory channel storage worker failed"
        ):
            await controller.reserve(
                QueueKey(data_type=PolicyInput, src="env", dst="rollout")
            )

    asyncio.run(run())


def test_trajectory_reward_service_uses_api_reward_override() -> None:
    class ResultWork:
        def __init__(self, result: Any = None, error: BaseException | None = None):
            self._result = result
            self._error = error

        async def async_wait(self) -> Any:
            if self._error is not None:
                raise self._error
            return self._result

    class Channel:
        def __init__(self, request: RewardRequest):
            self._request = request
            self._taken = False
            self.published: list[RewardResult] = []

        def take(self, *_: Any, **__: Any) -> ResultWork:
            if self._taken:
                return ResultWork(error=asyncio.CancelledError())
            self._taken = True
            return ResultWork(self._request)

        def publish(self, result: RewardResult, **_: Any) -> ResultWork:
            self.published.append(result)
            return ResultWork()

    async def run() -> None:
        worker = object.__new__(EmbodiedAPIRewardWorker)
        worker.enable_offload = False
        worker.interval_reward = 0.25
        worker.gt_success_bonus = 0.0
        worker.input_builder = type(
            "InputBuilder",
            (),
            {"get_valid_input_ids": lambda *_: []},
        )()
        request = RewardRequest(
            **_record_fields(slot_ids=(0, 1)),
            mode="per_step",
            inputs={"history_input": {"frames": {"camera": [[], []]}}},
        )
        channel = Channel(request)

        with pytest.raises(asyncio.CancelledError):
            await worker._serve_reward_requests(channel, channel)

        assert len(channel.published) == 1
        result = channel.published[0]
        assert result.mode == "per_step"
        torch.testing.assert_close(result.rewards, torch.full((2, 1), 0.25))

    asyncio.run(run())


def test_channel_worker_delivers_one_packed_message_and_advertises_it() -> None:
    class Controller:
        def __init__(self) -> None:
            self.notifications: list[tuple[QueueKey, WorkerAddress]] = []
            self.notify_available = _RemoteMethod(
                lambda key, address: self.notifications.append((key, address))
            )

    async def run() -> None:
        controller = Controller()
        worker = object.__new__(TrajectoryChannelWorker)
        worker._routes = {
            PolicyOutput: get_data_routes("ppo")[PolicyOutput],
        }
        worker._max_queue_size = 0
        worker._queues = {}
        worker._controller_worker = controller
        worker._worker_address = WorkerAddress("trajectory-channel", 0)

        queue_key = QueueKey(
            data_type=PolicyOutput,
            src="rollout",
            dst="env",
            extra_key=(1, "train"),
        )
        packed = PackedTrajectoryData(skeleton={"mode": "train"}, tensors={})
        worker.recv = lambda *_: (packed, queue_key)
        sent: list[tuple[PackedTrajectoryData, int]] = []
        worker.send = lambda **kwargs: (
            sent.append((kwargs["object"], kwargs["piggyback_payload"])),
            _CompletedWork(),
        )[1]

        await worker.publish(WorkerAddress("rollout", 0))
        await worker.take(WorkerAddress("env", 0), queue_key, query_id=19)

        assert controller.notifications == [(queue_key, worker.worker_address)]
        assert sent == [(packed, 19)]

    asyncio.run(run())


def test_storage_worker_releases_a_batch_owner_only_after_delivery() -> None:
    class Controller:
        def __init__(self) -> None:
            self.notifications: list[tuple[QueueKey, WorkerAddress]] = []
            self.released: list[BatchKey] = []
            self.failures: list[BaseException] = []
            self.notification_event = asyncio.Event()
            self.notify_available = _RemoteMethod(self._notify)
            self.release_storage_worker = _RemoteMethod(self.released.append)
            self.report_failure = _RemoteMethod(self.failures.append)

        def _notify(self, key: QueueKey, address: WorkerAddress) -> None:
            self.notifications.append((key, address))
            self.notification_event.set()

    class Storage:
        def __init__(self, output: TrajectoryBatch) -> None:
            self._outputs = asyncio.Queue()
            self._outputs.put_nowait(output)

        async def take(self) -> TrajectoryBatch:
            return await self._outputs.get()

    async def run() -> None:
        output = TrajectoryBatch.create(
            global_step=4,
            actor_rank=1,
            context=TrajectoryBatchContext(
                rollout_epochs=1,
                chunk_steps=1,
                slot_ids=(0,),
            ),
            progress_factory=create_embodied_progress,
        )
        output._seal()
        controller = Controller()
        worker = object.__new__(TrajectoryStorageWorker)
        route = get_data_routes("ppo")[TrajectoryBatch]
        queue_key = QueueKey(
            data_type=TrajectoryBatch,
            src=route.src,
            dst=route.dst,
        )
        worker._routes = {TrajectoryBatch: route}
        worker._queues = {}
        worker._tensor_compressors = {}
        worker._max_queue_size = 0
        worker._storage = Storage(output)
        worker._controller_worker = controller
        worker._worker_address = WorkerAddress("trajectory-storage", 0)
        sent: list[PackedTrajectoryData] = []
        worker.send = lambda **kwargs: (
            sent.append(kwargs["object"]),
            _CompletedWork(),
        )[1]

        output_task = asyncio.create_task(worker._publish_completed_outputs())
        try:
            await asyncio.wait_for(controller.notification_event.wait(), timeout=1)
            assert controller.released == []

            await worker.take(WorkerAddress("actor", 0), queue_key, query_id=7)

            assert len(sent) == 1
            assert controller.released == [BatchKey(global_step=4, actor_rank=1)]
            assert controller.failures == []
        finally:
            output_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await output_task

    asyncio.run(run())


def test_compression_round_trip_preserves_tensors_and_raw_fallback() -> None:
    compressor = TensorCompressor(
        CompressionConfig(
            min_bytes=16,
            block_bytes=32,
            num_threads=1,
            max_inflight=1,
            pin_memory=False,
        )
    )
    tensors = {
        ("compressed",): torch.zeros(128, dtype=torch.uint8),
        ("small",): torch.arange(1, dtype=torch.int64),
    }

    output = compressor.compress(tensors)
    try:
        assert ("compressed",) in output.metadata
        assert ("small",) not in output.metadata
    finally:
        if output.lease is not None:
            output.lease.release()
    restored = compressor.decompress(output.tensors, output.metadata)

    assert output.stats.wire_bytes <= output.stats.raw_bytes
    assert restored.keys() == tensors.keys()
    for key, tensor in tensors.items():
        torch.testing.assert_close(restored[key], tensor)


def test_cancelling_async_take_releases_its_result_slot() -> None:
    async def run() -> None:
        reserve_ref = asyncio.get_running_loop().create_future()
        work = AsyncTakeWork(
            reserve_ref=reserve_ref,
            channel=object(),
            queue_key=object(),
        )
        take = asyncio.create_task(work.async_wait())
        await asyncio.sleep(0)
        take.cancel()

        with pytest.raises(asyncio.CancelledError):
            await take

        assert work._query_id not in AsyncTakeWork._result_store

    asyncio.run(run())

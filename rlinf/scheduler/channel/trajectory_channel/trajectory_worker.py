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


import asyncio
from collections import defaultdict
from typing import TypeAlias

import torch
from omegaconf import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
    EmbodiedTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvResult,
    convert_trajectories_to_batch,
    split_batch_value,
    split_episode_data,
)
from rlinf.scheduler.channel.trajectory_channel.data import (
    TrajectoryEnd,
    TrajectoryEpochEnd,
    TrajectorySegment,
)
from rlinf.scheduler.worker.routing import CommMapper
from rlinf.scheduler.worker.worker import Worker, WorkerAddress
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)

TrajectoryCollector: TypeAlias = (
    EmbodiedTrajectoryBuilder | EmbodiedLerobotTrajectoryBuilder
)
SourceID: TypeAlias = tuple[int, int]


class TrajectoryWorker(Worker):
    """Assemble rollout segments and serve actor-ready training data."""

    def __init__(self, cfg: DictConfig, max_size: int = 0):
        """Initialize trajectory builders and output queues."""
        super().__init__()
        self._cfg = cfg

        self._collectors: dict[SourceID, TrajectoryCollector] = {}
        self._step_collectors: dict[int, dict[SourceID, TrajectoryCollector]] = {}
        self._pipeline_collectors: dict[
            SourceID, dict[SourceID, TrajectoryCollector]
        ] = {}
        self._finished_sources: dict[int, set[SourceID]] = {}
        self._finished_epoch_sources: dict[SourceID, set[SourceID]] = {}
        self._pipeline_generators: dict[SourceID, torch.Generator] = {}

        self._output_queues: dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_size)
        )

        from rlinf.scheduler.cluster import Cluster
        from rlinf.utils.placement import HybridComponentPlacement

        self._placement = HybridComponentPlacement(self._cfg, Cluster())

        rollout_ws = self._placement.get_world_size("rollout")
        env_ws = self._placement.get_world_size("env")
        actor_ws = self._placement.get_world_size("actor")
        self.producer_count = rollout_ws * self._cfg.rollout.pipeline_stage_num
        self.source_count = env_ws * self._cfg.rollout.pipeline_stage_num
        from rlinf.utils.metric_utils import compute_split_num

        self.output_count = (
            compute_split_num(env_ws * self._cfg.rollout.pipeline_stage_num, actor_ws)
            * actor_ws
        )

        if self.output_count % self.source_count:
            raise ValueError(
                "Trajectory routing requires each rollout source to have an equal "
                "number of actor shards."
            )
        self.shards_per_source = self.output_count // self.source_count
        self.collect_prev_infos = self._cfg.rollout.get("collect_prev_infos", True)
        self.collect_transitions = self._cfg.rollout.get("collect_transitions", False)
        self.enable_rlt = self._cfg.algorithm.get("loss_type") == "rlt_ac"
        online_lerobot_cfg = self._cfg.algorithm.get("dagger", {}).get(
            "online_lerobot", {}
        )
        self.enable_online_lerobot = bool(online_lerobot_cfg.get("enabled", False))
        self.lerobot_only_success = bool(online_lerobot_cfg.get("only_success", False))
        self.env_reward_weight = self._cfg.get("reward", {}).get(
            "env_reward_weight", 1.0
        )
        self.reward_weight = self._cfg.get("reward", {}).get("reward_weight", 1.0)
        self.use_training_pipeline = self._cfg.runner.get(
            "use_training_pipeline", False
        )
        self.shuffle_rollout = self._cfg.algorithm.get("shuffle_rollout", True)
        if self.use_training_pipeline and self.enable_online_lerobot:
            raise ValueError(
                "Training pipeline does not support online LeRobot trajectory data."
            )
        if self.use_training_pipeline and self._cfg.runner.get(
            "enable_decoupled_mode", False
        ):
            raise ValueError(
                "Training pipeline does not support decoupled environment mode."
            )

    def _collector(
        self, collectors: dict[SourceID, TrajectoryCollector], source: SourceID
    ) -> TrajectoryCollector:
        if source not in collectors:
            if self.enable_online_lerobot:
                collectors[source] = EmbodiedLerobotTrajectoryBuilder(
                    max_episode_length=self._cfg.env.train.max_episode_steps,
                    num_envs=self._cfg.env.train.total_num_envs // self.source_count,
                    only_success=self.lerobot_only_success,
                    num_action_chunks=self._cfg.actor.model.num_action_chunks,
                    action_dim=self._cfg.actor.model.action_dim,
                )
            else:
                collectors[source] = EmbodiedTrajectoryBuilder(
                    max_episode_length=self._cfg.env.train.max_episode_steps
                )
        return collectors[source]

    def _rewards(self, segment: TrajectorySegment) -> torch.Tensor | None:
        rewards = segment.env_result.rewards
        if rewards is None:
            return None
        if segment.env_result.reward_model_output is not None:
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight
                * segment.env_result.reward_model_output.to(rewards.dtype)
            )
        values = segment.rollout_result.bootstrap_values
        if (
            values is None
            or not self._cfg.env.train.auto_reset
            or segment.env_result.dones is None
        ):
            return rewards
        truncations = segment.env_result.truncations
        if self._cfg.algorithm.get("bootstrap_type", "standard") != "standard":
            truncations = segment.env_result.dones
        if truncations is None or not truncations[:, -1].any():
            return rewards
        rewards = rewards.clone()
        mask = truncations[:, -1]
        rewards[mask, -1] += self._cfg.algorithm.get("gamma", 1.0) * values[
            mask
        ].reshape(-1).to(rewards.dtype)
        return rewards

    def _append_final_values(self, event: TrajectoryEpochEnd) -> None:
        if self.enable_online_lerobot or not self.collect_prev_infos:
            return
        values = split_batch_value(
            event.final_prev_values, [size for _, _, size in event.sources]
        )
        for (rank, stage, _), value in zip(event.sources, values):
            if self.use_training_pipeline:
                collectors = self._pipeline_collectors[(event.step_id, event.epoch_id)]
            else:
                collectors = self._step_collectors.setdefault(event.step_id, {})
            self._collector(collectors, (rank, stage)).append_final_value(value)

    async def _flush(self, step_id: int) -> None:
        finished_sources = self._finished_sources[step_id]
        if len(finished_sources) != self.producer_count:
            return

        collectors = self._collectors
        if not self.enable_online_lerobot:
            collectors = self._step_collectors.pop(step_id)
        if len(collectors) != self.source_count:
            raise ValueError(
                f"Expected {self.source_count} collectors, but got {len(collectors)}."
            )
        if self.enable_online_lerobot:
            for collector in collectors.values():
                if not isinstance(collector, EmbodiedLerobotTrajectoryBuilder):
                    raise ValueError(
                        "Expected collector to be an instance of EmbodiedLerobotTrajectoryBuilder."
                    )
                episodes = collector.drain_episodes()
                for shard in range(self.shards_per_source):
                    episode_chunk = episodes[shard :: self.shards_per_source]
                    await self._output_queues["default"].put(episode_chunk)
        else:
            for collector in collectors.values():
                for trajectory in collector.to_splited_trajectories(
                    self.shards_per_source
                ):
                    await self._output_queues["default"].put(trajectory)
        del self._finished_sources[step_id]

    async def _flush_pipeline(self, key: SourceID) -> None:
        if len(self._finished_epoch_sources[key]) != self.producer_count:
            return
        collectors = self._pipeline_collectors[key]
        if len(collectors) != self.source_count:
            raise ValueError(
                f"Expected {self.source_count} collectors, but got {len(collectors)}."
            )
        batches_by_actor: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
        for (rollout_rank, stage_id), collector in collectors.items():
            logical_rank = (
                rollout_rank * self._cfg.rollout.pipeline_stage_num + stage_id
            )
            actor_splits = CommMapper.get_dst_ranks(
                batch_size=self._cfg.env.train.total_num_envs,
                src_world_size=self.source_count,
                dst_world_size=self._placement.get_world_size("actor"),
                src_rank=logical_rank,
            )
            trajectories = collector.to_splited_trajectories_by_sizes(
                [size for _, size in actor_splits]
            )
            for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                batches_by_actor[actor_rank].append(
                    self._prepare_pipeline_batch(trajectory)
                )

        if self._cfg.algorithm.get("normalize_advantages", True):
            for batches in batches_by_actor.values():
                stats = sum(
                    masked_stats(batch["advantages"], batch.get("loss_mask"))
                    for batch in batches
                )
                for batch in batches:
                    batch["advantages"] = normalize_from_stats(
                        batch["advantages"], stats
                    )

        for actor_rank, batches in batches_by_actor.items():
            for batch in batches:
                for micro_batch in self._pipeline_micro_batches(batch, actor_rank):
                    await self._output_queues[f"actor:{actor_rank}"].put(micro_batch)
        del self._pipeline_collectors[key]
        del self._finished_epoch_sources[key]

    def _prepare_pipeline_batch(self, trajectory) -> dict[str, torch.Tensor]:
        batch = preprocess_embodied_batch(
            convert_trajectories_to_batch([trajectory]),
            rollout_epoch=1,
            auto_reset=self._cfg.env.train.auto_reset,
            ignore_terminations=self._cfg.env.train.ignore_terminations,
            reward_type=self._cfg.algorithm.reward_type,
            filter_rewards=self._cfg.algorithm.get("filter_rewards", False),
            group_size=self._cfg.algorithm.group_size,
            rewards_lower_bound=self._cfg.algorithm.get("rewards_lower_bound"),
            rewards_upper_bound=self._cfg.algorithm.get("rewards_upper_bound"),
        )
        batch.update(
            calculate_adv_and_returns(
                task_type=self._cfg.runner.task_type,
                adv_type=self._cfg.algorithm.adv_type,
                rewards=batch["rewards"],
                dones=batch["dones"],
                values=batch.get("prev_values"),
                prev_logprobs=batch.get("prev_logprobs"),
                num_action_chunks=self._cfg.actor.model.num_action_chunks,
                gamma=self._cfg.algorithm.get("gamma", 1),
                gae_lambda=self._cfg.algorithm.get("gae_lambda", 1),
                group_size=self._cfg.algorithm.get("group_size", 8),
                reward_type=self._cfg.algorithm.reward_type,
                loss_mask=batch.get("loss_mask"),
                loss_mask_sum=batch.get("loss_mask_sum"),
                normalize_advantages=False,
            )
        )
        return batch

    def _pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        generator = self._pipeline_generators.setdefault(
            (actor_rank, 0),
            torch.Generator().manual_seed(self._cfg.actor.seed + actor_rank),
        )
        indices = (
            torch.randperm(batch_size, generator=generator)
            if self.shuffle_rollout
            else torch.arange(batch_size)
        )
        flat_batch = flatten_embodied_batch(batch, indices)
        micro_batch_size = self._cfg.actor.micro_batch_size
        if batch_size % micro_batch_size:
            raise ValueError(
                f"Pipeline batch size {batch_size} is not divisible by "
                f"micro batch size {micro_batch_size}."
            )
        return [
            pack_batch(micro_batch)
            for micro_batch in split_dict_to_chunk(
                flat_batch, batch_size // micro_batch_size, dim=0
            )
        ]

    def _split_segments(self, segment: TrajectorySegment):
        sizes = [size for _, _, size in segment.sources]
        if not sizes:
            raise ValueError("Trajectory segment has no logical source metadata.")
        result_fields = {
            name: split_batch_value(getattr(segment.rollout_result, name), sizes)
            for name in segment.rollout_result.__dataclass_fields__
        }
        env_fields = {
            name: split_batch_value(getattr(segment.env_result, name), sizes)
            for name in segment.env_result.__dataclass_fields__
            if name != "episode_data"
        }
        initial_env_fields = (
            {
                name: split_batch_value(
                    getattr(segment.initial_env_result, name), sizes
                )
                for name in segment.initial_env_result.__dataclass_fields__
                if name != "episode_data"
            }
            if segment.initial_env_result is not None
            else None
        )
        episode_data = split_episode_data(segment.env_result.episode_data, sizes)
        for index, (rank, stage, _) in enumerate(segment.sources):
            yield (
                (rank, stage),
                TrajectorySegment(
                    step_id=segment.step_id,
                    epoch_id=segment.epoch_id,
                    sources=[(rank, stage, sizes[index])],
                    obs=split_batch_value(segment.obs, sizes)[index],
                    next_obs=split_batch_value(segment.next_obs, sizes)[index],
                    rollout_result=EmbodiedRolloutResult(
                        **{
                            name: values[index]
                            for name, values in result_fields.items()
                        }
                    ),
                    env_result=EnvResult(
                        **{name: values[index] for name, values in env_fields.items()},
                        episode_data=episode_data[index],
                    ),
                    initial_env_result=(
                        EnvResult(
                            **{
                                name: values[index]
                                for name, values in initial_env_fields.items()
                            }
                        )
                        if initial_env_fields is not None
                        else None
                    ),
                    forward_inputs=split_batch_value(segment.forward_inputs, sizes)[
                        index
                    ],
                ),
            )

    def _append(self, segment: TrajectorySegment) -> None:
        collectors = self._collectors
        if not self.enable_online_lerobot:
            collectors = self._step_collectors.setdefault(segment.step_id, {})
        if self.use_training_pipeline:
            key = (segment.step_id, segment.epoch_id)
            collectors = self._pipeline_collectors.setdefault(key, {})
        for source, source_segment in self._split_segments(segment):
            self._append_one(collectors, source, source_segment)

    def _append_one(
        self,
        collectors: dict[SourceID, TrajectoryCollector],
        source: SourceID,
        segment: TrajectorySegment,
    ) -> None:
        collector = self._collector(collectors, source)
        result = segment.rollout_result
        env_result = segment.env_result
        if self.enable_online_lerobot:
            if env_result.episode_data is None:
                raise ValueError("Online LeRobot segment is missing episode data.")
            assert isinstance(collector, EmbodiedLerobotTrajectoryBuilder)
            collector.append_chunk_episode_data(
                policy_output=result, **env_result.episode_data
            )
            return
        if segment.initial_env_result is not None:
            collector.append_initial_state(
                dones=segment.initial_env_result.dones,
                truncations=segment.initial_env_result.truncations,
                terminations=segment.initial_env_result.terminations,
            )
        collector.append_step_result(
            ChunkStepResult(
                actions=result.forward_inputs.get("action"),
                prev_logprobs=(
                    result.prev_logprobs if self.collect_prev_infos else None
                ),
                prev_values=(result.prev_values if self.collect_prev_infos else None),
                forward_inputs=result.forward_inputs,
                versions=result.versions,
                dones=env_result.dones,
                truncations=env_result.truncations,
                terminations=env_result.terminations,
                rewards=self._rewards(segment),
            )
        )
        if env_result.reward_assign_lengths is not None:
            rewards = collector.rewards
            reward = self.reward_weight * env_result.reward_model_output
            for env_id, length in enumerate(env_result.reward_assign_lengths):
                for offset in range(2, min(length, len(rewards)) + 1):
                    rewards[-offset][env_id] += reward[env_id].to(
                        rewards[-offset].dtype
                    )
        if env_result.intervene_actions is not None:
            collector.update_last_actions(
                env_result.intervene_actions, env_result.intervene_flags
            )
        if result.intervene_flags is not None:
            collector.mark_last_step_with_intervene_flags(result.intervene_flags)
        if self.enable_rlt:
            from rlinf.algorithms.rlt.transition import (
                extract_rlt_obs_from_forward_inputs,
            )

            collector.append_transitions(
                extract_rlt_obs_from_forward_inputs(result.forward_inputs),
                extract_rlt_obs_from_forward_inputs(
                    segment.forward_inputs, transition=True
                ),
            )
        elif self.collect_transitions:
            collector.append_transitions(segment.obs, segment.next_obs)

    async def publish(self, worker_address: WorkerAddress) -> None:
        """Receive and apply one trajectory event from a producer."""
        data = await self.recv(
            src_group_name=worker_address.root_group_name,
            src_rank=worker_address.rank,
            async_op=True,
        ).async_wait()
        if isinstance(data, TrajectorySegment):
            await asyncio.to_thread(self._append, data)
        elif isinstance(data, TrajectoryEnd):
            if not self.use_training_pipeline:
                self._finished_sources.setdefault(data.step_id, set()).add(data.source)
                await self._flush(data.step_id)
        elif isinstance(data, TrajectoryEpochEnd):
            self._append_final_values(data)
            if self.use_training_pipeline:
                key: SourceID = (data.step_id, data.epoch_id)
                self._finished_epoch_sources.setdefault(key, set()).add(data.source)
                await self._flush_pipeline(key)
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")

    async def subscribe(
        self, worker_address: WorkerAddress, queue_key: str, query_id: int
    ) -> None:
        """Send the next queued item to a subscriber."""
        data = await self._output_queues[queue_key].get()
        await self.send(
            object=data,
            dst_group_name=worker_address.root_group_name,
            dst_rank=worker_address.rank,
            piggyback_payload=query_id,
            async_op=True,
        ).async_wait()

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
from dataclasses import dataclass
from typing import Any, TypeAlias

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
    TrajectoryKey,
    TrajectorySource,
    convert_trajectories_to_batch,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_episode_data,
)
from rlinf.scheduler.channel.trajectory_channel.data import (
    EnvStepResult,
    PolicyStep,
    TerminalResult,
    TrajectoryEnd,
    TrajectoryEpochEnd,
    TrajectoryStart,
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


@dataclass(kw_only=True)
class _TrajectoryStep:
    """Fully joined model and environment data for one action chunk."""

    step_id: int
    epoch_id: int
    obs: dict[str, Any]
    next_obs: dict[str, Any]
    env_result: EnvResult
    rollout_result: EmbodiedRolloutResult
    initial_env_result: EnvResult | None = None
    forward_inputs: dict[str, Any] | None = None


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
        self._policy_steps: dict[TrajectoryKey, PolicyStep] = {}
        self._env_results: dict[TrajectoryKey, EnvStepResult] = {}
        self._terminal_results: dict[TrajectoryKey, TerminalResult] = {}
        self._initial_results: dict[tuple[int, int, int, int], EnvResult] = {}
        self._completed_keys: set[TrajectoryKey] = set()
        self._fragments: dict[tuple[type, TrajectoryKey], list[object]] = defaultdict(
            list
        )
        self._event_lock = asyncio.Lock()

        self._output_queues: dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_size)
        )

        from rlinf.scheduler.cluster import Cluster
        from rlinf.utils.placement import HybridComponentPlacement

        self._placement = HybridComponentPlacement(self._cfg, Cluster())

        env_ws = self._placement.get_world_size("env")
        actor_ws = self._placement.get_world_size("actor")
        self.source_count = env_ws * self._cfg.rollout.pipeline_stage_num
        self.chunk_count = (
            self._cfg.env.train.max_steps_per_rollout_epoch
            // self._cfg.actor.model.num_action_chunks
        )
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

    def _rewards(self, segment: _TrajectoryStep) -> torch.Tensor | None:
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

    async def _flush(self, step_id: int) -> None:
        finished_sources = self._finished_sources[step_id]
        if len(finished_sources) != self.source_count:
            return
        expected = (
            self.source_count * self._cfg.env.train.rollout_epoch * self.chunk_count
        )
        if sum(key.step_id == step_id for key in self._completed_keys) != expected:
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
        self._completed_keys = {
            key for key in self._completed_keys if key.step_id != step_id
        }

    async def _flush_pipeline(self, key: SourceID) -> None:
        if len(self._finished_epoch_sources[key]) != self.source_count:
            return
        expected = self.source_count * self.chunk_count
        if (
            sum(
                item.step_id == key[0] and item.epoch_id == key[1]
                for item in self._completed_keys
            )
            != expected
        ):
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
        self._completed_keys = {
            item
            for item in self._completed_keys
            if not (item.step_id == key[0] and item.epoch_id == key[1])
        }

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

    @property
    def _source_batch_size(self) -> int:
        return self._cfg.env.train.total_num_envs // self.source_count

    def _split_event(self, event):
        sizes = [source.size for source in event.sources]
        if isinstance(event, PolicyStep):
            fields = {
                name: split_batch_value(getattr(event.rollout_result, name), sizes)
                for name in event.rollout_result.__dataclass_fields__
            }
            for index, source in enumerate(event.sources):
                yield PolicyStep(
                    sources=[source],
                    obs=split_batch_value(event.obs, sizes)[index],
                    rollout_result=EmbodiedRolloutResult(
                        **{name: values[index] for name, values in fields.items()}
                    ),
                )
            return
        if isinstance(event, EnvStepResult):
            fields = {
                name: split_batch_value(getattr(event.result, name), sizes)
                for name in event.result.__dataclass_fields__
                if name != "episode_data"
            }
            episodes = split_episode_data(event.result.episode_data, sizes)
            for index, source in enumerate(event.sources):
                yield EnvStepResult(
                    sources=[source],
                    result=EnvResult(
                        **{name: values[index] for name, values in fields.items()},
                        episode_data=episodes[index],
                    ),
                    needs_terminal=event.needs_terminal,
                )
            return
        if isinstance(event, TerminalResult):
            values = split_batch_value(event.bootstrap_values, sizes)
            inputs = split_batch_value(event.forward_inputs, sizes)
            observations = split_batch_value(event.obs, sizes)
            for index, source in enumerate(event.sources):
                yield TerminalResult(
                    sources=[source],
                    obs=observations[index],
                    bootstrap_values=values[index],
                    forward_inputs=inputs[index],
                )

    def _merge_fragments(self, event):
        source = event.sources[0]
        key = (type(event), source.key)
        fragments = self._fragments[key]
        fragments.append(event)
        received_size = sum(item.sources[0].size for item in fragments)
        if received_size < self._source_batch_size:
            return None
        if received_size > self._source_batch_size:
            raise ValueError(
                f"Trajectory fragments exceed source batch size for {source.key}."
            )
        fragments.sort(key=lambda item: item.sources[0].offset)
        offsets = [item.sources[0].offset for item in fragments]
        sizes = [item.sources[0].size for item in fragments]
        if any(offset != sum(sizes[:index]) for index, offset in enumerate(offsets)):
            raise ValueError(f"Non-contiguous trajectory fragments: {offsets}.")
        del self._fragments[key]
        full_source = TrajectorySource(source.key, self._source_batch_size)
        if isinstance(event, PolicyStep):
            return PolicyStep(
                sources=[full_source],
                obs=merge_batch_values([item.obs for item in fragments]),
                rollout_result=EmbodiedRolloutResult(
                    **{
                        name: merge_batch_values(
                            [getattr(item.rollout_result, name) for item in fragments]
                        )
                        for name in event.rollout_result.__dataclass_fields__
                    }
                ),
            )
        if isinstance(event, EnvStepResult):
            return EnvStepResult(
                sources=[full_source],
                result=EnvResult(
                    **{
                        name: merge_batch_values(
                            [getattr(item.result, name) for item in fragments]
                        )
                        for name in event.result.__dataclass_fields__
                        if name != "episode_data"
                    },
                    episode_data=(
                        merge_episode_data(
                            [item.result.episode_data for item in fragments]
                        )
                        if event.result.episode_data is not None
                        else None
                    ),
                ),
                needs_terminal=event.needs_terminal,
            )
        return TerminalResult(
            sources=[full_source],
            obs=merge_batch_values([item.obs for item in fragments]),
            bootstrap_values=merge_batch_values(
                [item.bootstrap_values for item in fragments]
            ),
            forward_inputs=merge_batch_values(
                [item.forward_inputs for item in fragments]
            ),
        )

    def _store_event(self, event) -> None:
        for fragment in self._split_event(event):
            complete = self._merge_fragments(fragment)
            if complete is None:
                continue
            key = complete.sources[0].key
            if key in self._completed_keys:
                raise ValueError(f"Received duplicate trajectory event for {key}.")
            if isinstance(complete, PolicyStep):
                self._policy_steps[key] = complete
                if key.chunk_id:
                    self._try_complete(
                        TrajectoryKey(
                            key.step_id,
                            key.epoch_id,
                            key.env_rank,
                            key.stage_id,
                            key.chunk_id - 1,
                        )
                    )
            elif isinstance(complete, EnvStepResult):
                self._env_results[key] = complete
            else:
                self._terminal_results[key] = complete
            self._try_complete(key)

    def _try_complete(self, key: TrajectoryKey) -> None:
        policy = self._policy_steps.get(key)
        env = self._env_results.get(key)
        if policy is None or env is None:
            return
        initial_key = (key.step_id, key.epoch_id, key.env_rank, key.stage_id)
        if key.chunk_id == 0 and initial_key not in self._initial_results:
            return
        terminal = self._terminal_results.get(key)
        next_policy = self._policy_steps.get(
            TrajectoryKey(
                key.step_id,
                key.epoch_id,
                key.env_rank,
                key.stage_id,
                key.chunk_id + 1,
            )
        )
        if self.enable_online_lerobot:
            next_obs = {}
            next_inputs = None
        elif env.needs_terminal:
            if terminal is None:
                return
            next_obs = terminal.obs
            next_inputs = terminal.forward_inputs
            policy.rollout_result.bootstrap_values = terminal.bootstrap_values
        else:
            if next_policy is None:
                return
            next_obs = next_policy.obs
            next_inputs = next_policy.rollout_result.forward_inputs

        segment = _TrajectoryStep(
            step_id=key.step_id,
            epoch_id=key.epoch_id,
            obs=policy.obs,
            next_obs=next_obs,
            rollout_result=policy.rollout_result,
            env_result=env.result,
            initial_env_result=self._initial_results.pop(initial_key, None),
            forward_inputs=next_inputs,
        )
        self._append_one(
            collectors := self._collectors_for(segment),
            (key.env_rank, key.stage_id),
            segment,
        )
        if (
            key.chunk_id == self.chunk_count - 1
            and not self.enable_online_lerobot
            and self.collect_prev_infos
        ):
            self._collector(
                collectors, (key.env_rank, key.stage_id)
            ).append_final_value(terminal.bootstrap_values)
        self._completed_keys.add(key)
        del self._policy_steps[key]
        del self._env_results[key]
        self._terminal_results.pop(key, None)

    def _collectors_for(
        self, segment: _TrajectoryStep
    ) -> dict[SourceID, TrajectoryCollector]:
        if self.use_training_pipeline:
            return self._pipeline_collectors.setdefault(
                (segment.step_id, segment.epoch_id), {}
            )
        if self.enable_online_lerobot:
            return self._collectors
        return self._step_collectors.setdefault(segment.step_id, {})

    def _append_one(
        self,
        collectors: dict[SourceID, TrajectoryCollector],
        source: SourceID,
        segment: _TrajectoryStep,
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
                apply_rlt_interventions,
                extract_rlt_obs_from_forward_inputs,
            )

            current_obs = extract_rlt_obs_from_forward_inputs(result.forward_inputs)
            apply_rlt_interventions(
                current_obs,
                env_result.intervene_actions,
                env_result.intervene_flags,
            )
            collector.append_transitions(
                current_obs,
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
        async with self._event_lock:
            await self._apply_event(data)

    async def _apply_event(self, data) -> None:
        """Apply one received event while holding the assembly lock."""
        if isinstance(data, (PolicyStep, EnvStepResult, TerminalResult)):
            self._store_event(data)
            step_ids = {source.key.step_id for source in data.sources}
            for step_id in step_ids:
                if step_id in self._finished_sources:
                    await self._flush(step_id)
            for source in data.sources:
                epoch_key = (source.key.step_id, source.key.epoch_id)
                if epoch_key in self._finished_epoch_sources:
                    await self._flush_pipeline(epoch_key)
        elif isinstance(data, TrajectoryStart):
            key = data.source.key
            self._initial_results[
                (
                    key.step_id,
                    key.epoch_id,
                    key.env_rank,
                    key.stage_id,
                )
            ] = data.result
            self._try_complete(key)
        elif isinstance(data, TrajectoryEnd):
            if not self.use_training_pipeline:
                self._finished_sources.setdefault(data.step_id, set()).add(data.source)
                await self._flush(data.step_id)
        elif isinstance(data, TrajectoryEpochEnd):
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

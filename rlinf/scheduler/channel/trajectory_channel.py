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
from typing import TYPE_CHECKING

import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedLerobotRolloutResult,
    EmbodiedRolloutResult,
    EnvResult,
    RolloutResult,
    TrajectoryEnd,
    TrajectoryEpochEnd,
    TrajectorySegment,
    convert_trajectories_to_batch,
    split_batch_value,
    split_episode_data,
)

from ..cluster import Cluster
from ..placement import NodePlacementStrategy
from ..worker import Worker
from .channel import Channel

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TrajectoryWorker(Worker):
    """Own the trajectory collector for one :class:`TrajectoryChannel`."""

    def __init__(self, cfg: "DictConfig", channel_name: str):
        """Create storage for the trajectory path identified by ``channel_name``."""
        super().__init__()
        self.cfg = cfg
        self.channel_name = channel_name
        self.collectors: dict[tuple[int, int], EmbodiedRolloutResult] = {}
        self.finished_sources: set[tuple[int, int]] = set()
        self.pipeline_collectors: dict[
            tuple[int, int], dict[tuple[int, int], EmbodiedRolloutResult]
        ] = {}
        self.finished_epoch_sources: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.pipeline_generators: dict[tuple[int, int], torch.Generator] = {}

        placement = self._component_placement()
        from rlinf.utils.metric_utils import compute_split_num

        self.producer_count = (
            placement.get_world_size("rollout") * cfg.rollout.pipeline_stage_num
        )
        self.source_count = (
            placement.get_world_size("env") * cfg.rollout.pipeline_stage_num
        )
        self.output_count = compute_split_num(
            placement.get_world_size("env") * cfg.rollout.pipeline_stage_num,
            placement.get_world_size("actor"),
        ) * placement.get_world_size("actor")
        if self.output_count % self.source_count:
            raise ValueError(
                "Trajectory routing requires each rollout source to have an equal "
                "number of actor shards."
            )
        self.shards_per_source = self.output_count // self.source_count

        self.collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)
        self.collect_transitions = cfg.rollout.get("collect_transitions", False)
        self.enable_rlt = cfg.algorithm.get("loss_type") == "rlt_ac"
        online_lerobot_cfg = cfg.algorithm.get("dagger", {}).get("online_lerobot", {})
        self.enable_online_lerobot = bool(online_lerobot_cfg.get("enabled", False))
        self.lerobot_only_success = bool(online_lerobot_cfg.get("only_success", False))
        self.env_reward_weight = cfg.get("reward", {}).get("env_reward_weight", 1.0)
        self.reward_weight = cfg.get("reward", {}).get("reward_weight", 1.0)
        self.use_training_pipeline = cfg.runner.get("use_training_pipeline", False)
        self.shuffle_rollout = cfg.algorithm.get("shuffle_rollout", True)
        if self.use_training_pipeline and self.enable_online_lerobot:
            raise ValueError(
                "Training pipeline does not support online LeRobot trajectory data."
            )
        if self.use_training_pipeline and cfg.runner.get(
            "enable_decoupled_mode", False
        ):
            raise ValueError(
                "Training pipeline does not support decoupled environment mode."
            )

    def _component_placement(self):
        from rlinf.utils.placement import HybridComponentPlacement

        return HybridComponentPlacement(self.cfg, Cluster())

    def init_worker(self) -> None:
        """Connect the private ingress and begin consuming trajectory events."""
        self.ingress = Channel.connect(f"{self.channel_name}.segments", self)
        self.output = Channel.connect(self.channel_name, self)
        self._serve_task = asyncio.create_task(self._serve())

    def _collector(
        self,
        collectors: dict[tuple[int, int], EmbodiedRolloutResult],
        source: tuple[int, int],
    ) -> EmbodiedRolloutResult:
        if source not in collectors:
            if self.enable_online_lerobot:
                collectors[source] = EmbodiedLerobotRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                    num_envs=self.cfg.env.train.total_num_envs // self.source_count,
                    only_success=self.lerobot_only_success,
                    num_action_chunks=self.cfg.actor.model.num_action_chunks,
                    action_dim=self.cfg.actor.model.action_dim,
                )
            else:
                collectors[source] = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps
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
            or not self.cfg.env.train.auto_reset
            or segment.env_result.dones is None
        ):
            return rewards
        truncations = segment.env_result.truncations
        if self.cfg.algorithm.get("bootstrap_type", "standard") != "standard":
            truncations = segment.env_result.dones
        if truncations is None or not truncations[:, -1].any():
            return rewards
        rewards = rewards.clone()
        mask = truncations[:, -1]
        rewards[mask, -1] += self.cfg.algorithm.get("gamma", 1.0) * values[
            mask
        ].reshape(-1).to(rewards.dtype)
        return rewards

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
                    rollout_result=RolloutResult(
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
                    next_forward_inputs=split_batch_value(
                        segment.next_forward_inputs, sizes
                    )[index],
                ),
            )

    def _append_one(
        self,
        collectors: dict[tuple[int, int], EmbodiedRolloutResult],
        source: tuple[int, int],
        segment: TrajectorySegment,
    ) -> None:
        collector = self._collector(collectors, source)
        result = segment.rollout_result
        env_result = segment.env_result
        if self.enable_online_lerobot:
            if env_result.episode_data is None:
                raise ValueError("Online LeRobot segment is missing episode data.")
            assert isinstance(collector, EmbodiedLerobotRolloutResult)
            collector.append_chunk_episode_data(
                rollout_result=result, **env_result.episode_data
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
                    segment.next_forward_inputs, transition=True
                ),
            )
        elif self.collect_transitions:
            collector.append_transitions(segment.obs, segment.next_obs)

    def _append_final_values(self, event: TrajectoryEpochEnd) -> None:
        if self.enable_online_lerobot or not self.collect_prev_infos:
            return
        values = split_batch_value(
            event.final_prev_values, [size for _, _, size in event.sources]
        )
        for (rank, stage, _), value in zip(event.sources, values):
            collectors = self.collectors
            if self.use_training_pipeline:
                collectors = self.pipeline_collectors[(event.step_id, event.epoch_id)]
            self._collector(collectors, (rank, stage)).append_final_value(value)

    def _append(self, segment: TrajectorySegment) -> None:
        collectors = self.collectors
        if self.use_training_pipeline:
            key = (segment.step_id, segment.epoch_id)
            collectors = self.pipeline_collectors.setdefault(key, {})
        for source, source_segment in self._split_segments(segment):
            self._append_one(collectors, source, source_segment)

    def _prepare_pipeline_batch(self, trajectories):
        """Build the same flattened actor input formerly prepared by EnvWorker."""
        from rlinf.algorithms.registry import calculate_adv_and_returns
        from rlinf.utils.utils import preprocess_embodied_batch

        batch = preprocess_embodied_batch(
            convert_trajectories_to_batch(trajectories),
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.get("group_size", 8),
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound"),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound"),
        )
        batch.update(
            calculate_adv_and_returns(
                task_type=self.cfg.runner.task_type,
                adv_type=self.cfg.algorithm.adv_type,
                rewards=batch["rewards"],
                dones=batch["dones"],
                values=batch.get("prev_values"),
                prev_logprobs=batch.get("prev_logprobs"),
                num_action_chunks=self.cfg.actor.model.num_action_chunks,
                gamma=self.cfg.algorithm.get("gamma", 1.0),
                gae_lambda=self.cfg.algorithm.get("gae_lambda", 1.0),
                group_size=self.cfg.algorithm.get("group_size", 8),
                reward_type=self.cfg.algorithm.reward_type,
                loss_mask=batch.get("loss_mask"),
                loss_mask_sum=batch.get("loss_mask_sum"),
                normalize_advantages=False,
            )
        )
        return batch

    def _publish_pipeline(
        self, collectors: dict[tuple[int, int], EmbodiedRolloutResult]
    ) -> None:
        """Preserve the old logical-env-to-actor pipeline routing."""
        from rlinf.scheduler import CommMapper
        from rlinf.utils.distributed import masked_stats, normalize_from_stats
        from rlinf.utils.nested_dict_process import split_dict_to_chunk
        from rlinf.utils.utils import flatten_embodied_batch, pack_batch

        actor_count = self._component_placement().get_world_size("actor")
        batches_by_actor: dict[int, list[dict]] = {
            rank: [] for rank in range(actor_count)
        }
        pending_batches: list[tuple[int, int, dict]] = []
        for source, collector in sorted(collectors.items()):
            source_rank = source[0] * self.cfg.rollout.pipeline_stage_num + source[1]
            actor_splits = CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=self.source_count,
                dst_world_size=actor_count,
                src_rank=source_rank,
            )
            trajectories = collector.to_splited_trajectories_by_sizes(
                [size for _, size in actor_splits]
            )
            for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                batch = self._prepare_pipeline_batch([trajectory])
                batches_by_actor[actor_rank].append(batch)
                pending_batches.append((source[0], actor_rank, batch))

        for actor_rank, batches in batches_by_actor.items():
            if self.cfg.algorithm.get("normalize_advantages", True):
                stats = sum(
                    masked_stats(batch["advantages"], batch.get("loss_mask"))
                    for batch in batches
                )
                for batch in batches:
                    batch["advantages"] = normalize_from_stats(
                        batch["advantages"], stats
                    )
        for env_rank, actor_rank, batch in pending_batches:
            generator = self.pipeline_generators.setdefault(
                (env_rank, actor_rank),
                torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + env_rank * actor_count
                ),
            )
            batch_size = (
                batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
            )
            order = (
                torch.randperm(batch_size, generator=generator)
                if self.shuffle_rollout
                else torch.arange(batch_size)
            )
            flat_batch = flatten_embodied_batch(batch, order)
            if batch_size % self.cfg.actor.micro_batch_size:
                raise ValueError(
                    "Pipeline source shard is not divisible into actor micro batches."
                )
            micro_batches = split_dict_to_chunk(
                flat_batch,
                batch_size // self.cfg.actor.micro_batch_size,
                dim=0,
            )
            key = CommMapper.build_channel_key(actor_rank, actor_rank, "pipeline_actor")
            for micro_batch in micro_batches:
                self.output.put(pack_batch(micro_batch), key=key, async_op=True)

    def _flush(self) -> None:
        if len(self.finished_sources) != self.producer_count:
            return
        if len(self.collectors) != self.source_count:
            raise ValueError(
                "Trajectory storage did not receive every logical env source before "
                "the rollout workers finished."
            )
        if self.enable_online_lerobot:
            from rlinf.utils.data_iter_utils import split_list

            for collector in self.collectors.values():
                assert isinstance(collector, EmbodiedLerobotRolloutResult)
                episodes = collector.drain_episodes()
                for episode_chunk in split_list(
                    episodes,
                    self.shards_per_source,
                    enforce_divisible_batch=False,
                ):
                    if episode_chunk:
                        self.output.put(episode_chunk, async_op=True)
        else:
            for collector in self.collectors.values():
                for trajectory in collector.to_splited_trajectories(
                    self.shards_per_source
                ):
                    self.output.put(trajectory, async_op=True)
        if not self.enable_online_lerobot:
            self.collectors.clear()
        self.finished_sources.clear()

    def _flush_pipeline_epoch(self, key: tuple[int, int]) -> None:
        if len(self.finished_epoch_sources[key]) != self.producer_count:
            return
        collectors = self.pipeline_collectors[key]
        if len(collectors) != self.source_count:
            raise ValueError(
                "Pipeline trajectory storage did not receive every logical env source."
            )
        self._publish_pipeline(collectors)
        del self.pipeline_collectors[key]
        del self.finished_epoch_sources[key]

    async def _serve(self) -> None:
        while True:
            item = await self.ingress.get(async_op=True).async_wait()
            if isinstance(item, TrajectorySegment):
                self._append(item)
            elif isinstance(item, TrajectoryEpochEnd):
                self._append_final_values(item)
                if self.use_training_pipeline:
                    key = (item.step_id, item.epoch_id)
                    self.finished_epoch_sources.setdefault(key, set()).add(item.source)
                    self._flush_pipeline_epoch(key)
            elif isinstance(item, TrajectoryEnd):
                if not self.use_training_pipeline:
                    self.finished_sources.add(item.source)
                    self._flush()
            else:
                raise TypeError(f"Unexpected trajectory item: {type(item)!r}")


class TrajectoryChannel(Channel):
    """A normal channel with a private rollout-to-actor trajectory path.

    ``put`` and ``get`` retain :class:`Channel` semantics. ``publish`` and
    ``take`` are worker-only APIs: rollout workers append segments to the
    dedicated storage worker, and actor workers consume its completed shards.
    """

    @classmethod
    def create(
        cls,
        name: str,
        *args,
        trajectory_cfg: "DictConfig | None" = None,
        trajectory_node_rank: int = 0,
        **kwargs,
    ) -> "TrajectoryChannel":
        """Create a normal channel and, optionally, its trajectory worker."""
        channel = super().create(name, *args, **kwargs)
        if trajectory_cfg is None:
            return channel
        cluster = Cluster()
        if not 0 <= trajectory_node_rank < cluster.num_nodes:
            raise ValueError(
                f"Trajectory worker node rank {trajectory_node_rank} is outside "
                f"the cluster's {cluster.num_nodes} nodes."
            )
        ingress_channel = Channel.create(
            f"{name}.segments", node_rank=trajectory_node_rank
        )
        ingress_channel._channel_worker_group._is_ready().wait()
        trajectory_worker_group = TrajectoryWorker.create_group(
            trajectory_cfg, name
        ).launch(
            cluster=cluster,
            name=f"{name}.trajectory",
            placement_strategy=NodePlacementStrategy([trajectory_node_rank]),
            max_concurrency=2**31 - 1,
        )
        # Keep the private ingress and its consumer alive with this channel.
        channel._trajectory_ingress_channel = ingress_channel
        channel._trajectory_worker_group = trajectory_worker_group
        trajectory_worker_group.init_worker().wait()
        return channel

    def _require_worker(self) -> None:
        if self._current_worker is None:
            raise RuntimeError(
                "TrajectoryChannel publish/take may only run in a worker."
            )

    def publish(self, segment: TrajectorySegment | TrajectoryEpochEnd | TrajectoryEnd):
        """Append a segment without routing it through Ray object transport."""
        self._require_worker()
        ingress = getattr(self, "_trajectory_ingress", None)
        if ingress is None:
            ingress = Channel.connect(
                f"{self._channel_name}.segments", self._current_worker
            )
            self._trajectory_ingress = ingress
        return ingress.put(segment, async_op=True)

    def take(self):
        """Return asynchronous work for the next completed trajectory shard."""
        self._require_worker()
        return self.get(async_op=True)

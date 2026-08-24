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

"""Collect trajectory parts and emit actor-ready outputs.

Rollout workers publish policy-owned and environment-owned parts. The one
public :class:`TrajectoryCollector` restores routing fragments, joins both
parts, accumulates chunks, and delegates final formatting to a private output
strategy selected by :class:`TrajectoryPlan`.
"""

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, TypeAlias

import torch
from omegaconf import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvPart,
    EnvResult,
    PolicyPart,
    TrajectoryKey,
    TrajectoryPart,
    TrajectorySource,
    convert_trajectories_to_batch,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_env_result,
    split_episode_data,
)
from rlinf.data.schema.trajectory_accumulator import (
    LerobotEpisodeAccumulator,
    TrajectoryAccumulator,
)
from rlinf.scheduler.channel import (
    ChannelContext,
    Collector,
    register_collector,
)
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.worker.routing import CommMapper
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)

# Logical source: ``(environment rank, pipeline stage)``.
SourceID: TypeAlias = tuple[int, int]
# Chunks that must complete before one output strategy may flush.
CollectionScope: TypeAlias = int | tuple[int, ...]
# Channel collector protocol: ``(queue key, payload)``.
CollectedItem: TypeAlias = tuple[str, Any]


@dataclass(kw_only=True)
class JoinedChunk:
    """Policy and environment data joined for one action chunk.

    ``source`` identifies the environment rank and pipeline stage. Only chunk
    zero carries ``initial_env_result``.
    """

    key: TrajectoryKey
    source: tuple[int, int]
    policy: PolicyPart
    env: EnvPart
    initial_env_result: EnvResult | None = None


class ChunkJoiner:
    """Restore source batches, then pair policy and environment parts by key."""

    def __init__(self, source_batch_size: int):
        """Initialize join state for one logical rollout source."""
        self._source_batch_size = source_batch_size
        self._policy_parts: dict[TrajectoryKey, PolicyPart] = {}
        self._env_parts: dict[TrajectoryKey, EnvPart] = {}
        self._fragments: dict[tuple[type, TrajectoryKey], list[object]] = defaultdict(
            list
        )

    def push(self, part: TrajectoryPart) -> list[JoinedChunk]:
        """Consume one part and return any newly joined chunks."""
        if not isinstance(part, (PolicyPart, EnvPart)):
            raise ValueError(f"Unexpected trajectory part: {type(part)}")

        # Most producers already send one complete source batch.
        if self._is_complete_source(part):
            chunk = self._accept_complete_part(part)
            return [chunk] if chunk is not None else []

        completed = []
        for fragment in self._split_part(part):
            merged = self._merge_fragments(fragment)
            if merged is None:
                continue
            chunk = self._accept_complete_part(merged)
            if chunk is not None:
                completed.append(chunk)
        return completed

    def acknowledge(self, key: TrajectoryKey) -> None:
        """Release a chunk after the collector accepts it."""
        del self._policy_parts[key]
        del self._env_parts[key]

    def _is_complete_source(self, part: TrajectoryPart) -> bool:
        """Return whether a part covers its entire logical source batch."""
        if len(part.sources) != 1:
            return False
        source = part.sources[0]
        return source.offset == 0 and source.size == self._source_batch_size

    def _accept_complete_part(self, part: TrajectoryPart) -> JoinedChunk | None:
        """Store a complete part and join it when its counterpart is present."""
        key = part.sources[0].key
        if isinstance(part, PolicyPart):
            self._policy_parts[key] = part
        else:
            self._env_parts[key] = part
        return self._try_complete(key)

    def _try_complete(self, key: TrajectoryKey) -> JoinedChunk | None:
        """Build a joined chunk once both parts for ``key`` are available."""
        policy = self._policy_parts.get(key)
        env = self._env_parts.get(key)
        if policy is None or env is None:
            return None
        if key.chunk_id == 0 and env.initial_result is None:
            raise ValueError(f"Chunk zero is missing its initial state: {key}.")
        if key.chunk_id != 0 and env.initial_result is not None:
            raise ValueError(f"Only chunk zero may carry initial state: {key}.")
        return JoinedChunk(
            key=key,
            source=(key.env_rank, key.stage_id),
            policy=policy,
            env=env,
            initial_env_result=env.initial_result,
        )

    def _split_part(self, part: TrajectoryPart):
        """Split a routed multi-source part into single-source fragments."""
        sizes = [source.size for source in part.sources]
        if isinstance(part, PolicyPart):
            observations = split_batch_value(part.obs, sizes)
            if part.rollout_result is None:
                actions = split_batch_value(part.external_actions, sizes)
                for index, source in enumerate(part.sources):
                    yield PolicyPart(
                        sources=[source],
                        obs=observations[index],
                        external_actions=actions[index],
                    )
                return
            fields = {
                name: split_batch_value(getattr(part.rollout_result, name), sizes)
                for name in part.rollout_result.__dataclass_fields__
            }
            for index, source in enumerate(part.sources):
                yield PolicyPart(
                    sources=[source],
                    obs=observations[index],
                    rollout_result=EmbodiedRolloutResult(
                        **{name: values[index] for name, values in fields.items()}
                    ),
                )
            return

        fields = {
            name: split_batch_value(getattr(part.result, name), sizes)
            for name in part.result.__dataclass_fields__
            if name != "episode_data"
        }
        episodes = split_episode_data(part.result.episode_data, sizes)
        next_observations = split_batch_value(part.next_obs, sizes)
        forward_inputs = split_batch_value(part.forward_inputs, sizes)
        bootstrap_values = split_batch_value(part.bootstrap_values, sizes)
        final_prev_values = split_batch_value(part.final_prev_values, sizes)
        initial_results = (
            split_env_result(part.initial_result, sizes)
            if part.initial_result is not None
            else [None] * len(sizes)
        )
        for index, source in enumerate(part.sources):
            yield EnvPart(
                sources=[source],
                result=EnvResult(
                    **{name: values[index] for name, values in fields.items()},
                    episode_data=episodes[index],
                ),
                next_obs=next_observations[index],
                forward_inputs=forward_inputs[index],
                bootstrap_values=bootstrap_values[index],
                final_prev_values=final_prev_values[index],
                initial_result=initial_results[index],
            )

    def _merge_fragments(self, part: TrajectoryPart):
        """Reassemble one source batch after all of its fragments arrive."""
        source = part.sources[0]
        key = (type(part), source.key)
        fragments = self._fragments[key]
        fragments.append(part)
        received_size = sum(item.sources[0].size for item in fragments)
        if received_size < self._source_batch_size:
            return None
        if received_size > self._source_batch_size:
            raise ValueError(
                f"Trajectory fragments exceed source batch size for {source.key}."
            )

        # Routing fragments may arrive out of order; offsets define batch order.
        fragments.sort(key=lambda item: item.sources[0].offset)
        offsets = [item.sources[0].offset for item in fragments]
        sizes = [item.sources[0].size for item in fragments]
        if any(offset != sum(sizes[:index]) for index, offset in enumerate(offsets)):
            raise ValueError(f"Non-contiguous trajectory fragments: {offsets}.")

        del self._fragments[key]
        full_source = TrajectorySource(source.key, self._source_batch_size)
        if isinstance(part, PolicyPart):
            if part.rollout_result is None:
                return PolicyPart(
                    sources=[full_source],
                    obs=merge_batch_values([item.obs for item in fragments]),
                    external_actions=merge_batch_values(
                        [item.external_actions for item in fragments]
                    ),
                )
            return PolicyPart(
                sources=[full_source],
                obs=merge_batch_values([item.obs for item in fragments]),
                rollout_result=EmbodiedRolloutResult(
                    **{
                        name: merge_batch_values(
                            [getattr(item.rollout_result, name) for item in fragments]
                        )
                        for name in part.rollout_result.__dataclass_fields__
                    }
                ),
            )

        initial_results = [item.initial_result for item in fragments]
        has_initial = [result is not None for result in initial_results]
        if any(has_initial) and not all(has_initial):
            raise ValueError(f"Incomplete initial-state fragments for {source.key}.")
        return EnvPart(
            sources=[full_source],
            result=EnvResult(
                **{
                    name: merge_batch_values(
                        [getattr(item.result, name) for item in fragments]
                    )
                    for name in part.result.__dataclass_fields__
                    if name != "episode_data"
                },
                episode_data=(
                    merge_episode_data([item.result.episode_data for item in fragments])
                    if part.result.episode_data is not None
                    else None
                ),
            ),
            next_obs=merge_batch_values([item.next_obs for item in fragments]),
            forward_inputs=merge_batch_values(
                [item.forward_inputs for item in fragments]
            ),
            bootstrap_values=merge_batch_values(
                [item.bootstrap_values for item in fragments]
            ),
            final_prev_values=merge_batch_values(
                [item.final_prev_values for item in fragments]
            ),
            initial_result=(
                EnvResult(
                    **{
                        name: merge_batch_values(
                            [getattr(item, name) for item in initial_results]
                        )
                        for name in initial_results[0].__dataclass_fields__
                        if name != "episode_data"
                    },
                    episode_data=(
                        merge_episode_data(
                            [item.episode_data for item in initial_results]
                        )
                        if initial_results[0].episode_data is not None
                        else None
                    ),
                )
                if all(has_initial)
                else None
            ),
        )


@dataclass(frozen=True)
class RolloutGeometry:
    """How one training step's rollout data is shaped and routed.

    Attributes:
        source_count: Number of logical rollout sources (env ranks x stages).
        chunk_count: Action chunks each source produces per rollout epoch.
        shards_per_source: Consumable items each source is split into.
        actor_world_size: Number of actor workers consuming the channel.
    """

    source_count: int
    chunk_count: int
    shards_per_source: int
    actor_world_size: int

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "RolloutGeometry":
        """Derive the geometry a run configuration implies.

        Args:
            cfg: The run configuration.

        Returns:
            The rollout geometry.

        Raises:
            ValueError: If sources cannot be split evenly across actors.
        """
        from rlinf.scheduler.cluster import Cluster
        from rlinf.utils.metric_utils import compute_split_num
        from rlinf.utils.placement import HybridComponentPlacement

        placement = HybridComponentPlacement(cfg, Cluster())
        actor_world_size = placement.get_world_size("actor")
        source_count = placement.get_world_size("env") * cfg.rollout.pipeline_stage_num
        output_count = compute_split_num(source_count, actor_world_size) * (
            actor_world_size
        )
        if output_count % source_count:
            raise ValueError(
                "Trajectory routing requires each rollout source to have an "
                "equal number of actor shards."
            )
        return cls(
            source_count=source_count,
            chunk_count=(
                cfg.env.train.max_steps_per_rollout_epoch
                // cfg.actor.model.num_action_chunks
            ),
            shards_per_source=output_count // source_count,
            actor_world_size=actor_world_size,
        )


class TrajectoryMode(str, Enum):
    """Actor-facing output mode for collected trajectory chunks."""

    ROLLOUT = "rollout"
    PIPELINE = "pipeline"
    LEROBOT = "lerobot"


@dataclass(frozen=True)
class TrajectoryPlan:
    """Validated trajectory mode and routing geometry for one run."""

    mode: TrajectoryMode
    geometry: RolloutGeometry

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "TrajectoryPlan":
        """Build the single source of truth for trajectory collection."""
        return cls(mode=cls.mode_from_cfg(cfg), geometry=RolloutGeometry.from_cfg(cfg))

    @staticmethod
    def mode_from_cfg(cfg: DictConfig) -> TrajectoryMode:
        """Validate mode combinations and return the configured output mode."""
        use_pipeline = cfg.runner.get("use_training_pipeline", False)
        use_lerobot = bool(
            cfg.algorithm.get("dagger", {})
            .get("online_lerobot", {})
            .get("enabled", False)
        )
        if use_pipeline and use_lerobot:
            raise ValueError(
                "Training pipeline does not support online LeRobot trajectory data."
            )
        if use_pipeline and cfg.runner.get("enable_decoupled_mode", False):
            raise ValueError(
                "Training pipeline does not support decoupled environment mode."
            )
        if use_pipeline and cfg.algorithm.get("adv_type") == "opd":
            raise ValueError(
                "Training pipeline does not support OPD because teacher log-probabilities "
                "are computed on actor workers after rollout."
            )
        if use_lerobot:
            return TrajectoryMode.LEROBOT
        if use_pipeline:
            return TrajectoryMode.PIPELINE
        return TrajectoryMode.ROLLOUT

    @property
    def dispatcher(self) -> str | None:
        """Return the channel dispatcher compatible with the output routing."""
        return None if self.mode is TrajectoryMode.PIPELINE else "least_loaded"


class TrajectoryOutput:
    """Base strategy for converting joined chunks into channel outputs."""

    def __init__(self, cfg: DictConfig, geometry: RolloutGeometry) -> None:
        """Store immutable run context and initialize duplicate tracking."""
        self._cfg = cfg
        self._geometry = geometry
        self._source_count = self._geometry.source_count
        self._chunk_count = self._geometry.chunk_count
        self._completed_keys: set[TrajectoryKey] = set()

    @abstractmethod
    def emit(self, chunk: JoinedChunk) -> Iterator[CollectedItem]:
        """Turn one completed action chunk into consumer-ready items."""


class AccumulatorOutput(TrajectoryOutput):
    """Shared trajectory-building strategy for rollout and pipeline modes."""

    def __init__(self, cfg: DictConfig, geometry: RolloutGeometry) -> None:
        """Initialize per-scope accumulators and collection options."""
        super().__init__(cfg, geometry)
        cfg = self._cfg
        self._accumulators: dict[
            CollectionScope, dict[SourceID, TrajectoryAccumulator]
        ] = {}
        self._collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)
        self._collect_transitions = cfg.rollout.get("collect_transitions", False)
        self._enable_rlt = cfg.algorithm.get("loss_type") in {"rlt_ac", "rlt_td3"}
        self._env_reward_weight = cfg.get("reward", {}).get("env_reward_weight", 1.0)
        self._reward_weight = cfg.get("reward", {}).get("reward_weight", 1.0)

    def emit(self, chunk: JoinedChunk) -> Iterator[CollectedItem]:
        """Append a policy chunk and flush its scope when complete."""
        if not chunk.policy.inferred:
            raise ValueError(
                "External policy actions are only supported for online LeRobot DAgger."
            )
        if chunk.key in self._completed_keys:
            raise ValueError(f"Received duplicate trajectory event for {chunk.key}.")

        # A scope is the smallest set that must complete before output is safe.
        scope = self._scope(chunk.key)
        accumulators = self._accumulators.setdefault(scope, {})
        accumulator = accumulators.setdefault(
            chunk.source,
            TrajectoryAccumulator(
                max_episode_length=self._cfg.env.train.max_episode_steps
            ),
        )
        self._append_chunk(accumulator, chunk)
        if chunk.key.chunk_id == self._chunk_count - 1 and self._collect_prev_infos:
            accumulator.append_final_value(chunk.env.final_prev_values)
        self._completed_keys.add(chunk.key)

        if self._scope_key_count(scope) != self._expected_key_count:
            return
        if len(accumulators) != self._expected_accumulator_count:
            raise ValueError(
                f"Expected {self._expected_accumulator_count} accumulators, but got "
                f"{len(accumulators)}."
            )

        outputs = list(self._flush(accumulators))
        del self._accumulators[scope]
        self._completed_keys = {
            key for key in self._completed_keys if self._scope(key) != scope
        }
        yield from outputs

    def _append_chunk(
        self, accumulator: TrajectoryAccumulator, chunk: JoinedChunk
    ) -> None:
        """Append model, environment, reward, and optional transition data."""
        policy = chunk.policy
        result = policy.rollout_result
        assert result is not None
        env_result = chunk.env.result
        result.bootstrap_values = chunk.env.bootstrap_values

        if chunk.initial_env_result is not None:
            accumulator.append_initial_state(
                dones=chunk.initial_env_result.dones,
                truncations=chunk.initial_env_result.truncations,
                terminations=chunk.initial_env_result.terminations,
            )
        accumulator.append_step_result(
            ChunkStepResult(
                actions=result.forward_inputs.get("action"),
                prev_logprobs=(
                    result.prev_logprobs if self._collect_prev_infos else None
                ),
                prev_values=(result.prev_values if self._collect_prev_infos else None),
                forward_inputs=result.forward_inputs,
                versions=result.versions,
                dones=env_result.dones,
                truncations=env_result.truncations,
                terminations=env_result.terminations,
                rewards=self._rewards(chunk),
            )
        )
        if env_result.reward_assign_lengths is not None:
            # History rewards are assigned backwards to the requested chunks.
            rewards = accumulator.rewards
            reward = self._reward_weight * env_result.reward_model_output
            for env_id, length in enumerate(env_result.reward_assign_lengths):
                for offset in range(2, min(length, len(rewards)) + 1):
                    rewards[-offset][env_id] += reward[env_id].to(
                        rewards[-offset].dtype
                    )
        if env_result.intervene_actions is not None:
            accumulator.update_last_actions(
                env_result.intervene_actions, env_result.intervene_flags
            )
        if result.intervene_flags is not None:
            accumulator.mark_last_step_with_intervene_flags(result.intervene_flags)
        if self._enable_rlt:
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
            accumulator.append_transitions(
                current_obs,
                extract_rlt_obs_from_forward_inputs(
                    chunk.env.forward_inputs, transition=True
                ),
            )
        elif self._collect_transitions:
            accumulator.append_transitions(policy.obs, chunk.env.next_obs)

    def _rewards(self, chunk: JoinedChunk) -> torch.Tensor | None:
        """Combine reward sources and apply terminal-value bootstrapping."""
        env_result = chunk.env.result
        rewards = env_result.rewards
        if rewards is None:
            return None
        if env_result.reward_model_output is not None:
            rewards = (
                self._env_reward_weight * rewards
                + self._reward_weight * env_result.reward_model_output.to(rewards.dtype)
            )
        values = chunk.env.bootstrap_values
        if (
            values is None
            or not self._cfg.env.train.auto_reset
            or env_result.dones is None
        ):
            return rewards
        truncations = env_result.truncations
        if self._cfg.algorithm.get("bootstrap_type", "standard") != "standard":
            truncations = env_result.dones
        if truncations is None or not truncations[:, -1].any():
            return rewards
        rewards = rewards.clone()
        mask = truncations[:, -1]
        rewards[mask, -1] += self._cfg.algorithm.get("gamma", 1.0) * values[
            mask
        ].reshape(-1).to(rewards.dtype)
        return rewards

    @property
    @abstractmethod
    def _expected_key_count(self) -> int:
        """Number of chunks required to flush one collection scope."""

    @property
    def _expected_accumulator_count(self) -> int:
        """Number of source accumulators required to flush one scope."""
        return self._source_count

    @abstractmethod
    def _scope(self, key: TrajectoryKey) -> CollectionScope:
        """Return the collection scope containing a trajectory key."""

    @abstractmethod
    def _flush(
        self, accumulators: dict[SourceID, TrajectoryAccumulator]
    ) -> Iterator[CollectedItem]:
        """Convert a completed scope into consumer-ready items."""

    def _scope_key_count(self, scope: CollectionScope) -> int:
        """Count completed chunk keys belonging to one collection scope."""
        return sum(self._scope(key) == scope for key in self._completed_keys)


class RolloutOutput(AccumulatorOutput):
    """Collect complete rollout steps and emit trajectory shards.

    Every shard goes to the shared key, so any actor may take any of them.
    """

    def _flush(
        self, accumulators: dict[SourceID, TrajectoryAccumulator]
    ) -> Iterator[CollectedItem]:
        """Split each source's trajectory into its shards."""
        # Stable source ordering makes output independent of arrival timing.
        for _, accumulator in sorted(accumulators.items()):
            shards = accumulator.to_splited_trajectories(
                self._geometry.shards_per_source
            )
            for trajectory in shards:
                yield DEFAULT_KEY, trajectory

    @property
    def _expected_key_count(self) -> int:
        """Return all chunks emitted by one source in a training step."""
        return self._cfg.env.train.rollout_epoch * self._chunk_count

    @property
    def _expected_accumulator_count(self) -> int:
        """Flush sources independently to preserve async main behavior."""
        return 1

    def _scope(self, key: TrajectoryKey) -> CollectionScope:
        """Group chunks by training step and logical rollout source."""
        # Main emits one trajectory as soon as one env source completes. Do not
        # turn async/decoupled collection into a cross-source barrier.
        return (key.step_id, key.env_rank, key.stage_id)


class PipelineOutput(AccumulatorOutput):
    """Collect one rollout epoch and emit actor-routed training micro-batches.

    Each micro-batch is keyed for the one actor that must train on it, so the
    key alone routes it and no dispatcher is involved.
    """

    def __init__(self, cfg: DictConfig, geometry: RolloutGeometry) -> None:
        """Initialize actor-specific shuffling and routing state."""
        super().__init__(cfg, geometry)
        self._actor_world_size = self._geometry.actor_world_size
        self._generators: dict[int, torch.Generator] = {}
        self._shuffle_rollout = self._cfg.algorithm.get("shuffle_rollout", True)

    def _flush(
        self, accumulators: dict[SourceID, TrajectoryAccumulator]
    ) -> Iterator[CollectedItem]:
        """Route each source's data to its actors, then split into micro-batches."""
        batches_by_actor: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
        # Sort by source so batch order does not depend on event arrival order.
        for (env_rank, stage_id), accumulator in sorted(accumulators.items()):
            logical_rank = env_rank * self._cfg.rollout.pipeline_stage_num + stage_id
            actor_splits = CommMapper.get_dst_ranks(
                batch_size=self._cfg.env.train.total_num_envs,
                src_world_size=self._source_count,
                dst_world_size=self._actor_world_size,
                src_rank=logical_rank,
            )
            trajectories = accumulator.to_splited_trajectories_by_sizes(
                [size for _, size in actor_splits]
            )
            for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                batches_by_actor[actor_rank].append(
                    self._prepare_pipeline_batch(trajectory)
                )

        # Normalize per actor after routing, matching its local training batch.
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
                    yield (
                        CommMapper.build_channel_key(
                            actor_rank, actor_rank, "pipeline_actor"
                        ),
                        micro_batch,
                    )

    def _prepare_pipeline_batch(self, trajectory) -> dict[str, torch.Tensor]:
        """Preprocess one actor shard and compute training targets."""
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
        """Shuffle deterministically and pack one actor batch into micro-batches."""
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        generator = self._generators.setdefault(
            actor_rank,
            torch.Generator().manual_seed(self._cfg.actor.seed + actor_rank),
        )
        indices = (
            torch.randperm(batch_size, generator=generator)
            if self._shuffle_rollout
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
    def _expected_key_count(self) -> int:
        """Return all source chunks required for one pipeline epoch."""
        return self._source_count * self._chunk_count

    def _scope(self, key: TrajectoryKey) -> CollectionScope:
        """Group every source participating in the same pipeline epoch."""
        return (key.step_id, key.epoch_id)


class LerobotOutput(TrajectoryOutput):
    """Collect online LeRobot episodes and emit episode shards per rollout step.

    Accumulators persist across steps because an episode may span several of them.
    """

    def __init__(self, cfg: DictConfig, geometry: RolloutGeometry) -> None:
        """Initialize persistent per-source episode accumulators."""
        super().__init__(cfg, geometry)
        self._accumulators: dict[SourceID, LerobotEpisodeAccumulator] = {}
        online_cfg = self._cfg.algorithm.dagger.online_lerobot
        self._only_success = bool(online_cfg.get("only_success", False))

    def emit(self, chunk: JoinedChunk) -> Iterator[CollectedItem]:
        """Append episode data and drain completed episodes at step boundaries."""
        if chunk.key in self._completed_keys:
            raise ValueError(f"Received duplicate trajectory event for {chunk.key}.")
        episode_data = chunk.env.result.episode_data
        if episode_data is None:
            raise ValueError("Online LeRobot segment is missing episode data.")

        accumulator = self._accumulators.setdefault(
            chunk.source,
            LerobotEpisodeAccumulator(
                max_episode_length=self._cfg.env.train.max_episode_steps,
                num_envs=self._cfg.env.train.total_num_envs // self._source_count,
                only_success=self._only_success,
                num_action_chunks=self._cfg.actor.model.num_action_chunks,
                action_dim=self._cfg.actor.model.action_dim,
            ),
        )
        policy_output = chunk.policy.rollout_result
        accumulator.append_chunk_episode_data(
            policy_output=policy_output, **episode_data
        )
        self._completed_keys.add(chunk.key)

        scope = (chunk.key.step_id, *chunk.source)
        expected = self._cfg.env.train.rollout_epoch * self._chunk_count
        if (
            sum(self._lerobot_scope(key) == scope for key in self._completed_keys)
            != expected
        ):
            return

        shards_per_source = self._geometry.shards_per_source
        episodes = accumulator.drain_episodes()
        outputs = [
            (DEFAULT_KEY, episodes[shard::shards_per_source])
            for shard in range(shards_per_source)
        ]
        self._completed_keys = {
            key for key in self._completed_keys if self._lerobot_scope(key) != scope
        }
        yield from outputs

    @staticmethod
    def _lerobot_scope(key: TrajectoryKey) -> tuple[int, int, int]:
        """Return the independently drainable scope used by main's env worker."""
        return (key.step_id, key.env_rank, key.stage_id)


@register_collector("embodied_trajectory")
class TrajectoryCollector(Collector):
    """Join trajectory parts and emit outputs selected by one validated plan."""

    def setup(self, ctx: ChannelContext) -> None:
        """Initialize the join state and private output strategy."""
        if ctx.cfg is None:
            raise ValueError(
                "TrajectoryCollector needs the run config. Pass cfg= when "
                "creating the channel."
            )
        self.plan = TrajectoryPlan.from_cfg(ctx.cfg)
        source_batch_size = (
            ctx.cfg.env.train.total_num_envs // self.plan.geometry.source_count
        )
        self._joiner = ChunkJoiner(source_batch_size=source_batch_size)
        # One collector owns joining; only final materialization varies by mode.
        output_types: dict[TrajectoryMode, type[TrajectoryOutput]] = {
            TrajectoryMode.ROLLOUT: RolloutOutput,
            TrajectoryMode.PIPELINE: PipelineOutput,
            TrajectoryMode.LEROBOT: LerobotOutput,
        }
        self._output = output_types[self.plan.mode](ctx.cfg, self.plan.geometry)

    def collect(self, item: Any, key: str) -> Iterable[CollectedItem]:
        """Accept one part and yield any actor outputs it completes."""
        del key
        for chunk in self._joiner.push(item):
            # Commit join state only after accumulation/materialization succeeds.
            outputs = list(self._output.emit(chunk))
            self._joiner.acknowledge(chunk.key)
            yield from outputs


def select_trajectory_collector(cfg: DictConfig) -> type[TrajectoryCollector]:
    """Validate the configured mode and return the one public collector."""
    TrajectoryPlan.mode_from_cfg(cfg)
    return TrajectoryCollector


def select_trajectory_dispatcher(cfg: DictConfig) -> str | None:
    """Return the dispatcher used for actor-facing trajectory output.

    Whole trajectories and LeRobot episode shards share one queue key. Dealing
    them evenly at enqueue time prevents an asynchronous ``get_nowait`` loop on
    one actor from draining work intended for its peers. Pipeline output is
    already routed by its canonical ``<rank>_<rank>_pipeline_actor`` key and
    must stay on shared key queues, because applying a second dispatcher could
    send an actor-specific key to a different consumer.

    Args:
        cfg: The run configuration.

    Returns:
        The registered dispatcher name, or ``None`` for pipeline output.
    """
    mode = TrajectoryPlan.mode_from_cfg(cfg)
    return None if mode is TrajectoryMode.PIPELINE else "least_loaded"

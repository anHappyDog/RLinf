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

"""Embodied trajectory assembly, as a channel collector.

Environment and rollout workers each publish their own half of a rollout as
typed lifecycle events. This collector joins the two streams by action chunk on
the channel worker and emits actor-ready trajectories, so no worker has to hold
partial trajectory state.
"""

from typing import Any, Iterable

from rlinf.data.schema.trajectory_assembler import TrajectoryEventAssembler
from rlinf.data.schema.trajectory_collectors import create_trajectory_collector
from rlinf.scheduler.channel import (
    ChannelContext,
    Collector,
    register_collector,
)
from rlinf.scheduler.channel.channel import DEFAULT_KEY

__all__ = ["EmbodiedTrajectoryCollector", "actor_queue_key"]

#: Queue key an actor rank subscribes to in training-pipeline mode.
ACTOR_KEY_PREFIX = "actor:"

#: Key the trajectory collectors use for their shared, unrouted output.
SHARED_OUTPUT_KEY = "default"


def actor_queue_key(actor_rank: int) -> str:
    """Return the channel key carrying micro-batches for one actor rank.

    Args:
        actor_rank: Rank of the actor worker.

    Returns:
        The key to pass to ``channel.get()``.
    """
    return f"{ACTOR_KEY_PREFIX}{actor_rank}"


@register_collector("embodied_trajectory")
class EmbodiedTrajectoryCollector(Collector):
    """Join embodied rollout events into actor-ready trajectories.

    Consumes ``TrajectoryStart``, ``PolicyStep``, ``DummyPolicyStep``, and
    ``EnvStepResult`` events, reassembles routed source fragments, and emits
    whatever the configured collection strategy produces: complete trajectories,
    online LeRobot episode shards, or per-actor training micro-batches.
    """

    def setup(self, ctx: ChannelContext) -> None:
        """Size the assembler and collection strategy against the run config.

        Args:
            ctx: Channel description. ``ctx.cfg`` must be the run config.

        Raises:
            ValueError: If the config asks for a routing that cannot give every
                rollout source an equal number of actor shards.
        """
        from rlinf.scheduler.cluster import Cluster
        from rlinf.utils.metric_utils import compute_split_num
        from rlinf.utils.placement import HybridComponentPlacement

        cfg = ctx.cfg
        if cfg is None:
            raise ValueError(
                "The embodied_trajectory collector needs the run config. Pass "
                "cfg= when creating the channel."
            )

        placement = HybridComponentPlacement(cfg, Cluster())
        actor_world_size = placement.get_world_size("actor")
        source_count = placement.get_world_size("env") * cfg.rollout.pipeline_stage_num
        chunk_count = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        output_count = compute_split_num(source_count, actor_world_size) * (
            actor_world_size
        )
        if output_count % source_count:
            raise ValueError(
                "Trajectory routing requires each rollout source to have an "
                "equal number of actor shards."
            )

        self._assembler = TrajectoryEventAssembler(
            source_batch_size=cfg.env.train.total_num_envs // source_count
        )
        self._collector = create_trajectory_collector(
            cfg,
            source_count=source_count,
            chunk_count=chunk_count,
            shards_per_source=output_count // source_count,
            actor_world_size=actor_world_size,
        )

    def collect(self, item: Any, key: str) -> Iterable[tuple[str, Any]]:
        """Assemble one event and emit any outputs it completed.

        Args:
            item: One trajectory lifecycle event.
            key: Ignored; outputs are keyed by their destination.

        Yields:
            ``(key, data)`` pairs, on the shared key for whole trajectories and
            episode shards, or on ``actor:<rank>`` for pipeline micro-batches.
        """
        for chunk in self._assembler.push(item):
            outputs = self._collector.collect(chunk)
            self._assembler.acknowledge(chunk.key)
            for output in outputs:
                queue_key = output.queue_key
                if queue_key == SHARED_OUTPUT_KEY:
                    queue_key = DEFAULT_KEY
                yield queue_key, output.data

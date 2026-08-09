Trajectory Channel
==================

Use ``TrajectoryChannel`` to move embodied rollout data from rollout workers to
actor workers without assembling complete trajectories in environment workers.
It keeps trajectory state in one dedicated ``TrajectoryWorker`` and transfers
payloads with worker-to-worker communication.

When RLinf Uses It
------------------

Embodied runners create a ``TrajectoryChannel`` for the rollout-to-actor data
path. The environment and rollout workers still exchange policy requests and
actions through ordinary :doc:`Channel <channel>` objects. The rollout worker
publishes the data needed for training after each action chunk, and the actor
subscribes to completed training items.

This division keeps each component focused:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Component
     - Responsibility
   * - ``EnvWorker``
     - Step environments and send ``PolicyInput`` values containing observations
       and the preceding ``EnvResult``.
   * - ``RolloutWorker``
     - Run policy inference, return ``PolicyOutput`` actions, and publish
       ``TrajectorySegment`` events.
   * - ``TrajectoryWorker``
     - Preserve source ordering, append segments, apply delayed updates, and emit
       actor-ready trajectories or pipeline micro-batches.
   * - ``ActorWorker``
     - Subscribe to completed items and perform the algorithm-specific update.

Data Flow
---------

The following sequence shows one rollout epoch. ``Channel`` traffic and
``TrajectoryChannel`` traffic are separate, so trajectory assembly does not add
payloads to the policy-response path.

.. code-block:: text

   EnvWorker              RolloutWorker         TrajectoryWorker       ActorWorker
       | PolicyInput            |                       |                    |
       |----------------------->|                       |                    |
       |     PolicyOutput       |                       |                    |
       |<-----------------------|                       |                    |
       |                        | TrajectorySegment     |                    |
       |                        |---------------------->| append by source   |
       |          ... repeat for each action chunk ... |                    |
       |                        | TrajectoryEpochEnd    |                    |
       |                        |---------------------->| finalize epoch     |
       |                        | TrajectoryEnd         |                    |
       |                        |---------------------->| flush completed    |
       |                        |                       | trajectory / batch |
       |                        |                       |------------------->|

The rollout worker keeps one pending inference result per pipeline stage. When
the next ``PolicyInput`` arrives, it now has both the result of the previous
action and the next environment state, so it can publish one complete segment.
The final policy input closes the last pending segment.

Events and Results
------------------

``TrajectoryChannel.publish()`` accepts one of three internal event types:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Event
     - Meaning
   * - ``TrajectorySegment``
     - One append operation containing observations, the next observation,
       environment results, ``EmbodiedRolloutResult``, and logical source
       metadata.
   * - ``TrajectoryEpochEnd``
     - One producer finished an epoch. It also carries final values needed by
       value-based advantage calculation.
   * - ``TrajectoryEnd``
     - One producer finished a training step. The worker flushes only after all
       expected producers report completion.

For ordinary training, ``subscribe()`` returns a complete ``Trajectory``. For
online LeRobot DAgger it returns completed episode dictionaries. With training
pipeline mode enabled, the actor subscribes to its ``actor:<rank>`` queue and
receives prepared micro-batches.

Routing and Ordering
--------------------

Each policy input records logical sources as ``(env_rank, stage_id, batch_size)``.
Channel routing may split or merge a policy batch, but these source records stay
with the data. The trajectory worker uses them to restore per-environment-rank,
per-stage streams before appending segments.

Completed data is split according to the actor world size. This has two useful
properties:

* Segments from one logical source always reach the same collector, so a
  trajectory is never assembled across workers.
* The number of output shards is divisible by the number of logical sources, so
  each source contributes the same number of actor-consumable items.

Pipeline mode uses ``CommMapper`` to map every logical source to actor ranks. It
calculates advantages after an epoch is complete, normalizes them over the
actor's assigned batches when configured, and then emits micro-batches in the
actor queue.

Placement
---------

Trajectory assembly currently uses exactly one physical ``TrajectoryWorker``.
Choose its node with ``cluster.trajectory_node_rank``:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     trajectory_node_rank: 1

The default is node rank ``0``. The value must be in
``[0, cluster.num_nodes - 1]``. Place the worker near the dominant trajectory
traffic when nodes are connected by a slower network. This setting does not
change ``ChannelWorker`` placement or ordinary ``Channel`` routing.

Communication Semantics
-----------------------

``publish()`` and ``subscribe()`` are worker-only APIs. They use a small Ray RPC
to coordinate each operation and RLinf P2P ``send``/``recv`` for the payload.
Calling either method outside an RLinf ``Worker`` raises ``RuntimeError``.

Both methods support synchronous and asynchronous completion. Pass
``async_op=True`` to receive an ``AsyncWork`` handle, then call ``wait()`` or
``async_wait()``. Completion covers both control and payload transfer, and
failures from either side are propagated to the caller.

Algorithm Behavior
------------------

The worker selects stored fields from configuration rather than exposing a
different channel API per algorithm:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mode
     - Additional behavior
   * - PPO / GRPO / NFT
     - Assemble rewards, done flags, policy inputs, log-probabilities, and values
       required by the configured loss and advantage function.
   * - SAC / CrossQ
     - Preserve observation transitions when ``rollout.collect_transitions`` is
       enabled and emit trajectories for replay-buffer insertion.
   * - DAgger
     - Apply intervention actions and flags; online LeRobot mode drains completed
       episodes instead of returning standard trajectories.
   * - RLT
     - Extract and store RLT transition observations from model forward inputs.
   * - Training pipeline
     - Finalize each epoch, calculate advantages, route actor-specific batches,
       and split them into configured micro-batches.

Training pipeline mode does not currently support online LeRobot data or
decoupled environment mode. Invalid combinations fail during
``TrajectoryWorker`` initialization.

Operational Checks
------------------

If training stops making progress, identify the boundary before changing queue
or placement settings:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Check
   * - Rollout waits for policy input
     - Inspect the ordinary Env-to-Rollout ``Channel`` path; no trajectory event
       has been produced yet.
   * - Actor waits after rollout completes
     - Confirm every rollout rank and pipeline stage published its matching end
       event for the same ``step_id``.
   * - Pipeline actor waits
     - Confirm the actor subscribes with ``actor:<rank>`` and that the configured
       batch size is divisible by ``actor.micro_batch_size``.
   * - Placement fails at startup
     - Check ``cluster.trajectory_node_rank`` against ``cluster.num_nodes`` and
       confirm Ray sees that node rank.

See :doc:`Channel <channel>` for ordinary queue communication and
:doc:`Placement <placement>` for the cluster resource model.

Trajectory Channel
==================

Use ``TrajectoryChannel`` to move embodied rollout data from rollout workers to
actor workers without assembling complete trajectories in environment workers.
It keeps trajectory state in one dedicated ``TrajectoryWorker`` and transfers
payloads with worker-to-worker communication.

When RLinf Uses It
------------------

Embodied runners create a ``TrajectoryChannel`` for the Env/Rollout-to-Actor
data path. Environment and rollout workers exchange inference requests through
ordinary :doc:`Channel <channel>` objects. They publish their own training data
directly to the trajectory worker, which joins the two streams by action chunk.

This division keeps each component focused:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Component
     - Responsibility
   * - ``EnvWorker``
     - Send ``PolicyInput`` values, execute actions, and publish the resulting
       rewards, done flags, and algorithm-specific environment data.
   * - ``RolloutWorker``
     - Run policy inference, return ``PolicyOutput`` actions, publish model
       outputs, and process terminal observations while environments simulate.
   * - ``TrajectoryWorker``
     - Reassemble routed source fragments, join environment and model results,
       apply delayed updates, and emit actor-ready data.
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
       |                        | PolicyStep            |                    |
       |                        |---------------------->|                    |
       |     PolicyOutput       |                       |                    |
       |<-----------------------|                       |                    |
       | simulate action        |                       |                    |
       | EnvStepResult          |                       |                    |
       |----------------------------------------------->| join by key        |
       | TerminalRequest*       |                       |                    |
       |----------------------->| TerminalResult*       |                    |
       |                        |---------------------->| append when ready  |
       |          ... repeat for each action chunk ... |                    |
       | TrajectoryEnd          |                       |                    |
       |----------------------------------------------->| flush completed    |
       |                        |                       | trajectory / batch |
       |                        |                       |------------------->|

``*`` marks terminal processing, which runs only for a terminal state or the
last action chunk. The rollout worker may receive either request from any
environment rank in decoupled mode. It does not keep cross-request trajectory
state.

Events and Results
------------------

``TrajectoryChannel.publish()`` accepts the following internal event types:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Event
     - Meaning
   * - ``TrajectoryStart``
     - Initial done and termination state for one logical environment source.
   * - ``PolicyStep``
     - Observation, action, log-probability, value, version, and model forward
       inputs produced by rollout inference.
   * - ``EnvStepResult``
     - Reward, done flags, interventions, reward-model output, and optional
       online LeRobot episode data produced by the environment.
   * - ``TerminalResult``
     - Terminal observation, bootstrap value, and terminal model inputs produced
       from a ``TerminalRequest``.
   * - ``TrajectoryEpochEnd``
     - One logical environment source finished an epoch.
   * - ``TrajectoryEnd``
     - One logical environment source finished a training step. The worker
       flushes only after every expected action key is complete.

For ordinary training, ``subscribe()`` returns a complete ``Trajectory``. For
online LeRobot DAgger it returns completed episode dictionaries. With training
pipeline mode enabled, the actor subscribes to its ``actor:<rank>`` queue and
receives prepared micro-batches.

Routing and Ordering
--------------------

``TrajectoryKey`` identifies an action with ``step_id``, ``epoch_id``,
``env_rank``, ``stage_id``, and ``chunk_id``. ``TrajectorySource`` adds the
source shard's batch size and offset. Channel routing may split one environment
batch across several rollout workers or merge unrelated sources. The trajectory
worker restores the original batch by offset before joining events with the same
key.

In decoupled mode, ``PolicyInput`` and ``TerminalRequest`` use separate ordinary
Channel tags. Either request may reach any available rollout worker. Only policy
requests record a return route because only actions return to the environment.
The trajectory worker, rather than rollout-worker affinity or a terminal
barrier, determines when an action chunk is complete.

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
     - Confirm Env published ``EnvStepResult`` for every key and Rollout published
       the matching ``PolicyStep`` and required ``TerminalResult``.
   * - Pipeline actor waits
     - Confirm the actor subscribes with ``actor:<rank>`` and that the configured
       batch size is divisible by ``actor.micro_batch_size``.
   * - Placement fails at startup
     - Check ``cluster.trajectory_node_rank`` against ``cluster.num_nodes`` and
       confirm Ray sees that node rank.

See :doc:`Channel <channel>` for ordinary queue communication and
:doc:`Placement <placement>` for the cluster resource model.

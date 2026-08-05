TrajectoryChannel: Transport Complete Trajectories
===================================================

TrajectoryChannel assembles rollout segments into complete training trajectories
before handing them to the actor. EnvWorker and RolloutWorker continue to
exchange small policy requests and actions, while EnvWorker no longer retains
the data for an entire training round. Embodied training runners create and use
the channel automatically, so you usually do not need to change training code.

Overview
--------

An embodied run uses two communication paths. Normal Channels handle stepwise
interaction; TrajectoryChannel handles complete trajectories. Keep these roles
separate.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Path
     - Payload
     - Purpose
   * - EnvWorker → RolloutWorker
     - ``PolicyInput``: the current observation, the previous action's ``EnvResult``, and logical-source metadata
     - Request the next policy inference.
   * - RolloutWorker → EnvWorker
     - ``PolicyOutput.actions``
     - Execute policy actions.
   * - RolloutWorker → TrajectoryWorker
     - ``TrajectorySegment``, ``TrajectoryEpochEnd``, and ``TrajectoryEnd``
     - Persist training data incrementally and mark rollout-epoch and training-step boundaries.
   * - TrajectoryWorker → ActorWorker
     - A complete ``Trajectory``, or actor micro-batches in pipeline mode
     - Compute advantages, update the policy, or fill a replay buffer.

Communication Flow
------------------

Policy interaction stays between EnvWorker and RolloutWorker. RolloutWorker
publishes segments and completion events to one TrajectoryWorker on a selected
node. TrajectoryWorker does not run environment steps or policy inference; it
emits only complete actor inputs.

Each ``TrajectorySegment`` contains the observation before execution, rollout
information such as actions, log-probabilities, and values, and the resulting
environment state. The first segment also carries the initial done state.
``TrajectoryEpochEnd`` supplies the value bootstrap for the final state.
TrajectoryWorker can therefore build a trajectory equivalent to the original
training path without accessing EnvWorker memory.

Routing and Completion Semantics
--------------------------------

``PolicyInput.sources`` records each logical source in a batch as
``(env_rank, stage_id, size)``. RolloutWorker carries that metadata into the
trajectory events. TrajectoryWorker splits mixed batches with the same metadata,
so every segment from one logical environment source reaches the same collector
without hashing.

Outside pipeline mode, TrajectoryWorker flushes a training step only after every
rollout worker has sent ``TrajectoryEnd``. It splits each logical source into the
number of shards required by the actors, so every actor receives the same number
of shards. An actor's ``take()`` therefore receives complete trajectories, not a
partial episode or rollout epoch.

In pipeline mode, the completion boundary is finer. When every producer has sent
``TrajectoryEpochEnd`` for one rollout epoch, TrajectoryWorker immediately
prepares and sends the actor micro-batches for that epoch. This preserves the
existing pipeline timing without requiring EnvWorker to retain or preprocess a
complete trajectory.

Usage
-----

Launch existing embodied training configurations as usual. ``EmbodiedRunner``
creates TrajectoryChannels for Env, Rollout, and Actor; the Actor channel also
starts TrajectoryWorker. Normal ``put/get`` behavior is unchanged, so existing
worker interfaces do not change.

For multi-node training, select the node that stores trajectories with
``trajectory_worker_node_rank`` under ``runner``:

.. code-block:: yaml

   runner:
     trajectory_worker_node_rank: 1

This is a Ray cluster node rank and defaults to ``0``. It must satisfy
``0 <= trajectory_worker_node_rank < cluster.num_nodes``. Place the worker on a
node close to rollout and actor networking with sufficient CPU memory: it holds
one training step of trajectories until the actor consumes them.

.. warning::

   Do not call ``publish()`` or ``take()`` from user code or a runner process.
   They are worker-internal APIs and require a TrajectoryChannel bound to a
   worker. Start training through the existing runner and configuration APIs.

Worker-Internal APIs
--------------------

Use the following APIs when extending rollout or actor workers. They are
internal interfaces. Do not forward trajectory events over Ray RPC or ordinary
``put``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Contract
   * - ``TrajectoryChannel.publish(event)``
     - RolloutWorker only. Asynchronously sends a ``TrajectorySegment``, ``TrajectoryEpochEnd``, or ``TrajectoryEnd`` to TrajectoryWorker.
   * - ``TrajectoryChannel.take()``
     - ActorWorker only. Returns asynchronous work that resolves to a complete ``Trajectory`` or a pipeline micro-batch.
   * - ``TrajectoryChannel.put/get``
     - Inherited from ``Channel`` with unchanged semantics. Use these for ordinary worker messages; they do not assemble trajectories.

A new rollout implementation must publish one ``TrajectorySegment`` for every
executed action, ``TrajectoryEpochEnd`` after every epoch, and
``TrajectoryEnd`` after all epochs in a training step. Omitting an end event
leaves actors waiting for an incomplete trajectory.

Algorithms and Modes
--------------------

TrajectoryWorker selects the training fields from configuration instead of
introducing a separate transport protocol for each algorithm.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Scenario
     - Handling
   * - PPO, GRPO, and actor-critic training
     - Collects actions, rewards, done flags, log-probabilities, values, and the final-state bootstrap. The actor computes advantages after receiving complete trajectories.
   * - SAC, CrossQ, DSRL, and RLT
     - The same trajectory path retains transitions or adjacent states required by RLT. The actor writes complete trajectories to a replay buffer or performs the corresponding update.
   * - Online LeRobot DAgger
     - Collects episode data and outputs episodes according to ``only_success``. It does not create a standard ``Trajectory``.
   * - ``runner.use_training_pipeline=True``
     - Supports only ``algorithm.adv_type: gae``. It does not support online LeRobot, decoupled environment mode, ``embodied_sac``, ``rlt_ac``, ``embodied_dagger``, or ``embodied_nft``.

See also :doc:`Channel <channel>`, :doc:`Worker and WorkerGroup <worker>`,
:doc:`Placement <placement>`, and :doc:`Execution Modes <execution_modes>`.

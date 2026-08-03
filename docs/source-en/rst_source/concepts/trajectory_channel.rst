TrajectoryChannel: Asynchronous Trajectory Data Flow
=====================================================

Use ``TrajectoryChannel`` to decouple environment interaction, policy inference, and training-data assembly into an asynchronous data flow. Environments and rollout workers exchange actions by request, while the system aggregates environment, rollout, value, and reward records into training batches in the background. The actor receives data only after a trajectory batch is complete.

Choose it for embodied training with asynchronous or decoupled execution, especially when environments, rollout workers, and actors run on different nodes. In most cases, enable it in your configuration. The training entrypoint creates the channel and gives each worker its permitted view; you do not call ``publish`` or ``take`` directly in a training script.

Overview
--------

``TrajectoryChannel`` defines the producer, consumer, and forwarding path for every trajectory message type. It uses three dedicated worker types:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Role
   * - ``TrajectoryChannelWorker``
     - Forwards short-lived online requests and responses, such as ``PolicyInput`` from an environment and ``PolicyOutput`` from rollout.
   * - ``TrajectoryStorageWorker``
     - Receives training records, aggregates them into ``TrajectoryBatch``, ``PipelineMicroBatch``, or ``LeRobotEpisodeBatch`` in the background, and delivers the result to an actor.
   * - ``TrajectoryControllerWorker``
     - Tracks which worker has data available and chooses one storage owner for each training batch.

Use it in these situations:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Situation
     - What you get
   * - Environments and rollout run asynchronously
     - Each environment shard requests actions independently instead of waiting for every environment to form one synchronous batch.
   * - Actors, environments, and rollout run separately
     - Requests prefer a local channel/storage worker and fall back to round-robin selection across all available workers.
   * - You use external rewards or value bootstrap
     - Reward and value results are aggregated with their matching environment and rollout records. The actor does not align them itself.
   * - You use online LeRobot DAgger
     - Consecutive step records stay on one storage owner and produce an actor-local episode batch after completion.

Use It
------

TrajectoryChannel is the embodied training data plane. By default, its channel and
storage workers share the actor placement. Add explicit ``trajectory_channel`` and
``trajectory_storage`` placements only when they should run elsewhere. The following
is a single-node example. For multiple nodes, replace the node numbers with
placements that match your actor, environment, and rollout deployment.
For eval-only runs without an actor group, they share the rollout placement instead.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor,env,rollout: 0
       trajectory_channel,trajectory_storage: 0

   trajectory_channel:
     max_queue_size: 0
     num_record_threads: 4

Launch the configuration through the existing embodied training entrypoint:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh <config_name>

This command loads ``examples/embodiment/config/<config_name>.yaml``.
``train_embodied_agent.py`` and ``train_async.py`` create ``TrajectoryChannel``
after creating worker groups and before training begins. The runner then gives env,
rollout, actor, and optional reward workers their routed channel views. The only
ordinary channels retained by the embodied runners carry metrics.

.. note::

   ``TrajectoryChannel`` supports ``runner.use_training_pipeline: true`` for the
   embodied FSDP pipeline. Pipeline mode currently requires
   ``algorithm.adv_type: gae`` and does not support ``algorithm.type: sac``,
   ``dsrl``, ``rlt_ac``, ``dagger``, ``nft``, or ``opd``. ``env.train.total_num_envs``
   must be divisible by the actor world size, so storage can construct fixed
   trajectory shards for each actor.

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Setting
     - Default
     - Meaning
   * - ``trajectory_channel.max_queue_size``
     - ``0``
     - Maximum number of items in each internal queue. ``0`` means an unbounded queue: producers do not block on capacity, but memory can grow when consumers continually fall behind.
   * - ``trajectory_channel.num_record_threads``
     - ``4``
     - Background threads per storage worker for writing and aggregating records. Increasing this can improve record-processing concurrency but also increases CPU contention.
   * - ``trajectory_channel`` in ``cluster.component_placement``
     - actor placement (rollout for eval-only)
     - Optionally places online request/response queue workers. Prefer placing them near high-frequency environment or rollout traffic.
   * - ``trajectory_storage`` in ``cluster.component_placement``
     - actor placement (rollout for eval-only)
     - Optionally places trajectory aggregation workers. Prefer placing them near actors to reduce cross-node transfers of completed training batches.

How It Works
------------

One training step has two concurrent data paths. The action-request path needs low latency. The training-record path needs complete, consistently ordered data for one batch. Separating them lets environments progress without waiting for an actor to collect trajectories from other shards.

.. code-block:: text

   Action request path (each chunk)

   Env                 ChannelWorker        Rollout             ChannelWorker        Env
    | PolicyInput             |                 |                       |               |
    |------------------------>|---------------->| inference             |               |
    |                         |                 | PolicyOutput          |               |
    |<----------------------------------------------------------------------------------|
    | apply action and step   |                 |                       |               |

   Training record path (in the background)

   Env / Rollout / Reward ---> StorageWorker(owner) ---> TrajectoryBatch ---> Actor
          EnvResult               aggregate by BatchKey          complete       train
          RolloutResult
          ValueResult / RewardResult

The environment first publishes ``PolicyInput`` with the current observations, training step, rollout epoch, chunk step, and environment-shard information. Rollout takes the request, runs inference, and publishes ``PolicyOutput`` with the same shard identity. The environment takes its action with the explicit ``(env_rank, "train")`` or ``(env_rank, "eval")`` partition, preventing responses from different environment workers from mixing.

While stepping the environment, env publishes ``EnvResult`` and rollout publishes ``RolloutResult``. When the value head is enabled, env also publishes ``ValueRequest`` and rollout returns ``ValueResult``. When an external reward model is enabled, env publishes ``RewardRequest`` and the reward worker returns ``RewardResult``. These records do not go directly to the actor. Storage aggregates them and emits one completed batch.

With ``runner.use_training_pipeline: true``, the completion boundary is one
``rollout_epoch`` rather than the whole training step. Storage reconstructs the
same actor-local source partitions used by the legacy pipeline, preprocesses the
trajectory, calculates advantages and returns, normalizes advantages across all
environment ranks that feed the actor, then shuffles and packs training
micro-batches. It delivers ``PipelineMicroBatch`` messages directly to the
actor. The actor starts training as soon as an epoch is ready and reuses each
micro-batch for its configured update epochs; data is not returned to an
environment worker.

Batch Ownership and Completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regular training records use ``BatchKey(global_step, actor_rank)`` as their owner key. The controller chooses one ``TrajectoryStorageWorker`` for each key, so the ``EnvResult``, ``RolloutResult``, ``ValueResult``, and ``RewardResult`` for one actor and training step are written in one place. Storage tracks the expected positions for every record type from ``rollout_epoch``, the chunk count, and the actor's environment slots. It emits ``TrajectoryBatch`` only after every enabled record is present.

An actor then consumes the batch, converts it to the training trajectory layout, computes advantages and returns, and trains. After the batch is sent successfully, the controller releases its owner so later steps can be balanced across storage workers again.

Pipeline records instead use ``PipelineBatchKey(global_step, rollout_epoch,
actor_rank)``. This pins every source shard for one actor and rollout epoch to
one storage worker. The owner is released only after the last
``PipelineMicroBatch`` has been delivered.

Online LeRobot DAgger uses a different long-lived owner: ``LeRobotOwnerKey(actor_rank)``. It keeps consecutive environment streams for one actor on the same storage worker so episodes can be assembled across steps. Storage emits ``LeRobotEpisodeBatch`` when every slot reaches a rollout boundary.

Supported Algorithms and Records
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``algorithm.type`` is required for embodied runs and selects the
``TrajectoryChannel`` protocol. It is independent of ``algorithm.loss_type``
and ``algorithm.adv_type``: those fields still select the policy loss and
advantage estimator, while the channel never infers its protocol from either
field. The supported values and their records are listed below.

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - Algorithm
     - Shared by every algorithm
     - Additional records
   * - ``ppo``
     - ``PolicyInput``, ``PolicyOutput``, ``EnvResult``, ``RolloutResult``, and ``TrajectoryBatch`` (or ``PipelineMicroBatch`` in pipeline mode)
     - ``ValueRequest`` / ``ValueResult``; ``RewardRequest`` / ``RewardResult`` when an external reward is used.
   * - ``nft`` / ``opd``
     - Same as PPO.
     - Same as PPO.
   * - ``sac``
     - Same as above
     - No additional channel records.
   * - ``grpo``
     - Same as above
     - ``RewardRequest`` / ``RewardResult``.
   * - ``dsrl``
     - Same as above
     - No additional channel records.
   * - ``dagger``
     - Same as above
     - ``LeRobotStepResult`` / ``LeRobotEpisodeBatch`` in online LeRobot mode.
   * - ``rlt_ac``
     - ``PolicyInput``, ``PolicyOutput``, ``EnvResult``, ``RolloutResult``, and ``TrajectoryBatch``.
     - No additional channel records.

Communication Protocol and Data Transfer
----------------------------------------

``TrajectoryChannel`` uses RLinf worker ``send``/``recv`` calls for the payload rather than passing large tensors as Ray RPC arguments. Each message is first split by ``TrajectoryData.flatten()`` into:

* structure data for non-tensor fields;
* CPU tensors keyed by field path; and
* a routing ``QueueKey``, sent as a piggyback payload with the transfer.

``QueueKey`` contains the message type, producer, consumer, and an optional partition. The controller tracks only which worker has a message for a matching key. A consumer first reserves an available worker from the controller, then receives the data from that worker. Producers and consumers therefore do not need to know each other's rank in advance.

For large, high-frequency tensors, ``PolicyInput``, ``ValueRequest``, ``RewardRequest``, ``TrajectoryBatch``, ``PipelineMicroBatch``, and LeRobot data attempt LZ4 compression before transfer. By default, only contiguous CPU tensors of at least 64 KiB are compressed, in 1 MiB blocks; data stays raw when compression is not beneficial. This needs no extra configuration, but custom participants that directly publish compressed messages must use contiguous CPU tensors.

Capacity and Placement
~~~~~~~~~~~~~~~~~~~~~~

Start with ``max_queue_size: 0`` and ``num_record_threads: 4``. After confirming stable operation, adjust them according to the bottleneck:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - What you observe
     - Adjust first
   * - Memory keeps growing while rollout or reward is much slower than the environment
     - Set a finite ``max_queue_size`` to propagate backpressure to producers. Then inspect rollout/reward throughput and batch size.
   * - CPUs spend sustained time recording or compressing
     - Increase ``num_record_threads`` gradually and watch CPU utilization. Do not use more threads to hide a rollout or network bottleneck.
   * - Cross-node communication is a large share of runtime
     - Place ``trajectory_channel`` near environments or rollout and ``trajectory_storage`` near actors. The channel automatically prefers available local workers.
   * - An actor waits a long time for training data
     - Check that its environment, rollout, value, and reward records are all being produced. Storage emits only a complete batch.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - What to check
   * - Startup cannot find placement for ``trajectory_channel`` or ``trajectory_storage``
     - Configure both component names under ``cluster.component_placement``. The names must match exactly.
   * - Storage creation says the environment count is not divisible by actor world size
     - Adjust ``env.train.total_num_envs`` or the actor placement world size so the former is divisible by the latter.
   * - Pipeline startup reports an unsupported algorithm
     - Use ``algorithm.adv_type: gae`` and an embodied FSDP PPO-style actor loss. Pipeline mode does not support ``algorithm.type: sac``, ``dsrl``, ``rlt_ac``, ``dagger``, ``nft``, or ``opd``.
   * - A consumer waits or a training batch does not appear
     - Confirm that the algorithm is in the support table, all required workers are running, and reward/value features have their corresponding service worker. A complete batch needs every enabled record.
   * - Storage writing, deserialization, or compression fails
     - Inspect the first storage-worker exception. The failure propagates to waiting channel consumers instead of leaving them waiting silently.

Relation to Other Concepts
--------------------------

``TrajectoryChannel`` builds on the generic :doc:`Channel <channel>`, workers, and collective communication, but defines fixed embodied message types, routes, batch ownership, and completion semantics. Use the generic ``Channel`` for a simple producer-consumer queue. Enable ``TrajectoryChannel`` when you need environment interaction, inference, rewards, and training records to become complete trajectory batches that can be consumed asynchronously. For deployment and cross-node placement, continue with :doc:`Cluster <cluster>` and :doc:`Execution Model <execution-model/index>`.

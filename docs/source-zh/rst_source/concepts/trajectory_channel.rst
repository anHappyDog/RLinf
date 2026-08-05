TrajectoryChannel：传输完整轨迹
=================================

TrajectoryChannel 将 rollout 产生的片段拼成完整训练轨迹，再交给 actor。
它让 EnvWorker 与 RolloutWorker 继续交换小的策略请求和动作，而不会由 EnvWorker
长期保存整轮训练数据。具身训练 runner 会自动创建并使用它；通常不需要修改训练代码。

概览
----

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - 路径
     - 传输内容
     - 用途
   * - EnvWorker → RolloutWorker
     - ``PolicyInput``：当前观测、上一动作的 ``EnvResult`` 和逻辑来源信息
     - 请求下一次策略推理。
   * - RolloutWorker → EnvWorker
     - ``PolicyOutput.actions``
     - 执行策略动作。
   * - RolloutWorker → TrajectoryWorker
     - ``TrajectorySegment``、``TrajectoryEpochEnd``、``TrajectoryEnd``
     - 逐段持久保留训练所需数据，并标识 rollout epoch 与训练 step 的边界。
   * - TrajectoryWorker → ActorWorker
     - 完整 ``Trajectory``，或 pipeline 的 actor micro-batch
     - 计算优势、更新策略或写入 replay buffer。

通信流程
--------

EnvWorker 与 RolloutWorker 之间只保留策略交互。RolloutWorker 向指定节点上的单个
TrajectoryWorker 发布片段和完成事件；它不参与模型推理或环境执行，只向 actor 输出完整训练数据。

每个 ``TrajectorySegment`` 同时包含执行前观测、rollout 的动作/对数概率/价值信息，
以及动作执行后的环境结果。第一个 segment 还带有初始 done 状态；
``TrajectoryEpochEnd`` 提供末状态的 value bootstrap。TrajectoryWorker 因而能在不访问
EnvWorker 内存的情况下构造与原有训练路径等价的轨迹。

路由与完成语义
--------------

``PolicyInput.sources`` 记录每个 batch 切片的逻辑来源：``(env_rank, stage_id, size)``。
RolloutWorker 将这份来源信息原样放入 trajectory event。TrajectoryWorker 按该信息拆分
混合 batch，因此一个逻辑环境来源的所有 segment 始终进入同一个 collector，不依赖 hash。

在非 pipeline 模式下，所有 rollout worker 都发出 ``TrajectoryEnd`` 后，
TrajectoryWorker 才会 flush 当前训练 step。它将每个逻辑来源按 actor 所需份数切分，
使每个 actor 收到相同数量的 shard。actor 的 ``take()`` 因此只会取得完整轨迹，不会看到
半个 episode 或半个 rollout epoch。

在 pipeline 模式下，完成边界更细：每个 rollout epoch 的 ``TrajectoryEpochEnd`` 到齐后，
TrajectoryWorker 立即准备并发送该 epoch 的 actor micro-batch。这样保留 pipeline 原有的
训练时序，而不需要 EnvWorker 保存或预处理完整轨迹。

使用方式
--------

对现有具身训练配置，直接照常启动即可。``EmbodiedRunner`` 会为 Env、Rollout 和 Actor
创建 TrajectoryChannel；其中 Actor channel 额外启动 TrajectoryWorker。普通的
``put/get`` 语义保持不变，现有 worker 接口不需要改动。

多节点训练时，如需指定轨迹存储所在节点，在 ``runner`` 中设置
``trajectory_worker_node_rank``：

.. code-block:: yaml

   runner:
     trajectory_worker_node_rank: 1

该值是 Ray 集群中的节点 rank，默认是 ``0``。它必须满足
``0 <= trajectory_worker_node_rank < cluster.num_nodes``。将 worker 放在 rollout 与 actor
之间网络较近、且有足够 CPU 内存的节点；它保存一整个训练 step 的轨迹，直到 actor 消费。

.. warning::

   不要在用户代码或 runner 进程中调用 ``publish()`` 或 ``take()``。它们仅供 worker
   内部使用，并且只接受已经绑定到 worker 的 TrajectoryChannel。用户代码应继续通过
   配置和现有 runner 接口启动训练。

Worker 内部接口
------------------

扩展 rollout 或 actor worker 时，使用以下接口。它们是内部接口；不要把 trajectory event
通过 Ray RPC 或普通 ``put`` 转发。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 接口
     - 契约
   * - ``TrajectoryChannel.publish(event)``
     - 仅由 RolloutWorker 调用。异步发送 ``TrajectorySegment``、``TrajectoryEpochEnd`` 或 ``TrajectoryEnd`` 到 TrajectoryWorker。
   * - ``TrajectoryChannel.take()``
     - 仅由 ActorWorker 调用。返回异步 work；完成后得到完整 ``Trajectory``，或 pipeline micro-batch。
   * - ``TrajectoryChannel.put/get``
     - 继承自 ``Channel``，语义未变。用于普通 worker 间消息，不参与轨迹拼接。

新 rollout 实现必须为每个执行过的动作发布一个 ``TrajectorySegment``，在每个 epoch 末发布
``TrajectoryEpochEnd``，并在训练 step 的所有 epoch 完成后发布 ``TrajectoryEnd``。遗漏结束
事件会让 actor 等待尚未完成的轨迹。

算法与模式
-------------

TrajectoryWorker 根据配置收集训练所需字段，而不是为每个算法建立一套传输协议。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 场景
     - 处理方式
   * - PPO、GRPO 与 actor-critic 类训练
     - 收集动作、奖励、done、logprob、value 和末状态 bootstrap；actor 在收到完整轨迹后计算优势。
   * - SAC、CrossQ、DSRL 与 RLT
     - 同一轨迹路径额外保留 transition 或 RLT 所需的相邻状态；actor 将完整轨迹写入 replay buffer 或执行对应更新。
   * - Online LeRobot DAgger
     - 收集 episode data，并按 ``only_success`` 设置输出 episode；不构造标准 ``Trajectory``。
   * - ``runner.use_training_pipeline=True``
     - 仅支持 ``algorithm.adv_type: gae``。不支持 online LeRobot、环境解耦模式，以及 ``embodied_sac``、``rlt_ac``、``embodied_dagger``、``embodied_nft``。


相关概念包括 :doc:`Channel <channel>`、:doc:`Worker 与 WorkerGroup <worker>`、
:doc:`放置策略 <placement>` 和 :doc:`执行模式 <execution_modes>`。

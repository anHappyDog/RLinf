Trajectory Channel
==================

使用 ``TrajectoryChannel`` 将具身 rollout 数据从 rollout worker 传给 actor
worker，而不在 environment worker 中组装完整 trajectory。它由一个独立的
``TrajectoryWorker`` 保存 trajectory 状态，并通过 worker 间通信传输数据。

RLinf 何时使用它
-----------------

具身 runner 为 Rollout 到 Actor 的数据流创建 ``TrajectoryChannel``。Environment
和 Rollout 仍通过普通 :doc:`Channel <channel>` 交换策略请求与动作。Rollout
worker 在每个 action chunk 后发布训练所需的数据，Actor 则订阅组装完成的训练项。

各组件的职责如下：

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 组件
     - 职责
   * - ``EnvWorker``
     - 执行环境，并发送包含 observation 与前一个 ``EnvResult`` 的
       ``PolicyInput``。
   * - ``RolloutWorker``
     - 执行策略推理、返回 ``PolicyOutput`` action，并发布
       ``TrajectorySegment`` 事件。
   * - ``TrajectoryWorker``
     - 保持 source 顺序、追加 segment、应用滞后更新，并输出可供 Actor 使用的
       trajectory 或 pipeline micro-batch。
   * - ``ActorWorker``
     - 订阅完成的数据并执行算法对应的更新。

数据流
------

下面展示一个 rollout epoch。``Channel`` 和 ``TrajectoryChannel`` 使用独立的
数据流，因此 trajectory 组装不会向策略响应通路增加 payload。

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

Rollout worker 为每个 pipeline stage 保留一个待完成的推理结果。收到下一个
``PolicyInput`` 时，它同时拥有前一个 action 的结果和下一个环境状态，因此可以发布
一个完整 segment。最后一个 policy input 用于关闭最后一个待完成 segment。

事件与结果
----------

``TrajectoryChannel.publish()`` 接受三种内部事件：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 事件
     - 含义
   * - ``TrajectorySegment``
     - 一次 append 操作，包含 observation、next observation、环境结果、rollout
       结果和逻辑 source 元数据。
   * - ``TrajectoryEpochEnd``
     - 一个 producer 已完成一个 epoch；同时携带基于 value 计算 advantage 所需的
       final value。
   * - ``TrajectoryEnd``
     - 一个 producer 已完成一个训练 step。只有全部预期 producer 都完成后，worker
       才会 flush 数据。

普通训练中，``subscribe()`` 返回完整 ``Trajectory``。Online LeRobot DAgger
中，它返回已完成的 episode 字典。启用 training pipeline 后，Actor 从
``actor:<rank>`` 队列订阅准备好的 micro-batch。

路由与顺序
----------

每个 policy input 使用 ``(env_rank, stage_id, batch_size)`` 记录逻辑 source。
Channel 路由可以拆分或合并策略 batch，但 source 记录会随数据一起传递。
Trajectory worker 使用这些记录恢复每个 environment rank、每个 stage 的数据流，
再按顺序追加 segment。

完成的数据按照 actor world size 分片。这带来两个性质：

* 一个逻辑 source 的 segment 始终进入同一个 collector，不会跨 worker 组装同一条
  trajectory。
* 输出 shard 数可以被逻辑 source 数整除，因此每个 source 产生相同数量的 Actor
  可消费数据项。

Pipeline 模式使用 ``CommMapper`` 将每个逻辑 source 映射到 actor rank。它在 epoch
完成后计算 advantage，按配置在 Actor 对应的 batch 上执行 normalization，然后向
Actor 队列输出 micro-batch。

Placement
---------

Trajectory 组装目前只使用一个物理 ``TrajectoryWorker``。通过
``cluster.trajectory_node_rank`` 选择其节点：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     trajectory_node_rank: 1

默认使用 node rank ``0``。取值必须位于 ``[0, cluster.num_nodes - 1]``。当节点间
网络较慢时，将该 worker 放在主要 trajectory 流量附近。这个配置不会改变
``ChannelWorker`` 的 placement 或普通 ``Channel`` 的路由。

通信语义
--------

``publish()`` 和 ``subscribe()`` 是仅供 worker 内部使用的接口。每次操作使用小型
Ray RPC 协调，并使用 RLinf P2P ``send``/``recv`` 传输 payload。在 RLinf
``Worker`` 外调用任一接口都会抛出 ``RuntimeError``。

两个接口都支持同步和异步完成。传入 ``async_op=True`` 可获得 ``AsyncWork``
句柄，再调用 ``wait()`` 或 ``async_wait()``。完成状态同时覆盖控制操作与 payload
传输；任一侧的错误都会传递给调用者。

算法行为
--------

Worker 根据配置选择保存字段，而不是为每个算法暴露不同的 channel 接口：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 模式
     - 额外行为
   * - PPO / GRPO / NFT
     - 组装配置的 loss 和 advantage 函数所需的 reward、done flag、policy input、
       log-probability 与 value。
   * - SAC / CrossQ
     - 当 ``rollout.collect_transitions`` 启用时保留 observation transition，并为
       replay buffer 输出 trajectory。
   * - DAgger
     - 应用 intervention action 和 flag；online LeRobot 模式输出完成的 episode，
       而非标准 trajectory。
   * - RLT
     - 从模型 forward input 中提取并保存 RLT transition observation。
   * - Training pipeline
     - 完成每个 epoch、计算 advantage、路由 Actor 专属 batch，并按照配置拆分
       micro-batch。

Training pipeline 当前不支持 online LeRobot 数据或 decoupled environment 模式。
无效组合会在 ``TrajectoryWorker`` 初始化时直接失败。

运行检查
--------

如果训练停止推进，先确定阻塞边界，再调整队列或 placement 配置：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 检查项
   * - Rollout 等待 policy input
     - 检查普通 Env 到 Rollout 的 ``Channel`` 通路；此时还没有产生 trajectory
       事件。
   * - Rollout 完成后 Actor 仍等待
     - 确认每个 rollout rank 和 pipeline stage 都为相同 ``step_id`` 发布了对应的
       end 事件。
   * - Pipeline Actor 等待
     - 确认 Actor 使用 ``actor:<rank>`` 订阅，并且配置的 batch size 可以被
       ``actor.micro_batch_size`` 整除。
   * - 启动时 placement 失败
     - 对照 ``cluster.num_nodes`` 检查 ``cluster.trajectory_node_rank``，并确认 Ray
       能识别该 node rank。

普通队列通信参见 :doc:`Channel <channel>`，集群资源模型参见
:doc:`Placement <placement>`。

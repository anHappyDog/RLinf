理解 Trajectory Collector
=========================

阅读本页，理解具身 rollout 数据如何从一个 action chunk 变成 Actor 训练 batch。
算法开发者应关注算法生产和消费的字段；Collector 负责 routing、配对和完成状态。

核心模型
--------

一个 action chunk 被拆成两个独立产生的 part：

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Part
     - 所有者
     - 核心数据
   * - ``PolicyPart``
     - Policy inference
     - Observation、action、log-probability、value、model version 和保留的
       forward inputs。
   * - ``EnvPart``
     - Environment interaction
     - Reward、termination state、next observation、intervention result 和可选的
       terminal inference data。

两个 part 携带相同的 ``TrajectoryKey``：

.. code-block:: text

   (step_id, epoch_id, env_rank, stage_id, chunk_id)

这个 key 标识 chunk。Routing 拆分或合并 source batch 时，``TrajectorySource``
额外保存 batch size 和 offset。

Actor Channel 看到的数据流如下：

.. code-block:: text

   PolicyPart -----------+
                         |
                         v
                    按 key 配对
                         |
                         v
   EnvPart --------------+----> 累积 ----> flush ----> Actor 输出

Environment 不直接写 Actor Channel。它通过 ``PolicyCompletion`` 将已完成 transition
返回 Rollout，再由 Rollout 发布 ``EnvPart``。因此 Actor Channel 只有一个 producer
和一个确定的顺序边界。

闭合一个 Action Chunk
---------------------

跟踪一个普通 inference chunk：

1. Env 使用当前 observation 和新的 ``TrajectoryKey`` 发送 ``PolicyInput``。
2. Rollout 执行 policy inference，将 ``PolicyOutput(actions)`` 返回 Env，并立即向
   Actor Channel 发布 ``PolicyPart``。
3. Env 执行 action chunk，记录 rewards、boundary flags、next observation 和
   intervention data。
4. Env 将结果放进 ``PolicyCompletion``。Chunk zero 还携带 ``initial_result``，即
   第一个 action 之前的边界。
5. Rollout 将 completion 转为 ``EnvPart``。如果算法需要 terminal value 或 terminal
   transition feature，Rollout 会先对 terminal observation 执行 inference。
6. ``ChunkJoiner`` 按 key 配对两个 part，抵达顺序不影响结果。
7. ``TrajectoryCollector`` 将完整 chunk 交给 ``TrajectoryPlan`` 选择的 output
   strategy。

时间语义如下：

.. code-block:: text

   initial_result                   EnvPart.result
         |                                |
         v                                v
      state_t ---- PolicyPart.action ----> state_t+1

只有 chunk zero 保存 ``initial_result``。这样无需单独的 start event，也能对齐边界
序列。

从 Chunk 到 Actor 输出
----------------------

Collector 执行三个核心操作。

配对
~~~~

``ChunkJoiner`` 恢复 routed source fragment，并暂存 policy 和 environment part，
直到一个 key 的两侧都到达。完整 chunk 包含一个 source id、一个 ``PolicyPart`` 和
一个 ``EnvPart``。

累积
~~~~

``TrajectoryAccumulator`` 按 rollout 顺序追加完整 chunk。Action 字段通常每个 chunk
有一个元素；boundary 字段包含第一个 action 之前的状态，value 字段还可能包含最终
bootstrap value：

.. code-block:: text

   actions:      [a0, a1, ...]
   rewards:      [r0, r1, ...]
   dones:        [d_before, d0, d1, ...]
   prev_values:  [V(s0), V(s1), ..., V(s_final)]

累积过程中，Collector 还会处理 reward-model weight、terminal bootstrap、延迟
history reward、实际执行的 intervention，以及可选 transition observation。

Flush
~~~~~

Output strategy 定义可以输出的最小完成 scope。Scope 完成后，它会物化连续 tensor、
删除 accumulator 状态，并返回可入队的数据项。

输出模式
--------

.. list-table::
   :header-rows: 1
   :widths: 20 32 48

   * - 模式
     - 完成 Scope
     - Actor 输出
   * - ``rollout``
     - 一个 logical source 的一个 training step
     - 默认 key 上的 ``Trajectory`` shard。不同 source 独立 flush。
   * - ``pipeline``
     - 所有 source 的一个 ``(step_id, epoch_id)``
     - 已完成预处理和 advantage 计算、按 Actor rank 绑定的 micro-batch。
   * - ``lerobot``
     - 一个 logical source 的一个 training step
     - 已完成 episode shard；未结束 episode 的状态跨 training step 保留。

同步、异步和 decoupled runner 使用相同的两个 part 和同一个 Collector。它们的
service loop 不同，但 chunk 数据契约不变。

算法开发者需要负责什么
------------------------

从算法需要的训练 tensor 出发，再沿着所有权路径检查。

.. list-table::
   :header-rows: 1
   :widths: 27 31 42

   * - 算法需求
     - 产生位置
     - Collector 职责
   * - Log-probability 或 value 等 policy statistic
     - Rollout 中的 ``EmbodiedRolloutResult``
     - 从 ``PolicyPart`` 复制到 ``ChunkStepResult``，并物化到 ``Trajectory``。
   * - Environment 或 reward-model signal
     - Env 中的 ``EnvResult``
     - 合并 reward source，并追加到同一个 chunk。
   * - Current 和 next observation
     - ``PolicyPart.obs`` 和 ``EnvPart.next_obs``
     - 启用 ``rollout.collect_transitions`` 时追加 transition。
   * - 实际执行的 intervention action
     - ``EnvResult.intervene_actions`` 和 ``intervene_flags``
     - 替换已保存的 policy action，并保留 intervention mask。
   * - RLT transition feature
     - 两侧保留的 ``forward_inputs``
     - 提取 ``z_rl``、``proprio`` 和 ``ref_chunk``，应用实际 intervention 后保存
       transition。
   * - Pipeline training target
     - Algorithm advantage/loss configuration
     - 在按 Actor 打包 micro-batch 之前计算 advantages 和 returns。

新增算法时检查以下五点：

1. 根据数据所有者，将字段添加到 ``EmbodiedRolloutResult`` 或 ``EnvResult``。
2. 在 Rollout 或 Env 中赋值，并保持 CPU transport 和 routed split/merge 行为。
3. 在 Collector 的 joined-chunk 路径中追加字段。
4. 将字段物化到 ``Trajectory`` 或对应模式的 Actor 输出。
5. 保持 pipeline preprocessing 与非 pipeline Actor 路径一致。

大多数算法无需修改 ``TrajectoryKey``、``TrajectorySource``、fragment restoration、
dispatcher selection 或 completion scope。只有算法改变数据所有权或 batch flush
条件时才修改这些部分。

最小完整流程
------------

假设只有一个 Env source、一个 chunk、一个 rollout epoch 和一个 Actor：

1. Bootstrap 产生 action 前的 observation 和 boundary state。
2. Rollout 发布一个 ``PolicyPart``，并将 action 发给 Env。
3. Env 执行 action，返回一个 ``PolicyCompletion``。
4. Rollout 使用同一个 key 发布一个 ``EnvPart``。
5. Joiner 创建一个完整 chunk。
6. Accumulator 追加 initial boundary、action data、environment data 和 final value。
7. Rollout scope 完成，Collector 在默认 queue key 上输出一个 ``Trajectory``。
8. Actor 将收到的 trajectory 转为 training batch，然后在非 pipeline 路径中计算
   advantages 和 returns。

存在更多 chunk 时，步骤 2–6 会重复执行后再 flush。存在更多独立 source 时，
rollout 模式可以在每个 source 完成后立即 flush。Pipeline 模式会等待 epoch 中所有
source，因为 routing 和可选 advantage normalization 作用于 Actor-specific batch。

阅读接口
--------

阅读 :doc:`channel` 了解 Collector 和 Dispatcher 的执行方式。阅读
:doc:`../reference/api/embodied_data` 查看准确的数据类和 Collector API。实现位于
``rlinf/data/schema/trajectory_collector.py`` 和
``rlinf/data/schema/trajectory_accumulator.py``。

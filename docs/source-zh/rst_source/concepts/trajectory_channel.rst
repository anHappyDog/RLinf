TrajectoryChannel：异步轨迹数据流
=================================

使用 ``TrajectoryChannel`` 将环境交互、策略推理和训练数据整理解耦为异步数据流。它让环境与 rollout worker 按请求交换动作，同时在后台按训练批次聚合环境、rollout、价值和奖励记录；actor 只在一批轨迹完整时接收训练数据。

选择它来运行启用了异步或解耦执行的 embodied 训练，特别是环境、rollout 和 actor 分布在不同节点时。通常只需在配置中启用它；训练入口会创建 channel 并把相应视图传给各个 worker，无需在训练脚本中手动调用 ``publish`` 或 ``take``。

Overview
--------

``TrajectoryChannel`` 为每种轨迹消息定义固定的生产者、消费者和转发路径。它包含三类专用 worker：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 组件
     - 作用
   * - ``TrajectoryChannelWorker``
     - 转发短生命周期的在线请求和响应，例如环境发出的 ``PolicyInput`` 与 rollout 返回的 ``PolicyOutput``。
   * - ``TrajectoryStorageWorker``
     - 接收训练记录，在后台聚合成 ``TrajectoryBatch``、``PipelineMicroBatch`` 或 ``LeRobotEpisodeBatch``，再交给 actor。
   * - ``TrajectoryControllerWorker``
     - 记录哪个 worker 有可取数据，并为同一训练批次选择唯一的 storage owner。

它适合以下场景：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 场景
     - 你得到的效果
   * - 环境与 rollout 异步运行
     - 每个环境分片独立请求动作，不必等待所有环境拼成一个同步批次。
   * - actor、环境和 rollout 分开部署
     - 请求优先发往本节点的 channel/storage worker；没有本地 worker 时才在所有可用 worker 中轮询。
   * - 需要外部奖励或 value bootstrap
     - 奖励和价值结果与对应的环境、rollout 记录按同一批次聚合，actor 无需自行对齐。
   * - online LeRobot DAgger
     - 连续 step 记录会保留在同一个 storage owner 中，完成后输出 actor 本地的 episode 批次。

使用方式
--------

TrajectoryChannel 是 embodied 训练的数据面。默认情况下，它的 channel 和
storage worker 与 actor 使用相同 placement。只有需要将它们部署到其他位置时，才需要显式添加 ``trajectory_channel`` 和 ``trajectory_storage`` placement。下面是单节点示例；多节点时把节点编号替换为与你的 actor、环境和 rollout 部署相匹配的 placement。
对于不含 actor group 的纯评估运行，它们会改为使用 rollout placement。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor,env,rollout: 0
       trajectory_channel,trajectory_storage: 0

   trajectory_channel:
     max_queue_size: 0
     num_record_threads: 4

然后使用现有的 embodied 训练入口启动你的配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh <config_name>

该命令会加载 ``examples/embodiment/config/<config_name>.yaml``。
``train_embodied_agent.py`` 和 ``train_async.py`` 会在创建工作组后、训练开始前创建 ``TrajectoryChannel``；runner 随后把 env、rollout、actor 和可选 reward 所需的路由视图传入各自的 worker。embodied runner 中保留的普通 Channel 仅用于 metrics。

.. note::

   ``TrajectoryChannel`` 支持具身 FSDP 的
   ``runner.use_training_pipeline: true``。pipeline 模式目前要求
   ``algorithm.adv_type: gae``，且不支持 ``algorithm.type: sac``、
   ``dsrl``、``rlt_ac``、``dagger``、``nft`` 和 ``opd``。``env.train.total_num_envs``
   必须能被 actor world size 整除，storage 才能为每个 actor 构造固定的轨迹分片。

配置项
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - 配置
     - 默认值
     - 含义
   * - ``trajectory_channel.max_queue_size``
     - ``0``
     - 每个内部队列的最大项目数。``0`` 表示无界队列：不会因队列容量阻塞生产者，但当消费者持续落后时会增长内存占用。
   * - ``trajectory_channel.num_record_threads``
     - ``4``
     - 每个 storage worker 用于写入和聚合轨迹记录的后台线程数。提高该值可增加记录处理并发，但也会增加 CPU 竞争。
   * - ``cluster.component_placement`` 中的 ``trajectory_channel``
     - actor placement（纯评估时为 rollout）
     - 可选地放置在线请求/响应队列 worker。建议优先与高频的环境或 rollout 流量同节点部署。
   * - ``cluster.component_placement`` 中的 ``trajectory_storage``
     - actor placement（纯评估时为 rollout）
     - 可选地放置轨迹聚合 worker。建议优先与 actor 同节点部署，减少完整训练批次的跨节点传输。

运行机制
--------

一次训练 step 同时有两条数据路径：动作请求路径要求低延迟；训练记录路径要求同一批数据完整且顺序一致。两者分开后，环境不必等待 actor 收集其他分片的轨迹。

.. code-block:: text

   动作请求路径（每个 chunk）

   Env                 ChannelWorker        Rollout             ChannelWorker        Env
    | PolicyInput             |                 |                       |               |
    |------------------------>|---------------->| 推理                  |               |
    |                         |                 | PolicyOutput          |               |
    |<----------------------------------------------------------------------------------|
    | 执行动作、推进环境       |                 |                       |               |

   训练记录路径（后台并行）

   Env / Rollout / Reward ---> StorageWorker(owner) ---> TrajectoryBatch ---> Actor
          EnvResult               按 BatchKey 聚合            完整后输出       训练
          RolloutResult
          ValueResult / RewardResult

环境先发布 ``PolicyInput``，其中包含当前观测、训练 step、rollout epoch、chunk step 和环境分片信息。rollout 取到请求后执行推理，发布带有相同分片标识的 ``PolicyOutput``；环境使用显式分区 ``(env_rank, "train")`` 或 ``(env_rank, "eval")`` 取回属于自己的动作，避免不同环境 worker 的响应混淆。

在推进环境的同时，env 发布 ``EnvResult``，rollout 发布 ``RolloutResult``。启用 value head 时，env 还会发布 ``ValueRequest``，rollout 消费后返回 ``ValueResult``。启用外部奖励模型时，env 发布 ``RewardRequest``，reward worker 返回 ``RewardResult``。这些记录不直接交给 actor，而是由 storage 聚合后一次性输出。

启用 ``runner.use_training_pipeline: true`` 时，完成边界从整个训练 step
变成单个 ``rollout_epoch``。storage 会重建旧 pipeline 使用的 actor 本地源分片，
预处理轨迹、计算 advantage/return、汇总所有向该 actor 供数的环境 rank 的
advantage 统计量，再完成归一化、shuffle 和训练 micro-batch 打包。它将
``PipelineMicroBatch`` 直接交给 actor；actor 在一个 epoch 就绪后立即开始训练，
并按配置的 update epoch 重用 micro-batch，数据不会回传给 env worker。

批次所有权和完成条件
~~~~~~~~~~~~~~~~~~~~~~

普通训练记录以 ``BatchKey(global_step, actor_rank)`` 作为 owner key。controller 为每个 key 选择一个 ``TrajectoryStorageWorker``，因此同一 actor 的同一训练 step 的 ``EnvResult``、``RolloutResult``、``ValueResult`` 和 ``RewardResult`` 会在同一处写入。storage 根据 ``rollout_epoch``、chunk 数和该 actor 的环境 slot 数追踪每类记录的预期位置；所有启用的记录齐全时，才产生 ``TrajectoryBatch``。

该批次随后被 actor 消费。actor 将它转换为训练所需的轨迹布局、计算 advantage/return 并训练。批次成功发送后，controller 释放对应的 owner，后续 step 可以重新均衡到其他 storage worker。

pipeline 记录使用 ``PipelineBatchKey(global_step, rollout_epoch, actor_rank)``。
它将一个 actor 的一个 rollout epoch 的全部源分片固定到同一个 storage worker；
只有最后一个 ``PipelineMicroBatch`` 成功交付后，controller 才释放 owner。

online LeRobot DAgger 使用不同的长期 owner：``LeRobotOwnerKey(actor_rank)``。它让同一 actor 的连续环境流始终落在同一个 storage worker，以便跨 step 组装 episode；当所有 slot 到达一个 rollout 边界后，storage 输出 ``LeRobotEpisodeBatch``。

支持的算法和记录
~~~~~~~~~~~~~~~~~~

embodied 运行必须配置 ``algorithm.type``，它用于选择
``TrajectoryChannel`` 协议。它与 ``algorithm.loss_type``、
``algorithm.adv_type`` 相互独立：后两者仍分别选择策略损失和优势估计器，
channel 不会从其中任一字段推断协议。支持的取值及其记录如下。

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - 算法
     - 所有算法共有
     - 额外记录
   * - ``ppo``
     - ``PolicyInput``、``PolicyOutput``、``EnvResult``、``RolloutResult``、``TrajectoryBatch``（pipeline 模式下为 ``PipelineMicroBatch``）
     - ``ValueRequest`` / ``ValueResult``；使用外部奖励时还有 ``RewardRequest`` / ``RewardResult``。
   * - ``nft`` / ``opd``
     - 与 PPO 相同。
     - 与 PPO 相同。
   * - ``sac``
     - 同上
     - 无额外的 channel 记录。
   * - ``grpo``
     - 同上
     - ``RewardRequest`` / ``RewardResult``。
   * - ``dsrl``
     - 同上
     - 无额外的 channel 记录。
   * - ``dagger``
     - 同上
     - online LeRobot 模式还使用 ``LeRobotStepResult`` / ``LeRobotEpisodeBatch``。
   * - ``rlt_ac``
     - ``PolicyInput``、``PolicyOutput``、``EnvResult``、``RolloutResult`` 和 ``TrajectoryBatch``。
     - 无额外的 channel 记录。

通信协议和数据传输
------------------

``TrajectoryChannel`` 使用 RLinf 的 worker ``send``/``recv`` 传输实际数据，而不是把大 tensor 作为 Ray RPC 参数。每条消息先由 ``TrajectoryData.flatten()`` 拆为：

* 非 tensor 字段组成的结构信息；
* 以字段路径为键的 CPU tensor；
* 用于路由的 ``QueueKey``，作为通信 piggyback payload 一并发送。

``QueueKey`` 包含消息类型、生产者、消费者和可选分区。controller 只维护“哪个 worker 有匹配 key 的消息”；消费者先向 controller reserve 一个可用 worker，再从该 worker 接收数据。因此生产者和消费者不需要预先知道对方的具体 rank。

对于较大的高频 tensor，``PolicyInput``、``ValueRequest``、``RewardRequest``、``TrajectoryBatch``、``PipelineMicroBatch`` 和 LeRobot 数据会在传输前尝试 LZ4 压缩。默认仅压缩不小于 64 KiB 的连续 CPU tensor，按 1 MiB 分块，并在压缩收益不足时保留原始数据。这个实现细节不需要额外配置，但意味着自定义的参与方在直接发布压缩消息前应保证 tensor 位于 CPU 且连续。

容量与放置建议
~~~~~~~~~~~~~~~~

先从 ``max_queue_size: 0`` 和 ``num_record_threads: 4`` 开始。确认稳定后，再根据瓶颈调整：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 观察到的现象
     - 优先调整
   * - 内存持续增长，且 rollout 或奖励明显慢于环境
     - 设置有限的 ``max_queue_size``，让慢消费者把背压传回生产者；同时检查 rollout/reward 的吞吐和 batch 大小。
   * - CPU 长时间忙于记录或压缩
     - 逐步增加 ``num_record_threads``，并观察 CPU 利用率。不要仅靠增加线程掩盖 rollout 或网络瓶颈。
   * - 跨节点通信占比高
     - 将 ``trajectory_channel`` 放在环境或 rollout 附近，将 ``trajectory_storage`` 放在 actor 附近；channel 会自动优先选择本节点的可用 worker。
   * - 某个 actor 长时间等不到训练数据
     - 检查该 actor 分片的环境、rollout、value 和奖励记录是否都在产生；storage 只会在一批记录完整后输出。

故障定位
--------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 症状
     - 检查项
   * - 启动时找不到 ``trajectory_channel`` 或 ``trajectory_storage`` placement
     - 在 ``cluster.component_placement`` 中同时配置两个组件名。名称必须与上述配置完全一致。
   * - 创建 storage 时提示环境数不能整除 actor world size
     - 调整 ``env.train.total_num_envs`` 或 actor placement 的 world size，使前者能被后者整除。
   * - pipeline 启动时报算法不支持
     - 使用 ``algorithm.adv_type: gae`` 和具身 FSDP 的 PPO 风格 actor loss。pipeline 模式不支持 ``algorithm.type: sac``、``dsrl``、``rlt_ac``、``dagger``、``nft`` 和 ``opd``。
   * - 消费者等待或训练批次迟迟不出现
     - 确认算法在支持表中，确认每个必要 worker 已启动，并检查 reward/value 功能是否有对应的服务 worker。完整批次需要全部已启用的记录。
   * - storage 写入、反序列化或压缩失败
     - 查看最先出现的 storage worker 异常。失败会传播到等待中的 channel 消费者，而不是继续静默等待。

与其他概念的关系
------------------

``TrajectoryChannel`` 建立在通用 :doc:`Channel <channel>`、worker 和集合通信之上，但它为 embodied trajectory 定义了固定的消息类型、路由、批次所有权和完成语义。若你只需要简单的生产者—消费者队列，请使用通用 ``Channel``；若你需要把环境交互、推理、奖励和训练记录组织成可异步消费的完整轨迹批次，请启用 ``TrajectoryChannel``。部署和跨节点 placement 的背景可继续阅读 :doc:`Cluster <cluster>` 与 :doc:`执行模型 <execution-model/index>`。

Nsight Systems
==============================

本文介绍 RLinf 中基于 ``cluster.nsight`` 的系统级 Profiling 配置，用于通过
NVIDIA Nsight Systems 对指定 Ray worker group 执行 ``nsys profile`` 包装。

借助这套机制，你可以采集 CUDA kernel、cuDNN、cuBLAS、NVTX，以及可选的
CPU runtime 相关时间线。

如何启用
------------------------------

在具身 YAML 的 ``defaults`` 中引入 Nsight 预设：

.. code-block:: yaml

   defaults:
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - nsight/default@cluster.nsight

对应的配置文件是：

- ``examples/embodiment/config/nsight/default.yaml``


默认预设
------------------------------

内置的默认预设如下：

.. code-block:: yaml

   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup, Actor, Rollout, Env]
   options:
     t: cuda,cudnn,cublas,nvtx,osrt
     sample: process-tree
     cpuctxsw: process-tree
     osrt-threshold: 1000

这份默认配置会优先采样具身训练里最常见的计算 worker 和通信 worker：

- ``ActorGroup``
- ``RolloutGroup``
- ``EnvGroup``
- ``Actor``
- ``Rollout``
- ``Env``

这里的名字必须和真实的 worker group 名一致，例如 ``actor.group_name``、
``rollout.group_name``，而不是组件别名 ``actor`` 或 ``rollout``。


``enabled`` 开关
------------------------------

``enabled`` 是 Nsight 的总开关：

.. code-block:: yaml

   cluster:
     nsight:
       enabled: false

当 ``enabled: false`` 时：

- RLinf 不会用 ``nsys profile`` 包装 worker
- RLinf 不会预留默认的 Nsight 输出目录
- 其余 profiling 配置可以保留，方便后续再次开启

因此不需要单独维护一份 ``disabled.yaml``，直接在主 YAML 里覆盖
``cluster.nsight.enabled: false`` 即可。


如何覆盖 worker_groups
------------------------------

你可以直接在主 YAML 里覆盖这份预设：

.. code-block:: yaml

   cluster:
     nsight:
       worker_groups: [EnvGroup, RolloutGroup, ActorGroup, Env, Rollout, Actor]

这对以下场景很有用：

- 采 actor / rollout 这类计算 worker
- 采 ``Env``、``Rollout``、``Actor`` 这类 channel worker
- 采 ``EnvGroup`` 这类环境 worker

如果省略 ``worker_groups``，RLinf 会对所有 worker group 开启 profiling。

这里有一个容易混淆的点：当前实现里的 ``ChannelWorker`` 不是
``ActorGroup`` / ``RolloutGroup`` 某个 rank 的子进程，而是通过
``Channel.create(name)`` 单独 launch 出来的独立 worker group，名字通常就是
``Env``、``Rollout``、``Actor``。因此只 profile ``ActorGroup`` 并不会自动
覆盖 ``Actor`` 这个 channel worker；如果你想看 channel 本身，需要把这些名字
显式加进 ``worker_groups``。


如何覆盖 Nsight 参数
------------------------------

``cluster.nsight.options`` 会被直接映射到 ``nsys profile`` 的 CLI 参数：

.. code-block:: yaml

   cluster:
     nsight:
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         cpuctxsw: process-tree

常用参数包括：

- ``t``: 需要采集的 API，例如 ``cuda``、``cudnn``、``cublas``、``nvtx``、``osrt``
- ``sample``: CPU sampling 模式
- ``cpuctxsw``: CPU 线程调度时间线
- ``capture-range`` 和 ``capture-range-end``: 用 NVTX 或 CUDA profiler API 控制采样窗口
- ``o`` 或 ``output``: 显式指定输出前缀

如果你开启了 ``capture-range: nvtx``，请确认代码里确实发出了 NVTX range；
否则 Nsight 很可能只会生成几乎没有内容的空 report。


输出路径
------------------------------

当 ``cluster.nsight.enabled`` 为 true，且没有显式指定 ``o`` / ``output`` 时，
RLinf 默认会把 report 写到：

.. code-block:: text

   runner.logger.log_path/runner.logger.experiment_name/nsights

例如：

.. code-block:: text

   ../results/libero_spatial_ppo_openpi/nsights/

如果你希望写入固定目录，可以显式覆盖：

.. code-block:: yaml

   cluster:
     nsight:
       options:
         o: /mnt/public/profiles/my_run/worker_trace


推荐使用方式
------------------------------

第一轮定位问题时，最简单的用法通常是：

- 先用 ``nsight/default@cluster.nsight``
- 保持 ``enabled: true``
- 如果你既想看 CUDA timeline，也想看 CPU/channel 侧 runtime 行为，默认 preset 可以直接用
- 在确认目标 worker 已经打出 NVTX 之前，不要急着加 ``capture-range: nvtx``

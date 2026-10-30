Franka Pickup 混合数据监督微调
================================

使用本页流程，可以选择已整理的全部或部分 Franka pickup 数据源，在 RLinf 中微调 π₀.₅。流程读取 LeRobot v3.0 分区与共享 catalog，生成按来源分组的训练集、验证集和归一化统计，再让一个 policy 学习多种原生控制格式。各来源保留原始频率与动作含义，部署时需选择匹配的控制器和归一化统计。

概览
----

单臂与双臂数据共用一个 loader；每个 source profile 指定相机映射与采样权重。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 方法
      :text-align: center

      全参数 SFT · 带 mask 的 Flow Matching

   .. grid-item-card:: 数据
      :text-align: center

      LeRobot v3.0 · 共享 Catalog

   .. grid-item-card:: 硬件
      :text-align: center

      2 张 NVIDIA GPU · FSDP

任务
~~~~

数据集合包含真机与仿真机器人的 pickup 片段。准备阶段接受源数据成功标记、干净 pickup 标注与视觉确认三类质量级别，排除仅有运动线索的候选片段，以及语义核心边界未知的片段。

.. list-table::
   :header-rows: 1
   :widths: 28 35 37

   * - Profile
     - 数据
     - 默认权重
   * - ``molmobot``
     - 仿真单臂
     - 4
   * - ``habit``
     - 真机双臂
     - 4
   * - ``fmb`` · ``droid``
     - 真机单臂
     - 各 1
   * - ``duobench_real`` · ``duobench_sim``
     - 真机 / 仿真双臂
     - 各 1
   * - ``gr00t_transport``
     - 已复核的仿真双臂装置片段
     - 1

观测与动作
~~~~~~~~~~

每个训练样本使用当前时刻的相机图像与 state，预测未来 50 步原生动作。源数据保留片段上下文，只有 pickup 监督锚点与未来仍位于核心动作范围内的有效目标参与 loss。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 训练表示
   * - Observation
     - 最多 3 路当前 RGB 图像，缩放到 224 × 224；缺失视角的 image mask 为 false。原生 state 按 schema 使用 ``tanh((state - mean) / std)``，并补齐至 32 维。
   * - Action
     - 50 × 32 张量。原生 7 / 8 / 14 / 16 维动作按 schema 归一化。补齐维度、片段末尾越界与核心范围外的目标不参与 loss。
   * - Prompt
     - 任务文本、schema、控制模式和原生 FPS；包含离散化 state 输入在内，最多 200 个 token。
   * - History
     - ``data.num_obs_steps: 1``。本配方使用当前观测，其他值会被明确拒绝。
   * - Loss
     - 先按每个样本有效动作元素求 flow-matching 误差均值，再对 batch 求均值。

50 步预测在 10 Hz 下覆盖 5 秒，在 30 Hz 下约覆盖 1.67 秒。loader 保留原生帧索引，不插值动作，也不缩放 delta 指令。MolmoBot 媒体为 15.15 Hz，其名义 0.066 秒控制周期仍作为独立源元数据保留。

准备数据
--------

从包含 ``catalog/episodes.jsonl`` 及所引用分区的转换后集合开始。原始视频需保持可访问：分区中可能保存相对链接，loader 会读取每个 episode 记录的视频起始偏移。

在训练环境中，从仓库根目录运行准备命令：

.. code:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/prepare_pickup_mixture.py \
     --catalog-root /path/to/unified_lerobot_v1 \
     --output-dir /path/to/prepared-pickup-mixture \
     --allow-development

命令生成 ``train.jsonl``、``val.jsonl``、``normalization.json`` 和 ``summary.json``，不复制媒体。``--allow-development`` 显式允许历史开发片段进入训练，使小规模 pilot 来源也能参与。省略该参数后，仅使用符合要求的非开发数据；添加 ``--sources molmobot habit`` 可选择部分 profile。每个选中的 profile 都必须有符合条件的训练 episode。

``--seed`` 与 ``--val-fraction`` 决定可复现的划分。同一场景的 episode 保持在同一 split；缺少场景标识时按 source group 分组。开发组始终保留在训练侧。归一化统计按 schema 分开计算，仅使用训练集核心动作行，验证数据不参与统计。Action 使用 mean/std 归一化；state 张量与离散 state tokenizer 统一使用 ``tanh((state - mean) / std)``，保持在 tokenizer 的有界输入范围内。常数列 std 取 1。推理时必须复用相同 schema 的统计与变换，并对动作执行逆变换，再发送给对应控制器。

小规模试训可添加 ``--max-train-episodes 2 --max-val-episodes 1``。这些限制在分组划分后按 profile 分别应用。启动训练前检查 ``summary.json``，确认实际 episode 与监督锚点数量；小规模来源可能没有验证 episode。

安装
----

按照 :doc:`OpenPI SFT <sft_openpi>` 创建独立训练环境，再准备数据和训练。旧 OpenPI 环境可能包含 LeRobot v2.1；本 loader 需要 v3 布局与 ``lerobot.datasets.video_utils``。本配方已验证的版本如下：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 包
     - 已验证版本
   * - torch
     - 2.6.0
   * - torchvision
     - 0.21.0
   * - transformers
     - 4.53.2
   * - lerobot
     - 0.4.3
   * - datasets
     - 4.1.1
   * - av
     - 15.1.0
   * - pyarrow
     - 24.0.0

完成基础 OpenPI 安装后，安装混合数据 loader 所需包并检查解码器导入。``--no-deps`` 保留现有 Torch 与 transformers 版本；应在已安装基础运行依赖的独立环境中执行。

.. code:: bash

   uv pip install --python /path/to/isolated-env/bin/python --no-deps \
     'lerobot==0.4.3' 'datasets==4.1.1' 'av==15.1.0' 'pyarrow<26'
   /path/to/isolated-env/bin/python -c \
     'from lerobot.datasets.video_utils import decode_video_frames; print(decode_video_frames.__name__)'

命令应输出 ``decode_video_frames``。本次 v3 读取支持属于新增 mixture loader，现有 OpenPI loader 仍各自遵循原有数据格式要求。

下载模型
--------

使用模型动作维度为 32 的 π₀.₅ base checkpoint。按照 :doc:`sft_openpi` 的下载和转换说明准备 RLinf 支持的布局，并将 ``actor.model.model_path`` 指向包含 ``model.safetensors`` 的目录。加载器也支持在加载时转换官方 OpenPI PyTorch checkpoint 的 key 名称。

本配方加载完整 base 模型并训练全部参数，使用 FP32 主参数和优化器更新、BF16 激活、FSDP ``full_shard`` 与 gradient checkpointing。训练读取混合数据生成的 ``normalization.json``，避免沿用其他机器人任务的统计。

运行
----

将训练与验证路径指向准备好的目录，使用标准 VLA SFT 入口启动。示例配置将 actor rank 放置到 GPU 0 和 1，micro batch size 为 1，global batch size 为 4。

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-path "$EMBODIED_PATH/config" \
     --config-name franka_pickup_sft_openpi_pi05 \
     data.train_data_paths=/path/to/prepared-pickup-mixture \
     data.val_data_paths=/path/to/prepared-pickup-mixture \
     actor.model.model_path=/path/to/pi05-base \
     runner.logger.log_path=/path/to/pickup-sft-results

默认运行 1,000 个优化器 step，每 100 步验证一次，并在第 1,000 步保存 checkpoint。短试训可追加 ``runner.max_steps=3 runner.val_check_interval=1 runner.save_interval=3``。``data.samples_per_epoch`` 控制每个 epoch 采样的训练锚点数，``data.eval_max_samples`` 限制验证成本。

采样器先按配置权重选择 schema，再从该 schema 中选择有效锚点，因此数据源的原始大小不会直接决定混合占比。调整 ``examples/sft/config/data/franka_pickup_sources.yaml`` 中的权重或相机映射后，重新准备数据，记录新的 mixture。

扩展数据源
----------

新集合沿用相同 catalog 与 LeRobot 布局时，新增 source profile 即可。为它指定唯一 ``name``、匹配的 ``source_dataset``、可选 ``domain``、正数 ``weight``，以及支持的相机槽位到图像 feature 的映射。一个 profile 必须对应一种控制模式与 FPS，异构来源需拆成多个 profile。

共享 loader 从 manifest 读取 episode、源视频偏移、数值 schema 与监督 mask。兼容的新来源通过 profile 接入，无需新增来源专属 loader 分支。新增观测模态或物理动作表示时，则需明确实现模型 adapter。

.. warning::

   归一化与维度补齐让一个模型学习多种原生 schema，但并不建立统一的物理动作空间。将所有来源转换到目标机器人控制器前，需要核实坐标系、单位、夹爪约定和运动学，再决定重采样或 retargeting。

可视化与结果
------------

在 ``runner.logger.log_path`` 下查看 TensorBoard 日志，checkpoint 保存在 ``checkpoints/global_step_<N>/``。验证使用固定 seed，结束后恢复训练 RNG 状态，并关闭图像增强。验证 flow loss 衡量模型对留出示范的预测误差；pickup 成功率需通过匹配控制器的独立闭环评测确认。共享日志约定见 :doc:`训练指标 </rst_source/reference/metrics>`。

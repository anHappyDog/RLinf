Franka Pickup 混合数据监督微调
================================

使用本页流程，可以选择已整理的全部或部分 Franka pickup 数据源，在 RLinf 中微调 π₀.₅。流程读取 LeRobot v3.0 分区与共享 catalog，生成按来源分组的训练集、验证集、测试集与归一化统计，再通过明确的控制 adapter 微调一个 policy。选择旧版原生动作配方或 v2 关节相对动作配方后，推理复用相同的观测与动作变换，各来源仍保留原始频率。

使用 MolmoBot checkpoint 时，按下文「使用官方 MolmoBot 模型」的流程操作；该入口保留 checkpoint 原始 processor、归一化与 H16 动作窗口。

概览
----

单臂与双臂数据共用一个 loader；每个 source profile 指定相机映射与采样权重。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅ · MolmoBot π₀

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
     - 旧版默认权重
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
     - 最多 3 路当前 RGB 图像，缩放到 224 × 224；缺失视角的 image mask 为 false。State 按 schema 编码、归一化后补齐至 32 维。旧配方使用 ``tanh((state - mean) / std)``，v2 关节 profile 使用分位数和规范后的夹爪值。
   * - Action
     - 50 × 32 张量。原生 7 / 8 / 14 / 16 维动作先经过选定的 adapter，再按 schema 归一化。补齐维度、片段末尾越界与核心范围外的目标不参与 loss。
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

对已经核验的 DROID 与 DuoBench 关节接口，使用 v2 source 配置。新的 pickup 集合与旧 catalog 分开存放时，可一次传入多个转换后根目录：

.. code:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/prepare_pickup_mixture.py \
     --catalog-root /path/to/unified_lerobot_v1 /path/to/duobench_pickup_v2 \
     --source-config examples/sft/config/data/franka_pickup_sources_v2.yaml \
     --output-dir /path/to/prepared-pickup-mixture-v2 \
     --action-horizon 50 --action-dim 32 \
     --sampling-unit episode --statistics-max-anchors 8192 \
     --allow-development

命令生成 ``train.jsonl``、``val.jsonl``、``test.jsonl``、``normalization.json``、``contract.json`` 和 ``summary.json``，不复制媒体。添加 ``--sources duobench_real duobench_sim`` 可选择部分 profile，每个选中的 profile 都需要符合要求的训练 episode。``--allow-development`` 允许历史 pilot 片段进入训练；省略它会排除这些片段。

Catalog 中已有的 ``assigned_split`` 可以固定为 ``train``、``val`` 或 ``test``。其余数据由 ``--seed`` 与 ``--val-fraction`` 按场景划分；缺少场景标识时，按 source group 分组。同一原始 episode 的所有切片保持在同一集合，开发组始终留在训练侧。测试 manifest 留给独立评测，SFT loader 只读取训练和验证集合。小规模试训可添加 ``--max-train-episodes 2 --max-val-episodes 1``，这些限制在划分后按 profile 分别应用。启动前检查 ``summary.json`` 中的实际 episode 和监督锚点数。

V2 的 DROID 和 DuoBench profile 将动作块中每一步未来关节目标减去当前观测关节位置。整段使用同一参考状态，与逐步计算相邻差值不同。夹爪映射为 0=open、1=closed，并跳过归一化。包含已核验 ``adapter`` 或 ``normalization`` 配置的 profile，在变换后计算统计量，只纳入仍有效且有监督的未来目标。确定性 hash 抽样从每个原始 episode 最多取 32 个锚点，每个 schema 最多取 ``--statistics-max-anchors`` 个锚点。分位数归一化使用 q01/q99，不做截断；manifest 记录抽样规模、变换与统计量。

保留原生动作配方时，使用 ``examples/sft/config/data/franka_pickup_sources.yaml``。没有 adapter 或 normalization 配置的 profile 继续使用动作 mean/std 和状态 ``tanh((state - mean) / std)``，统计来自训练集核心动作行。V2 source 文件只选择 DROID 和 DuoBench；其他来源在控制 contract 核验完成前，使用旧版配置。验证集、测试集始终不参与统计。

准备结果中的 contract 对划分 manifest 与归一化文件计算哈希。使用 adapter 的准备结果还绑定动作 horizon 和模型维度；改变任一项都需要重新计算统计。每个 checkpoint 在 ``mixture_assets`` 中保存归一化、summary、contract 和曝光记录。恢复训练时若准备数据发生变化，会明确拒绝继续，避免悄然更换动作语义。

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

使用 ``--sampling-unit episode`` 时，采样器先按权重选择 schema，再均匀选择该来源的原始 episode，最后从它的有效锚点中均匀选择。同一原始 episode 的多个切片共用这条 episode 的抽样概率。``--sampling-unit frame`` 保留旧版按 schema 内锚点均匀抽样的行为。权重表示来源占比，不随数据源大小变化；抽样允许重复。修改选用的 source 配置后，重新准备数据以记录新的 mixture。

远端主机的数据链路较慢时，可先物化模型实际接收的样本，再迁移缓存。设置 ``data.mixture.sample_cache_dir=/path/to/sample-cache`` 后，loader 分别读取其中的 ``train`` 和 ``val`` 子目录。每个子目录包含 ``metadata.json``，其中 ``contract`` 来自对应 dataset 的 ``sample_cache_contract()``，``indices`` 列出已缓存的 dataset 索引。将未经修改的 ``dataset[index]`` 字典使用 gzip 与 pickle protocol 5 保存为 ``<index>.pkl.gz``。缓存从已准备的数据集生成，传输后核验文件哈希。

缓存保留原始 dataset 长度、采样权重、来源分组、数值数组、像素值、token 和 mask。缓存 contract 绑定选定的 manifest 行、schema、维度、token 长度和预处理源代码哈希。缓存应覆盖所选 sampler seed、epoch、采样预算与验证样本上限会请求的全部索引；输入 contract 发生变化或缺少缓存索引时，读取会失败。修改训练配置后，需重新生成受影响的缓存。

训练与推理保持一致
------------------

推理使用同一份准备结果，将原生观测转为模型输入，再把预测动作块恢复为控制器命令。对于控制接口匹配、动作维度为 16 的 DuoBench 环境，在评测配置中设置：

.. code:: yaml

   actor:
     model:
       action_dim: 16
       openpi:
         task: eval
         config_name: native_mixture
         action_horizon: 50
         model_action_dim: 32
         mixture:
           manifest_dir: /path/to/prepared-pickup-mixture-v2
           schema_id: duobench_sim

配置会安装 ``MixturePolicyIO``，使用选定 schema 的变换与统计。``manifest_dir`` 也接受 checkpoint 的 ``mixture_assets`` 目录，并校验归一化文件与保存 contract 的哈希是否一致。环境按 batch 提供原生 ``states`` 和 ``task_descriptions``；RGB 采用 HWC 布局，可放在 ``images`` 的模型相机槽位下，或使用常规 ``main_images``、``wrist_images``、``extra_view_images``。图像值可以是 [0, 255] 的 uint8，或 [0, 1] 的浮点数。Schema 选择所需视角，训练与推理共用缩放、补边、prompt、状态编码和归一化。

动作解码先去除补齐维度，逆转归一化和夹爪映射，再为相对关节动作加回原始观测关节位置。原始状态显式随请求传递，归一化状态不会作为关节参考。``native_mixture`` 在线 I/O 当前支持普通 ``task: eval``。``rtc_enabled: true`` 会被明确拒绝；上一请求的相对动作块需要先转换到新观测的关节参考，才能用于 RTC。数值接口已有 CPU 覆盖；接入仿真器时仍需核验状态顺序、相机映射、控制命令和执行时间。Pickup 成功率需要闭环评测确认。

扩展数据源
----------

新集合沿用相同 catalog 与 LeRobot 布局时，新增 source profile 即可。为它指定唯一 ``name``、匹配的 ``source_dataset``、可选 ``domain``、正数 ``weight``，以及支持的相机槽位到图像 feature 的映射。一个 profile 必须对应一种控制模式与 FPS，异构来源需拆成多个 profile。可逆控制变换通过 ``adapter`` 指定，变换后的统计通过 ``normalization`` 指定。Source contract 应明确关节与状态索引、夹爪端点、单位和坐标系。

共享 loader 从 manifest 读取 episode、源视频偏移、数值 schema 与监督 mask。兼容的新来源通过 profile 接入，无需新增来源专属 loader 分支。新增观测模态或物理动作表示时，则需明确实现模型 adapter。

.. warning::

   归一化与维度补齐让一个模型学习多种原生 schema，但并不建立统一的物理动作空间。将所有来源转换到目标机器人控制器前，需要核实坐标系、单位、夹爪约定和运动学，再决定重采样或 retargeting。

可视化与结果
------------

在 ``runner.logger.log_path`` 下查看 TensorBoard 日志，checkpoint 保存在 ``checkpoints/global_step_<N>/``。验证使用固定 seed，结束后恢复训练 RNG 状态，并关闭图像增强。验证 flow loss 衡量模型对留出示范的预测误差；pickup 成功率需通过匹配控制器的独立闭环评测确认。共享日志约定见 :doc:`训练指标 </rst_source/reference/metrics>`。

使用官方 MolmoBot 模型
----------------------

使用 ``franka_pickup_sft_openpi_official``，可以通过原始 OpenPI PyTorch core 微调 MolmoBot checkpoint。该入口复用 checkpoint 的数据变换与归一化，输出绝对关节目标，动作窗口为 H16/D8。选择 MolmoBot 原生单臂 manifest，或使用下文的 DROID native8 reader；其他机器人接口需要各自明确的 adapter。

安装官方运行环境
~~~~~~~~~~~~~~~~

为官方模型创建独立的 Python 3.11 环境。安装脚本选择 Torch 2.7.1，并固定此配方使用的 OpenPI、MolmoBot 和 LeRobot 源码版本：

.. code:: bash

   UV_TORCH_BACKEND=cu126 bash requirements/install.sh embodied \
     --venv .venv-openpi-official --model openpi_official --env dummy
   source .venv-openpi-official/bin/activate

安装脚本在此环境应用原始 OpenPI transformers replacement，并安装模型与数据依赖。JAX 使用 CPU，模型计算由 Torch 执行。``dummy`` 目标选择 SFT 运行环境，对应的 Docker target 为 ``embodied-openpi-official``。

下载 checkpoint 时，一并保留训练 config 和归一化资产：

.. code:: bash

   hf download allenai/MolmoBot-Pi0-DROID \
     --revision 48c8e2fc1868fd34dba1dd059278e7a2430a5383 \
     --local-dir /path/to/molmobot-pi0-droid

模型目录必须包含 ``assets/train_config.pkl`` 和 ``assets/molmobot/norm_stats.json``。Loader 读取可信 checkpoint 的 config，重建原始 processor。

使用 Checkpoint 的数据变换训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将官方配方的数据路径指向已准备的原生 manifest，并指定 checkpoint 目录：

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-path "$EMBODIED_PATH/config" \
     --config-name franka_pickup_sft_openpi_official \
     data.train_data_paths.target=/path/to/prepared-native-pickup \
     data.train_data_paths.auxiliary=/path/to/prepared-molmo-pickup \
     data.val_data_paths.target=/path/to/prepared-native-pickup \
     actor.model.model_path=/path/to/molmobot-pi0-droid \
     runner.logger.log_path=/path/to/official-pickup-sft-results

模型沿用原始 mean/std 归一化和对全部动作元素计算的 Flow Matching loss。Loader 只选择未来 H16 步均有有效监督的 anchor，并记录每个样本的来源与原始 episode，用于统计学习次数。选择训练预算前，先核对实际保留的样本数。

读取 DROID native8 manifest 时，其 ``normalization.json`` 中的 prepared schema 应声明 ``native_control_mode: droid_absolute_joint_position_and_normalized_closure``。State 与 action 都由七个绝对关节位置和一个连续夹爪闭合量组成，0 表示打开，1 表示闭合。MolmoBot 路线保留原生 state9/action8 布局以及 driver/actuator-command 尺度。

DROID adapter 要求 checkpoint 使用 MolmoBot 的 ``MlSpacesDatasetConfigFactory``，且 ``joint_pos_actions=True``。DROID 已提供归一化闭合量，因此只跳过 ``GripperStateTransform`` 和 ``GripperActionTransform``；其余 checkpoint 数据变换、统计归一化和模型变换保持原有顺序。DROID 与 MolmoBot 的变换链分别缓存。统一存储格式并不使不同动作接口可以互换。

将准备好的 DROID 数据加入上述命名来源映射时，追加 ``+data.train_data_paths.droid=/path/to/prepared-droid``。该来源的 schema 名为 ``droid/<schema_id>``，使用现有辅助采样流。Reader 支持属于数据准备阶段；真实 SFT 与闭环 pickup 表现需要使用选定 checkpoint 分别验证。

示例使用 ``openpi_cosine``，峰值学习率为 5e-5，warmup 为 1000 次更新。第一次 optimizer 更新实际使用 5e-5 / 1001，约 4.995e-8；每次 optimizer 更新推进一次 scheduler。配置的 schedule 长度为 300000 次更新，``min_lr: 5e-5`` 使 warmup 后的学习率保持在峰值。

命名目录将来源 schema 分别放入 ``target/molmobot`` 和 ``auxiliary/molmobot`` 命名空间。``data.official.sampling: target_exposure`` 为目标与辅助来源使用独立的随机数生成器。通过 ``data.official`` 下的 ``target_samples_per_epoch`` 和 ``auxiliary_samples_per_epoch`` 控制两者次数；改变辅助样本数时，每个 epoch 内目标来源的抽样顺序和学习次数保持一致。

比较目标曝光一致的单训与混训时，运行相同数量的完整 epoch，并报告辅助数据增加的 optimizer 更新次数。固定 ``runner.max_steps`` 会让两组在不同位置结束目标数据流；需要核对保存的各来源样本计数。 单训时追加 ``runner.max_epochs=1 runner.max_steps=-1 data.official.target_samples_per_epoch=512 data.official.auxiliary_samples_per_epoch=0``；混训沿用相同覆盖值，仅将 ``data.official.auxiliary_samples_per_epoch=512``。Global batch size 为 4 时，两组分别更新 128 次和 256 次，各自学习 512 个目标样本。

加入辅助样本还会增加 optimizer 更新次数，并使 warmup 推进得更远。比较时，另外加入与混训更新次数相同的目标数据单训控制组，并报告实际目标样本数。每次 optimizer 更新前记录实际使用的学习率；现有 ``learning_rate`` 指标在 scheduler 推进后读取，表示下一次更新的学习率。

运行两次更新的目标数据单训接入测试时，将 ``OPENPI_OFFICIAL_MODEL_PATH`` 和 ``OPENPI_OFFICIAL_DATA_PATH`` 设为 checkpoint 与目标数据路径，设置 ``REPO_PATH`` 为仓库目录，再运行 ``bash tests/e2e_tests/sft/run_vla_sft.sh franka_pickup_sft_openpi_official_test``。对应的 SFT CI job 仅在仓库变量中配置这两个 runner 资产路径后启用。

导出到原始 Policy Loader
~~~~~~~~~~~~~~~~~~~~~~~~~

使用原始 OpenPI 或 MolmoBot policy 加载训练结果前，将 actor 导出到一个新目录：

.. code:: bash

   python -m rlinf.models.embodiment.openpi_official.checkpoint \
     --actor-dir /path/to/experiment/checkpoints/global_step_N/actor \
     --output-dir /path/to/exported-policy

导出器从 actor 目录读取 ``full_weights.pt`` 和 ``upstream_assets``，将带 ``core.*`` 前缀的张量严格加载到原始 CPU 模型，再保存含共享权重元数据的 ``model.safetensors``。原始训练 config 和数据归一化统计会复制到 ``assets``，导出完成后写入 ``EXPORT.json``。传输训练结果时，保留包含 ``EXPORT.json`` 的完整导出目录；RLinf 的 ``full_weights.pt`` 使用不同的 key，不能直接当作原始模型的 safetensors 文件。

RLinf 识别 ``EXPORT.json`` 后，自动通过 ``rlinf.models.embodiment.openpi_official.policy.load_exported_policy`` 加载导出权重。外部 OpenPI 或 MolmoBot 服务应将训练 config 和导出目录传给同一 helper。上游 factory 会先把 FP32 归一化层权重转为 BF16；helper 在最终 dtype 建立后严格重载导出张量，保留训练产生的 FP32 更新。发布方原始 checkpoint 仍沿用上游加载流程。

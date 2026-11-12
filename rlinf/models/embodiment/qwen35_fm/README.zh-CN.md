# Qwen3.5 Flow Matching Action Expert

使用这套实现，通过 [Knowledge Insulation](#使用-knowledge-insulation-预训练) 预训练 Qwen3.5 VLA，并比较 Attention、Gated DeltaNet（GDN）和 VLM 状态直接读取三种连续动作建模方式。Qwen 保留原生多模态骨干；下面的冻结 backbone FM 流程用于架构对照，KI 入口则通过不同目标训练 VLM 和 Expert。这些 standalone 入口使用原生 PyTorch 分布式执行，接入 RLinf 分布式 SFT worker 仍是后续工作。

## 选择架构

架构依据 [Qwen3.5-2B 官方配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) 和 [Transformers 原生 Qwen3.5 实现](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。

五份配置位于 `examples/sft/config/qwen35_fm/`。

| 配置文件 | Expert | 每次向量场评估的 cross-attention 次数 | Hidden / FFN 宽度 |
| --- | --- | --- | --- |
| `qwen35_gqa6.yaml` | 6 层双向 GQA | 6 | 1024 / 4096 |
| `qwen35_gqa6_gdn1.yaml` | 6 层 GQA 后叠加 1 层独立因果 GDN | 6 | 1024 / 4096 |
| `qwen35_gqa6_gdn2.yaml` | 6 层 GQA 后叠加 2 层独立因果 GDN | 6 | 1024 / 4096 |
| `qwen35_gqa6_gdn3.yaml` | 6 层 GQA 后叠加 3 层独立因果 GDN | 6 | 1024 / 4096 |
| `qwen35_gqa6_vlm_state.yaml` | 6 层 GQA 加 fast-weight 状态读取 | 6 | 1024 / 4096 |

三个叠加 GDN 的 Expert 都保留完整 GQA 基线：每个 GQA block 包含动作 self-attention、读取 VLM hidden states 的 cross-attention 和 FFN。六层骨干之后，再依次执行独立参数的 GDN block；每个 GDN block 处理动作特征，并带有自己的 FFN。这些模块不替换 GQA block，也不共享预训练 VLM 的 GDN 权重。V3 则在六层 GQA 中额外读取预训练 VLM 的 recurrent state。

五份预设统一使用 hidden size 1024、FFN size 4096、action horizon 16、action dimension 8，以及 16 个 query head 和 4 个 KV head。叠加 GDN 统一使用 8 个 key/value head，key/value head dimension 均为 128。参数量由实际增加的模块自然决定，不再通过调整宽度凑目标预算。比较时还应固定数据、seed、归一化、优化和 conditioning 设置。

`depth` 控制 GQA 骨干深度，`gdn_depth` 控制后置 GDN 层数；默认分别为六层 GQA 和 variant 名称中的 GDN 层数。`action_self_attn: false` 用于 GQA 只保留 cross-attention 的消融。每层都通过 adaptive normalization 接收 FM timestep，动作 embedding 还加入可学习的物理时间位置和 timestep embedding。

## 编码输入并预测动作

准备 PyTorch、Transformers 5.19.0、OmegaConf 和 PyYAML。SFT 还使用 PyArrow、PyAV、NumPy、Pillow 及原生 Qwen processor 的依赖。模型与 processor 文件须已在本地；`from_pretrained` 强制使用 `local_files_only=True`。

修改下面的 `config_path` 即可实例化表中的任意配置。VLM 默认冻结，设为 `trainable=True` 后保留 VLM 梯度。

```python
import torch
import yaml
from rlinf.models.embodiment.qwen35_fm.action_expert import ActionExpertConfig, build_action_expert
from rlinf.models.embodiment.qwen35_fm.policy import FlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper

config_path = "examples/sft/config/qwen35_fm/qwen35_gqa6_vlm_state.yaml"
config = ActionExpertConfig(**yaml.safe_load(open(config_path))["expert"])
vlm = Qwen35VLMWrapper.from_pretrained(
    "/mnt/public/daibo/models/Qwen3.5-2B", trainable=False,
    torch_dtype=torch.bfloat16, attn_implementation="eager",
).to("cuda:0")
expert = build_action_expert(config, vlm.state_specs).to("cuda:0", torch.bfloat16)
policy = FlowMatchingPolicy(vlm, expert)
# image is a PIL image; native processed image/video/text kwargs also work.
condition = vlm.encode(
    observation=image, instruction="Pick up the cup.",
    extract_states=config.variant == "gqa6_vlm_state",
)
actions = policy.sample(condition, steps=10, method="euler")
```

`encode` 返回 token 特征及有效性 mask；`predict_velocity(noisy_actions, timestep, condition)` 在同一噪声水平同时评估全部动作位置。`loss(actions, condition, action_mask=...)` 采样高斯噪声和均匀时间，使用 `A_tau=(1-tau)*epsilon+tau*A`、目标速度 `A-epsilon`，计算有效 action coordinate 上的 MSE。`sample` 用 Euler 或 Heun 从时间 0 的噪声积分到时间 1 的动作。训练前先归一化动作，机器人执行前再反归一化。

| 接口 | Shape |
| --- | --- |
| 动作、噪声、速度 | `[B,H,D_action]` |
| FM timestep | 预测接受 scalar 或 `[B]`；loss 接受 `[B]` |
| 动作 embedding | `[B,H,D_hidden]` |
| VLM 最终 hidden states | Qwen3.5-2B 为 `[B,L,2048]` |
| VLM token mask | `[B,L]`，非零表示有效 |
| 训练 action mask | `[B,H]` 或 `[B,H,D_action]` |
| Expert token mask | `[B,H]` |
| 选定的 recurrent state | `gdn_states[layer]: [B,H_v,K,V]` |
| 状态读取 query / 结果 | `[B,H,H_v,K]` / `[B,H,H_v,V]` |

## 读取原生多模态记忆

Qwen3.5-2B 的文本骨干共 24 层。按零起始索引，3、7、11、15、19、23 层使用 full attention，配置为 8 个 query head、2 个 KV head、head dimension 256；其余 18 层使用 GDN，配置为 16 个 key head、16 个 value head、key/value dimension 128、卷积宽度 4。原生 full attention 从 projection 中拆出 query 和 output gate。GDN 则有独立的 QKV、output gate、beta 和 decay projection，依次经过因果 depthwise convolution、Q/K normalization、gated delta rule 和 gated RMS normalization。

Wrapper 调用原生多模态骨干，直接取得特征而不计算语言 logits。`extract_states=True` 为当前调用新建 native cache，请求最终状态。支持的 Transformers 5.19 将状态放在 `layers[layer].recurrent_states[0]`，与 `conv_states[0]`、普通 attention KV 分开存储。已检查的旧 native cache 使用 `recurrent_states[layer]`，提取逻辑也支持该布局。状态尺寸从实际 text config 读取，并与返回 tensor 核对。

原生矩阵的方向为 `[B,H_v,K,V]`，动作 query 归一化后通过 `einsum("bthk,bhkv->bthv", q, S)` 读取。每个选定的 VLM 层都有独立的可学习 query/output projection，通过 `state_adapter_dim` 控制中间 bottleneck 宽度。Softmax layer weights 聚合各层 readout，再经 projection 和可学习 sigmoid gate，在 block FFN 前作为 residual 注入。Gate 初值为 `sigmoid(-4)`，约 0.018。`state_injection: per_block` 为每个 block 建立独立 adapter；`shared` 在全部 block 中复用同一个 adapter。状态矩阵不会展平为 VLM token。

通过 `Qwen35VLMWrapper(..., state_layers=[2,10,22])` 选择层，留空则提取所有 GDN 层。新 cache 中的 tensor 保留计算图，包括 native `copy_` 更新，trainable VLM 的状态不会 detach。可微状态提取要求关闭 gradient checkpointing。提取状态时必须左 padding，因为尾部 padding 仍会继续更新 recurrent decay；wrapper 将原生 processor 设为左 padding，并拒绝右 padding 的状态提取。

当前支持 `current_vlm_forward` scope。外部旧 cache 和 `persistent_stream` 会报错，直到明确实现增量输入与 episode reset 协议。Action GDN block 同样不保留跨调用 cache，每次向量场评估都独立初始化，包括每次 Euler/Heun 调用。GPU 默认复用 native chunkwise 实现；`gdn_kernel: reference` 使用保持同一状态方向的可微逐 token scan。CPU Expert 路径在需要时调用 native unwrapped PyTorch chunk 实现。环境安装 FLA 后，原生 VLM 的 CPU 执行可能选择 CUDA kernel，此时应在 GPU 上运行 native VLM 测试。缺少 `causal-conv1d` 时，因果卷积使用原生 PyTorch fallback。可选 PEFT adapter 接口已经暴露，本次 pilot 尚未验证。

全部 18 层的 FP32 recurrent matrix 存储量为 `B*18*16*128*128*4` bytes，即每个样本 18 MiB。六层 full attention 的 BF16 KV 存储量为 `B*L*6*2*2*256*2` bytes。每个 GDN 层的 convolution buffer 为 `[B,6144,4]`。这些解析估计不包含 activation、allocator overhead 和临时 cache；即使只返回少数状态，原生提取过程仍会为所有层建立 cache。

## 运行测试与小规模 SFT 验证

在仓库根目录使用上述 Python 环境运行：

```bash
PYTHONPATH=. python -m pytest tests/unit_tests/test_qwen35_fm.py -q
QWEN35_CHECKPOINT=/mnt/public/daibo/models/Qwen3.5-2B \
  PYTHONPATH=. python -m pytest tests/unit_tests/test_qwen35_fm.py -q -k local_pretrained_multimodal_checkpoint
```

Native VLM 测试要求使用已验证的 Transformers 5.19.0 环境，安装 FLA 时还需要 GPU。常规 CI 固定使用较旧的 Transformers，可能跳过或不支持可选 Qwen3.5 路径；本功能没有升级共享 CI 环境。

第一条命令测试 synthetic Expert 和 tiny native 多模态骨干；环境变量额外启用本地 checkpoint 验证。覆盖项包括 mask、reference recurrence、梯度、state readout、原生数值保持、层映射、状态隔离、optimizer step 和 ODE 输出 shape。

Pilot 使用已有的 LeRobot v3 MolmoBot 转换数据，保留 native train/validation 划分，并检查 `obs[i]` / `command[i+1]` 对齐。Action 包含七个关节目标位置和一个夹爪命令，loader 保留 8D action 及 supervision mask。实验从四个训练 episode、两个验证 episode 各选两个 horizon-16 chunk，即 8 个训练 chunk 和 4 个验证 chunk。Exterior 相机图像 resize 为 224×224，归一化参数只由选定训练 chunk 拟合。输出会分别记录源数据 metadata 的总帧数和本次真正使用的样本数。

```bash
PYTHONPATH=. python examples/sft/qwen35_fm_sft.py \
  --config examples/sft/config/qwen35_fm/qwen35_gqa6.yaml \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --output results/qwen35_fm_pilot_new \
  --variants gqa6_gdn1,gqa6_gdn2,gqa6_gdn3 \
  --steps 10 --batch-size 2 --train-episodes 4 --val-episodes 2 \
  --chunks-per-episode 2 --image-size 224 --precision bfloat16 --device cuda:0 \
  --seed 42 --lr 1e-4 --cache-conditions \
  --hidden-size 128 --intermediate-size 512 --num-heads 4 --num-kv-heads 2 \
  --gdn-num-key-heads 2 --gdn-num-value-heads 2 \
  --gdn-key-head-dim 32 --gdn-value-head-dim 32 --state-layers 2 10 22
```

该命令沿用已有的小宽度训练流程验证协议，仅重跑三个改动后的叠加架构：冻结预训练 VLM，随机初始化各个 Expert，每种架构执行 10 次 AdamW 更新，共抽取 20 个训练 chunk。`gqa6` 和 `gqa6_vlm_state` 的小宽度配置与实现未变，复用前一轮 pilot 结果，不重复训练。Condition cache 仅适用于冻结 VLM；使用 `--train-vlm` 时须关闭该选项，并分别从同一 checkpoint 启动匹配实验。输出目录必须尚不存在，其中保存选定数据 manifest、归一化参数、结果 JSON、Expert/optimizer checkpoint；训练 VLM 时还保存 VLM 权重。该子集上的 offline FM loss 不能代表闭环任务成功率，也不能说明 recurrent state 已具有有用记忆。

## 统计参数量与计算成本

读取本地模型配置，统计精确参数量及 state/cache 估计，不加载预训练权重：

```bash
PYTHONPATH=. python examples/sft/qwen35_fm_benchmark.py \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --config-dir examples/sft/config/qwen35_fm --counts-only \
  --output results/qwen35_fm_counts.json
```

在相同 conditioning shape 和硬件上，测量 Expert forward、forward/backward 延迟与 GPU 峰值内存：

```bash
PYTHONPATH=. python examples/sft/qwen35_fm_benchmark.py \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --config-dir examples/sft/config/qwen35_fm --device cuda:0 \
  --batch-size 2 --horizon 16 --sequence-length 128 \
  --hidden-size 128 --intermediate-size 512 --iterations 10 \
  --output results/qwen35_fm_benchmark.json
```

Benchmark 使用实际 VLM 配置构造 synthetic conditioning，测量范围为 Expert，不含图像编码或完整控制周期。报告对比时，应同时保留参数量、cross-attention 次数、状态层选择、precision 和计时范围。实现这些架构本身，并不能说明 GDN 或 VLM 状态读取提高了机器人性能。

统一宽度预设的精确参数报告位于 [parameter_counts.json](../../../../results/qwen35_fm_additive_20261112/parameter_counts.json)，Expert 实测成本位于 [benchmark.json](../../../../results/qwen35_fm_additive_20261112/benchmark.json)。报告包含 Expert、可训练及总参数量、head 配置、状态尺寸和 cache 内存解析估计。VLM 使用 tied weights，共 2,213,241,664 个参数；各架构的共同 GQA 骨干和每个新增 GDN block 均保持相同尺寸。

小宽度 pilot 的参数量与预设不同。更新后的叠加实验在 [data_manifest.json](../../../../results/qwen35_fm_additive_20261112/molmobot_pilot/data_manifest.json) 中记录实际数据和配置，在 [results.json](../../../../results/qwen35_fm_additive_20261112/molmobot_pilot/results.json) 中记录训练结果。未改动的 `gqa6` 和 `gqa6_vlm_state` 复用[前一轮 pilot](../../../../results/qwen35_fm_20261112/molmobot_pilot/results.json) 结果；此前替换式 GDN 的结果属于另一种架构，不能作为本次叠加 Expert 的验证证据。

## 两节点冻结 Backbone 对照训练

使用 PyTorch 2.11，在两台各有四张 GPU 的机器上运行 `qwen35_fm_hybrid_sft`。FSDP2 mesh 为 `(replicate=2, shard=4)`：Action Expert 在节点内分片、节点间复制，冻结的原生 VLM 则在各 rank 独立编码。启动前须按机器映射设置网络接口、RDMA 设备和 GID；本次运行的 launcher 保留了已验证可用的设置。

两端使用同一个共享输出目录和相同训练参数，分别设置 node rank：

```bash
# Set NODE_RANK=0 on the first node and NODE_RANK=1 on the second.
NODE_RANK=0
MASTER_ADDR=10.208.38.28
PYTHONPATH=. python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29671 \
  -m examples.sft.qwen35_fm_hybrid_sft \
  --config examples/sft/config/qwen35_fm/qwen35_gqa6_vlm_state.yaml \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --output results/qwen35_hybrid_new \
  --steps 1000 --max-seconds 1200 --batch-size 128 \
  --train-episodes 4096 --val-episodes 64 --chunks-per-episode 8 \
  --workers 4 --checkpoint-every 100 --log-every 5
```

`--batch-size` 表示每个 rank 的样本数，因此该命令的 global batch 为 1,024。当前选择实际包含 32,387 个训练 chunk 和 512 个 validation chunk，只使用选定训练动作拟合归一化。每个模型在达到 1,000 次更新或 1,200 秒训练时间后保存分布式 checkpoint，再评估原始 validation split；准备与验证耗时分别记录。

使用 `--resume /absolute/path/to/checkpoint_XXXXXX` 恢复 Expert、optimizer、sampler 位置和各 rank RNG。数据、conditioning、优化、precision 和 batch 设置须与原运行一致，入口会对照已保存的 manifest 校验。`status.json`、`metrics.jsonl` 和 `checkpoint_*/metadata.json` 记录实际进展。此前冻结对照队列按 `gqa6_vlm_state`、`gqa6_gdn3`、`gqa6` 排序，保留的状态见 [queue_status.json](../../../../results/qwen35_hybrid_20261112T122822Z/queue_status.json)。

## 使用 Knowledge Insulation 预训练

使用 `examples/sft/qwen35_ki_pretrain.py`，同时预训练可学习的 Qwen backbone、连续 Action Expert，并加入真实视觉语言数据。前面的冻结 backbone FM 实验保留为架构对照。[Knowledge Insulation（KI）](https://arxiv.org/html/2505.23705v1) 隔离不同学习信号，而不是冻结整个 VLM：

| 更新类型 | Loss | 更新的参数 |
| --- | --- | --- |
| Robot | FAST 动作 token CE | 原生 Qwen，包括 vision 和 GDN 层 |
| Robot | 连续动作 FM | Action Expert 及其 state-readout adapter |
| 视觉语言 | 答案 token CE | 原生 Qwen |

`--vl-every 4` 表示每四次更新中有一次 VL 更新，其余三次同时计算 robot CE 和 FM。各 loss 分别按自己的全局有效 target 数归一化。FM 读取 detach 后的 prefix hidden states 和 GDN 矩阵，readout 自身仍接收梯度。Prefix-only 原生 forward 在 evaluation mode 和 `no_grad` 下运行；独立的 teacher-forced CE forward 保留训练梯度并启用 checkpointing。FAST 真值不进入 FM prefix。额外执行 prefix forward 是为了避免读取已经处理过 FAST 真值后缀的 GDN 最终状态而泄漏动作答案，Qwen 的原生 attention 和 recurrent kernel 均保留。

### 准备数据与 Token Target

当前 MolmoBot 选择包含 32,387 个训练 chunk 和 512 个 validation chunk，每个形状为 `[16,8]`：七个以 rad 表示的关节目标及原生 0–255 夹爪指令。当前观测 state 保留 9 维，即七个 arm joint 和两个 gripper-driver position。FM 使用只由训练集拟合的 mean/std；FAST 独立拟合训练动作的 1%/99% 分位数并映射到 `[-1,1]`，仅编码开头连续、全部有效的时间步。九维 state 使用自己的训练分位数，量化为九个 `[0,255]` 的整数文本 bin。

四路外部视角依次为 `exterior`、`randomized_gopro_analogue_1`、`randomized_zed2_analogue_1`、`randomized_zed2_analogue_2`，顺序固定。Prefix 中为每幅图像保留相机标签，图像 resize 到 224×224；数据中的 wrist 视角不属于这份四视角配方。沿用 native 转换已完成的 observation/action 对齐。

使用官方 [FAST tokenizer](https://huggingface.co/physical-intelligence/fast)，本次固定 revision 为 `ec4d7aa71691cac0b8bed6942be45684db2110f4`，下载的 tokenizer/code 资产约 682 KiB。Bridge 增加 2,048 个 code symbol 及 begin/end marker：tokenizer 长度从 248,077 增至 250,127，原生 embedding 行数从 248,320 增至 250,127。初始化时保留全部已有行、多模态 token ID 和 input/output weight tying，原始 Qwen checkpoint 文件保持不变。

VL co-training 使用机器人视觉问答数据 [Robo2VLM-1](https://huggingface.co/datasets/keplerccc/Robo2VLM-1)。本地有 262 个 train shard 中的 71 个，共 183,748 条记录，以及完整的 6,676 条官方 test 索引。[官方 exporter](https://github.com/KeplerC/robo2VLM/blob/main/scripts/create_huggingface_dataset.py) 按 question 划分，未保护 episode-directory group；本地审计发现其 train/test 有 1,536 个 group 重叠。Loader 先排除所有官方 test group，再按 seed=42 的稳定 hash 留出 5% group。实际训练/验证选取 8,192/512 条，覆盖 2,674/143 个 exporter group，选定图像 resize 后的 RGB hash 交集为零。这里的 group 是从 ID 恢复的 sanitized directory name，不是已核实的原生物理 episode，也不代表 scene/object OOD 隔离；这份机器人 VQA 数据也不同于 KI 论文的广泛 web-data mixture。

选择新的输出目录，在 CPU 上一次准备两种 cache：

```bash
KI_RUN_ROOT=results/qwen35_ki_new
PYTHONPATH=. python -m examples.sft.qwen35_fm_image_cache \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --output "$KI_RUN_ROOT/robot_images4" --workers 12 \
  --train-episodes 4096 --val-episodes 64 --chunks-per-episode 8 \
  --horizon 16 --image-size 224 \
  --cameras observation.images.exterior \
    observation.images.randomized_gopro_analogue_1 \
    observation.images.randomized_zed2_analogue_1 \
    observation.images.randomized_zed2_analogue_2
PYTHONPATH=. python -m examples.sft.qwen35_ki_data \
  --data-path /mnt/public/daibo/datasets/robo2vlm-1/data \
  --output "$KI_RUN_ROOT/vl_cache" --workers 8 \
  --train-limit 8192 --val-limit 512 --image-size 224 --seed 42
```

第一条命令按照原生 timestamp 规则解码机器人视角，并对照未缓存 loader 检查选定 RGB 像素。第二条命令生成 group-disjoint 的 VQA 元数据与图像 memmap。Cache builder 要求新输出目录，后续架构可以复用已经完成的 cache。本次 session 的准备路径为 `results/qwen35_ki_20261112/robot_images4` 和 `results/qwen35_ki_20261112/vl_cache`，元数据记录实际选择；不会跨 optimizer update 缓存 VLM feature 或 recurrent state。

### 运行两节点 KI Smoke

在两台各四卡机器上使用 PyTorch 2.11 和 Transformers 5.19.0。FSDP2 mesh 为 `(replicate=2, shard=4)`，同时分片可训练 VLM 和 Expert。下面使用本次 session 已验证的 RDMA 映射；换机器时须调整设备、GID、地址，并选择空闲 rendezvous port。两端设置不同 `NODE_RANK`，其余共享路径和参数保持一致：

```bash
# Set NODE_RANK=0 locally and NODE_RANK=1 on nxb_dev_2.
NODE_RANK=0
MASTER_ADDR=10.208.38.28
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA='=mlx5_1'
export NCCL_IB_GID_INDEX=$((5 + 4 * NODE_RANK))
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
KI_RUN_ROOT=results/qwen35_ki_new
PYTHONPATH=. python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29675 \
  -m examples.sft.qwen35_ki_pretrain \
  --config examples/sft/config/qwen35_fm/qwen35_gqa6_vlm_state.yaml \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --fast-path /mnt/public/daibo/models/physical-intelligence-fast-tokenizer \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --vl-data-path /mnt/public/daibo/datasets/robo2vlm-1/data --vl-format robo2vlm \
  --image-cache "$KI_RUN_ROOT/robot_images4" \
  --vl-image-cache "$KI_RUN_ROOT/vl_cache" --output "$KI_RUN_ROOT/smoke_v3" \
  --train-episodes 4096 --val-episodes 64 --chunks-per-episode 8 \
  --vl-limit 8192 --vl-val-limit 512 --vl-every 4 \
  --cameras observation.images.exterior \
    observation.images.randomized_gopro_analogue_1 \
    observation.images.randomized_zed2_analogue_1 \
    observation.images.randomized_zed2_analogue_2 \
  --image-size 224 --batch-size 1 --workers 2 --steps 4 --max-seconds 600 \
  --vlm-lr 2.5e-5 --expert-lr 1e-4 --warmup-steps 50 --schedule-steps 1000 \
  --checkpoint-every 100 --log-every 1
```

该命令运行四次更新，即三次 robot 更新和一次 VL 更新，global batch 为八。这是训练链路 smoke，不是正式 batch 的推荐值。先测量完整四相机 KI step，再为 `gqa6_vlm_state`、`gqa6_gdn3`、`gqa6` 选取共同 batch；冻结 VLM 的吞吐和显存结果不能直接用于可训练 Qwen 的 KI。

保留 FP32 master parameter 和 AdamW state，使用 BF16 计算、FP32 梯度归约与 loss，遵循 [FSDP2 的 mixed-precision 接口](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html)。VLM 与 Expert 分别使用 optimizer，分别做 norm=1 clipping。默认 VLM/Expert peak LR 为 `2.5e-5`/`1e-4`，AdamW β=`(0.9,0.95)`、ε=`1e-8`、weight decay=`1e-10`。VLM LR 和 AdamW 参数参考 [OpenPI optimizer 配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py)；Expert LR、短运行的 50 次更新 warmup 与 cosine schedule 属于明确记录的实验选择。`--schedule-steps 1000` 固定衰减 horizon，最终降至 peak LR 的 0.1 倍。FM timestep 保留用户要求的均匀采样，而 KI 论文采用变换后的 Beta 分布，因此这里不声称逐项复现论文超参数。

运行 `PYTHONPATH=. python -m pytest tests/unit_tests/test_qwen35_fm.py -q -k ki_`，检查 KI 梯度路由、prefix 隔离、原生 masked CE 等价性、FAST mapping 及多模态 CE；若审核过的本地 tokenizer 路径不同，设置 `QWEN35_FAST_TOKENIZER`。本次 session 已通过十一项 KI 测试，以及八 rank、完整 Qwen 的四次更新 smoke；首次 smoke 使用单 exterior 视角。[状态记录](../../../../results/qwen35_ki_20261112/smoke_v3/status.json) 保留了实际 robot/VL 更新数和 loss，四视角正式运行仍需自己的实测结果。

分布式 checkpoint 保存 Qwen、Expert、两个 optimizer、归一化、FAST mapping、sampler/update 位置和各 rank RNG，完整模型 smoke checkpoint 约 30 GB。使用 `--resume /absolute/path/to/checkpoint_XXXXXX` 时，保持数据、相机顺序、batch、precision、optimizer 和 schedule 协议一致。日志分开记录 FM、action CE、VL CE 及各自的 validation 分母；这些 offline 目标验证学习和梯度路径，不能代表闭环成功率或通用 web 知识保留。

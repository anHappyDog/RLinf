# π0.5 Multi-Scale Embodied Memory 复现计划

> 状态：R0/R1 已通过；K=6 v3 memory-critical curriculum 已完成 50-step 四卡 pilot；history-content gate 通过但 temporal-order gate 失败，当前 recipe 已停止扩训。GroundSG Oracle、FrameSamp+Modulator、temporal RoPE 和 recurrent memory 均处于方案讨论/候选实验阶段，尚未在 B1K 上验证  
> 最近更新：2026-08-31  
> 工作目录：`/mnt/public/daibo/timeline/0812/RLinf`  
> 论文：本地 `./2603.03596v2.pdf`，对应 π-MEM / Multi-Scale Embodied Memory

## 跨对话接续说明

本文件是该项目的长期设计记录。开启新对话后，应先读取本文件，再检查代码和数据的实际状态；不要重新从零推导方案。

每次完成设计决策、实现或实验后，应至少更新以下三处：

1. “当前状态”；
2. “决策记录”；
3. 对应阶段的完成情况、实验结果或未决问题。

当前已经使用真实 π0.5 base checkpoint 完成 R0-A/R0-B。R1 已从 10,000 个 B1K episode 构建 130,882 条样本，按每个 task 内的完整 episode 划分为 8,000/1,000/1,000 train/validation/test，且没有 canonicalization skip。通用 base 上的 task-0000 三 seed 选型最终采用 state-only、Gemma 最后 2 层、500 steps：validation exact match 为 `88.33% ± 2.04%`，action relative RMSE 为 `5.97% ± 1.84%`。正式主线现已切换到 OpenPI-Comet `pi05_b1kpt50_pt`：同配方的 7-task pilot 为 80.00% exact match、动作漂移 1.05%；progress-sensitive 4-task pilot 为 56.25%、动作漂移 2.26%。R2 oracle primitive prompt 已接入 action SFT，并完成全量标注审计。`/mnt/public/daibo/venv/behavior_openpi` 已通过真实 streaming、OpenPI transform/collate 和 πLL flow-loss forward；test split 尚未使用。R2 mixed πLL seed=42 已在本机 4×A100 完成 2,000 steps，并保存每 100 steps 的完整 checkpoint。旧 short-memory v1 使用 `K=4, stride=16`，且 temporal PE 在 LayerNorm 后加入、只连续化过去 `K-1` 个 state；其闭环结果只能作为诊断。v2 已修正为 `K=6, stride=30`、PE 在 LayerNorm 前加入并保留在 residual、全部 K 个 state 连续化；官方 B1K 评测为 30 Hz，因此连续 policy decision 天然相隔约 1.067 秒，不再使用 60 Hz 下的 decision stride 2。真实 B1K batch、相关单测、Hydra compose、完整 3B 模型前向和 4×A100 FSDP 反向 smoke 均已通过。seed=42 正式训练已在本机 4×A100 完成 2,000 steps，输出目录为 `/mnt/public/daibo/results/mem_short_k6_v2_task0000_seed42`。long memory 尚未实现。

2026-08-31 对 RoboMME 的对照研究没有启动新训练或改变既有 checkpoint。当前只形成了候选方案：用 GroundSG Oracle 区分目标 grounding 与低层控制瓶颈；把 FrameSamp+Modulator 作为公开的强 perceptual-memory baseline；把真实时间 temporal RoPE 作为独立消融；recurrent memory 暂不替代 symbolic/perceptual 主线。详细证据、实现要点和实验矩阵见第 12 节。用户当前希望先消化方案；恢复工作时不得把这些候选项写成已验证结论。

后续审计确认该轮 `mixed_oracle` 使用了训练标签中不存在的初始 prompt `move to radio`，所以它不能作为有效 Oracle-HL gate。task-0000 的 200 条 demonstration 只有 `pick up radio from coffee table`、`press radio`、`place radio on coffee table` 三类 canonical primitive；第一类从 episode 起点开始，已包含 navigation。正确映射应让 simulator 的 `move_to_radio` 和 `pickup_from_support` 都条件于 `pick up radio from coffee table`。`mixed_task` 相对 base 的退化仍然成立。

映射修正后的 step-2000 Oracle-only gate 已完成：`mixed_task / corrected mixed_oracle` 均为 `0/4` success，平均 completed stage 均为 `0.50`，平均 return 为 `316.086 / 304.787`。step 100/200/300/400/500 的双条件 sweep、step-100 sensitivity check 和 public-20 也已完成；R2b/R2c screening 均未通过原定 gate。2026-08-29 首先发现这些运行使用 `skip_intermediate_obs_in_chunk=true`；2026-08-30 又确认 RLinf base config 的 60 Hz 与官方 B1K `eval_utils.py` 的 `action/render=30 Hz, physics=120 Hz` 不一致。此前 4,096@60 Hz 虽与 2,048@30 Hz 物理时长相同，但一个 32-action chunk 只有 0.533 秒，改变了动作执行和重规划分布，所以所有旧 60 Hz 数字都只保留作诊断。canonical `turning_on_radio` 现固定为 2,048 action steps、30 Hz、`skip=false`。在该协议下，step-1100 Oracle primitive 四实例均到达 pickup stage、但 success 仍为 `0/4`；decision trace 证明闭环中左右 gripper action 全程没有负值、实际夹爪保持全开。demo grasp-event generation gate 表明，同一 checkpoint 在示范抓取状态上会闭合正确右手。随后发现 `EnvOutput.prepare_observations()` 会丢弃全部 `history_*` 字段，因此修复前的所有闭环 `correct/repeat/shuffle` 其实都退化为 current-only，不能用于 memory 因果结论。修复后的精确示范事件三条件 action-fidelity 评测显示：correct 显著抑制 repeat-current 的 open-control 假闭合，但与 shuffle-past 基本等价；当前证据只支持“模型使用过去内容校准夹爪”，不支持“模型学习了历史顺序”。

### 评测协议纠错（2026-08-29 至 2026-08-30）

- 旧 MEM eval 使用 RLinf `BehaviorEnv`，并继承/显式设置 `skip_intermediate_obs_in_chunk=true`；
- OmniGibson `VectorEnvironment.step` 在该模式下仍逐 action 执行 `_pre_step → sim.step → _post_step → task.step`，所以物理、reward 和 termination 没有被跳过；变化是中间步使用 `render=false, get_obs=false`；
- 官方 B1K OmniGibson `generate_basic_environment_config()` 固定 `action_frequency=30`、`rendering_frequency=30`、`physics_frequency=120`，dataset 的视频和 action timestamps 也是 30 Hz；RLinf base Behavior config 的 60 Hz 不是本项目的 acceptance cadence；
- π0.5 action chunk 为 32。`2048@30 Hz` 提供 64 次策略重规划和约 68.3 秒控制，与用户记忆中的 radio horizon 一致；此前 `4096@60 Hz` 物理时长相同，但每个 chunk 从 1.067 秒缩短到 0.533 秒，不能视作等价策略评测；
- canonical acceptance protocol 固定为 `max_episode_steps=max_steps_per_rollout_epoch=2048`、`action/render=30 Hz`、`physics=120 Hz`、`skip_intermediate_obs_in_chunk=false`、视频 30 fps；launcher 显式写入这些值，避免再继承错误默认；
- `skip=false` 原本会让 short-memory 每个 action step 都写 history。现已把视频帧收集与 policy-memory 更新解耦：中间帧只用于完整视频，history 仍只在 chunk 最后一帧更新一次；在 30 Hz 下 `history_decision_stride=1` 即约 1.067 秒间隔，K=6 覆盖约 5.33 秒；
- 旧的 skip=true 以及所有 60 Hz R2/short-memory 闭环结果均标记为 diagnostic，不作为最终验收结论；必须用新协议重跑因果矩阵；
- 更严重的 transport 审计发现，`EnvOutput.prepare_observations()` 在构造 rollout model input 时没有转发 `history_images/history_image_masks/history_state/history_state_mask`。修复前的全部 short-memory 闭环运行，无论日志标为 `correct/repeat/shuffle`，模型实际收到的都是 current-only 输入；这些运行仍可诊断 horizon、simulator 和 πLL 行为，但其条件差异与 return 排序全部作废，不能作为 temporal causal gate；
- step-1100 Oracle primitive 已用 30 Hz/2,048 steps/skip=false 重跑。IDs `[242,295,211,203]` 的 completed stage 均为 1，return 分别为 `446.453/429.133/393.476/533.997`，success 全为 false；30 Hz 修正让 ID 211 从旧 60 Hz 的 stage 0 提升到 stage 1，但仍没有 pickup；
- 新增逐 policy-decision trace。四个 episode 的左右 gripper command 负值比例均为 0，最小 command 约 `0.97--0.99`，实际 gripper width 约 `0.098--0.100 m`；`in_hand/has_left_support` 从未成立，radio 从未离开 support。最小 EEF-center 到 radio 距离为 `0.268/0.239/0.278/0.196 m`，说明模型既没有进入稳定抓取位姿，也没有发出闭爪；
- step-1100 的 30 Hz 四条件配对输出位于 `/mnt/public/daibo/results/mem_short_k6_v2_step1100_paired_h2048_f30_fullobs`，16 段视频和四份 raw metrics 完整，无 traceback。记录的 `memoryless/correct/repeat/shuffle` success 均为 `0/4`，平均 stage 为 `1.00/1.00/0.75/1.00`，平均 return 为 `408.608/453.215/447.636/456.767`；但该运行发生在 history transport 修复前，后三个条件实际都是 current-only，数值只能作为 simulator stochasticity/闭环失败的诊断，不能比较 memory；
- 审计 task-0000 全部 200 demonstrations 后，canonical pickup primitive 有 249,169 个 boundary-safe 32-step 样本，其中仅 `9.955%` 的 chunk 含任一闭爪 action；press/place 则分别为 `99.975%/97.579%`。pickup 标签覆盖长时间 navigation/reach，uniform per-frame SFT 会让“保持张开”成为约 90% 的主模态，这是下一轮 πLL 训练必须处理的具体数据失衡。
- demo grasp-event gate 从 episodes 10/20/30/40 各取 2 个远端 open-control、close-onset 和 closed-hold 样本，每样本使用 4 个固定 flow-noise draw。step-1100 的事件闭合率/正确右手率为 `79.6875%`、closed-step recall 为 `73.1531%`，距离真实闭合约 64 帧的 open-control 假闭合率仅 `3.125%`，通过预先固定的 `50%/50%/50%/25%` gate；
- step-2000 在同一缓存上的事件率为 `78.125%`、recall 为 `67.1760%`、open 假闭合为 `3.125%`，也通过但略弱于 step-1100。故不启动 event-balanced πLL SFT；下一步采用 mid-stage/demo-state reset 定位 rollout 到 demo grasp manifold 的偏差，再决定 recovery/DAgger；
- 本次事件筛选还暴露出 short-memory streaming 数据缺陷：旧 dataset 把 episode 切成 250-frame chunks，并在每次 chunk reload 时清空 `_stream_history`。对 `K=6, stride=30`，每块前 150 帧只有 padded/incomplete history，即最多约 60% chunk 帧无法获得完整窗口；radio 的若干 grasp event 也恰好位于该区域。现已改为仅在 short-memory 模式按完整 episode 分配 rank/worker、只打乱 episode 顺序、episode 内连续读取 chunk，并仅在 episode 切换时清空 history；memoryless 原 chunk 采样保持不变；
- 为区分模型误差与 simulator takeover 误差，新增 ranked、one-shot、shape-validated 的 exact observation snapshot 和 model-action override。episodes 10/20/30/40 的 onset 前 15 帧 observation 与离线样本逐字段 bit-exact，模型 action override 也逐值一致；同一精确输入下 step-1100 的右臂 MAE 为 `0.1044`、右夹爪 MAE 为 `0.8575`，非夹爪动作 cosine 为 `0.9838`，说明 reach 方向大体正确但夹爪幅值/时序误差最大；
- raw HDF5 的 simulator state 是带 UUID 的 dump-filter 稀疏序列，必须先用该 episode 的 `scene_file` 恢复场景，再加载 onset 前历史 state；不能用 live full-state 长度判断其完整性。加入 scene restore 后，expert action takeover 仍只有 episode 10 达到 `in_hand=true`，其余 3 个没有形成抓取，故 mid-stage replay 本身不够稳定，不能据此启动 DAgger 或作为 policy acceptance gate；
- 修复 transport 后，使用完全相同的 24 个 `open-control/close-onset/closed-hold` 样本和每样本 4 个固定 flow-noise draw，完成 `correct/repeat_current/shuffle_past` 三条件 action-fidelity。step-1100 的 correct/repeat/shuffle open-FP 为 `3.125%/31.25%/3.125%`，右夹爪 MAE 为 `0.3903/0.4875/0.3924`，右臂 MAE 为 `0.1158/0.1131/0.1153`；step-2000 为 `3.125%/28.125%/3.125%`、右夹爪 `0.4178/0.4566/0.4131`、右臂 `0.0680/0.0720/0.0672`。两者都稳定证明 past content 能抑制无条件闭爪，step-2000 的非夹爪拟合更好；但 shuffle 与 correct 几乎相同甚至略优，未证明 temporal order sensitivity；
- K=6 v3 训练流仅将完整 K=6 history 且 32-step target 中包含 pickup 闭爪的窗口标为 memory-critical；它们全部保留，非 critical 窗口以确定性 20% 概率保留作为 navigation、reach 和 open-control。真实 task-0000 连续 stream 的 256 个输出中，`17.19%` 为 critical，`88.28%` 已填满 K=6；首个 critical 是 episode 10 frame 1064，比实际闭爪 onset 早 31 帧，符合 action horizon 覆盖；
- v3 对每个 critical 样本以相同 action、flow noise 和 flow time 比较 `correct / repeat_current / shuffle_past`。辅助项为 `relu(margin + L_correct - stopgrad(L_control))`，margin `0.01`、weight `0.25`；control 分支 stop-gradient，避免通过故意破坏 corrupted-history 预测来满足排序。若任一 FSDP rank 含 critical，distributed MAX 会让全部 rank 执行相同 control forwards；本地无 critical 的 rank 贡献可微零项，从而避免 FSDP collective 错配；
- 正常 v3 两步四卡 smoke 因处于 episode 开头而没有 critical 样本，但已验证新 sampler 与标准 flow loss。另一个强制 full-history/critical 的 4×A100 smoke 实际执行两个 control forward 和 backward：critical fraction `1.0`，repeat/shuffle loss delta 为 `+0.0414/+0.00346`，auxiliary loss `0.0088`，total/VLA loss `0.120/0.118`，gradient norm `4.28`，无 OOM 或 deadlock；这只验证训练计算图，不代表 temporal-order gate 已通过；
- v3 50-step early-gate pilot 已完成，step 25/50 checkpoint 均完整。首个真实 critical batch 出现在 step 10；step 50 的 loss/VLA loss 为 `0.0423/0.0420`、grad norm `0.382`、critical fraction `18.8%`，训练 batch 的 repeat/shuffle delta 为 `+0.00142/+0.00093`，无 OOM、NCCL 错误或分布式死锁；
- 首轮 step-25/50 action-fidelity 误复用了 stream-fix 前保存的 `selection_samples.pt`。该文件缓存的是 observation tensor 而不仅是 frame index，24 个样本中只有 7 个完整 K=6，因此这轮数字只作诊断。用修复后的 episode-contiguous loader 重建缓存后，24/24 均为完整 K=6；有效缓存位于 `/mnt/public/daibo/results/mem_short_k6_streamfix_grasp_event_selection/selection_samples.pt`；
- 有效缓存上，v3 step-50 的 correct/repeat/shuffle event rate 为 `73.44%/43.75%/75.00%`，closed-step recall 为 `59.34%/38.18%/59.71%`，open-FP 为 `6.25%/15.625%/3.125%`。correct 相对 repeat 的右夹爪 MAE 改善 `0.0581`、win rate `75%`，history-content gate 明确通过；但 shuffle 的 event/recall/open-FP、右臂 MAE和 cosine 均不差于 correct，correct 仅在右夹爪 MAE 上微弱改善 `0.0048`。因此 temporal-order gate 失败，当前 v3 不续训到 100/200/2000 steps；

## 1. 项目目标与复现边界

最终目标是在开源 π0.5 的基础上复现 Multi-Scale Embodied Memory，包括：

- short memory：利用历史视频帧和历史机器人状态改善动作生成；
- long memory：利用语言形式的语义记忆跟踪已经完成的步骤、对象状态和剩余目标；
- multi-scale memory：同时启用 short memory 和 long memory，并验证二者能够互补。

本项目追求的是**功能性复现**，而不是声称严格复现论文中的 π0.6 结果。主要差异是：

- π-MEM 使用未开源的 π0.6；
- π-MEM 的视觉语言模型约为 Gemma 3 4B，action expert 约为 860M；
- 当前可用的开源 π0.5 使用较小的 PaliGemma/Gemma 2B 和约 300M action expert；
- 开源 π0.5 代码没有完整提供论文中的高层 subtask 文本生成接口和训练路径。

因此，验收标准不是复现论文绝对成功率，而是在选定的 B1K 任务上，用受控消融证明 short memory、long memory 和二者组合相对同源基线带来可重复的性能提升。

## 2. 已确认的模型事实

### 2.1 π0.5 论文中的 πHL 与 πLL

π0.5 的 hierarchical inference 不是两个独立训练、权重不同的 VLM。πHL 和 πLL 是**同一个 π0.5 模型、同一套 SigLIP + PaliGemma/Gemma 权重的两次调用**：

1. 高层调用 πHL 根据任务、当前观测和状态生成文本 subtask；
2. 低层调用 πLL 将该 subtask 作为语言条件，由 action expert 生成动作。

两次调用的输入频率和摄像头集合可以不同，但这不意味着二者使用不同的 backbone 权重。低层路径额外使用 action expert；高层文本生成使用共享语言模型的 token decoding 能力。

论文中的大致组成是：

- 共享 SigLIP 视觉编码器；
- 共享 PaliGemma/Gemma 2B 语言模型；
- 低层动作生成额外使用约 300M action expert；
- 预训练阶段包含 FAST 离散动作 token / 文本建模；
- 后训练阶段联合 next-token prediction 与 flow matching，论文中两者具有损失权重设置。

这意味着忠实补回开源 π0.5 的首选方案，应是恢复同一个模型的高层文本输出，再复用同一权重执行低层动作生成，而不是默认增加一个独立 4B 高层模型。

### 2.2 当前开源 π0.5 的实际情况

RLinf 中现有 `openpi_rlinf` 路径的关键特征包括：

- Gemma 2B 语言模型；
- 约 300M action expert；
- SigLIP So400m/14，27 层；
- 图像分辨率 224 × 224，每个摄像头产生 256 个 patch token；
- B1K 配置使用 3 个摄像头；
- action horizon 为 32；
- B1K action/state policy 维度为 23。

开源路径主要实现了 action-only flow matching，没有完整暴露 πHL 所需的：

- 文本目标构造；
- next-token cross-entropy loss；
- 自回归 subtask generation；
- 高层/低层两次调用的统一推理接口。

但文本能力并非从权重中完全消失：Gemma embedder 已提供 tied decoding，checkpoint 转换也会把 PaliGemma language head 合入共享 embedding。因此恢复 πHL 的主要工作是补回训练目标、forward/generation 接口和数据组织，而不是从零训练一个语言模型。

π0.5 低层路径中，图像和语言构成 prefix；pi0.5 的机器人 state 通常被离散化进 prompt，而非像早期连续 state 一样进入 action suffix。这一点会影响历史 state 的兼容设计。

本机还存在一个更适合作为动作基线的第三方 checkpoint：

```text
/mnt/public/daibo/models/pi05_b1kpt50_pt
```

它已确认对应 [OpenPI-Comet `pi05-b1kpt50-cs32`](https://huggingface.co/sunshk/openpi_comet/tree/main/pi05-b1kpt50-cs32)。OpenPI-Comet 文档说明该模型在 B1K tasks 0–49 上预训练，action chunk 为 32；本地目录还包含 Behavior norm stats。权重结构与 `pi05_base` 相同，但抽样层的 relative RMSE 为 2.9%–12.9%，不是简单 dtype 转换。后续正式 baseline 和 MEM 主线应从该 B1K checkpoint 分叉；通用 `pi05_base` 保留为“从零做域适配”的消融。该模型是第三方 π0.5 B1K checkpoint，不应误称为 Physical Intelligence 官方 B1K release。

### 2.3 π-MEM 相对 π0.5 的变化

π-MEM 以 π0.6 为基础，不能直接等价映射到当前 π0.5。论文中的主要变化包括：

- 更大的 Gemma 3 4B 视觉语言模型；
- 约 860M action expert；
- 448 分辨率视觉输入；
- 最多四个摄像头；
- 同时保留文本/FAST 与 flow matching 能力；
- action expert 的梯度不直接更新 VLM。

short memory 的视频编码器大体执行：

1. 输入 patch token 形状为 `[B, T, P, D]`；
2. 每一层对每帧分别做 spatial attention；
3. 每隔若干层，对相同 patch 位置沿时间做 causal temporal attention；
4. temporal attention 复用该层已有 Q/K/V 和 attention 参数，不额外增加一套 attention 权重；
5. 使用固定的时间位置编码，并令当前时刻的位置编码为零；
6. 在视觉编码器中间层丢弃所有历史帧 token；
7. 后续层和 VLM 只接收当前帧 token，因此不会让语言模型上下文长度随历史帧数增长。

论文预训练使用 6 个观测，即 5 个历史观测加当前观测，约按 1 秒间隔采样；后训练最长扩展到 18 帧、约 54 秒历史。论文消融表明，只在后训练阶段加入历史帧弱于先经过 video pretraining。

历史机器人状态使用连续投影，每个历史状态形成一个 token。long memory 的高层输出同时包含 subtask 和更新后的语义记忆，并使用成功/失败记录及语言模型摘要构造训练监督。

## 3. 核心执行原则

### 3.1 先恢复基础能力，再验证 memory

memory 不能替代基本的任务理解和动作能力。如果 π0.5 在目标 B1K 任务上几乎没有成功率，则：

- long memory 无法获得有意义的闭环进度；
- short memory 即使学到了历史表示，也很难转化为完整任务成功；
- 最终的性能差异容易被基础模型能力不足掩盖。

因此训练顺序应是：

1. 补回 π0.5 高层 subtask 文本生成；
2. 使用 oracle subtask 训练并验证低层动作能力；
3. 建立 generated-subtask 的层级闭环基线；
4. 在同源基线上分别加入 short memory 和 long memory；
5. 最后组合 multi-scale memory。

“先做 subtask 文本生成”与“先建立成功基线”不是冲突关系。subtask 生成正是恢复 π0.5 层级能力和建立 B1K 基线的一部分。

### 3.2 暂不优先扩大模型规模

当前不把独立 4B πHL 或整体模型扩容作为第一阶段，原因是：

- 它偏离 π0.5 共享模型、两次调用的层级设计；
- 没有与更大低层模型匹配的开源 VLA/action checkpoint；
- 随机或弱初始化的大模型可能增加训练成本，却不能解决 B1K 动作域适配问题；
- 会让 memory 收益与容量收益难以区分。

独立 4B 高层模型可以保留为工程 baseline 或 fallback，但不作为主复现路线。只有在 oracle-subtask 低层实验显示小模型容量是明确瓶颈后，才评估扩大模型规模。

## 4. B1K 数据及其监督价值

数据目录：

```text
/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos/
```

当前已检查到的数据规模和结构大致为：

- 50 个任务；
- 每个任务 200 个 demonstration，共约 10,000 个 episode；
- 约 119,094,660 帧，按 30 FPS 估算约 1,102.7 小时；
- 每个 episode 中位时长约 370.5 秒；
- 3 路 RGB，并带 depth / segmentation 等信息；
- 原始 state 约 256 维，需映射为 π0.5 B1K policy 使用的 23 维 state；
- action 为 23 维。

标注中约有：

- 235,492 条 skill annotation；
- 65,441 条 primitive annotation；
- `skill_annotation`；
- `primitive_annotation`；
- `skill_idxes`；
- `memory_prefix`；
- `spatial_prefix`；
- object instance id、manipulating object id 和 frame duration。

### 4.1 Skill 与 primitive 的用途不同

`skill_annotation` 更接近原子动作，例如：

- move to；
- pick up from；
- press；
- place on。

`primitive_annotation` 通常聚合一个或多个 skill，更接近 π0.5 高层应该输出的 subtask。`skill_idxes` 可以建立 primitive 到 skill 的映射。

例如 `turning_on_radio` 可形成如下 primitive：

1. pick up radio from coffee table；
2. press radio；
3. place radio on coffee table。

因此主训练目标应使用 canonicalized primitive 作为 πHL subtask；skill 可作为低层 curriculum、辅助监督或细粒度评估标签。

对于 `picking_up_trash` 这类任务，一个 episode 可以包含多次对不同罐子的抓取和放置。重复结构以及不同 demonstration 中的步骤变化，正适合测试进度记忆和历史消歧。

### 4.2 标注规范化

B1K 的 instance id 不能直接作为文本目标。需要统一 canonicalizer，例如：

- `radio_89` → `radio`；
- `coffee_table_koagbh_0` → `coffee table`。

第一版 canonical subtask 应采用稳定模板，尽量避免同义改写带来的额外标签熵，例如：

```text
pick up <object> from <source>
place <object> on <destination>
place <object> in <destination>
press <object>
open <object>
close <object>
```

数据中可能存在空 primitive、格式异常或边界缺口。处理策略应是：

1. 可修复时根据结构化字段生成 canonical primitive；
2. primitive 不可用时回退到对应 skill；
3. 仍不可解析时跳过样本并记录统计，不静默产生错误文本。

### 4.3 `memory_prefix` 的限制

`memory_prefix` 常是 `back`、`the other` 等指代或空间片段，而不是完整的语义记忆标签。因此：

- 可以把它用作“当前指令依赖历史上下文”的信号；
- 可以用于筛选 long-memory-sensitive 样本；
- 不应直接把它当成 π-MEM long memory 的完整监督目标。

## 5. 补回开源 π0.5 的方案

### 5.1 高层训练样本

每条高层样本建议包含：

```text
task instruction + current observations + current state
    -> canonical primitive subtask + EOS
```

高层样本不应直接按 30 FPS 枚举，否则相邻重复帧会压倒有效的 primitive 边界。应按 primitive 区间均匀抽取少量 anchor，并提高边界附近样本的比例。

输入 prompt 可先沿用普通文本，而不引入新的 special token：

```text
Task: <full task instruction>
Subtask:
```

这样可以最大程度复用原 Gemma tokenizer 和已有 embedding。

### 5.2 恢复文本 forward 与 generation

高层 forward 应实现：

1. 图像、任务和当前 state 构成 prefix；
2. prefix 内采用与现有多模态路径兼容的 attention mask；
3. subtask target 采用 causal mask；
4. 通过 tied embedding decoder 得到 token logits；
5. cross-entropy 只计算 subtask target 和 EOS，不计算 prefix；
6. 支持 teacher forcing 训练。

推理阶段增加自回归 generation：

1. 编码高层 prefix；
2. 缓存 prefix KV；
3. 从 `Subtask:` 后逐 token 解码；
4. 遇到 EOS 或长度上限停止；
5. 将生成的 canonical subtask 作为同一个模型低层调用的语言条件。

### 5.3 低层 oracle-subtask 训练

低层样本为：

```text
current observations + current state + oracle canonical primitive
    -> action chunk [32, 23]
```

为了公平测量 primitive prompt 的作用，主 checkpoint 使用相同的 boundary-safe action chunks 混合训练 full-task 与 primitive prompt，初始比例为 `50% / 50%`。随后保持这一份 πLL 权重不变，分别以 full task 和 Oracle primitive 做闭环推理；二者差值才是层级 prompt 的净收益。额外训练一个 primitive-only checkpoint 可以作为部署消融，但不能单独用于上述因果比较。

Skill prompt 可在 Oracle gate 通过后作为更细粒度 curriculum，而不应在第一轮 task/primitive 对照中引入第三种接口。

#### R2 早期 checkpoint 闭环结果

固定官方实例 `[242, 295, 211, 203]`、seed 42 和 2,048 simulator steps。表中 stage 为平均 completed stage，Δ 为同一 checkpoint 的 `Oracle - task`：

| step | task stage | Oracle stage | Δ stage | task return | Oracle return | Δ return |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.50 | 0.75 | +0.25 | 371.215 | 410.733 | +39.518 |
| 200 | 0.25 | 0.50 | +0.25 | 278.131 | 308.493 | +30.362 |
| 300 | 1.00 | 1.00 | 0.00 | 399.477 | 396.808 | -2.669 |
| 400 | 0.00 | 0.25 | +0.25 | 281.546 | 291.459 | +9.913 |
| 500 | 0.00 | 0.00 | 0.00 | 203.238 | 227.151 | +23.913 |

全部条件均为 `0/4` success。step-100 是 Oracle 净收益与绝对 return 的最佳折中；step-300 保留的 task-prompt stage 最好，但 Oracle 没有增益。现有结果支持“primitive conditioning 对部分 checkpoint 有方向性帮助”，尚不足以通过 full-task Oracle gate。

step-100 的 4,096-step sensitivity check 中，task/Oracle 的平均 stage 都从 2,048-step 下的 `0.50/0.75` 增至 `1.25/1.25`，per-instance stage 均为 `[1,1,2,1]`，仍无 success；return 为 `855.902/935.886`。延长 horizon 能提高绝对进度，但没有让 Oracle 完成额外离散阶段，因此 2,048-step 的正 stage delta 尚不能解释为稳定 HL 收益。

完整 20 个官方 public-test instances 的 2,048-step 聚合为：

| condition | success | mean stage | mean return |
|---|---:|---:|---:|
| original base task | 1/20 | 1.20 | 367.721 |
| step-100 task | 0/20 | 0.85 | 382.286 |
| step-100 Oracle | 0/20 | 1.20 | 398.207 |
| Oracle - step-100 task | 0 | +0.35 | +15.921 |

逐实例 stage 为 `8 improved / 1 degraded / 11 tied`，非 tie 的双侧 exact sign test 为 `p=0.0391`，Wilcoxon 为 `p=0.0196`；return 为 `14 improved / 6 degraded`，Wilcoxon `p=0.0826`。因此 Oracle primitive 对阶段推进已有方向性和统计证据，但没有产生 full-task success，不能按原验收标准宣告 gate 通过。

相对原始 base，step-100 task 的 stage 为 `-0.35`（3 正/8 负/9 平），Oracle 的平均 stage 差为 `0.00`（7 正/6 负/7 平）；Oracle return 比 base 高 `30.486`（Wilcoxon `p=0.0107`），但 success 从 `1/20` 降为 `0/20`。这说明 primitive prompt 能补回 mixed-SFT 的平均阶段退化，但当前训练没有保留原始 full-task 完成能力。

### 5.4 联合训练

高层 batch 和低层 batch 应由同一个 checkpoint、同一套共享 SigLIP + Gemma 权重处理：

- πHL batch：更新共享视觉/语言 backbone，不经过 action expert；
- πLL batch：更新 action expert，并以更小学习率更新共享 backbone；
- 初始可按 1:1 的 batch 次数交替，而不是按原始样本数量混合；
- 联合目标为 `L = λ_HL L_CE + λ_LL L_flow`；
- FAST 离散动作目标可作为后续里程碑，不必阻塞第一版层级闭环。

正式 baseline 与 MEM 模型必须从同一个原始 π0.5 checkpoint 分叉，并使用相同数据量、训练步数和采样规则。开发过程中可以顺序验证模块，但最终不能把一个训练更久的模型与较弱 baseline 直接比较。

### 5.5 能力恢复阶段

| 阶段 | 目标 | 退出条件 |
|---|---|---|
| R0 | 验证 tied text decoding、tokenizer、CE mask 和 EOS | 小数据过拟合，能精确生成 canonical subtask |
| R1 | 训练 πHL | held-out episode 上 primitive verb/object/destination 达到可用精度 |
| R2 | oracle-subtask πLL B1K SFT | 原子/primitive 执行出现稳定非零成功率 |
| R3 | πHL/πLL 联合训练 | 共享 backbone 不发生明显灾难性遗忘 |
| R4 | generated-subtask 闭环 | 完整层级策略在选定任务上达到可评估区间 |

必须同时比较：

1. full task instruction 直接条件化 πLL；
2. oracle subtask 条件化 πLL；
3. generated subtask 条件化 πLL。

如果 oracle subtask 仍无法使低层获得可靠执行能力，问题在 B1K 动作域适配或模型容量，暂时不应归因于 high-level generation 或 memory。

## 6. Short Memory 设计

### 6.1 模型命名和定位

实现应命名为 `pi05_short_memory` 或 `pi05_mem`，并在文档和实验中明确标注其为 π0.5-based functional reproduction，不能称为严格的 π0.6/π-MEM checkpoint 复现。

### 6.2 视频编码器接口

每个摄像头分别处理自己的时间序列，多个摄像头共享同一套视频编码器参数，但不在 temporal attention 中混合不同摄像头：

```text
per-camera input:   [B, K, H, W, C]
patch tokens:       [B, K, 256, 1152]
current output:     [B, 256, 2048]
```

3 个摄像头最终仍向 Gemma 提供 `3 × 256` 个 image token，因此丢弃历史帧后不会增加 VLM 的 prefix 长度。

### 6.3 Spatial/temporal attention

初始配置：

- `K = 6`，5 个历史帧加当前帧；
- SigLIP 共 27 层；
- zero-based temporal layers 为 `[3, 7, 11, 15]`；
- 在第 15 层输出之后丢弃全部历史 token；
- 剩余 11 层只处理当前帧。

每层对每一帧独立执行原 spatial attention。在 temporal layer 中，将相同空间 patch 位置跨 K 帧重排为时间序列，执行 causal temporal attention，并复用该层的 normalization、Q/K/V、output projection 和 attention 参数。

论文没有完全消除 temporal residual ordering 的实现歧义，因此该部分必须：

- 作为显式配置/模块封装；
- 写单元测试验证因果性和形状；
- 在实验记录中注明采用的 residual 顺序；
- 对替代顺序做小规模消融，而不是声称唯一忠实实现。

`K = 1` 时应有显式 fast path，直接调用原始 SigLIP 路径，数值上与未修改模型一致。这既是兼容要求，也是最重要的回归测试。

### 6.4 时间位置编码

使用相对当前帧的真实时间间隔，而不是只使用 frame index。例如 1 秒间隔的 6 帧为：

```text
[-5, -4, -3, -2, -1, 0] seconds
```

固定时间编码应平移为 `PE(t) - PE(0)`，保证当前帧的时间编码严格为零。数据管线必须保留实际 timestamp/offset，以支持未来改变采样间隔或使用不规则历史。

论文预训练使用总共 6 个 observation（5 个过去帧加当前帧），间隔 1 秒；`K=18 / 54 秒` 是后训练扩展上限，不是预训练默认值。本项目先严格对齐 `K=6`：B1K 30 Hz 数据使用 `history_frame_stride=30`；BEHAVIOR 评测也使用官方 30 Hz，每 32 个 action 做一次策略决策并保留每次 decision，得到约 1.067 秒间隔和 5.33 秒 oldest-to-current 窗口。只有 K=6 causal gate 通过后才做 K=12→K=18 curriculum。

### 6.5 历史 state

主线采用论文对齐方案：

- 将全部 K 个归一化 23 维 state 分别投影成 2048 维连续 token；
- 包含当前 state，并对缺失历史 state 提供 mask；
- 设置 `discrete_state_input=false`，不再把当前 state 重复编码进 π0.5 文本 prompt；
- 保留旧的“离散 current + 连续 past”作为兼容性消融，不再作为默认实现。

这会改变开源 π0.5 的输入分布，因此必须从共同 base 重新训练；旧 K=4 checkpoint 不能在新计算图下直接作为最终 short-memory checkpoint。

### 6.6 Short-memory 数据样本

一个逻辑样本建议包含：

```text
rgb:            3 cameras × K frames
state:          K × 23
time_offsets:   K
task:           full task instruction
subtask:        oracle/generated primitive
actions:        [32, 23]
```

构造约束：

- 历史只能来自 anchor 时刻及以前；
- action target 为 `[t, t + 32)`；
- 禁止使用 future observation；
- 禁止跨 episode 拼接历史；
- episode 开头不足 K 帧时使用 mask，不伪造为真实历史；
- 对所有摄像头和对应 state 使用一致的时间采样和时序增强。

建议使用 task-balanced sampler：先采 task，再采 episode，最后采 anchor，避免长 episode 或高频 primitive 支配训练。

初始 batch mixture：

- 50%：正常 K=6 历史；
- 25%：K=1，维持原始单帧能力；
- 25%：K=6，但对当前帧做遮挡或 current-camera dropout，增强对历史的真实依赖。

比例需要通过 clean performance 和 memory-sensitive evaluation 调整。

### 6.7 梯度与预训练问题

π-MEM 可以阻断 action expert 到 VLM 的梯度，是因为其 VLM 仍有文本/FAST 等训练信号。若当前开源 π0.5 只有 action flow loss，却同时完全阻断该梯度，新加入的视频编码器将无法学习。

因此第一版应在补回文本训练后再选择梯度策略：

- Gemma 参数可先冻结或使用很小学习率；
- 允许梯度穿过 Gemma 到 SigLIP/video path；
- SigLIP 使用较小学习率；
- action expert 使用正常低层学习率；
- 新增历史 state projection 使用相对更大学习率。

可增加两个辅助信号：

1. K=1 teacher distillation：约束修改后的模型保持原 π0.5 表示或动作分布；
2. current-frame feature reconstruction：用冻结原 SigLIP 的干净当前帧 patch feature 作为 teacher，让 history + masked-current student 预测该表示。

对专门构造的 memory-critical 遮挡样本，不应强制模型模仿只看当前帧的 teacher 动作，否则会压制历史信息的收益。

## 7. Long Memory 设计

### 7.1 输出格式

在恢复 πHL 后，将高层目标从仅输出 subtask 扩展为：

```text
Subtask: <next primitive>
Memory: <updated semantic memory>
```

同一个共享模型的高层调用读取 task、当前观测、current state 和上一轮 memory，输出下一 subtask 及更新后的 memory；低层调用读取 subtask 并生成动作。

### 7.2 第一版 memory 监督

B1K expert demonstration 可以可靠提供成功轨迹的进度监督，但不能天然提供策略失败后的纠错记忆。第一版应从已完成 primitive event log 确定性生成结构化自然语言 memory，包括：

- 已完成的对象和步骤；
- 对象被放置的目标位置；
- 容器/设备的 open、closed、on、off 等状态；
- 已处理数量和剩余数量；
- 重复对象中已经处理的实例或类别。

时间对齐规则：

- memory at step `i` 只包含 `i` 之前已完成的 primitive；
- current subtask 为 primitive `i`；
- memory 在 primitive 边界更新；
- 同一 primitive 内可保持不变；
- 不得把未来步骤结果泄漏到当前 memory。

第一阶段优先使用可验证的模板化自然语言。等结构化版本稳定后，再尝试自由摘要或外部 LLM 生成的多样化表达。

### 7.3 训练稳健性

仅使用 ground-truth previous memory 的 teacher forcing 会导致闭环误差累积。训练应逐步加入：

- scheduled sampling；
- 部分删除、顺序扰动或计数错误的 corrupted memory；
- 模型自生成 memory 回灌；
- subtask 和 memory 分字段评估。

### 7.4 失败与纠错记忆

B1K expert demos 主要支持“成功进度记忆”，不足以复现 π-MEM 中围绕失败、恢复和 in-context adaptation 的完整长记忆能力。

在基础策略具备一定成功率后，应从仿真 rollout 收集：

- subtask 成功/失败；
- 重试次数；
- simulator predicate 的状态变化；
- 失败类型；
- 恢复动作或替代 subtask；
- 必要时加入人工核验或语言标注。

这部分是 long memory 第二阶段，不应阻塞第一版基于 expert progress 的功能性验证。

## 8. 训练路线图

### Phase A：数据和层级能力恢复

- [ ] 完成 B1K 下载与完整性统计；
- [x] 实现全量 instance id 和 34 类 skill action canonicalizer；
- [x] 建立 primitive/skill/episode boundary manifest；
- [x] 生成复合 canonical subtask targets，并对空 primitive 回退到 skill；
- [x] 实现 R0 版 πHL masked text loss 和 greedy generation；
- [x] 完成 R0-A 固定标签和 R0-B 视觉条件小数据过拟合测试；
- [ ] 将 πHL text batch 接入正式 mixed SFT pipeline；
- [ ] 训练 πHL 和 oracle-subtask πLL；
- [ ] 建立 generated-subtask 层级闭环。

### Phase B：Short memory

- [x] 将单帧 SigLIP 扩展为 `[B, K, P, D]`；
- [x] 实现 per-frame spatial attention；
- [x] 实现 causal same-patch temporal attention；
- [x] 实现固定相对时间编码；
- [x] 实现指定层 drop past tokens；
- [x] 实现历史 23 维 state token；
- [x] 完成 K=1 等价、因果性、mask 和 shape 测试；
- [ ] 完成 short-memory SFT（seed=42 正式训练中）；
- [ ] 进行正确、重复、打乱历史的反事实评估。

### Phase C：Long memory

- [ ] 从 primitive event log 构造 progress memory；
- [ ] 扩展 πHL 输出 subtask + memory；
- [ ] 实现 memory state 的闭环传递；
- [ ] 加入 corrupted/generated-memory training；
- [ ] 评估进度、顺序、计数和重复操作；
- [ ] 收集 policy failure/correction rollout。

### Phase D：Multi-scale memory

- [ ] 固定同源四组模型和训练预算；
- [ ] 在选定任务上完成 2 × 2 消融；
- [ ] 报告 clean、occlusion、long-horizon 和反事实历史结果；
- [ ] 分析 short/long memory 的独立贡献与交互。

## 9. 评估与验收标准

### 9.1 能力门槛

进入正式 memory 对比前，oracle-subtask πLL 应在目标 primitive/skill 上达到约 50%–70% 的执行成功率。该范围是工程 gate，不是论文指标；具体阈值可随任务难度调整。

最终主任务应满足 memoryless hierarchical baseline 的 full-task success 大致处于 10%–60%：

- 接近 0% 时，memory 改进难以显现；
- 接近 100% 时，任务没有足够提升空间。

初始任务建议：

- 校准任务：`turning_on_radio`；
- short + long memory 主任务：`picking_up_trash`；
- 第二个复杂任务候选：`putting_dishes_away_after_cleaning` 或 `preparing_lunch_box`。

最终任务必须由 oracle pilot 和 memoryless baseline 结果决定，而不是只按语义复杂度决定。

### 9.2 核心 2 × 2 实验

| 高层语言记忆 | 低层视频历史 | 实验含义 |
|---|---|---|
| 关闭 | K=1 | 同源 memoryless baseline |
| 关闭 | K=6 | short memory only |
| 开启 | K=1 | long memory only |
| 开启 | K=6 | full multi-scale memory |

四组实验必须共享：

- 原始 π0.5 初始化；
- B1K 训练数据和采样规则；
- 总训练步数或等价 compute budget；
- evaluation seeds；
- subtask 标注规范；
- action horizon 和环境设置。

### 9.3 指标

主要指标：

- full task success；
- task progress / completed primitives；
- skill completion rate；
- repeated failure / redundant action count；
- generated subtask 的 verb、object、source/destination accuracy；
- memory 中 completed/remaining object、count 和 state accuracy。

short-memory 专项评估：

- 当前帧遮挡；
- 临时 camera dropout；
- 需要速度/运动方向判断的片段；
- correct history 对比 repeated-current、shuffled-history 和 K=1；
- 不同 history length 和 drop layer。

如果模型真正利用历史，正确历史应优于重复当前帧和乱序历史；否则仅仅增加 K 带来的提升不能证明 memory 机制有效。

### 9.4 初始验收目标

- short memory 在预先定义的 occlusion/history-sensitive 指标上相对 K=1 提升至少 10 个百分点；
- long memory 在 progress/order-sensitive 指标上相对无语言记忆提升至少 10 个百分点；
- full multi-scale memory 在主指标上优于两个单独模块；
- clean、无需记忆场景相对 memoryless baseline 的退化不超过 3 个百分点；
- 改进应跨多个 seed，并报告均值、方差和 episode 数量。

这些是项目验收目标，不是对原论文数字的宣称；可以根据 pilot 的统计置信区间调整，但调整必须在查看最终 test 结果前完成。

## 10. 必须保留的消融

### 层级能力

- full task → πLL；
- oracle subtask → πLL；
- generated subtask → πLL；
- skill condition 对比 primitive condition；
- πHL 单独模型对比共享模型，仅作为非主线 baseline。

### Short memory

- K=1 / K=3 / K=6；
- history pretraining + posttraining 对比 posttraining only；
- spatial only 对比 spatial + temporal；
- temporal layer / drop layer sweep；
- 正确、重复当前帧、打乱历史；
- image history only / state history only / both；
- current discrete state + past continuous 对比 all-continuous；
- teacher distillation on/off。

### Long memory

- no memory；
- ground-truth structured memory；
- generated memory；
- shuffled/wrong memory；
- teacher forcing only 对比 corrupted/scheduled sampling；
- expert progress memory 对比 rollout failure/correction memory。

## 11. 未决问题

以下问题需要通过实现检查或 pilot 决定，不能在没有证据时默认为已解决：

1. temporal attention 与 spatial attention 的精确 residual/norm 顺序；
2. FAST 离散动作目标放在第一阶段还是第二阶段；
3. 历史 state 采用兼容优先混合表示，还是直接全部连续化；
4. πHL 的调用频率和 subtask boundary 触发方式；
5. high-level 是否也读取 short video，还是只让 πLL 使用视频历史；
6. shared SigLIP/Gemma 的冻结范围、学习率和 LoRA/full fine-tuning 选择；
7. 最终主任务及其成功率区间；
8. short-memory 预训练是否需要额外 generic video，还是 B1K 加 feature teacher 已足够；
9. B1K 全量下载后的缺失 episode、损坏标注和摄像头完整性；
10. action expert 梯度与 VLM 隔离到什么阶段才可安全启用；
11. `turning_on_radio` 是否真的包含当前 RGB/state/primitive 无法消除的历史歧义，还是主要是 grounding 与低层抓取能力问题；
12. B1K 的 oracle primitive 是否应扩展为 grounded primitive，以及 2D object center、affordance point、mask 或 3D target 中哪种表示最适合 πLL；
13. 历史信息经 SigLIP 融合后是否被 VLM/action pathway 稀释，action-expert Modulator 是否能提供更有效的直接通路；
14. additive time encoding 与作用在 Q/K 上的 real-time temporal RoPE 哪一种能带来可验证的顺序敏感性；
15. recurrent memory 是否有足够的 episode-sequential 数据、辅助监督和 recurrence-oriented pretraining，避免再次学成 current-observation shortcut。

## 12. RoboMME 对照研究与候选扩展（2026-08-31）

本节记录文献事实、对本项目的解释和候选实验。除明确标记为“本项目已验证”的内容外，RoboMME 的结果不能直接外推为 B1K 结论。

参考资料：

- RoboMME paper：<https://arxiv.org/abs/2603.04639>；
- 官方 MME-VLA code：<https://github.com/RoboMME/robomme_policy_learning>；
- FrameSamp+Modulator config：<https://github.com/RoboMME/robomme_policy_learning/blob/main/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml>；
- Modulator implementation：<https://github.com/RoboMME/robomme_policy_learning/blob/main/src/mme_vla_suite/models/integration/history_gemma.py>。
- TemporalFlow-VLA：<https://arxiv.org/abs/2608.26821>。

### 12.1 两套分类轴不能混为一谈

RoboMME 将模型中的 memory representation 分为：

- symbolic memory：用 language subgoal 等离散、高层表示概括历史；
- perceptual memory：保留历史帧的 visual tokens 或 memory bank；
- recurrent memory：用 RMT、TTT 等迭代机制把历史压缩为固定大小的 latent state。

它同时按任务需求区分 temporal、spatial、object 和 procedural memory。π-MEM 的 short/long memory 则主要是时间尺度和系统职责划分。三者不是互斥关系：RMT 可以压缩 perceptual tokens，symbolic progress 可以与 recurrent state 并存，FrameSamp 也可能覆盖整个 episode 而不只是“short”窗口。因此后续设计不应把 recurrent 当成 symbolic/perceptual 的直接替代品。

### 12.2 RoboMME 的受控结果

RoboMME 在同一 π0.5 backbone 上报告的总体成功率如下：

| 方法 | AVG success | 解释 |
| --- | ---: | --- |
| π0.5，无 memory | 17.93% | current-observation baseline |
| SimpleSG+Oracle | 49.58% | simulator 提供正确 subgoal 文本 |
| GroundSG+Oracle | 84.08% | simulator 提供正确 subgoal 和目标图像坐标 |
| GroundSG+QwenVL | 32.70% | learned grounded symbolic memory |
| FrameSamp+Modulator | 44.51% | 最佳非-oracle MME-VLA variant |
| MemER | 42.38% | perceptual keyframe + symbolic subgoal hybrid |
| TTT（最佳 integration） | 22.35% | recurrent fast-weight memory |
| RMT（最佳 integration） | 20.17% | recurrent latent-slot memory |

可下的文献结论是：不存在所有任务都占优的 memory 表示；symbolic 对 counting/显著事件强，perceptual 对 motion-centric、time-sensitive 和长时视觉任务强；在该训练配方中 recurrent variants 最弱。不能据此得出“B1K 上 FrameSamp 必然达到更高成功率”或“recurrent 永远无效”。

### 12.3 FrameSamp+Modulator 的实际结构

FrameSamp 不是固定最近 K 帧，而是从截至当前的整个 episode history 中均匀选择最多 32 帧：

1. memory 只使用 front-view；
2. 冻结 π0.5 SigLIP，独立提取每帧视觉 tokens；
3. 每帧池化为 `4×4=16` tokens；
4. `32×16=512` 构成固定 memory-token budget；
5. 为每个 token 附加 time/y/x 三维位置特征，再投影到 action-expert width 1024；
6. 官方配置不使用 historical proprioception；
7. 论文公式写 `MaxPool`，released config/buffer 默认是 mean pooling。若移植，应以 mean 作为 checkpoint-faithful 默认，并把 pooling 做成显式消融。

Modulator 不把 512 个 memory tokens 拼入 VLM prefix。每个 action-expert layer 在 FFN 前执行：

```text
action tokens --Q--> cross-attention <--K,V-- memory tokens
                         |
                         v
                per-action scale / shift
                         |
                         v
               memory-conditioned RMSNorm
                         |
                         v
                  original action FFN
```

官方实现使用 4 个 query heads、1 个 KV head、head dimension 256，并在每个 action-expert layer 使用独立 modulator，总增量约 80M 参数。scale/shift 输出采用很小的初始化，使新模块起点接近 identity；VLM stream 保持原样。RoboMME 的 FrameSamp 三种 integration 成绩为 Context `30.68%`、Modulator `44.51%`、Expert `36.25%`，说明结果不能只归因于均匀采帧，直接调制 action pathway 也是主要变量。

本项目当前 `ShortMemoryVisionEncoder` 的路径不同：最近 K=6 帧在 SigLIP 内做 same-patch temporal attention，第 15 层后丢弃 history，只有融合后的 current tokens 继续进入 VLM；historical state 作为额外连续 tokens 加入 prefix。因此 FrameSamp+Modulator 应作为独立 baseline，不能被描述成现有 v3 的简单 K 扩展。

FrameSamp 本身不在 memory tokens 之间做 temporal self-attention。Cross-attention 对同时置换 K/V 行是集合不变的；若 shuffle 时连同视觉内容和已经绑定的 timestamp 一起移动，输出理论上可以不变。正确的顺序消融必须固定 temporal positions，只把 frame content 重新分配给这些位置。成功率提升但 `correct≈shuffle` 只能说明有效的历史内容检索，不能证明 temporal-order reasoning。

### 12.4 GroundSG+Oracle 为什么高，以及 B1K 如何借鉴

GroundSG 的输入不只是“当前做什么”，还包含被操作对象在 front image 中的点坐标，例如：

```text
SimpleSG:  pick up the green cube
GroundSG:  pick up the green cube at <63, 152>
```

Oracle evaluation 每一步直接读取 simulator 的 `grounded_subgoal_online`，action 仍由 π0.5 policy 生成；这与论文另一个“oracle planner 执行高层选择”的评测不是同一件事。`SimpleSG+Oracle 49.58% → GroundSG+Oracle 84.08%` 的主要增益集中在 Permanence `21.56→93.31` 和 Reference `32.28→95.17`，因为正确坐标直接消除了遮挡后的对象身份和位置歧义。它仍在 StopCube `49.67%`、InsertPeg `15.56%`、RouteStick `55.56%` 上受限，说明 precise visuomotor control 没有被语言 grounding 解决。

本项目现有 `pick up radio from coffee table` 更接近 SimpleSG+Oracle。它在正确 30 Hz/2,048-step 闭环中没有带来成功，trace 显示 EEF 最近仍距 radio 约 `0.196--0.278 m` 且从未闭爪。因此 GroundSG Oracle 是一个成本较低、解释力较强的候选诊断：

- 若 grounded prompt 改善 approach/grasp，则当前瓶颈包含目标定位；
- 若仍不能接近或闭爪，则主要瓶颈仍在 πLL/recovery/control；
- 若只改善 approach、不改善 grasp，则 grounding 与接触控制都是瓶颈。

B1K primitive 元数据包含 `manipulating_object_id` 和 support object，可从 simulator instance mask 或 3D→camera projection 构造 ground-truth point。第一版候选输出为：

```json
{
  "primitive": "pick up radio from coffee table",
  "target_object": "radio",
  "target_point": [142, 91],
  "support_object": "coffee table"
}
```

复杂 B1K 任务中 object center 可能不是可执行 affordance：pickup 更需要 graspable point，press 更需要 button/link，place 更需要 support region。因此应先区分 object-center grounding 和 affordance grounding。现有 πLL 没有在坐标 prompt 上训练，不能只在 evaluation 临时追加坐标；必须构造同源 grounded primitive SFT。GroundSG Oracle 只作为 privileged upper bound 和故障定位，不计作已经完成 long memory。

### 12.5 TemporalFlow-VLA 对当前结果的外部佐证

TemporalFlow-VLA 报告了一个与本项目非常接近的 baseline 现象：模型使用按顺序训练的 3 帧 history；evaluation 固定 current frame、打乱 3 个 historical observations 后，offline action flow-matching loss 几乎不变，而完全移除 history 会使 loss 增加约 `4.6%`。这说明普通多帧模型使用了历史内容，却没有形成对正确顺序的可靠依赖，与本项目修复后观察到的 `correct > repeat-current`、`correct ≈ shuffle-past` 在因果含义上相同。两者数据和模型不同，因此只能视为相互支持的现象，不能合并数值。

TemporalFlow-VLA 没有继续单纯增加 history frames，而是使用 previous action chunk 对齐的 `t-15/t-8/t` 三帧和两个 temporal queries：

- `Q8` 只读取 recent frame、current frame、language 和自己，表示最近半个 chunk 的变化；
- `Q15` 额外读取更早 frame 和 `Q8`，形成 `Q8→Q15` 的有向两级摘要；
- action tokens 只能访问 `Q8/Q15`，不能直接访问 historical patch tokens；
- training 使用 robot states、URDF geometry 和 calibrated camera 离线生成 robot-surface temporal-flow target；
- deployment 不运行 geometry/flow teacher，只保留 temporal queries。

这项工作支持一个重要判断：action loss 和无结构多帧输入可能足以学习“过去有用”，却不足以规定“过去的物理变化应如何表示”。它提供了比单纯 masked-current 更有物理含义的候选监督。B1K 同样有 robot state、simulator geometry 和 camera calibration，原则上可以生成类似 teacher，但必须先审计双臂/相机投影、遮挡和 object-contact 情况；robot-surface flow 只直接监督机器人运动，并不完整覆盖被操作物体的运动。

若后续采用该思路，π0.5 action chunk 为 32，可先研究 `t-31/t-16/t` 的 chunk-aligned queries，而不是沿用 K=6、1 秒 stride 的长窗口。它和 FrameSamp 解决不同尺度：TemporalFlow-style query 表示最近一个 chunk 的物理执行结果，FrameSamp 表示 whole-episode 稀疏视觉历史，symbolic memory 表示任务进度。

### 12.6 Temporal RoPE 候选

当前 K=6 video encoder 使用 additive relative-time encoding，即把 `PE(t)-PE(0)` 加到 patch token 后再做 temporal attention。真正的 temporal RoPE 应作用于 attention projection 后的 Q/K：

```text
q_i = R(t_i) Wq x_i
k_j = R(t_j) Wk x_j
score(i,j) depends on t_j - t_i
```

候选位置有两个：

1. 现有 same-patch temporal attention：同一 spatial patch 跨 K 帧使用真实秒数做 Q/K rotation，并继续保留 causal mask；
2. FrameSamp Modulator：memory key 使用历史真实时间，action query 使用当前时间或 `current + future action-step time`，空间 y/x 仍用 additive encoding。

RoPE 只提供相对时间归纳偏置，不会自动产生顺序依赖；任务和监督仍必须让不同历史对应不同正确动作。为了避免混淆，候选实验应区分 `FrameSamp+Modulator-official`（additive time/y/x）和 `FrameSamp+Modulator-temporal-rope`（additive spatial + Q/K real-time RoPE），不应首次移植时同时更改两者。

### 12.7 Recurrent memory 的定位

Recurrent memory 的理论优势是 episode 增长时 state/token 数仍固定，并天然支持在线更新；但压缩可能丢失早期信息，远距离 action loss 难以训练写入策略，状态漂移会累积，而且 current RGB/state/primitive shortcut 仍可能使 recurrent state 被完全忽略。

RoboMME 的 TTT/RMT 只比无-memory π0.5 略高，明显弱于 symbolic/perceptual。作者将其归因于浅层 recurrence 插件微调不稳定，并认为需要更深的架构融合和 recurrence-oriented pretraining。对本项目而言：

- recurrent 不替代 symbolic long memory；
- 在未证明 B1K task 真正需要历史、现有 πLL full-task success 仍低时，不应把 RMT/TTT 作为当前主线；
- 若后续试验，先做易于 reset/checkpoint/并行的 RMT，再做每 env 需要独立 fast weights 的 TTT；
- 应按 action chunk 或 subgoal boundary 更新，并配合 next-primitive、boundary、next-feature 或 temporal-order auxiliary supervision。

### 12.8 候选实验矩阵与证据边界

以下均为 proposed，而不是已启动或已通过：

| 优先级 | 候选实验 | 回答的问题 |
| ---: | --- | --- |
| 1 | SimpleSG Oracle vs GroundSG Oracle πLL | radio failure 是否包含目标 grounding 瓶颈 |
| 2 | Recent-K+Modulator | 当前失败是否主要来自 history→action 注入路径 |
| 3 | FrameSamp+Modulator-official | 稀疏 whole-episode perceptual memory 是否优于最近 K 帧 |
| 4 | TemporalFlow-style supervised queries | 物理 temporal target 是否能建立 action-usable order representation |
| 5 | FrameSamp+Modulator-temporal-rope | real-time Q/K rotation 是否增加顺序敏感性 |
| 6 | RMT pilot | 固定大小 recurrent compressor 是否值得更深投入 |

若实施 perceptual 分支，配置应使三个变量正交：

```yaml
history:
  sampling: recent_k | uniform_episode
  integration: vision_temporal | action_modulator
  time_encoding: additive | temporal_rope
```

所有分支必须从同源 πLL 初始化、使用同一数据和训练预算，并保留 `current-only/zero-memory`、`repeat-current`、固定 timestamp 的 content-shuffle、correct-history、offline event/action fidelity 和最终 closed-loop stage/success。仅有 loss 降低、参数能够 backward 或 memory attention 非零都不构成 memory acceptance。

截至 2026-08-31，本项目已经证明的是：

- 修复后 history content 会影响 grasp-event action，correct 明显优于 repeat-current；
- correct 不优于 shuffle，因此没有证明 temporal order；
- `turning_on_radio` 的所有当前 memory/Oracle-primitive policy full-task success 仍接近零并卡在 pickup；
- 当前 primitive-conditioned πLL 在示范 grasp manifold 上能闭爪，但闭环到达/状态分布偏移明显。

尚未证明的是：

- GroundSG Oracle 能改善 B1K；
- FrameSamp/Modulator 能改善 B1K；
- TemporalFlow-style supervision 能在 B1K 建立 temporal order；
- temporal RoPE 能建立顺序敏感性；
- recurrent memory 比 symbolic/perceptual 更适合当前项目；
- `turning_on_radio` 本身足以作为严格的 memory benchmark。

## 13. 决策记录

### 2026-08-29

- 项目目标确定为在开源 π0.5 上完成功能性 multi-scale memory 复现，而非宣称严格复现 π0.6；
- short memory 和 long memory 都属于最终验收范围；
- 首先恢复 π0.5 论文中的共享 πHL/πLL 层级路径；
- 主路线不额外引入独立 4B πHL，扩容只作为后续有证据支持的选项；
- B1K primitive 作为主要 high-level subtask，skill 作为 curriculum/辅助监督；
- 先建立有一定成功率的 B1K 层级基线，再评估 memory；
- short memory 默认从 K=6、temporal layers `[3, 7, 11, 15]`、第 15 层后丢弃历史开始；
- long memory 第一阶段使用由已完成 primitive 确定性生成的 structured progress memory；
- 最终采用同源 2 × 2 消融验收 short、long 和 multi-scale memory。
- 选用 `lerobot/pi05_base` 的上游 OpenPI PyTorch checkpoint 作为共同初始化；它可由 `openpi_rlinf` loader 在内存中转换，R0 不需要伪造 action norm stats 或落盘第二份权重；
- R0-A 在 200 steps 后达到 100% exact match；R0-B 在 500 steps 后达到 100% exact match，将图像跨 primitive 轮换后降为 0%，因此决定通过 R0 gate 并进入 R1。
- R1 对每个 task 独立做 80/10/10 episode split，而不做随机帧划分；对复合 primitive 使用 `then` 连接结构化动作，并只从其对应 manipulation skill 区间采样；
- task-0000 R1 pilot 使用 78 个 train episode 和 15 个不重叠 validation episode，300 steps 后 held-out exact match 达到 83.33%；test split 仍保持未使用。
- R1 current state 严格复用现有 Behavior policy 的 256D proprio → 23D 映射，并只从所选 train frames 计算 1%/99% quantile normalization；validation/test 不参与统计；
- task-0000 配对输入消融中，image-only/state-only/image+state exact match 分别为 83.33%/91.67%/75.00%。这表明当前小样本主要依赖 state 识别阶段，不能把结果解释成充分的视觉理解；先扩大数据再判断 image+state；
- 增加 task-balanced 两级 sampler 和 macro-task、verb、object、destination、step-count 指标；
- 使用相同 observation/noise 的 5-step action denoising gate 检查文本微调后的动作漂移。image-only/state-only/image+state tail 的 relative RMSE 分别为 1.49%/1.94%/3.22%，均通过暂定 10% gate，故暂不强制切换到独立 πHL 或 text-only adapter。
- 修复了初始 validation 误用 training sampler 导致 sampler epoch 前移的问题；旧 `state_seed42` 实际训练顺序与 seed 43 相同，已用独立 evaluation loader 重跑为 `state_seed42_corrected`，三 seed 汇总只采用相互独立的训练序列；
- task-0000 最后 2 层、1000 steps 虽达到 `91.11% ± 2.19%` exact match，但 action relative RMSE 为 `12.04% ± 3.76%`，只有 1/3 通过，故拒绝该配方；
- task-0000 最后 2 层、500 steps 达到 `88.33% ± 2.04%` exact match，action relative RMSE 为 `5.97% ± 1.84%`，3/3 通过，选为当前 R1 安全配方；
- 同预算 image+state 的 exact match 同为 `88.33% ± 2.04%`，但 CE 更高、动作漂移为 `8.10% ± 4.11%` 且只有 2/3 通过，因此当前不作为默认配方；
- image+state seed 43 的配对反事实中，aligned exact match 为 90.83%，错配 image 后为 39.17%，错配 state 后为 20.00%。这证明两种输入确实被使用，但不能把 task-0000 的 clean 指标解释为 RGB 带来的增益；
- 7-task、每 task 100 train / 20 validation、500-step task-balanced pilot 达到 86.43% micro/macro exact match；verb/object/destination 分别为 95.81%/98.95%/89.63%，32 样本 action relative RMSE 为 3.13%，通过 gate。
- progress-sensitive task `[0, 1, 11, 12]` 的 400/80、500-step pilot 达到 50.00% macro exact match；per-task 为 radio 65%、trash 70%、dishes 30%、lunch-box 35%，verb/object/destination 为 76.92%/76.92%/69.23%。其 32 样本 action relative RMSE 为 5.33%、余弦相似度为 0.9985，通过 gate。它可作为 R2 起点，但不足以宣称复杂任务 high-level 已解决；
- 发现并确认本地 `pi05_b1kpt50_pt` 是 OpenPI-Comet 在 B1K 0–49 tasks 上预训练的 π0.5 checkpoint。正式 baseline/MEM 改为从它共同初始化，`pi05_base` 的既有 R0/R1 结果作为接口与消融证据保留；
- R2 oracle primitive prompt 不采用 primitive 外层 duration 包络，因为 1,254 个 episode 存在嵌套或重叠。改用 primitive 引用的全部 skill 区间并集；10,000 个 episode 均可解析为 86,351 个区间，607 个 episode 的残余歧义帧会被排除；
- 从 `pi05_b1kpt50_pt` 恢复文本 tail 后，7-task exact match 为 80.00%，32 样本 action relative RMSE 为 1.05%；progress-sensitive 4-task exact match 为 56.25%，per-task radio/trash/dishes/lunch-box 为 70%/80%/30%/45%，动作漂移为 2.26%。两者均通过 gate；
- B1K checkpoint 的简单 7-task 文本精度低于通用 base（80.00% vs 86.43%），但复杂 4-task 更高（56.25% vs 50.00%）且动作保持更好。主线选择 B1K checkpoint 的依据是动作域能力与复杂任务表现，不是所有离线文本指标均占优；
- 验证 `/mnt/public/daibo/venv/behavior_openpi` 可作为 R2 环境：Python 3.10、PyTorch 2.5.1+cu124、OmniGibson 3.7.2、OpenPI、Ray 和 4×A100 均可用；
- production primitive streaming 返回三路 `[3,720,720]` RGB 和 `[32,23]` action；完整 transform/collate 返回三路 `[B,224,224,3]`、`[B,32]` state、`[B,200]` prompt 和 `[B,32,32]` action，并正确读取 B1K norm stats；
- `pi05_b1kpt50_pt` 在一个真实 oracle-primitive batch 上完成 flow-loss forward，loss shape 为 `[1,32]`、mean 为 `0.00507` 且全部有限，因此解除 R2 数据/模型接口阻塞；
- R2 采用 predicate-driven Oracle-HL：直接读取 OmniGibson `turning_on_radio` task-specific sequential reward 的 `current_stage_name`，不使用 demonstration timestamp 切换 prompt；
- 三条件因果评测冻结为 `base_task / mixed_task / mixed_oracle_stage`，每个 condition 使用完全相同的 ordered instance ids 和 eval seeds；逐 episode 指标包含 instance id、success、stage progress，并在聚合前保存 JSON；
- `openpi_rlinf` eval loader 已真实加载 RLinf FSDP `global_step_1/actor/model_state_dict/full_weights.pt`，3.353B 参数全部匹配，证明 SFT checkpoint 可以直接进入 rollout；norm stats 显式固定到 B1Kpt50；
- short-memory v1 采用 residual 顺序 `spatial attention → temporal attention → MLP`，temporal layer 为 `[3,7,11,15]`，第 15 层后丢弃历史 image tokens；temporal attention 复用 SigLIP 当前层权重，不新增 Q/K/V；
- short-memory v1 保留当前 π0.5 离散 state prompt，仅把过去 `K-1` 个归一化 23D state 投影为连续 Gemma-width token；`K=1` 显式调用原 SigLIP，单测达到逐元素精确一致；
- 第一版 short-memory recipe 暂用 `K=4`，B1K 30 Hz 训练数据 stride=16；该 stride 与后来确认的官方 30 Hz、32-action policy cadence 不一致，也是 v1 降级为诊断数据的原因之一。真实 B1K batch 已验证为三路 `[1,4,224,224,3]`、state `[1,4,23]`、mask/time `[1,4]`、action `[1,32,32]`；
- seed=42 首次正式训练在 step 391 因本机 Ray 被误清理而中断，当时尚未到原定 step-500 首次保存点，无法无损 resume；保留旧日志，使用相同 B1Kpt50 初始化、数据、seed 和优化配置从 step 0 重启，仅把 checkpoint interval 从 500 缩短为 100。重启后的 step-100 checkpoint 已完整写入 `run_restart/checkpoints/global_step_100`（约 50GB，含 13.4GB `full_weights.pt`），训练已继续；
- 重新审计发现 `nxb_4090` 与本机共享 NFS，demos 下载本身完整；OmniGibson 数据另有 `omni_data` 与拼写相近的旧 `omini_data` 两套目录。正确的 `/mnt/public/daibo/datasets/omni_data` 已包含 `behavior-1k-assets 3.7.2rc1`、key、robot assets 和完整 task instances；旧 `omini_data` 只有 `3.7.0rc23`，此前版本错误来自指向旧目录；
- turning-on-radio 在正确 task-instance repo 中有共享 base template 0 和 301 个 `*_template-tro_state.json`。正式配对改用官方 public-test IDs `[242,295,211,203]`。修复 launcher：基础环境必须先以 `activity_instance_id=0` 创建，再由 `ActivityInstanceLoader` 在 reset 前应用 fixed public-test tro_state；不能把首个 public ID 当作基础 template id；
- step-500 三条件闭环已在 `nxb_4090` 完成，输出为 `/mnt/public/daibo/results/mem_r2_oracle_gate_step500_official/paired_4x_seed42`。四个官方实例 `[242,295,211,203]` 在三条件中的顺序完全一致，每项均有 raw JSON 和4段视频，日志无 traceback；
- 该轮 `base_task / mixed_task / mixed_oracle_stage` 的 success 均为 `0/4`，但全部 episode 都精确在 511 simulator steps 截断。π0.5 每次输出 32 个 action，故 512 只允许 16 次 policy decision；这轮只能证明 prompt 改变了 rollout，不能用于 Oracle-HL gate。turning-on-radio 的配对协议改为 2,048 simulator steps（64 次 policy decision），并把 horizon 显式写入 manifest；其他 task 的 horizon 后续单独确定；
- memoryless mixed πLL seed=42 已完成 2,000/2,000 steps，最终 loss `0.00338`、grad norm `0.0479`，学习率按计划衰减到 0；`global_step_100` 至 `global_step_2000` 每 100 steps 均有约 50GB 的完整 FSDP checkpoint；
- step-2000 的 2,048-step 三条件配对已在 `nxb_4090` 完成，输出为 `/mnt/public/daibo/results/mem_r2_oracle_gate_step2000_h2048/paired_4x_seed42`。`base_task / mixed_task / mixed_oracle_stage` 的 success 均为 `0/4`，平均 completed stage 为 `1.25 / 0.50 / 0.25`，平均 return 为 `412.388 / 316.086 / 300.639`；
- mixed 相对 base 的平均 stage/return 差为 `-0.75 / -96.302`，Oracle 相对 mixed 为 `-0.25 / -15.447`。三条件均保存4段视频且日志无 traceback。mixed 的闭环退化有效，但 Oracle 条件初始使用了训练中不存在的 `move to radio`，因此 Oracle-HL gate 结论无效，必须按 B1K primitive 粒度修正后重跑；
- 已将 simulator 的 `move_to_radio` 与 `pickup_from_support` 都映射到 canonical primitive `pick up radio from coffee table`，并让配对 launcher 支持只运行指定 condition。修正后的 step-2000 输出位于 `/mnt/public/daibo/results/mem_r2_oracle_gate_step2000_h2048_corrected/paired_4x_seed42`：Oracle 仍为 `0/4` success、平均 stage `0.50`、return `304.787`，相对同 checkpoint task prompt 的 stage 差为 `0`、return 差为 `-11.299`，有效 Oracle gate 未通过；
- step 100/200/300/400/500 的 corrected 2,048-step sweep 已完整产出 10 份 raw metrics、40 段视频和 5 份 manifest，日志无 traceback。step 100/200/400 的 Oracle stage 相对 task prompt 均为 `+0.25`，step-100 的 paired return 增益最大（`+39.518`）；所有条件仍为 `0/4` success，因此选择 step-100 做 4,096-step sensitivity check，而不视为已经通过 gate；
- step-100 的 4,096-step sensitivity check 已完成，task/Oracle 均为 `0/4` success、stage `1.25`，Oracle return 高 `79.984`；说明 2,048-step 会截断后续进度，但 Oracle 尚未带来额外 stage。turning-on-radio 的 20 个官方 public-test IDs 已从 `metadata/test_instances.csv` 核对；
- step-100 的 public-20 评测已完成：Oracle 相对 task 的平均 stage 为 `+0.35`（8 正/1 负/11 平，exact sign `p=0.0391`），return 为 `+15.921`（Wilcoxon `p=0.0826`），但两者 success 均为 `0/20`。补跑部分完整生成 8 份 raw metrics、32 段视频、4 份 manifest，日志无 traceback；
- 原始 base 的 public-20 补充评测已完成：base 为 `1/20` success、stage `1.20`、return `367.721`；step-100 task 为 `0/20`、`0.85`、`382.286`，Oracle 为 `0/20`、`1.20`、`398.207`。base 补跑部分有 4 份 raw metrics、16 段视频、4 份 manifest，日志无 traceback；
- R2b conservative recipe 冻结 SigLIP、token embedder 与 Gemma expert-0，只训练 action expert/projections；学习率从 `2.5e-5` 降为 `2.5e-6`，预算为 100 steps、每 25 steps 保存。mixed 模式新增显式 `mixed_boundary_fallback_to_task`，跨 primitive boundary 的 chunk 回退 full-task prompt，从而保留 transition action 监督；旧 recipe 默认仍为 false。5-step 4×A100 smoke 稳定 step 约 `7.5s`，权重审计确认冻结张量 `0/477` 改变、可训练张量 `190/190` 改变。正式 step 25/50/75/100 均完成 task/Oracle 闭环：success 全为 `0/4`，Oracle−task stage 为 `-0.25/-0.75/+0.25/-0.25`，return 为 `-52.30/-130.95/-66.59/-44.48`，故 R2b gate 失败；
- R2c 在 R2b 上只改变冻结范围：新增 `freeze_vision_encoder`，固定 SigLIP，但允许 token embedder、Gemma expert-0、action expert 与 projections 更新。5-step smoke 与正式 100-step seed=42 均完成，稳定 step 约 `7.6s`。最终权重审计为 vision `0/331` 张量变化、Gemma expert-0 `150/156`（relative L2 `0.0164%`）、action expert `172/172`（`0.0525%`）、action projections `8/8`（`0.0509%`），证明实际更新范围符合实验设计；
- R2c step 25/50/75/100 的 task stage 为 `0.75/1.00/1.25/1.25`，Oracle stage 为 `0.50/1.00/1.00/1.00`；Oracle−task return 为 `-78.04/-117.35/-30.26/-62.55`，所有条件均为 `0/4` success。32 段视频完整且日志无 traceback，故 vision-only freeze 仍未建立稳定 primitive routing，R2c gate 失败；
- short-memory 闭环新增 `history_ablation: none/repeat_current/shuffle_past`。后两者分别以当前 observation 替换所有有效历史内容、以及确定性反转有效过去帧，同时保持当前帧、padding mask 和 time offsets 不变；新增 paired launcher 固定同一 checkpoint/seed/instance order。相关 11 项 short-memory/launcher 单测通过；
- short-memory step-500 四条件闭环已完成：memoryless / 标称真实历史 / 标称重复当前 / 标称打乱过去的 stage 为 `0/0.50/1.00/0.75`，return 为 `205.371/375.188/385.378/384.117`，success 全为 `0/4`；16 段视频完整。后续 transport 审计证明后三条件在进入模型前都丢失 history，故本条只保留为历史记录，不再解释为 memory causal gate；
- 参数审计显示 SFT 更新了 vision、VLM expert-0、action expert 和 action projection 的全部张量。step `100 / 500 / 2000` 的相对 L2 漂移分别为：vision `0.157% / 0.544% / 0.732%`，VLM `0.290% / 1.015% / 1.354%`，action expert `0.270% / 0.894% / 1.175%`，action projection `0.351% / 1.737% / 2.188%`。单 task、全参数、无 validation/early stopping 的 2,000-step SFT 造成了单调的行为参数漂移；
- mixed loader 对 full-task 和 primitive 两种 prompt 都排除跨 primitive boundary 的 32-step chunk。task-0000 的 404,187 个 valid-duration frames 中仅 375,972 个满足 boundary-safe 条件，排除 28,215 帧（6.98%）；其中 21,049 帧由 action horizon/边界歧义造成。阶段转换监督的缺失是闭环退化的一个具体风险；
- 北京集群与本机不共享存储。仅将运行所需的 `rlinf/`、`examples/`、`evaluations/`、`toolkits/`、`tests/` 及项目文档同步到 `bjd_dev_2:/mnt/public/daibo/timeline/0812/RLinf`，共 1,761 个文件、约 11.65MB，没有覆盖远端既有仓库；同步后 11 项 short-memory/Behavior 单测全部通过；
- `bjd_dev` 与 `bjd_dev_2` 实际均为 4×A800-SXM4-80GB。`bjd_dev` 已有连接外部 head 的 Ray worker，故保持不动；short-memory 训练只使用空闲的 `bjd_dev_2`。该机可用环境实际位于 `/mnt/public/daibo/venv/behavior_openpi`，B1K demos 位于 `/mnt/public/daibo/dataset/behavior-1k/2025-challenge-demos`；
- short-memory `K=4, stride=16` 的 4×A800 FULL_SHARD smoke 在 global batch 32 时稳定 step 约 `1.98s`，loss `0.00908`、grad norm `1.89`；为了与 memoryless πLL 的每步样本预算一致，最终 recipe 改为 micro batch 4、global batch 256，即每 rank 累积 16 个 micro-batch；
- global-batch-256 smoke 的稳定 step 为 `14.8s`，loss `0.0165`、grad norm `0.725`，采样显存约 `66.0--66.5 GiB/GPU`，没有 OOM。首步约 143 秒主要是 32 个 dataloader worker 冷启动；正式 seed=42 已在 `bjd_dev_2` 的 tmux `mem_short_seed42` 启动，输出为 `/mnt/public/daibo/results/mem_short_task0000_seed42`，2000 steps、每 100 steps 保存。正式 run 已连续通过前 4 steps，稳定 step 为 `14.8--16.3s`，loss 与梯度均有限；
- 论文复核确认预训练是 6 个 observation、1 秒 stride，K=18/54 秒只用于后训练扩展。代码审查发现 v1 把 temporal PE 加在 LayerNorm 之后，实际为 `LN(z)+e(t)`；v2 改为 `LN(z+e(t))`，并让 temporal residual 从 `z+e(t)` 继续传播，当前帧仍因 `e(0)=0` 保持不变；
- v2 将所有 K 个 normalized proprio state 投影为连续 token，并通过 transform override 设置 `discrete_state_input=false`，避免当前 state 同时出现在文本和连续 token 中；
- v2 训练 recipe 为 B1K `K=6, stride=30`；官方 30 Hz 评测为 `history_length=6, history_decision_stride=1`，约为 1.067 秒间隔、5.33 秒窗口。旧 K=4 checkpoint 的参数形状虽可加载，但激活语义已经不同，不能继续作为 v2 checkpoint；
- v2 的 17 项 short-memory/Behavior/launcher 单测、Ruff lint/format、真实 B1K `[1,6,23]` batch、Hydra train/eval compose 均通过。完整 3.35B 模型的 K=6 前向 prefix 为 `[1,974,2048]`、flow loss `1.171875` 且有限，峰值显存约 6.63 GiB（no-grad）；
- v2 的 4×A100 FULL_SHARD、global-batch-32 两步反向 smoke 无 OOM；第一个 step 含 worker 冷启动为 106 秒，稳定第二步为 2.03 秒，loss `0.477→0.169`、grad norm `46.2→6.34`，均为有限值；
- v2 seed=42 正式训练已在本机 tmux `mem_short_k6_v2_seed42` 启动：micro batch 4、global batch 256、2000 steps、每 100 steps 保存，日志为 `/mnt/public/daibo/results/mem_short_k6_v2_task0000_seed42/train.log`。正式 run 已通过 step 4，稳定 step 为 `15.1--16.7s`，显存约 `70.6--70.9 GiB/GPU`，loss/grad norm 均有限。step-100 先做离线/闭环 causal gate，不直接等待 final；
- 2026-08-30 03:03 UTC 核验时，v2 seed=42 训练仍在运行而非已经完成：进度为 `1413/2000`，无 traceback/OOM/NCCL error，step 100--1400 的 `full_weights.pt` 均完整写入且每份约 13.4GB；稳定 step 约 `15.6s`，预计还需约 2.6--3 小时；
- 新增 demonstration flow-loss causal gate，并以固定样本、相同 noise/time 比较 `correct / repeat_current / shuffle_past`。16 样本 checkpoint sweep 中 step 100/500/1100 通过方向性 gate，step 200/800/1400 未通过；step-1400 的 shuffle loss 甚至略低于 correct，说明 temporal sensitivity 随训练并不单调，不能默认 final checkpoint 最优；
- step-100 与 step-1100 的 64 样本复核均通过 gate。step-100 的 correct/repeat/shuffle loss 为 `0.023278 / 0.031984 / 0.023730`，correct 胜数为 `60/64 / 38/64`；step-1100 为 `0.022408 / 0.026010 / 0.023443`，胜数为 `53/64 / 46/64`。step-1100 具有更低 clean loss和更均衡的内容/时序敏感性，选为第一轮闭环 checkpoint；
- step-1100 的 4,096-step、`skip_intermediate_obs=false` 四条件闭环已在 `nxb_4090` 完成，输出为 `/mnt/public/daibo/results/mem_short_k6_v2_step1100_paired_h4096_fullobs`；memoryless 对照使用同预算 R2 step-1100，实例固定为 `[242,295,211,203]`、seed 42。四份 raw metrics、16 段视频齐全，日志无 traceback。启动时发现远端 `/tmp` 仅余 1.69GB，故 Ray 临时目录改为短路径 `/mnt/public/daibo/rtk6`，并显式设置 `OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data`；
- 该闭环记录的 memoryless / correct / repeat-current / shuffle-past success 均为 `0/4`，平均 completed stage 为 `0.75 / 1.00 / 1.00 / 0.75`，平均 return 为 `733.162 / 896.998 / 896.117 / 873.651`。由于后三条件的 history 后来被确认在 model input transport 中丢失，条件间差值不再解释为 v2 memory 效果；
- 对 correct-history 的四段 4097-frame/60Hz 视频做均匀时间轴检查后，四个实例都完成了 `move_to_radio`，随后约 50 秒持续在桌边伸手、调整腕部或振荡，但 radio 从未离开 coffee table；因此全部卡在 `pickup_from_support`，没有进入 `press_radio` 或 `place_on_support`。reward 的 pickup gate 要求目标先离开 support 且进入 robot hand，stage count=1 证明整个 episode 从未形成有效 pickup；
- 到桌后的 ego video 中，1.07 秒相邻样本灰度 MAE 中位数为 `4.6%--6.8%`，5.33 秒窗口为 `5.7%--9.8%`，后者有 `30.0%--48.5%` 像素变化超过 10/255；这只说明环境视觉并非静止。由于该轮模型没有收到 history，不能再从这些轨迹解释 `repeat_current≈correct`；
- 本轮实际 Hydra 配置的 `num_env_subprocess=1`，但 EnvGroup world size 为 4、每个 EnvWorker 只拥有一个 local env，因此实际拓扑是 4 个独立 BehaviorProcess、每 GPU 一个，并非 4 个环境共享同一 OmniGibson 进程。在当前 `total_num_envs=4`、4 EnvWorker 拓扑中，每 worker 的 local total 为 1，设置 `num_env_subprocess>1` 会因不可整除而报错；这不是重跑理由。若未来改成单 EnvWorker 管 4 个环境，则应相应设置 4 个 subprocess，并把拓扑作为评测协议的一部分固定；
- 2026-08-30 进一步核对官方 B1K `omnigibson.learning.utils.eval_utils` 与 dataset timestamps，确认 acceptance cadence 应为 action/render 30 Hz、physics 120 Hz。旧 4,096@60 Hz 与 2,048@30 Hz 虽同为约 68 秒，但 policy chunk 动力学不同；旧 step-1100 四条件只能保留作诊断。两个 eval YAML 和两个 paired launcher 已统一为显式 30/30/120 Hz、2,048 steps、30 fps；short-memory 固定 decision stride 1；
- 为定位 pickup 失败，Behavior eval 新增可选 decision trace，逐 chunk 保存 stage/prompt、双臂 gripper command、实际 gripper width、EEF-object distance、radio pose 及 `in_hand/on_support/has_left_support`。相关单测通过，trace 关闭时不改变默认 rollout；
- step-1100 的 corrected Oracle 30 Hz trace 输出为 `/mnt/public/daibo/results/mem_short_k6_v2_step1100_oracle_stage_trace_h2048_f30_fullobs`。四实例均完成 move-to-radio 但没有 pickup；64 次 decision 中左右 gripper action 全程保持正值，夹爪全开，radio 未进入 hand。ID 295 的 radio 在桌面水平滑动约 0.318 m 但仍保持 on-support，属于碰撞/推动而非抓取；
- corrected 30 Hz step-1100 标称四条件已完成：`memoryless/correct/repeat/shuffle` 的 return 为 `408.608/453.215/447.636/456.767`，stage 为 `1.00/1.00/0.75/1.00`，success 全为零。频率和 full-observation 协议正确，但 history transport 当时仍有缺陷，不能据此比较 correct 与 shuffle；
- v2 seed=42 已完成 `2000/2000`，final loss `0.0111`、grad norm `0.0969`、learning rate 0；`global_step_2000` 的四份 DCP shard、metadata 和 `13,414,188,487`-byte `full_weights.pt` 完整。训练结束时仅有 Python resource-tracker 的 leaked semaphore warning，没有训练错误；
- final 64-sample offline gate 的 correct/repeat/shuffle loss 为 `0.013736/0.014280/0.013808`，correct wins 为 `46/64` 与 `35/64`。它虽然保持 mean directional gate 且 clean loss 低于 step-1100，但控制项差距与 win count 都更弱（step-1100 为 `53/64`、`46/64`），因此不追加 final 闭环，step-1100 继续作为 causal early-stop 候选；
- task-0000 全 200 demonstrations 的 pickup primitive 虽有闭爪监督，但 boundary-safe chunk 中仅 `9.955%` 含任一闭爪 action；因此当前证据不是“grasp action 被 boundary filter 全删掉”，而是长 navigation/reach 区间导致动作事件高度失衡；
- 新增 `grasp_event_offline_eval.py`，按 primitive 精确过滤 demo，分别抽取远端 open-control、close-onset 和 closed-hold，并把 observation/action 样本缓存为可跨 checkpoint 复用的 `selection_samples.pt`。最初选择 onset 前 1--2 帧作为 control 会把真实闭合刚好落在 horizon 外的样本误算为假阳性；固定 `open_control_margin=32` 后，control 距真实闭合约 64 帧，gate 定义不再含糊；
- step-1100/2000 在相同 24 样本、每样本 4 draw 上均通过 offline grasp gate。step-1100 的 event/correct-hand/closed-step/open-FP 为 `79.69%/79.69%/73.15%/3.125%`；step-2000 为 `78.125%/78.125%/67.18%/3.125%`。这证明 πLL 在 demo grasp manifold 上会闭合正确右手，闭环全程开爪来自到达/状态分布偏移，暂不进行 event-balanced SFT；
- grasp-event 扫描发现 250-frame streaming chunk reload 会清空 short-memory history。`K=6, stride=30` 需要 151 帧才能填满窗口，意味着每个 chunk 前约 60% 区域使用 padded history；这是本轮 short-memory temporal causal gate 失败的重要混杂因素，下一轮训练前必须修复；
- 修复 `EnvOutput.prepare_observations()` 的 history transport，并增加 schema/eval 覆盖，确保四路 history tensor/mask 真正进入 OpenPI policy；修复前的所有闭环 memory 条件比较均已在本文件降级；
- 新增 exact-observation snapshot、ranked one-shot model-action override、raw HDF5 `scene_file + sparse UUID state` mid-stage reset 和对应测试。四个 expert takeover 仅 `1/4` 达到 `in_hand`，证明 state takeover 不能稳定复现示范接触轨迹；
- 新增三条件 grasp-event action-fidelity：step-1100/2000 的 correct 相对 repeat-current 均显著降低 open false-positive 和右夹爪 MAE，但 shuffle-past 与 correct 等价。当前 checkpoint 只通过 history-content gate，未通过 temporal-order gate；
- 真实 task-0000 stream 审计通过：episode 10 的 frame `149/150/151` valid history 为 `5/6/6`，frame `249→250→251` 跨 250-frame chunk 边界仍保持 `6/6/6`；新增单测同时验证 same-episode 不 reload、切换 episode 才 reload；
- 修复后的生产拓扑 smoke 通过：本机 4×A100、每 rank 2 个 spawn DataLoader workers、global batch 32、FULL_SHARD + gradient checkpointing 完成两步反向；loss `0.126→0.116`、grad norm `3.79→3.32`，稳定第二步 `2.03s`，无空 worker、分片错误、OOM 或死锁。首步约 30 秒为 worker 冷启动；产物位于 `/mnt/public/daibo/results/mem_short_k6_streamfix_smoke`；
- 实现 K=6 v3 memory-critical curriculum：pickup primitive 中 action horizon 含闭爪的完整历史窗口全保留，非 critical 确定性保留 20%；真实 256-sample stream 中 critical/full-history 为 `17.19%/88.28%`，首个 critical 在 episode 10 frame 1064；
- 实现共享 noise/time 的 `repeat_current + shuffle_past` 配对 ranking loss，并对 control loss stop-gradient。跨 rank critical 标记用 distributed MAX 对齐所有 FSDP forward 次数；强制 critical 的 4×A100 smoke 得到 repeat/shuffle delta `+0.0414/+0.00346`、aux loss `0.0088`、grad norm `4.28`，完整 backward 无 OOM/deadlock；
- v3 50-step early-gate pilot 完成，四卡 FSDP 与两个 checkpoint 均正常；重建 stream-fix 后的 24-sample 全 K=6 缓存并完成三条件评测。correct 明显优于 repeat-current，但不优于 shuffle-past，因此按 gate 停止当前 recipe；
- `nxb_4090` 的 root filesystem 曾因 OmniGibson 解密场景临时目录耗尽。仅删除 342 个超过一天、含 `*template.json.usd` 标记的可重建旧目录，释放约 30 GB；目标清单保存在 `/mnt/public/daibo/old_omnigibson_tmp_dirs_20260830.txt`，未删除数据集或实验产物；

### 2026-08-31

- RoboMME 对照研究确认为方案讨论，不把其 π0.5/RoboMME 成功率直接外推到 B1K；
- 不将 recurrent memory 升为当前主线；若未来投入，优先 RMT，TTT 置后；
- 将 GroundSG Oracle 记为 radio pickup 的优先候选诊断，但尚未生成 B1K grounded labels，也未训练 grounded πLL；
- 将 FrameSamp+Modulator 记为后续必须具备的强 perceptual-memory baseline，但尚未移植或训练；
- FrameSamp 与 Modulator 是两个独立变量，后续至少保留 Recent-K+Modulator 与 FrameSamp+Modulator 对照；
- TemporalFlow-VLA 的 controlled intervention 与本项目的 content/order gate 方向一致；将 physically supervised temporal queries 记为比单纯 masked-current 更有解释力的候选，但尚未审计 B1K teacher 可行性；
- temporal RoPE 只作为独立 time-encoding 消融；首次 FrameSamp+Modulator 移植应先保持官方 additive position 方案；
- v3.1 masked-current 不再被写成已经确定的下一主线。它仍可作为研究“如何制造 temporal learning signal”的受控 probe，但必须与真实非-Markov task、GroundSG 和 Modulator 假设分开；
- 当前没有因为本轮讨论启动新训练。用户先消化方案，下一次继续前应先确认优先实验。

## 14. 当前状态

- [x] 阅读并讨论 π-MEM 设计；
- [x] 确认 π0.5 论文中 πHL/πLL 共享同一套 SigLIP + PaliGemma/Gemma 权重；
- [x] 检查开源 π0.5 action-only 路径与 tied text decoder 的可恢复性；
- [x] 初步检查 B1K 的任务、skill、primitive 和 memory-related annotation；
- [x] 确定能力恢复 → short memory → long memory → multi-scale memory 的执行顺序；
- [x] 实现 R0 masked text loss、causal mask 和 greedy generation；
- [x] 实现 task-0000 的 12 样本 manifest、视频解码和 overfit harness；
- [x] 下载并校验 `lerobot/pi05_base` checkpoint，路径为 `/mnt/public/daibo/models/pi05_base_pytorch`；
- [x] 完成 R0-A：final loss `6.62e-9`，token/exact-match/EOS 均为 100%；
- [x] 完成 R0-B：final loss `1.15e-7`，token/exact-match/EOS 均为 100%，shuffled-image exact match 为 0%；
- [x] R0 实验产物已保存到 `/mnt/public/daibo/experiments/pi_mem/r0_fixed_label` 和 `/mnt/public/daibo/experiments/pi_mem/r0_visual_subtasks`；
- [x] 验证 10,000 个 annotation、10,000 个 parquet 以及三路各 10,000 个 RGB video 在结构上齐全；
- [x] 生成 R1 全量 manifest：104,720 train / 13,044 val / 13,118 test，667 个 canonical subtask，0 条 skip；
- [x] 完成 task-0000 R1 held-out pilot：val loss `5.1341 → 0.0959`，exact match `0% → 83.33%`；
- [x] 实现官方兼容的 23D current-state reader 和 train-only quantile normalization；
- [x] 实现 `image` / `state` / `image_state` 三种输入消融；
- [x] 实现 task-balanced sampler 与分 task/语义组件 generation metrics；
- [x] 实现 action-preservation evaluator，task-0000 三份 pilot tail 均通过 10% relative-RMSE gate；
- [x] 完成 task-0000 960/120 样本、3 seed 的 state 与 image+state 验证并冻结 500-step state-only 配方；
- [x] 完成 image/state 配对反事实，确认模型会使用 RGB，但当前 RGB 不提高 task-0000 clean accuracy；
- [x] 完成 7-task task-balanced pilot：macro exact match `86.43%`，32 样本 action relative RMSE `3.13%`；
- [x] 完成 progress-sensitive task 0/1/11/12 text pilot：macro exact match `50.00%`；
- [x] 完成 progress-sensitive tail 的 32 样本 action-preservation gate：relative RMSE `5.33%`；
- [x] 确认 OpenPI-Comet `pi05_b1kpt50_pt` 为 B1K 50-task、chunk-32 checkpoint，并核验本地 norm stats；
- [x] 实现 R2 `data.prompt_source: primitive`，使用 canonical primitive 作为 action SFT oracle prompt；
- [x] 实现 R2 `data.prompt_source: mixed`，在同一个 πLL checkpoint 中按确定性比例混合 full-task / primitive prompt；
- [x] 排除跨 primitive 边界的 32-step action chunk 和残余重叠歧义帧；
- [x] 完成 10,000 episode oracle interval 审计：86,351 intervals，0 个不可解析 episode；
- [x] 在 B1K checkpoint 上完成 7-task/progress-sensitive R1 迁移；exact match 分别为 `80.00%` / `56.25%`；
- [x] B1K 两份 tail 均通过 32 样本 action-preservation gate：relative RMSE `1.05%` / `2.26%`；
- [x] 确认 `/mnt/public/daibo/venv/behavior_openpi` 同时包含可用的 Behavior、OpenPI、CUDA 和当前 RLinf；
- [x] 完成 production oracle-primitive streaming 与 OpenPI transform/collate smoke test；
- [x] 完成真实 B1K batch 的 πLL flow-loss forward：mean `0.00507`，全部 finite；
- [x] 真实 task-0000 mixed streaming 验证：8 个 boundary-safe chunk 中 task/primitive prompt 各 4 个，action 均为 `[32,23]`；
- [x] 真实 2-sample mixed batch 通过 production SFT wrapper：B1Kpt50 scalar flow loss `0.03032`，全部 finite；
- [x] 本机 4×A100-80GB 完成 20-step FULL_SHARD mixed πLL benchmark：global batch 256、每 rank micro-batch 32；去除首步 DataLoader 冷启动后平均 `7.655 s/step`、中位数 `7.613 s/step`，采样峰值约 `66--67 GiB/GPU`；
- [x] 修复 `shuffle=false` 时 Behavior streaming 未初始化 rank/worker `_active_chunks` 的评测阻塞；
- [x] 实现 RLinf SFT checkpoint → OpenPI eval rollout 的自动路径解析和真实 full-weight 加载；
- [x] 实现 R2 三条件配对评测、固定 instance order、rollout/env seed 与逐 episode raw metrics；
- [x] 修正配对评测为官方 public-test IDs，并修复 tro_state 必须从 base template 0 初始化的 launcher 错误；
- [x] 实现 simulator predicate-driven Oracle-stage prompt controller；
- [x] 实现 short-memory v1 的 causal video encoder、时间编码、history drop 与历史 state tokens；
- [x] 接通 short-memory 的 BEHAVIOR env history、真实 SFT streaming/transform/collate 和 eval transform；
- [x] short-memory 与配对评测新增测试均通过；K=1、causality、mask、real-shape collate 和 paired-metrics 均有覆盖；
- [x] 旧 v1 真实 B1K short-memory batch 通过：`K=4, stride=16`；该配方现仅作历史诊断；
- [x] 修正 temporal PE/LN/residual 顺序，并增加对应公式级单元测试；
- [x] 将 state memory 改为全部 K 个连续 token，并关闭重复的离散 current-state prompt；
- [x] 将训练改为 `K=6, stride=30`，官方 30 Hz 评测改为 `K=6, decision stride=1`；
- [x] 完成 v2 的真实 B1K batch、完整模型前向和 4×A100 FSDP 反向 smoke；
- [x] R2 mixed πLL 已完成 4×A100 save/resume smoke；正式 seed=42 在 step 391 意外中断后已按相同配方重启，日志位于 `/mnt/public/daibo/results/mem_r2_task0000_mixed_seed42/train_restart.log`，每 100 steps 保存；重启后的 `global_step_100` checkpoint 已验证完整；
- [x] R2 mixed πLL 已完成 2,000 steps，并产出每 100 steps 的完整 checkpoint；
- [x] 完成 step-500 官方4实例配对闭环；发现 512-step horizon 使所有 episode 截断，该结果不再作为 Oracle gate 结论；
- [x] 使用 2,048-step horizon 重跑 step-2000 官方4实例三条件评测；确认 mixed 弱于 base，并发现 Oracle 初始 prompt 与训练 primitive 不一致；
- [x] 将 `move_to_radio` 与 `pickup_from_support` 都映射到训练标签 `pick up radio from coffee table`，并重跑 step-2000 Oracle-HL gate；有效 Oracle 没有改善 success 或 stage；
- [x] 完成 step 100/200/300/400/500 的 2,048-step mixed-task / corrected-Oracle sweep；step-100 是当前 Oracle 候选；
- [x] 完成 step-100 的 4,096-step task/Oracle horizon sensitivity check；两者 stage 相同且均无 success；
- [x] 完成 step-100 的 20 个 public-test instances、2,048-step task/Oracle 统计复核；stage 有显著方向性提升，但 success 仍为零；
- [x] 补齐原始 B1Kpt50 base 的 public-20、2,048-step 结果；base 保有唯一 success，当前 Oracle 仅恢复其平均 stage；
- [x] 实现 R2b conservative πLL recipe：expert-only、10× 低学习率、boundary-to-task fallback；
- [x] 完成 R2b 5-step save smoke、冻结权重审计和 100-step 正式训练；
- [x] 完成 R2b step 25/50/75/100 的 task/Oracle screening；四个 checkpoint 均未通过 Oracle gate，不扩大到 public20；
- [x] 完成 R2c vision-frozen 5-step smoke、100-step 正式训练、权重审计和四 checkpoint task/Oracle screening；R2c gate 失败；
- [x] 实现 short-memory 真实/重复当前/打乱过去三种历史条件及 memoryless 四条件配对 launcher；
- [x] 完成 short-memory step-500 标称四条件闭环；后续确认 history transport 丢失，条件比较已作废；
- [x] 审计 `skip_intermediate_obs_in_chunk` 与模拟频率：canonical turning-on-radio 协议最终固定为 full obs/render、2,048 action steps、30 Hz、physics 120 Hz；旧 60 Hz 结果降级为诊断；
- [x] 解耦完整视频帧与 short-memory 更新；每个 32-action chunk 仍只记录最后一个 policy history frame；
- [x] 将 short-memory 运行所需代码同步到 `bjd_dev_2` 独立存储，并在远端环境完成 11 项单测；
- [x] 完成 4×A800 short-memory global-batch-32/256 FSDP smoke；严格对照配方稳定 step `14.8s`、约 `66.0--66.5 GiB/GPU`；
- [x] 旧 v1 short-memory seed=42 SFT：`K=4`、stride 16；checkpoint 不再用于 v2 结论；
- [x] 启动 short-memory v2 seed=42 正式 SFT：`K=6`、stride 30、micro batch 4、global batch 256、2000 steps、每 100 steps 保存；
- [x] 实现并运行 short-memory demonstration causal gate；完成 step 100/200/500/800/1100/1400 的 16 样本 sweep，以及 step-100/1100 的 64 样本复核；
- [x] 完成 short-memory v2 seed=42 的 2,000-step 训练；final checkpoint 文件完整；
- [x] 完成 short-memory v2 step-1100 的旧 60 Hz/4,096-step 四条件闭环；产物完整，但现仅作诊断；
- [x] 实现 pickup-stage decision trace，并完成 step-1100 Oracle 的 30 Hz/2,048-step 复测；确认模型从未闭爪、从未抓起 radio；
- [x] 审计 task-0000 三类 primitive 的 gripper action 分布；pickup 的 boundary-safe chunk 只有 `9.955%` 含闭爪，而 press/place 约为 `99.975%/97.579%`；
- [x] 使用正确 30 Hz 协议重跑 step-1100 标称四条件矩阵；后续确认 history transport 丢失，不能作为 memory 因果矩阵；
- [x] 完成 final step-2000 的 64-sample offline gate；temporal sensitivity 弱于 step-1100，按预定规则不追加 final 闭环；
- [x] 实现并运行 demo grasp-event action-generation gate；step-1100/2000 均证明示范抓取状态下能闭合正确右手；
- [x] 为 grasp-event 样本增加可复用 torch cache、selection manifest、无歧义 open-control margin 和单元测试；
- [x] 修复 rollout schema 丢弃 `history_*` 的 transport bug，并为四路 tensor/mask 增加回归测试；
- [x] 实现 exact observation/model-action takeover 与 raw HDF5 scene/state mid-stage reset；4 条 expert takeover 仅 1 条形成 `in_hand`，不作为 policy gate；
- [x] 完成 step-1100/2000 的三条件 grasp-event action-fidelity；证明 history content 参与夹爪校准，但没有 temporal order sensitivity；
- [x] 修复 250-frame streaming chunk 边界清空 K=6 history 的训练数据问题；short-memory 按 episode 分片并通过真实 frame-250 审计；
- [x] 完成 stream-fix 的 4 rank × 2 DataLoader worker 两步 FSDP smoke；
- [x] 实现 v3 memory-critical event curriculum、确定性 noncritical subsampling 与完整 history gate；
- [x] 实现 correct/repeat-current/shuffle-past 共享 noise/time 的 stop-gradient paired ranking loss；
- [x] 完成 v3 真实 stream 审计、53 项相关回归和强制 critical 四卡 FSDP backward smoke；
- [x] 完成 v3 50-step early-gate pilot，并在 stream-fix 后的 24-sample 全 K=6 exact-demo action-fidelity 上筛选 checkpoint；结论为 content gate 通过、order gate 失败，不扩训当前 recipe；
- [x] 完成 RoboMME symbolic/perceptual/recurrent memory 对照研究，并记录 GroundSG、FrameSamp+Modulator、temporal RoPE 和 recurrent 候选边界；
- [ ] 审计 B1K GroundSG Oracle 的 object/affordance grounding 标注可行性；
- [ ] 实现并测试 Recent-K+Modulator 与 FrameSamp+Modulator 独立分支；
- [ ] 审计 B1K robot-surface temporal-flow teacher 和 chunk-aligned temporal queries 的可行性；
- [ ] 在官方风格 FrameSamp baseline 后实现 temporal-RoPE 消融；
- [ ] 完成 B1K 数据下载和最终完整性报告；
- [ ] 将 πHL text forward/loss/generation 接入正式 SFT worker；
- [x] 实现 short-memory video encoder；
- [ ] 构造 long-memory training targets；
- [x] 开始 R2 mixed πLL 与 short-memory seed=42 正式实验。

## 15. 下一步

当前没有正在运行的新训练。用户希望先消化已有实验和方案；恢复项目时应先确认优先项，不得自动启动下面任何候选实验。

当前事实是：mid-stage exact takeover 的 expert action 也只有 `1/4` 重建轨迹形成 `in_hand`，所以它只证明输入/action 接口一致，不能单独把闭环失败归因于 policy；v3 early gate 证明 correct history 优于 repeat-current，但没有证明 correct 优于 shuffle-past。现有 v2/v3 通过的是实现、反向计算和 history-content gate，不是 temporal-order 或 full-task memory acceptance。

建议的恢复顺序如下：

1. **先做 GroundSG Oracle 设计审计，不立即训练。** 确认 B1K 中 object instance mask、camera projection 和 affordance link 能否稳定生成当前视角 grounded target；冻结 SimpleSG/GroundSG 的 prompt schema、坐标系和配对评测。
2. **若标注审计通过，做小规模 grounded πLL pilot。** 从同一个 B1Kpt50/πLL base 比较 full-task、SimpleSG Oracle 和 GroundSG Oracle；先看 approach distance、gripper event 和 pickup stage，再决定是否扩大闭环 episode。
3. **随后移植 action-expert Modulator。** 先用 Recent-K+Modulator 与当前 K=6 vision-temporal branch 对比，以隔离 integration effect；再实现官方风格 FrameSamp+Modulator，以隔离 whole-episode sampling effect。
4. **审计 TemporalFlow-style teacher。** 核对 B1K robot geometry、双臂 link、camera calibration 和遮挡可见性，判断能否为 `t-31/t-16/t` 生成稳定 robot-surface flow；审计通过后再决定是否实现两级 temporal queries。
5. **temporal RoPE 置于官方 FrameSamp baseline 之后。** 使用真实时间作用于 Q/K，并与 additive time encoding 单独比较；shuffle 必须固定 timestamp、交换 content。
6. **v3.1 masked-current 保留为受控 probe，而非默认主线。** 只有在选定 task/样本确实存在当前观测歧义、且不会把训练目标变成人工遮挡恢复任务时才运行。
7. **RMT/TTT 暂缓。** 只有 GroundSG/Modulator/perceptual 分支仍无法满足固定 token budget 或长 episode 需求，且已建立真正 history-dependent 的数据与评测后，才做 RMT pilot；TTT 继续后置。
8. **最终仍采用同源 2×2 验收。** 在有非零低层 full-task 能力后，比较 `baseline / short / long / short+long`，并保留 current-only、repeat、content-shuffle、wrong-memory 等因果控制。

现有 state-only R1 只证明 current-state 中存在强阶段信号。黑图下 held-out episode 的高准确率同时兼容 state shortcut 和小样本记忆，不能单独判定过拟合；必须结合跨 episode、错 state、错图和闭环边界预测。`turning_on_radio` 是否足以验收 memory 仍是未决问题，不能因为已有大量结果就默认继续只使用该任务。

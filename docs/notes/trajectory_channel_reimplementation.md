# Trajectory Channel 重构实施基线

本文是 `/mnt/public/daibo/timeline/0703/RLinf` 中 trajectory channel 重构的唯一实施入口。每次只实现并验收一个 subgoal。新对话必须先阅读本文、检查工作区，再继续当前 subgoal。

> 状态日期：2026-07-20
>
> 当前阶段：SG-16 路径互斥、回归与实现收口已完成，等待人工验收
>
> 文档性质：内部设计、实施与交接记录，不接入 Sphinx toctree
>
> 语义参考配置：目标仓库 `libero_spatial_ppo_openpi`
>
> 性能数据规模：参考仓库 `libero_spatial_ppo_openpi_trajectory_channel`

## 1. 执行规则

1. 只在 `/mnt/public/daibo/timeline/0703/RLinf` 中工作。
2. 开始前运行 `git status --short`，保留所有已有修改和未跟踪文件。
3. 每次只实现“当前阶段”标记的一个 subgoal，不顺带实现后续阶段。
4. 开始实现前，先把该 subgoal 的设计决定、修改范围和验收命令写入本文。
5. 完成后记录代码变更、测试结果、未决问题和提交前 diff 摘要，然后停止并等待人工验收。
6. 只有人工确认通过后，才把下一 subgoal 标成当前阶段。
7. 正确性优先于性能。raw fixed-frame 路径没有通过前，不接入压缩。
8. benchmark 必须保存配置、命令、环境、原始样本和汇总结果。`/tmp` 中未归档的数据不能成为正式基线。

## 2. 信息来源与适用范围

本设计整合了以下参考资料，但以目标仓库的实际代码和本文后续决策日志为准：

- `/mnt/public/daibo/RLinf/docs/notes/trajectory_channel_architecture.md`
- `/mnt/public/daibo/RLinf/docs/notes/lossless_compression_handoff.md`
- `/mnt/public/daibo/timeline/0703/RLinf/rlinf/data/embodied_io_struct.py`
- `/mnt/public/daibo/timeline/0703/RLinf/rlinf/workers/trajectory/`

前两份文档来自另一个工作目录，其中提到的“当前实现”不能直接视为本目标仓库已经具备的能力。移植任何代码前都要重新核对依赖、调用链和测试。

本轮只重构 embodied trajectory channel。reasoning、agent、replay buffer 和无关 scheduler 行为不在范围内，除非某个已验收 subgoal 明确扩大范围。

### 2.1 OpenPI 验证环境

SG-02 及后续 OpenPI 集成验证使用以下本地环境：

| 项目 | 已验证值 |
|---|---|
| Python environment | `/opt/venv/openpi` |
| Python | 3.11 |
| PyTorch | `2.6.0+cu124` |
| OpenPI package | `/opt/venv/openpi/lib/python3.11/site-packages/openpi` |
| Checkpoint | `/mnt/public/daibo/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT` |
| Checkpoint weights | `model.safetensors`，约 6.6 GiB |
| Norm stats | `physical-intelligence/libero/norm_stats.json` |
| Verified accelerator | NVIDIA A100-SXM4-80GB |

用户消息中的 checkpoint 名含有 `R Linf` 空格；文件系统中实际存在并完成验证的目录名是 `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`，本文统一使用实际路径。

## 3. 当前仓库基线

截至 SG-00 审计完成时，目标目录没有可执行的 trajectory-channel 实现：

| 文件 | 当前状态 |
|---|---|
| `rlinf/data/embodied_io_struct.py` | 保留旧 `EnvOutput`、`RolloutResult`、`ChunkStepResult`、`Trajectory` 和 `EmbodiedRolloutResult`；没有新的 trajectory-channel 协议 |
| `rlinf/workers/trajectory/` | 不存在 |
| trajectory 专用单测 | 尚未建立 |

用户已确认主动删除了此前未跟踪的 `rlinf/workers/trajectory/` 骨架和 `embodied_io_struct.py` 尾部协议草案，因为旧设计不适合作为重构基础。这是预期的 clean-slate 状态。后续实现不恢复或兼容这些草案，只复用目标仓库中经过审计的通用 Worker、Channel 和模型能力。

因此 SG-01 将从协议定义开始，而不是在现有 trajectory 实现上修补。已冻结的兼容性要求：

- Python 3.10 不使用 `enum.StrEnum`，字符串枚举继承 `str, Enum`。
- Python 3.10 的 self type 使用 `typing_extensions.Self`，不从 `typing` 导入 `Self`。
- 新协议与旧 `Trajectory`、`EmbodiedRolloutResult` 在迁移期并存，但新 record 不继承旧聚合容器。

## 4. SG-00 调用链审计

### 4.1 当前同步 PPO 时序

目标仓库的当前同步训练使用三个普通 `Channel`。Runner 在每个 `global_step` 同时启动 Env interaction 和 Rollout generation，等待 Actor 收齐 Env 组装的 `Trajectory`，随后计算 advantage 并训练：

```text
Runner sets global_step on Actor and Rollout
    -> Env.interact(env_channel, rollout_channel, reward_channel, actor_channel)
    -> Rollout.generate(rollout_channel, env_channel)
    -> optional Reward.compute_rewards(reward_channel, env_channel)
    -> Actor.recv_rollout_trajectories(actor_channel)
    -> Actor.compute_advantages_and_returns()
    -> Actor.run_training()
```

证据：

| 行为 | 代码证据 |
|---|---|
| Runner 创建 Env/Rollout/Actor/Reward 普通 Channel | `rlinf/runners/embodied_runner.py:93-100` |
| `global_step` 在一次 generation 前设置到 Actor/Rollout | `rlinf/runners/embodied_runner.py:482-510` |
| Env 负责把 trajectory 发到 Actor | `rlinf/runners/embodied_runner.py:502-520`、`rlinf/workers/env/env_worker.py:1223-1232` |
| Actor 收齐后才计算 advantage | `rlinf/runners/embodied_runner.py:525-532` |

这证明当前 Env 同时是环境执行器和 trajectory assembler。新设计必须移走后一个职责。

### 4.2 当前 step 数据流

每个 rollout epoch 先发送初始 observation，然后执行 48 个 action-chunk transitions。每个普通 step 的真实顺序是：

```text
Env sends obs + optional final_obs
    -> Rollout full policy forward
    -> RolloutResult(actions, logprobs, values, forward_inputs, versions,
                     optional bootstrap_values)
    -> Env receives the whole RolloutResult
    -> Env combines reward and optional bootstrap correction
    -> Env appends rollout statistics and env transition fields
    -> Env executes actions
    -> Env sends next obs
```

证据：

| 行为 | 代码证据 |
|---|---|
| Env live request 当前包含 `obs` 和 `final_obs` | `rlinf/workers/env/env_worker.py:921-929` |
| Rollout 同一次返回 action、statistics 和 Actor `forward_inputs` | `rlinf/workers/rollout/hf/huggingface_worker.py:575-602` |
| Env 接收后组装 `ChunkStepResult` | `rlinf/workers/env/env_worker.py:1059-1092` |
| Env 之后才执行 action 并发送下一 observation | `rlinf/workers/env/env_worker.py:1114-1132` |
| OpenPI `forward_inputs` 包含 chains、denoise indices、token、action/model_action 和 observation leaves | `rlinf/models/embodiment/openpi/openpi_action_model.py:889-920` |

结论：`forward_inputs`、`prev_logprobs`、`prev_values` 和 versions 是 Actor record，不是 Rollout→Env live response。它们必须由 Rollout 直接提交给 StorageWorker。

### 4.3 当前 boundary value 与 bootstrap

当前配置 `auto_reset=false`。每个 rollout epoch 的 48 个 transitions 结束后，Env/Rollout 额外进行一次完整 policy forward；返回 action 不执行，只把末状态 value 和末状态 done flags append 到 trajectory。因此当前 shape 是：

```text
actions/rewards/logprobs/forward_inputs: T
values/dones/terminations/truncations:    T + 1
```

证据：

| 行为 | 代码证据 |
|---|---|
| 48 来自 `240 / num_action_chunks(5)` | `rlinf/workers/env/env_worker.py:146-151` 和配置 |
| Rollout 在每个 epoch 的普通循环后额外接收/forward 一次 | `rlinf/workers/rollout/hf/huggingface_worker.py:668-744` |
| Env 对最后结果只 append value、done 和 reward，不执行 action | `rlinf/workers/env/env_worker.py:1152-1209` |
| 聚合容器明确要求 value/done 为 trajectory length + rollout epoch | `rlinf/data/embodied_io_struct.py:533-554` |
| GAE 当前读取 `values[step + 1]` 和 `dones[step + 1]` | `rlinf/algorithms/advantages.py:66-79` |

auto-reset 的另一条路径把 `final_obs` 传给 Rollout，调用现有 full policy inference 取得 `prev_values`，再由 Env 把 `gamma * final_value` 加到 reward。非-auto-reset truncation 当前没有对应 reward correction：

| 行为 | 代码证据 |
|---|---|
| Rollout 使用现有模型的 value/q head 并通过 `_predict_rollout_actions(final_obs)` 取得 value | `rlinf/workers/rollout/hf/huggingface_worker.py:604-619` |
| Env 对 auto-reset truncation 修改 reward | `rlinf/workers/env/env_worker.py:685-725` |
| 当前 GAE 明确声明不支持 auto-reset | `rlinf/algorithms/advantages.py:36-39` |

目标设计不保留这两种隐式表达：

- transition 的 reward/done/termination/truncation 都是 `[T, B, ...]`；
- `state_values[t] = V(s_t)` 与 transition 对齐，也是 `[T, B, ...]`；
- epoch 内下一次 `PolicyInput` 触发的 inference 所产生的 state value 可作为前一 transition 的 continuation value；
- segment 末仍存活 slot 通过显式 `SegmentValueRequest(s_T)` 获得 `next_values`；
- auto-reset truncation 通过显式 `TimeoutValueRequest(final_obs_t)` 获得 `timeout_values[t]`；
- 非-auto-reset truncation 使用 step 返回的 terminal `next_obs` 发起同一种 `TimeoutValueRequest`；
- true termination 不请求 boundary value；
- 不再为获得 `V(s_T)` 做一个返回无用 action/forward_inputs 的完整 live policy response；
- Env 和 Storage 都不修改 reward，Actor advantage 只应用一次 bootstrap。

### 4.4 Reward 调用链

当前 external RewardWorker 从 Env 接收 observation/history，结果返回 Env；Env 再组合 env reward 和 reward-model output，history mode 还会回写此前保存的 rewards：

| 行为 | 代码证据 |
|---|---|
| Env 选择 final/current observation、env infos 和 history input | `rlinf/workers/env/env_worker.py:745-808` |
| RewardWorker 把结果发回 Env | `rlinf/workers/reward/reward_worker.py:357-392` |
| Env 组合 reward 权重 | `rlinf/workers/env/env_worker.py:696-703` |
| history assignment 修改此前 rollout rewards | `rlinf/workers/env/env_worker.py:842-863` |

目标决策：external reward inference 不属于下一 action 的 live critical path。Env 仍负责构造 reward inference 输入，因为它拥有 observation/history；RewardWorker 直接向 StorageWorker 提交 `RewardResult`。Storage 保存并对齐独立的 `env_rewards` 与 reward-source results，不原地组合。一个纯 `RewardComposer` 在 Actor advantage 准备阶段按配置生成 effective rewards，且每个 generation 只执行一次。环境控制逻辑若未来真的消费 reward-model output，应建立独立明确路径，不能复用 trajectory result 的返回链路。

### 4.5 OpenPI value API

OpenPI 在 `add_value_head=true` 时创建实际 `ValueHead`，policy inference 返回 `prev_values`；Actor 重新 forward 时通过 `compute_values` 得到 values：

- value-head 初始化：`rlinf/models/embodiment/openpi/openpi_action_model.py:165-190`；
- inference result：`rlinf/models/embodiment/openpi/openpi_action_model.py:877-921`；
- Actor forward values：`rlinf/models/embodiment/openpi/openpi_action_model.py:708-755`。

因此新 `ValueRequest` 的 rollout-side handler 应复用模型现有 observation preprocessing 和 `value_head` 计算路径。设计和代码中不得引入没有实现依据的 `predict_values()`。SG-02/SG-10 决定是否抽取一个正式的 `compute_state_values()` wrapper；它只是现有 `value_head` 路径的封装，不是新模型语义。

### 4.6 数据规模

SG-00 成功解析了目标仓库 `libero_spatial_ppo_openpi` 配置。它用于验证行为语义：

```text
train envs:               64
rollout epochs:            8
env steps per epoch:     240
actions per policy call:   5
transitions per epoch:    48
transitions per env:     384
camera size:         256 x 256
action dim:                 7
OpenPI denoise steps:       4
value head:              true
auto reset:             false
external reward:        false
```

用户要求的性能 profile 采用参考仓库 `libero_spatial_ppo_openpi_trajectory_channel` 的规模：32 train envs、4 trajectory/storage shards，因此每个 shard `B=8`，每个 generation 有 `8 * 48 = 384` 个 transitions。main/wrist images 均按 `[B, 256, 256, 3] uint8` 计入。历史文档估计每 shard 完整 records 约 3.512 GiB，但原始结果没有归档；SG-15 必须重跑后才能把该数字作为正式 baseline。

这里的 `chunk_step`/`t` 是一个 5-action macro transition，不是 primitive env step。LIBERO `chunk_step()` 在 chunk 内禁止 auto-reset，收集 5 个 rewards/masks，最后才执行一次 batch auto-reset；auto-reset/ignore-termination 模式还把 chunk 内累计的 termination/truncation 标到最后一个 action 位置。证据见 `rlinf/envs/libero/libero_env.py:749-813`。当前 `reward_type=chunk_level` 会把 5 个 rewards 求和并把 done 归约为一个 macro mask，见 `rlinf/algorithms/utils.py:79-119`。SG-05 应保持该 macro-transition 语义，除非另立 subgoal 改成 primitive-step critic。

目标仓库目前不存在名为 `libero_spatial_ppo_openpi_channel` 或 `libero_spatial_ppo_openpi_trajectory_channel` 的配置。SG-12 负责在目标仓库建立最终配置；在此之前不得把参考仓库配置当成目标仓库文件。

### 4.7 冻结的逻辑坐标与 Identity 语义

SG-00 只冻结逻辑含义，不冻结承载它们的 class。每条数据需要能够表达以下坐标：

```text
generation_id
rollout_epoch
chunk_step
```

- `generation_id` 是 Runner 的 `global_step`，由 control plane 在 generation 开始前明确下发；它不是模型权重 version。
- `rollout_epoch` 是一个 generation 内的索引，范围为 `[0, rollout_epoch_count)`。
- 对 transition records，`chunk_step` 是产生 `s_t -> s_{t+1}` 的 transition index，范围为 `[0, T)`。
- `SegmentValueResponse` 使用 `chunk_step=T` 表示 segment-tail state `s_T`。
- timeout value 使用发生 truncation 的 transition key `chunk_step=t`，不使用 reset state 的 key。
- `stage_id` 不属于逻辑时间坐标。当前代码中的 stage 是 `rollout.pipeline_stage_num` 对应的 source batch partition；稳定 `slot_ids` 已表达数据 ownership。source stage 只属于 source/route metadata。

一条逻辑写入必须由以下信息唯一确定：

```text
(generation_id, rollout_epoch, chunk_step,
 data role, source, slot_ids)
```

`data role` 区分环境结果、rollout 训练数据、reward 和两类 boundary value。`source` 包含 component、rank 和可选 source stage，用于 ownership 验证和 retry。`slot_ids` 是有序稳定全局 slots。wire sequence 和 schema identity 不属于 logical identity。

SG-01 可以把这些字段直接放在现有业务对象里，也可以提取一个小 metadata dataclass；只有出现真实的复用和 validation 需求时才新增 class。SG-00 中的 request/response/record 名称只是角色标签；SG-01 最终使用 `PolicyInput`、`PolicyOutput`、`EnvResult` 等业务名称。

### 4.8 冻结的 value-request 拓扑

Env 触发 boundary value request，Rollout 计算，response 直接进入 StorageWorker：

```text
any truncation:
    Env -> TimeoutValueRequest(truncated terminal obs only) -> Rollout
    Rollout -> ValueResult(kind="timeout") -> Storage
    if auto-reset: Env concurrently continues PolicyInput(reset obs)

segment end:
    Env -> SegmentValueRequest(alive next_obs only) -> Rollout
    Rollout -> ValueResult(kind="tail") -> Storage

true termination:
    no value request
```

选择 Env 作为触发者，因为 Env 是 `final_obs`、done masks 和 next observation 的原始所有者，并且已经拥有 Env→Rollout 路由。Storage 不应为了发起 inference 而理解 Rollout placement；Rollout response 直接写 Storage，Env 不需要 value result。value endpoint 使用独立有界队列，mid-segment timeout request 不阻塞下一次 policy request；segment-tail value 只阻塞 trajectory readiness，不阻塞当前 generation 的环境执行。

## 5. 目标架构

目标架构按数据是否阻塞下一步环境交互拆分，不按现有类名机械拆分。

### 5.1 Live Critical Path

```text
EnvWorker
    -> PolicyInput
    -> TrajectoryChannelWorker
    -> RolloutWorker
        +-> PolicyOutput -> TrajectoryChannelWorker -> EnvWorker
        +-> RolloutResult --------------------------> TrajectoryStorageWorker
```

约束：

- 只传下一步 policy inference 和环境执行立即需要的数据。
- `PolicyInput` 只包含 observation、rollout 坐标以及确实改变本次动作生成的条件字段；普通请求不包含 `final_obs`。
- `PolicyOutput` 只包含 Env 执行下一步所需的 actions，不携带训练副产物。
- Rollout 的同一次 forward 生成 `PolicyOutput` 和旁路的 `RolloutResult`；两者共享逻辑坐标，但发往不同消费者。
- `forward_inputs`、`prev_logprobs`、`prev_values`、versions 和 trajectory-only masks 不得为组装 trajectory 而绕经 Env。
- 普通 policy request 不携带 `final_obs`。
- `TrajectoryChannelWorker` 不执行 storage assembly、GAE、reward correction 或 trajectory compression。

### 5.2 Record/Bypass Path

```text
EnvWorker ---- EnvResult -------+
RolloutWorker - RolloutResult --+--> TrajectoryStorageWorker
RewardWorker -- RewardResult ---+
```

约束：

- record 不经过 `TrajectoryChannelWorker`。
- producer 通过固定 frame 直接提交给其 route 对应的 StorageWorker。
- `TrajectoryStorageWorker` 是独立进程，不是 ChannelWorker event loop 中的 coroutine 或 thread。
- record queue、receive buffer、completed trajectory queue 和 compression queue 都必须有界。
- Storage 的吞吐不足必须产生明确 backpressure，不能通过无界排队隐藏。

Reward inference 使用辅助非关键路径：

```text
EnvWorker -> RewardRequest -> RewardWorker -> RewardResult -> StorageWorker
```

RewardRequest 的有序提交不能阻塞下一步 policy/action。只有未来某个环境明确使用 reward-model output 做控制决策时，才为它建立单独的 live response；当前 PPO trajectory assembly 不需要 Reward→Env 返回。

### 5.3 Value/Bootstrap Path

```text
timeout final_obs --------------------+
alive segment-tail next_obs ----------+--> Rollout value_head
                                           -> aligned value response
                                           -> TrajectoryStorageWorker
```

Env 按 slot mask 触发请求，Rollout 计算，response 直接提交给 StorageWorker：

- 模型通过实际 `value_head` 路径计算 value，不引入虚构的 `predict_values()` API。
- 任意 standard-bootstrap truncation 使用 terminal observation；auto-reset 时必须使用 `final_obs`，不能使用 reset 后 observation。
- segment 末仍存活 slot 使用 `next_obs` 的 value。
- value response 不返回 Env 修改 reward。
- 新协议不定义含糊的 Env/Rollout bootstrap 聚合消息。末 transition 仍由环境侧数据表达；timeout value 和 alive-tail value 使用两个明确语义，具体 class 名在 SG-01 决定。

### 5.4 Actor Output Path

```text
TrajectoryStorageWorker
    -> validate readiness
    -> assemble actor shard
    -> optional lossless compression
    -> Actor receive/decompress/merge
    -> calculate advantages and returns
    -> update model
```

Storage 只保存和对齐 raw reward、value 与 mask。advantage calculator 是 bootstrap 数学的唯一所有者。

## 6. Bootstrap 与 Reward 不变量

这些规则已经由 SG-00 确认，成为不可被后续性能优化改变的正确性约束：

```text
state_values[t]   = V(s_t)
timeout_values[t] = V(terminal_obs_t)  # 仅 truncation slot 有效
next_values       = V(s_T)          # segment 末仍存活 slot
```

Storage 必须保留 raw reward，不在 Env 或 Storage 中加入 bootstrap reward。Actor 的 advantage 计算只应用一次 boundary value：

```text
continuation_t = (~done_t) * V(s_{t+1})
timeout_t      = truncation_t * V(final_obs_t)
delta_t        = raw_reward_t + gamma * (continuation_t + timeout_t) - V(s_t)
gae_t          = delta_t + gamma * lambda * (~done_t) * gae_{t+1}
```

必须用测试覆盖：

- true termination；
- auto-reset truncation；
- alive segment cutoff；
- 同一 batch 混合三种边界；
- rollout 中途发生 timeout；
- reset observation 不被写入上一 episode 的 timeout target；
- external reward 与 env reward 组合后仍只应用一次 bootstrap。

SG-00 已确认当前实现使用偏移一位的 `T+1 dones`，而目标协议使用与 transition 对齐的 `[T]` masks。SG-05 必须同时修改预处理和 GAE 调用方，不能只替换公式。

## 7. 数据协议原则

### 7.1 身份分层

协议应区分四种身份：

| 身份 | 用途 |
|---|---|
| logical step key | 定位 rollout/segment 中的逻辑时刻 |
| record kind/source | 区分同一时刻来自 Env、Rollout、Reward 或 Value 的记录 |
| slot coverage | 标识 record 覆盖的稳定全局 env slots |
| endpoint sequence | 标识 wire endpoint 上的顺序、retry 和 buffer ownership |

逻辑时间坐标不同时承担所有四种职责。字段语义已在 SG-00 冻结；SG-01 决定是否需要独立 metadata 类型以及 validation 放在哪里。

同一逻辑 record 的重试必须幂等：相同 identity 和相同内容返回既有 ack；identity 相同但内容冲突必须失败。

### 7.2 Required 与 Optional

optional 只能由明确配置或运行场景决定，不能表示调用方忘记填字段。例如：

```text
requires_values                -> state/timeout/tail value record
requires_external_reward       -> RewardResult
intervention enabled           -> intervention fields
RLT enabled                    -> RLT fields
reward_mode == history_buffer  -> history metadata
truncation present             -> terminal observation and timeout value coverage
```

固定 endpoint 的 steady-state tensor layout 不能随意增删 leaf。不同条件组合应映射为不同 schema，或使用固定 tensor 与 validity mask。

### 7.3 `ForwardInputs`

`ForwardInputs` 是 Actor 重新 forward 所需模型输入的 typed schema，不是任意 `dict[str, Tensor]`。每个模型实现至少提供：

```python
validate()
tensor_fields()
select(indices)
to_model_kwargs()
```

需要 schema name/version、batch-axis 约束、required/optional leaves 和稳定 field order。SG-02 先覆盖性能参考配置实际使用的 OpenPI 输入，再扩展其他模型。

### 7.4 命名与抽象原则

命名应贴近 RLinf 已有业务语言，不把设计文档里的概念图直接翻译成一组新 class：

- 优先沿用 `EnvOutput`、`RolloutResult`、`Trajectory`、`observations`、`actions`、`rewards`、`values` 等现有术语。
- 不因为数据经过一条通信边，就机械新增 `*Frame`、`*Envelope`、`*Message`、`*Payload` class。
- 不为三个整数机械新增 `StepKey`；先看它们是否可以自然地放进已有 metadata 或方法参数。
- 只有一个类型拥有独立 validation、行为或被多个组件稳定复用时，才提取 dataclass。
- transport frame 是底层 wire 实现术语，不进入 Env/Rollout/Actor 的业务 API，也不决定业务数据 class 名。
- class 名描述业务含义，不描述当前 transport、queue 或 worker 拓扑，以便后续替换 transport 时不改业务协议。
- SG-01 提交具体类型名前，先列出最小对象集合及未抽 class 的备选方案，由人工验收命名。

目标不是追求最少 class，而是让每个 class 都有无法由普通字段或现有类型清楚表达的职责。

## 8. 组件职责与接口方向

### 8.1 `TrajectoryChannel`

调用侧 façade，只封装 live publish/receive 和路由细节。不应以一个含糊的 `put()` 同时触发 live relay 与 trajectory record。

### 8.2 `TrajectoryChannelWorker`

继承 `ChannelWorker`，只管理 live queue、live frame relay、backpressure、health 和 live-path metrics。

### 8.3 `TrajectoryStorageWorker`

继承基础 `Worker`。建议接口按 control plane 和 data plane 分离：

```python
class TrajectoryStorageWorker(Worker):
    def configure(self, config: StorageWorkerConfig) -> None: ...
    def ready(self) -> bool: ...

    async def submit(
        self,
        src_addr: WorkerAddress,
        schema_id: int,
    ) -> SubmitAck: ...

    async def get(
        self,
        dst_addr: WorkerAddress,
        query_id: int,
        actor_rank: int,
    ) -> None: ...

    async def drain(self) -> None: ...
    def shutdown(self) -> None: ...
```

`submit()` 本身就是 fixed-frame receive API，所以生产接口不需要再分为 `submit_frame()` 和 `submit_via_ray()`。`src_addr` 用于从 sender 接收 data-plane frame；`schema_id` 选择初始化阶段注册的 wire schema，它不是 trajectory key。

`SubmitAck` 至少区分：

| 状态 | 保证 |
|---|---|
| `RECEIVED` | frame 已进入 StorageWorker 所有的 buffer，sender 可以复用发送 buffer |
| `INGESTED` | record 已写入 `TrajectoryStorage` |

SG-09 验收后的性能审计确认 producer 默认等待 `INGESTED` 会把 Storage assembly 延迟反向传入 producer，因此已修正为：`submit()` 在 frame 进入 Storage-owned buffers 后默认返回 `RECEIVED`，record 进入有界后台 ingestion queue；测试、drain 和恢复可在同一 API 请求 `INGESTED`。SG-11 选择 Actor 主动 pull 且不建立 output queue，因此 submit ack 不定义含糊的 `COMPLETED`；Actor 成功返回 `TrajectoryReader.pull()` 才表示该 Actor shard 已接收并完成 slot-order merge。

### 8.4 `TrajectoryStorage`

纯 CPU 聚合对象，不依赖 Worker、Ray、Gloo、RPC、codec 或 endpoint。负责：

- 将 typed record 映射到 time/slot；
- 校验 ownership、shape、dtype 和 batch axis；
- 支持合法乱序到达与幂等写入；
- 维护分字段 readiness；
- 保持 raw reward/value/mask 的对齐；
- 导出 Actor 所需 shard。

### 8.5 `RoutePlan`

建立稳定映射：

```text
component local batch
    -> global slot_ids
    -> storage worker shard
    -> storage local slot indices
    -> actor rank and actor-local order
```

每个 global slot 在每个 ownership 维度上必须不重不漏。route 结果不能依赖 record 到达顺序。

## 9. Transport 与 Compression

### 9.1 Transport v1

初始化阶段注册 endpoint schema，steady state 发送固定 header 和 tensor lanes。header 至少需要：

```text
protocol_version
schema_id
sequence_id
logical record identity
slot coverage or route id
payload sizes/flags
```

shape、dtype、field order 和最大容量属于 schema，不应每帧重复发送。receiver 当前按 frame 分配独立 buffers；未来只有在 Storage 释放对应 generation 的只读 ownership 后，才可通过有界 lease pool 复用。sender 在 `RECEIVED` 前不得覆盖 in-flight buffer。

第一版先实现 raw fixed-frame。generic Python object/pickle 只能用于测试或 control plane，不能成为生产 steady-state data plane。

### 9.2 Compression

压缩必须无损，并在 SG-14 才接入。第一候选是 LZ4-fast，第二候选是 Zstd level 1；最终选择由真实数据端到端实验决定。

```text
T_raw        = raw_bytes / bandwidth
T_compressed = encode + compressed_bytes / bandwidth + decode
```

只有 `T_compressed < T_raw` 才启用压缩。每个 field/block 支持 raw fallback：压缩结果不小于原始数据时发送 raw。

实现要求：

- codec 直接读写 contiguous CPU tensor buffer；
- 不经过 Tensor → NumPy → bytes → Tensor；
- receiver 尽量直接解压到预分配的最终或 pinned destination；
- 每个并发 lane 使用独立 codec context；
- 从 2 个 image-field lanes 开始实验，不假设 CPU core 越多单次 codec 就越快；
- buffer pool 有固定上限并遵守 send/decode/H2D completion ownership；
- temporal XOR、keyframe 和 resync 不属于第一版。

## 10. Subgoal 与验收门槛

### SG-00：冻结架构和语义

交付：

- 审计真实 Env/Rollout/Reward/Actor/advantage 调用链；
- 列出旧路径每处 reward/bootstrap 修改；
- 冻结三条输入路径和 Actor output path；
- 决定 value request 的触发者与拓扑；
- 冻结逻辑时间坐标各字段的语义和 Python 3.10 支持策略，不预设 class 名；
- 解析目标仓库普通配置，并对照参考仓库 trajectory-channel 性能规模；
- 把未决问题变成明确的 decision 或后续 subgoal，不留隐含假设。

验收：本文件的设计决定区不再含影响 SG-01 的未决语义；提供调用链证据和文件/行号；不改变运行时代码。

### SG-01：定义数据协议 v1

交付：typed records、metadata、value request/response、validation 和 Python 3.10 兼容实现。

验收：required/optional、shape、dtype、slot coverage 和 identity 单测通过；非法输入明确失败。

本阶段采用以下最小对象集合：

| 类型 | 单独存在的理由 |
|---|---|
| `TrajectoryData` | live 与 bypass 数据重复使用 rollout 坐标、`slot_ids` 和基础 validation；它不是 wire key 或 frame |
| `PolicyInput` | Env→Rollout 的 live policy 输入，只包含当前 observation、`rlt_switch_flags` 和 `intervene_requested` |
| `PolicyOutput` | Rollout→Env 的 live 输出，只包含待执行 actions |
| `EnvResult` | 绕过 live channel，保存一个 action-chunk transition 的 raw env 结果 |
| `RolloutResult` | 绕过 Env，保存同一 transition 的 policy/Actor 训练输入 |
| `RewardResult` | reward 可能乱序返回，并具有 per-step、terminal、history 三种对齐语义 |
| `ValueRequest` | Env→Rollout inference 输入，不写入 Storage |
| `ValueResult` | Rollout→Storage 的 timeout 或 alive-tail value |

不新增 observation、metadata、key、frame、envelope 或 response wrapper class。observations 和 OpenPI `forward_inputs` 暂时保持已有 nested dict 表达；`ForwardInputs` 的模型封装属于 SG-02。`global_step`、`rollout_epoch`、`chunk_step` 和 `slot_ids` 直接放在 `TrajectoryData`，避免只包装几个整数的 key class。

这些名称沿用此前讨论和 RLinf 已有 `Trajectory`/`EnvOutput` 语言。它们是当前最小实现，不是 wire protocol 名；SG-01 验收时仍可根据实际可读性做一次统一重命名。

字段分为两层校验。数据类型立即校验坐标、slot coverage、tensor dtype/shape 和字段内部条件；算法/配置决定的 required fields 由 SG-04 的 storage config 在注册 schema 时校验。这样不会把所有算法硬编码进基础 dataclass，也不会把“调用方漏填”静默解释成 optional。

| 字段 | 基础类型 | 何时 required |
|---|---|---|
| observations + action-generation controls | `PolicyInput` | 每次 live policy inference |
| executable actions | `PolicyOutput` | 每次 live policy inference |
| raw rewards、done/termination/truncation masks | `EnvResult` | 每个环境 transition |
| observations/next observations | `EnvResult` | SAC/RLT 或其他 transition-observation 训练 |
| intervention/RLT fields | `EnvResult`/`RolloutResult` | 对应功能启用 |
| actions | `RolloutResult` | 每个 policy transition |
| `forward_inputs` | `RolloutResult` | Actor 需要重新 forward；目标 OpenPI PPO 必选 |
| previous logprobs | `RolloutResult` | PPO ratio 使用旧 logprob；目标配置必选 |
| state values | `RolloutResult` | GAE/actor-critic；目标配置必选 |
| versions | `RolloutResult` | 异步 staleness/version 或 model-weights identity 使用 |
| `RewardResult` | external reward enabled | 当前目标配置不出现 |
| history lengths | `RewardResult` | `reward_mode=history_buffer` |
| `ValueRequest`/`ValueResult` | value-based boundary bootstrap | timeout slots 或 alive segment-tail slots 非空时 |

结构校验不会自动 `.cpu()`、`.contiguous()` 或复制 tensor。数据 ownership 和 canonicalization 属于 Storage/transport subgoal，构造业务对象不能产生隐藏的大内存复制。

### SG-02：实现 `ForwardInputs` 模型封装

交付：schema registry、基类和 OpenPI 实现。

验收：`validate/select/tensor_fields/to_model_kwargs` 单测通过；select 后重组与原始 batch 完全一致。

本阶段只实现性能参考配置实际产生的 OpenPI + LIBERO PPO schema，不用一个含 optional leaves 的宽泛 OpenPI class 假装覆盖其他环境或 NFT/DSRL。v1 固定字段为：

```text
chains
denoise_inds
tokenized_prompt
tokenized_prompt_mask
action
model_action
observation/image
observation/wrist_image
observation/state
```

决定：

- schema identity 为 `openpi_libero` + version `1`；registry 使用 `(name, version)` 精确查找。
- `tensor_fields()` 按上述顺序返回字段，作为后续 fixed-frame schema 的稳定输入。
- `select(indices)` 只沿 batch axis 选择，不移动其他 tensor、不改变 dtype，也不隐式转 CPU。
- `to_model_kwargs()` 返回当前模型真实调用形式 `{"forward_inputs": <flat dict>}`。
- `RolloutResult.forward_inputs` 改为 typed `ForwardInputs`，不再接受任意 nested dict。
- 其他 OpenPI 环境、NFT 和 DSRL 字段在出现真实目标配置时注册独立 schema；不在 v1 中用 optional field 改变 steady-state layout。

修改范围：

- `rlinf/data/forward_inputs.py`
- `rlinf/data/trajectory.py`
- `rlinf/models/embodiment/openpi/forward_inputs.py`
- `tests/unit_tests/test_openpi_forward_inputs.py`
- `tests/unit_tests/test_trajectory_data.py`
- 本文的阶段与验收记录

计划验收命令：

```text
python -m pytest -q tests/unit_tests/test_openpi_forward_inputs.py tests/unit_tests/test_trajectory_data.py
ruff check <SG-02 Python files>
ruff format --check <SG-02 Python files>
Python 3.10 py_compile <SG-02 Python files>
git diff --no-index --check <new/untracked files>
```

### SG-03：实现确定性 `RoutePlan`

交付：producer/storage/actor 的 slot ownership 与局部索引转换。

验收：覆盖不整除、空 shard、跨多个 storage 的 source batch；slot 不重不漏且顺序可恢复。

本阶段采用统一的 slot-partition 模型，不为 Env、Rollout、Reward、Storage 和 Actor 各写一套映射：

- `total_slots` 是稳定全局 env slot 数。
- 每个参与方只声明 rank 数；所有参与方使用相同的 contiguous balanced partition 规则，前 `total_slots % world_size` 个 rank 多拥有一个 slot。
- `RoutePlan.slot_range()`、`owner()`、`local_index()` 和 `global_slot()` 提供 ownership 与双向局部索引转换。
- `RoutePlan.routes()` 把某个 source rank 的完整 batch 映射到 destination ranks。
- `RoutePlan.route_slots()` 支持 timeout/value 等稀疏、非连续、任意顺序的 slot batch，并验证每个 slot 确实属于声明的 source rank。
- 每条 `Route` 同时保存 destination rank、source indices、destination indices 和 global slot IDs；调用方可以按 source indices 恢复原始顺序。
- 空 shard 合法且产生空 routes；不存在 slot 的 rank 不发送空 record。
- `RoutePlan` 是纯 Python 数据对象，不依赖 WorkerGroup、Ray、Channel、Storage 或 tensor。

修改范围：

- `rlinf/workers/trajectory/__init__.py`
- `rlinf/workers/trajectory/route_plan.py`
- `tests/unit_tests/test_trajectory_route_plan.py`
- 本文的阶段与验收记录

计划验收命令：

```text
python -m pytest -q tests/unit_tests/test_trajectory_route_plan.py
ruff check rlinf/workers/trajectory tests/unit_tests/test_trajectory_route_plan.py
ruff format --check rlinf/workers/trajectory tests/unit_tests/test_trajectory_route_plan.py
Python 3.10 py_compile <SG-03 Python files>
git diff --no-index --check <new/untracked files>
```

### SG-04：实现纯内存 `TrajectoryStorage`

交付：typed write、幂等、readiness 和 Actor shard export。

验收：record 任意合法乱序得到相同输出；重复写幂等；内容冲突失败；缺 record 不会 ready。

本阶段只实现内存中的确定性聚合，不引入 Worker、Ray、Channel、RPC、transport、codec、队列或后台进程。`TrajectoryStorage` 接收已经 decode 并完成 destination routing 的业务对象；source address/sequence/ack 属于 SG-06/SG-09 的 transport 边界，而不是 storage key。

决定：

- 一个 storage 实例只拥有一个 `global_step` 和一组有序全局 `slot_ids`；`rollout_epochs`、`chunk_steps` 与所需 optional fields 由 immutable config 冻结。
- Env 与 Rollout transition 必须覆盖每个 `(rollout_epoch, chunk_step, local slot)`；external reward 只在 config 显式启用的 steps 上要求覆盖。
- optional field 采用 exact schema：配置为 required 的字段缺失会失败，未配置字段却出现也会失败，避免把 producer 漏填静默解释为 optional。
- exact logical identity 的相同内容重试是幂等写；相同 identity 的不同内容、或不同 slot batch 对同一 role/time 产生重叠覆盖，均为冲突。
- `ValueResult(kind="timeout")` 只允许覆盖同 transition 中 `truncations=true` 的 slots；`ValueResult(kind="tail")` 使用 `chunk_step=T`，只覆盖最后 transition 后 `dones=false` 的 slots。value 可以早于 Env result 到达，因此 expected coverage 在 readiness 阶段统一核对。
- Storage 保存 raw env rewards、external rewards、state values 和 boundary values，不修改或组合它们。SG-05/Actor advantage 是 bootstrap 的唯一应用位置。
- Actor shard tensors 按 `[rollout_epoch, chunk_step, local_slot, ...]` 导出；typed `ForwardInputs` 采用同一顺序折叠前三维为 leading batch，保留具体模型 schema。
- Storage 接管成功写入对象的只读 ownership；本阶段不隐式 `.cpu()`、`.contiguous()` 或 clone。SG-06/SG-09 在 ack 前负责稳定 receive buffer 生命周期。
- SG-03 的逻辑 rank 已由调用方给定。从真实 placement/WorkerGroup 推导 data-owning ranks 属于 SG-07 configure，SG-12 用实际 placement 做最终验证。

修改范围：

- `rlinf/data/forward_inputs.py`
- `rlinf/workers/trajectory/storage.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_storage.py`
- 本文的阶段与验收记录

计划验收命令：

```text
python -m pytest -q tests/unit_tests/test_trajectory_storage.py tests/unit_tests/test_openpi_forward_inputs.py tests/unit_tests/test_trajectory_data.py tests/unit_tests/test_trajectory_route_plan.py
ruff check <SG-01..04 Python files>
ruff format --check <SG-01..04 Python files>
Python 3.10 py_compile <SG-04 Python files>
git diff --no-index --check <new/untracked files>
```

### SG-05：验证时间轴和 bootstrap 正确性

交付：边界 value 对齐和 advantage 侧唯一 bootstrap 实现/测试。

验收：Bootstrap 与 Reward 不变量一节列出的所有边界测试通过；raw reward 在 Storage 前后相等。

本阶段不替换现有普通 channel 使用的 legacy
`compute_gae_advantages_and_returns()`，因为它仍接收偏移一位的 `T+1
values/dones`。新 trajectory path 使用独立的 transition-aligned embodied GAE；
两套接口按 Runner execution path 隔离。普通 Channel 暂不迁移或删除。

决定：

- 一个 rollout epoch 是独立 segment；GAE 不跨 epoch 递推。目标 `auto_reset=false` 每个 epoch reset，`auto_reset=true` 时 done mask 也会切断 episode recursion。
- raw action-chunk rewards 在 advantage 侧沿最后一维求和为 macro-transition reward；done/termination/truncation 沿最后一维做 `any`。这与目标配置 `reward_type=chunk_level` 一致，不引入缺少 primitive-state values 的 action-level critic。
- env reward 与 aligned external reward 先分别归约，再按显式权重组合；validity mask 为 false 的 external reward 不参与计算。输入 tensors 不原地修改。
- `state_values[e,t]=V(s_t)`；中间 alive transition 使用 `state_values[e,t+1]`，segment 最后 alive transition 使用 `tail_values[e]`。
- truncation 只使用同 transition 的 `timeout_values[e,t]`；done mask 同时阻止 reset observation 的 `state_values[e,t+1]` 成为上一 episode continuation。
- true termination 没有 boundary value；timeout mask 必须精确等于 reduced truncations，tail mask 必须精确等于最后 transition 的 alive slots。Storage readiness 和 advantage API 双层校验这一不变量。
- bootstrap 不写回 reward。returns 始终是 `state_values + advantages`，boundary value 只进入一次 TD delta。
- 新 API 返回 `[E,S,B,1]` advantages/returns，保持 OpenPI chunk-level Actor 的 broadcast 方向；可选 normalization 只改变 advantages。

修改范围：

- `rlinf/algorithms/embodied_gae.py`
- `tests/unit_tests/test_embodied_gae.py`
- 本文的阶段与验收记录

计划验收命令：

```text
python -m pytest -q tests/unit_tests/test_embodied_gae.py tests/unit_tests/test_trajectory_storage.py
python -m pytest -q <SG-01..05 unit tests>
ruff check <SG-01..05 Python files>
ruff format --check <SG-01..05 Python files>
Python 3.10 py_compile <SG-05 Python files>
git diff --no-index --check <SG-05 new/untracked files and this document>
```

### SG-06：实现 raw fixed-frame transport

交付：endpoint schema、header、encode/decode、预分配 receive buffers 和 sequence/ack 基础。

验收：所有 record 逐字段 `torch.equal`；无 pickle tensor payload；schema/sequence 错误明确失败。

本阶段实现纯 CPU transport codec 与 endpoint state，不连接 Worker、WorkerAddress、collective group 或后台队列。SG-07/SG-09 负责把这些 API 接到真实 send/recv；因此本阶段测试用 tensor `copy_` 模拟 data plane，不把 Ray object transfer 冒充 fixed-frame 验证。

决定：

- `EndpointSchema` 在 configure/control plane 从一个已验证业务对象建立固定 tensor paths、batch-excluded shapes、dtypes、常量字段和 typed `ForwardInputs` schema；steady-state header 不重复发送这些 metadata。
- 一个 endpoint schema 精确对应一种 `EnvResult`、`RolloutResult`、`RewardResult` 或 `ValueResult` layout；timeout/tail、reward mode 和 optional-field 组合不同即使用不同 schema ID。
- raw sender 只接受 contiguous CPU tensors，不隐式 `.cpu()`/`.contiguous()`；producer/staging lane 明确承担设备搬运和 canonicalization 成本。
- sender payload 直接引用业务对象 tensors，直到收到 ack 前不得覆盖；header 是独立固定 `int64` CPU tensor。
- header 包含 magic、protocol/schema/sequence、trajectory coordinates、actual batch、tensor count、reserved flags、固定容量 slot IDs 与每个 lane 的 raw bytes。unused slot entries 固定为 `-1`。
- receiver 先收 header，再用 actual batch 取得 max-capacity receive buffers 的 prefix views，随后原位接收各 tensor lane；decode 返回的 tensors 继续引用这些 buffers，不做隐藏 clone。
- sequence 在每个 endpoint lane 上从 0 单调递增；future/gap sequence 失败，任意已接收旧 sequence 标记为 retry 并重新 decode，最终内容幂等/冲突继续由 SG-04 Storage 判断。
- `TransportAck(schema_id, sequence_id)` 只表示 wire receive/decode 后 sender 可以释放当前 send ownership；`INGESTED` 等更强状态属于 SG-07/SG-09 worker API。
- transport 不使用 pickle、NumPy 或 tensor→bytes 转换；每个 raw tensor 保持原 dtype/shape，SG-14 才在同一 lane 边界加入 compression/raw fallback。
- 当前 endpoint 覆盖 storage record 所需 tensor 与 nested tensor mappings。batch string lists（例如尚未 tokenized 的 task descriptions）明确拒绝，不静默走 Python serialization；live request 的字符串策略在 SG-08 结合真实 model preprocessing 决定。

修改范围：

- `rlinf/workers/trajectory/transport.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_transport.py`
- 本文的阶段与验收记录

计划验收命令：

```text
python -m pytest -q tests/unit_tests/test_trajectory_transport.py tests/unit_tests/test_trajectory_storage.py
python -m pytest -q <SG-01..06 unit tests>
ruff check <SG-01..06 Python files>
ruff format --check <SG-01..06 Python files>
Python 3.10 py_compile <SG-06 Python files>
git diff --no-index --check <SG-06 new/untracked files and this document>
```

### SG-07：实现 Worker 生命周期

交付：独立 ChannelWorker/StorageWorker 的 launch、configure、ready、drain、shutdown 和错误传播。

验收：两者为独立 Ray actors；StorageWorker 阻塞或退出不会静默挂死 ChannelWorker。

实现决定：

- `TrajectoryChannelWorker` 继承 `ChannelWorker`，只持有 live queue 和自身生命周期；它不持有 Storage actor handle，也不等待 Storage future。
- `TrajectoryStorageWorker` 继承基础 `Worker`，configure 时创建本地 `TrajectoryStorage`、导入配置指定的 `ForwardInputs` registry modules，并注册 SG-06 endpoint state。
- `WorkerLayout.data_ranks` 显式给出拥有独立数据 shard 的物理 worker ranks；tuple 顺序就是逻辑 data rank。不能使用当前未生效的 `WorkerGroup._data_io_ranks`，也不能把 model-parallel world size 当成 data world size。
- Storage configure 同时校验 `len(data_ranks)`、`RoutePlan` participant world size 和当前 logical rank 的连续 `slot_ids`。真实 placement 负责生成 `data_ranks` 的 wiring 仍由 SG-12 验证。
- 生命周期为 `created -> ready -> draining -> stopped`；configure 异常进入 `failed` 并保留错误摘要。`ready()` 只表示 actor control plane 可用，trajectory 是否完整由 `trajectory_ready()` 单独表达。SG-07 尚未新增 live façade，因此继承的通用 Channel API 不作为新协议入口；SG-08 的 typed façade 必须按 lifecycle 拒绝 draining/stopped 状态的新请求。
- drain 不做无限等待：发现 live queue 或 transport in-flight 数据时立即显式失败。真正的 bounded wait、timeout 和 metrics 属于 SG-13。
- Channel shutdown 会取消继承的 memory-cleaner task；Storage shutdown 只释放它自己拥有的 endpoint/storage state。Ray group 的最终 kill 仍由现有 `WorkerGroup._close()` 管理。

修改范围：

- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_workers.py`
- 本文的阶段与验收记录

计划验收命令：

```text
/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_workers.py
/opt/venv/openpi/bin/python -m pytest -q <SG-01..07 unit tests>
ruff check <SG-01..07 Python files>
ruff format --check <SG-01..07 Python files>
Python 3.10 py_compile <SG-07 Python files>
git diff --no-index --check <SG-07 new/untracked files and this document>
```

### SG-08：实现 live critical path

交付：`PolicyInput` 和 `PolicyOutput` 的 live façade、route 与 relay。

验收：小规模 Env↔Rollout 往返正确；live frame 不含 trajectory-only 字段。

实现决定：

- `TrajectoryChannel` 是 Env/Rollout 看到的 typed façade，只公开 `publish/take_policy_input` 和 `publish/take_policy_output`；generic Channel `put/get` 不进入新协议调用面。
- Channel actor 使用 `RoutePlan.route_slots()` 按稳定 global slots 将 Env→Rollout、Rollout→Env 数据切分到目标 logical data rank 的独立 FIFO key。
- façade 使用 `ChannelConfig.env_layout/rollout_layout` 将当前 physical worker rank 映射为 logical data rank，禁止 model-parallel rank 直接污染 route。
- 业务对象保持 `PolicyInput`/`PolicyOutput`。collective 内部使用私有 representation，把嵌套 observation tensors 提升为顶层 tensor tuple；RLinf collective 因而走 dataclass tensor fast path，Python skeleton 只包含坐标、tensor paths、optional markers 和 `task_descriptions` 等小型常量。
- tensor 提升不复制 tensor；接收端按 path 重建并再次执行业务类型 validation。OpenPI 的 batch task strings 无损保留，不在 Env 侧引入 tokenizer。
- Ray variants 仅用于 driver/control tests；真实 Worker façade 调用使用 Worker collective。Storage record、reward、value 和 training fields 无法通过 typed live methods 发布。
- live methods 只在 `READY` 状态工作；draining/stopped 后拒绝新 publish/take。

修改范围：

- `rlinf/workers/trajectory/live.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_live.py`
- `tests/unit_tests/test_trajectory_workers.py`
- 本文的阶段与验收记录

计划验收命令：

```text
/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_live.py tests/unit_tests/test_trajectory_workers.py
/opt/venv/openpi/bin/python -m pytest -q <SG-01..08 unit tests>
ruff check <SG-01..08 Python files>
ruff format --check <SG-01..08 Python files>
Python 3.10 py_compile <SG-08 Python files>
git diff --check
```

### SG-09：实现 bypass record path

交付：Env/Rollout/Reward producer 直接向 StorageWorker 提交 record。

验收：三类 record 可乱序组装；ChannelWorker 不承载 record；retry/ack 幂等。

实现决定：

- `TrajectoryWriter` 是 producer-side façade；Env、Rollout、Reward 分别构造绑定自身 participant/schema 的 writer，只能提交对应业务类型。
- writer 用 source `WorkerLayout` 将 physical producer rank 转为 logical source rank，经 RoutePlan 按 slots 切分，再用 Storage `WorkerLayout` 将 logical destination 转回真实 Storage actor physical rank。
- 每个 `(schema_id, destination physical rank)` 拥有独立 sender `TransportEndpoint`；Storage receiver 每个 `(schema_id, src_addr)` 拥有独立 endpoint lane，避免多个 producer 都从 sequence 0 开始时互相误判 retry。
- 生产接口只有 `TrajectoryStorageWorker.submit(src_addr, schema_id)`。writer 先启动该 remote receive，然后按 header、payload 顺序调用 `send_tensor()`；Storage 预分配 max-batch buffers，按 header batch prefix 原位接收、decode、write。
- receive buffers 在本次 write 后由 Storage record 持有，不在 record 生命周期内复用；exact duplicate 的新 buffers 在 `write=False` 后释放。receive/ingest 已通过有界后台 queue 解耦，buffer pool/generation lease 属于 SG-13。
- SG-09 初始版本接受 `EnvResult`、`RolloutResult`、`RewardResult`；SG-10 将同一 bypass allowlist 扩展到 ValueResult。ChannelWorker 不新增 record submit API。
- 默认 `SubmitAck.status=RECEIVED`，此时 `inserted/trajectory_ready=None`；请求强保证时返回 `INGESTED`，`inserted=False` 表示 exact logical retry，冲突 retry 失败。
- `StorageWorkerConfig.ingest_queue_size` 建立有界后台 ingestion queue；drain 先切换 DRAINING 拒绝新 submit，再等待 `queue.join()`。后台 write 错误进入 worker FAILED health，并通过等待 INGESTED 的 future 传播。
- 不实现 receive buffer pool：成功写入的 tensors 被 Storage trajectory 持有，直到消费前复用会破坏不可变 ownership。pool 必须与多 generation 生命周期/lease 一起在 SG-13 设计。

修改范围：

- `rlinf/workers/trajectory/bypass.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_bypass.py`
- `tests/e2e_tests/embodied/verify_trajectory_bypass.py`
- 本文的阶段与验收记录

计划验收命令：

```text
/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_bypass.py
PYTHONPATH=<repo> /opt/venv/openpi/bin/python tests/e2e_tests/embodied/verify_trajectory_bypass.py
/opt/venv/openpi/bin/python -m pytest -q <SG-01..09 unit tests>
ruff check <SG-01..09 Python files>
ruff format --check <SG-01..09 Python files>
Python 3.10 py_compile <SG-09 Python files>
git diff --check
```

### SG-10：实现独立 value path

交付：timeout 和 segment-tail value request/response，调用实际 model `value_head`。

验收：只计算需要的 slots；结果直接进入 Storage；Env 不接收 bootstrap value 修改 reward。

实现决定：

- `TrajectoryChannel` 同时提供 typed policy API 和 value-request API；`ValueRequest` 复用同一个 `TrajectoryChannelWorker`，但使用独立的 `("value_request", rollout_rank)` 有界队列，不与 policy 消息共用 FIFO。
- request 按 Env/ Rollout logical ranks 和 sparse `slot_ids` 路由；timeout 请求携带 terminal/final observations，tail 请求携带 segment 末 alive observations。请求继续使用 SG-08 collective tensor extraction，图像不进入 Python serialization skeleton。
- Rollout 通过 `infer_value_request(model, request)` 调用真实 `model.predict_action_batch(env_obs=..., compute_values=True)`，从实际 value-head 路径取得 `prev_values`。没有 value/q head 或未返回 values 时明确失败，不生成虚构 `predict_values()`，也不使用零值 fallback。
- `MultiStepRolloutWorker.infer_value_request()` 直接委托上述真实模型路径；SG-12 再把 worker loop 与 `TrajectoryChannel`/Writer 接线。
- `ValueResult` 保留 request 的 kind、坐标和 sparse slots，values 规范为 CPU contiguous `[B,1]`；它通过 rollout-owned `TrajectoryWriter` fixed-frame bypass 直接进入 Storage。
- ValueResult 不返回 Env；Env raw reward 不修改。timeout/tail 最终只由 SG-05 advantage 公式消费一次。

修改范围：

- `rlinf/workers/trajectory/value.py`
- `rlinf/workers/trajectory/live.py`
- `rlinf/workers/trajectory/bypass.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `tests/unit_tests/test_trajectory_value.py`
- `tests/unit_tests/test_trajectory_bypass.py`
- `tests/e2e_tests/embodied/verify_openpi_forward_inputs.py`
- 本文的阶段与验收记录

计划验收命令：

```text
/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_value.py tests/unit_tests/test_trajectory_bypass.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<repo> /opt/venv/openpi/bin/python tests/e2e_tests/embodied/verify_openpi_forward_inputs.py --model-path <checkpoint> --device cuda:0 --batch-size 2
/opt/venv/openpi/bin/python -m pytest -q <SG-01..10 unit tests>
ruff check <SG-01..10 Python files>
ruff format --check <SG-01..10 Python files>
Python 3.10 py_compile <SG-09 amendment and SG-10 Python files>
git diff --check
```

### SG-11：实现 Storage 到 Actor 输出

交付：ready trajectory 的 actor shard、pull/push 接口、slot-order merge。

验收：Actor 输入与 reference assembly 等价；advantage 是 bootstrap 的唯一消费者。

实现决定：

- 采用 Actor-initiated pull，不建立 completed-trajectory mailbox。Actor Worker 内使用 `TrajectoryReader.pull()`；测试和控制面诊断使用显式 logical rank 的 `pull_via_ray(actor_rank)`，不暗自假定 physical rank 0。Reader 找出所有有交集的 Storage ranks，Storage 在确认本地 trajectory ready 后直接返回局部 contribution。
- 控制语义是 pull，数据语义是 Storage 应请求 push 给 Actor。Worker 内路径使用 `pull_actor_shard(dst_addr, actor_rank)` 和现有 Worker transport；非 Worker/测试路径使用 `pull_actor_shard_via_ray()`。
- `select_trajectory_batch()` 沿显式 slot axis 选择 tensor、nested observation、tail value 和 flattened typed `ForwardInputs`；`merge_trajectory_batches()` 要求 shard 不重叠且恰好覆盖 Actor slots，再恢复 Actor 的全局 slot 顺序。
- Storage 只导出 raw env/external rewards、state values、timeout/tail values 和 masks。SG-11 不组合 reward、不修改 bootstrap，也不计算 advantage；SG-05 advantage 实现仍是 bootstrap 的唯一数学消费者。
- Worker data plane 使用两个纯函数而不新增 transfer class：`flatten_trajectory()` 单次递归把包括 typed `ForwardInputs` 和 mixed nested observations 在内的所有 tensor leaves 提取为 `dict[str, torch.Tensor]`，其余小数据形成 skeleton；Storage 调用 `Worker.send(tensors, piggyback_payload=skeleton)`，Actor 用 `restore_trajectory()` 重建。生产 flatten 遍历本身保证 tensor 必被提取、未知类型立即失败，不再追加第二次 tensor-free 扫描；完整 skeleton scan 只在单元测试中执行。SG-14 在 tensor dict values 上加入无损压缩，`TrajectoryReader` 控制接口不随 codec 改变。
- pull 在 trajectory 尚未 ready 时明确失败，不等待无界 future。跨 generation 的等待、timeout、重试、消费后释放和 active trajectory 上限属于 SG-13。

修改范围：

- `rlinf/workers/trajectory/storage.py`
- `rlinf/workers/trajectory/output.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_output.py`
- `tests/e2e_tests/embodied/verify_trajectory_output.py`
- 本文的阶段与验收记录

计划验收命令：

```text
/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_output.py tests/unit_tests/test_trajectory_storage.py tests/unit_tests/test_embodied_gae.py
/opt/venv/openpi/bin/python -m pytest -q <SG-01..11 unit tests>
ruff check <SG-01..11 Python files>
ruff format --check <SG-01..11 Python files>
Python 3.10 py_compile <SG-11 Python files>
git diff --check
```

### SG-12：迁移真实组件

交付：Env、Rollout、Reward、Actor 使用新协议，保留最小必要兼容层。

验收：SG-12 新增的 trajectory-channel 配置可完成小规模 rollout/update；字段审计无无意义绕行。

#### SG-12 实现边界与当前决策

- `PolicyInput` 只携带规范化 observations 和实际启用的 RLT/intervention 控制；不携带 `final_obs`。
- `PolicyOutput` 只携带 Env 执行所需 actions。`forward_inputs`、old logprobs、state values 和 versions 由 Rollout 直接提交 Storage，不再经过 Env。
- Env 直接提交原始 reward/done/termination/truncation；不再调用 legacy `compute_bootstrap_rewards()`，也不追加 `T+1` trajectory item。
- timeout 使用 terminal `final_obs`（缺失时使用非 auto-reset 的 terminal current observation）构造稀疏 `ValueRequest`；segment 末仍存活的 slots 使用 tail observation。两者生成独立 `ValueResult`，由 Storage 按语义校验。
- Actor 从 Storage pull 后在唯一 compatibility boundary 中把 `[E,S,B]` 转为现有训练入口的 `[S,E*B]`，并调用 transition-aligned reward composition 与 GAE；不再调用 legacy `T+1` GAE。
- endpoint schema 必须在 runtime 初始化阶段完整注册；steady-state submit 只查找既有 schema，未知 record/layout 立即失败。schema identity 包含 record 语义和 source logical rank，因此 timeout/tail、Reward mode 以及不等分 source batch 不会错误共享 layout。
- SG-12 为保证正确性使用 `INGESTED` ack。early `RECEIVED`、overlap、buffer pool 和有界背压仍属于 SG-13，不能根据当前正确性实现推断性能。
- 当前真实配置限制为 `pipeline_stage_num=1`、Env/Rollout 等 world size、无外部 RewardWorker、单 generation (`runner.max_steps=1`)。多 generation reset/release、非等 world-size live merge、pipeline 和 RewardWorker history alignment 尚未闭合，继续作为 SG-12 验收风险，不得挪到 SG-13 后假装真实组件迁移完成。

新增配置：`examples/embodiment/config/libero_spatial_ppo_openpi_trajectory_channel.yaml`，使用 32 env、4 Storage shards、真实 OpenPI checkpoint 与一个 240-step rollout/update。

### SG-13：实现背压、错误与可观测性

交付：有界队列、容量预算、timeout/eviction、失败状态和 latency/bytes/queue metrics。

验收：慢 consumer 下内存有上限且不静默丢 record；producer 能观测 backpressure；drain 后无 in-flight frame。

#### SG-13 实现决定

- bypass 改为两阶段 `reserve → submit`。Storage 在 producer 发送任何 tensor 前按 lane 预留 frame slot 和 receive-buffer bytes；预算不足时有界等待并明确 timeout，不能先接收大 frame 再在 ingestion queue 后隐藏无界内存。
- reservation 使用整数 ID，不新增业务数据 class；超时 reservation 自动回收 buffer/预算。`SubmitAck` 回传 reserve wait time，Storage metrics 同时累计 backpressure、receive、ingest latency、bytes、queue depth 和 eviction/failure count。
- `max_inflight_frames` 限制 reserve/receive/ingest 中的并发 frame；`max_resident_bytes` 同时覆盖 active-generation receive buffers、reservations 和 buffer pool。配置必须显式为正数。
- 成功写入的 receive buffer 归 active generation 持有，Actor consumer 完成 pull 前不复用。所有相关 Actor ranks 消费后释放 `TrajectoryStorage` records，并将 buffers 放入同一预算内的有界 lane pool；放不下的 buffer 直接释放并记录 eviction。
- incomplete generation 永不因容量压力被静默 eviction。写入错误进入 `FAILED`，当前及排队的 strong-ack futures 均收到异常；后续 reserve/submit 明确拒绝。
- drain 先拒绝新 reservation，再在配置 timeout 内等待 reservations、active frames、ingestion queue 和 sender in-flight 全部清零；超时明确失败。Writer 增加本地 drain 检查。
- SG-13 先保证一个 active generation 的确定性 ownership。完成消费后可显式开始下一 generation；不允许覆盖尚未消费的 generation。

修改范围：

- `rlinf/workers/trajectory/bypass.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/storage.py`
- `rlinf/workers/trajectory/output.py`
- `rlinf/workers/trajectory/runtime.py`
- `rlinf/runners/embodied_runner.py`
- trajectory channel 配置与 SG-13 unit/e2e tests
- 本文验收记录

计划验收：

```text
/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_capacity.py tests/unit_tests/test_trajectory_bypass.py tests/unit_tests/test_trajectory_workers.py tests/unit_tests/test_trajectory_output.py
/opt/venv/openpi/bin/python -m pytest -q <SG-01..13 unit tests>
真实 OpenPI 两个连续 generation rollout/update
ruff check/format --check, py_compile, git diff --check
```

#### SG-13 验收记录（2026-07-20）

- 新增 `reserve → submit` 两阶段协议。producer 在发送 tensor 前获取 Storage 的 frame/byte capacity；慢 consumer 下第二个 reserve 会等待并最终按配置 timeout，而不是先占用未计量内存。
- reservation timeout、lane buffer pool、pool eviction、generation consumer release、连续 generation 和 drain failure 均有确定性单元测试。
- SG-01..13 共 129 个 trajectory 相关单元测试通过；真实 Ray bypass round-trip 通过。
- 使用 `/opt/venv/openpi` 和真实 OpenPI checkpoint 完成两个连续 generation 的 rollout/update：日志包含 `Global Step: 1/2`、`Global Step: 2/2` 以及两次 Actor/Critic update。该短流程只验证生命周期正确性，不作为性能 baseline。
- Runner 每步采集 Storage 的累计 frame/byte、backpressure、receive/ingest latency、queue depth、resident bytes、pool 和 generation metrics，并写入 `trajectory/` namespace。

当前保留风险：

- `max_resident_bytes` 精确约束 Storage receive buffers、reservations 和 pool；它不包含 `TrajectoryStorage.export()` 组装出的 batch、Actor 选择结果或 Ray object store。generation 的 record 数量由静态 schema 有界，因此不会无界增长，但进程峰值 RSS 不等同于该预算。目标配置完整 240-step 的 RSS 与 4 GiB 默认值必须在 SG-15 实测。
- reserve 被 submit 消费后，collective receive 的中途 peer failure 依赖现有 process-group / `RLINF_TIMEOUT`，尚无 trajectory 层可取消的 per-frame receive timeout。该失败不会静默丢 record，但可能等到底层通信 timeout；SG-15 故障注入后再决定是否值得引入可取消 transport。
- generation 只在全部预期 Actor consumers 成功 pull 后释放；consumer 永久丢失时不会自动 eviction，而由 drain timeout 将 worker 置为 `FAILED`。这是保正确性的选择，恢复/重试协议尚未实现。
- metrics 是进程内累计 counter/gauge；跨 worker 聚合目前由 Runner 完成，尚未定义长期 histogram bucket 或外部 tracing schema。
- 外部 RewardWorker、非等 world-size live merge 和 pipeline stage 的真实组件迁移仍是 SG-12 留下的范围限制，不因 SG-13 容量控制而自动解决。

### SG-14：接入无损压缩

交付：direct-tensor codec、block policy、raw fallback 和独立 CPU lane。

验收：bitwise round-trip；不可压缩数据不膨胀；live worker 不执行 trajectory codec 工作。

#### SG-14 实现决定

- 压缩边界是生产 Storage→Actor collective transfer。`TrajectoryStorageWorker` 先用既有 `flatten_trajectory()` 得到 `dict[str, torch.Tensor]`，随后压缩 tensor values；tensor-free skeleton 只追加 block metadata。诊断用途的 Ray object transfer 保持 raw，不冒充生产数据面。
- codec 直接使用连续 CPU tensor 的 `data_ptr()` 调用系统 `liblz4` / `libzstd`，输入、压缩输出和解压目标均为 tensor，不经过 NumPy、Python `bytes` 或 pickle tensor payload。安装依赖由 `requirements/sys_deps.sh` 显式提供 `liblz4-1` 和 `libzstd1`。
- 静态 `CompressionConfig` 在 runtime 初始化期同时下发 Storage 与 Actor reader。当前字段为 `enabled`、`codec`、`level`、`min_bytes` 和 `block_bytes`；steady state 不注册或修改 schema/codec。
- 每个 tensor 按固定 `block_bytes` 切分。每块只有在 encoded bytes 严格小于 raw bytes 时才发送 compressed tensor，否则发送 raw tensor view；若一个 tensor 的所有 blocks 都不可压缩，则退回原始单 tensor，不增加 payload bytes 或 block 数量。
- Actor 收到 wire tensors 后直接解压到按原 shape/dtype 分配的连续 CPU tensor，再使用原 skeleton 恢复 `TrajectoryBatch`。round-trip 对 uint8、bool、int64、float32 和 nested typed `ForwardInputs` 使用 `torch.equal` 验证。
- codec 只由独立 `TrajectoryStorageWorker` 进程和最终 Actor consumer 创建。`TrajectoryChannelWorker`/live module 不导入、不创建、也不执行 trajectory codec；因此 Storage 侧压缩 CPU 工作不会占用 live channel event loop。
- Storage metrics 新增 raw/wire bytes 和 compressed/raw block counters，继续由 Runner 聚合到 `trajectory/` namespace。

SG-14 correctness E2E 最初显式启用以下配置：

```yaml
trajectory:
  compression:
    enabled: true
    codec: lz4
    level: 1
    min_bytes: 65536
    block_bytes: 1048576
```

SG-15 的同机/跨机 A/B 证明目标高带宽 placement 上 raw 更快，因此生产示例当前默认 `enabled: false`；上述配置只作为显式 opt-in compression 参考。

#### SG-14 验收记录（2026-07-20）

- direct LZ4/Zstd codec、block policy、mixed dtype bitwise round-trip 和 incompressible raw fallback 单元测试通过。
- 真实 `TrajectoryStorageWorker → Actor Worker` collective round-trip 在启用 LZ4 后通过；测试同时确认 `compression_wire_bytes < compression_raw_bytes` 且 Actor 恢复的 nested observations、typed `ForwardInputs` 和 actions 与原值一致。
- 使用 `/opt/venv/openpi`、真实 OpenPI checkpoint、32 env 和四个 Storage/Actor shards 完成两个连续 compressed generations。日志包含 `Global Step: 1/2`、`Global Step: 2/2` 以及两轮 Actor/Critic update。该短流程验证集成正确性，不作为 codec 性能结论。
- SG-01..14 共 141 个 trajectory 相关单元测试通过；Ruff、format、compileall 和 `git diff --check` 通过。

当前保留风险：

- Storage 压缩与 Actor 解压目前分别在各自 worker 的 pull 调用内同步执行。它们已经与 live `TrajectoryChannelWorker` 进程隔离，但尚未与 Storage export/network 或 Actor H2D overlap；是否需要线程 lane、CPU affinity 或预分配 codec buffers 必须由 SG-15 profile 决定。
- 每个 compressed block 当前分配一个 worst-case temporary tensor，压缩输出 view 会持有该 allocation 到 send 完成。它避免 Python/NumPy copy，但不是最终 buffer-pool 方案；temporary/output peak 不计入 SG-13 的 receive-buffer `max_resident_bytes`。
- raw fallback 保证不可压缩 tensor 使用原始单 tensor、payload bytes 不增长。mixed tensor 的 payload bytes 也不超过 raw，但 block metadata 和额外 collective tensor 数可能抵消很小的压缩收益；SG-15 应基于端到端时间确定最小 savings threshold，而非只看 byte ratio。
- 默认 LZ4 参数、64 KiB threshold 和 1 MiB block 只是正确性起点，不是性能最优结论。LZ4/Zstd、block size、lane 数、NUMA 和跨机带宽选择均留给 SG-15。
- 现协议是独立 block，无 temporal state，因此没有断帧重同步问题；temporal XOR/keyframe 不在 SG-14 范围。

### SG-15：性能和隔离实验

交付：本机及 `bjd_dev` ↔ `bjd_dev_2` 跨机 benchmark artifact。

验收：使用目标配置真实数据量；报告 p50/p95/p99/p99.9、吞吐、CPU、RSS、网络和压缩率；比较 inline async、thread、独立 process、raw 和 compressed。

#### SG-15 实验决定

- critical Env↔Rollout live path 保持 raw，不在本阶段实现 XOR/LZ4。SG-15 只测量其延迟是否受到 trajectory 工作干扰；不得用“性能实验”名义改变 live 协议。
- 建立同 checkout、同 workload 的 feature-toggle A/B，不比较不同 SG 的单次 Runner wall time。短 OpenPI E2E 继续作为 correctness evidence，不作为 performance gate。
- target shard 固定为 8 slots、48 chunk steps、256×256 RGB main/wrist observations，并包含 OpenPI forward-input image/chains 等实际大字段。每份 artifact 必须记录最终 tensor schema 和 raw bytes，不能只写配置名。
- local codec 测试比较 raw、LZ4 和 Zstd；报告 encode/decode/end-to-end 的 p50/p95/p99/p99.9、CPU time、RSS peak、wire bytes、ratio 和 throughput。
- isolation 测试比较 no-work baseline、event-loop inline、async task、background thread 和独立 process。async task 不被假定为并行；实验应直接显示同步 native/memory work是否阻塞 event loop。
- cross-host 测试使用同一 tensor/block 协议在 `bjd_dev` 与 `bjd_dev_2` 间运行，原始结果、环境和命令归档。不得把本地 codec microbenchmark 推算成跨机结论。
- regression gate 分两层：确定性 gate 检查 raw fallback 不膨胀、bitwise correctness 和 artifact completeness；性能 gate 在本轮样本量足以估计机器噪声后制定。未测噪声前不武断写死 5%。
- 正式结果写入 `artifacts/trajectory_channel/<run_id>/`，遵循本文 Benchmark Artifact Contract；`/tmp` 只允许作为运行中日志。

计划修改范围：

- 当前协议 benchmark 工具及其单元测试
- `artifacts/trajectory_channel/<run_id>/` 的 manifest、resolved config、environment、commands、raw data 与 report
- 本设计文档的结果、回归 gate 和保留风险

计划验收：

```text
/opt/venv/openpi/bin/python -m pytest -q <SG-15 benchmark tests>
本机 smoke + target-scale raw/LZ4/Zstd/isolation
bjd_dev(rank 0) ↔ bjd_dev_2(rank 1) target-scale cross-host transfer
artifact completeness check
SG-01..15 unit regression, Ruff, format, compileall, git diff --check
```

#### SG-15 验收记录（2026-07-20）

正式 artifact：`artifacts/trajectory_channel/20260720_sg15_openpi_libero/`。

Workload 与边界：

- 单 Storage shard 为 8 slots × 48 chunk steps，包含四路 `[48,8,256,256,3] uint8` image leaves、OpenPI chains/token/action/value 等主要 tensor leaves；manifest 中逐 tensor 保存 shape、dtype 和 bytes。
- 总 raw tensor payload 为 314,511,360 bytes（299.94 MiB）。这纠正了旧实现“约 3.512 GiB/shard”的未归档估计；旧数字包含已经移除的字段/epoch 布局，不能用于当前协议容量规划。
- image 内容是确定性的 spatially coherent synthetic tiles，用于固定 byte volume、schema 与可压缩 workload。它不是真实 camera distribution，因此 12.64×/18.92× 压缩率不能外推到真实训练图像。
- 每项 2 次 warmup、10 次正式采样；p99/p99.9 是工程回归观测，不是统计充分的长期 SLO。

本机 codec round-trip：

| mode | p50 | p95 | p99 | p99.9 | wire bytes | ratio |
|---|---:|---:|---:|---:|---:|---:|
| raw | 13.14 ms | 13.21 ms | 13.24 ms | 13.25 ms | 314,511,360 | 1.00× |
| LZ4 | 223.47 ms | 225.57 ms | 225.81 ms | 225.87 ms | 24,873,812 | 12.64× |
| Zstd | 313.70 ms | 405.35 ms | 408.08 ms | 408.70 ms | 16,624,024 | 18.92× |

`bjd_dev → bjd_dev_2` Gloo tensor round-trip：

| mode | p50 | p95 | p99 | p99.9 | effective raw throughput |
|---|---:|---:|---:|---:|---:|
| raw | 164.18 ms | 167.94 ms | 168.22 ms | 168.28 ms | 1.78 GiB/s |
| LZ4 | 258.06 ms | 307.80 ms | 313.25 ms | 314.48 ms | 1.14 GiB/s |
| Zstd | 335.64 ms | 348.47 ms | 350.79 ms | 351.31 ms | 0.87 GiB/s |

结论：该 placement 上 raw 明确快于 compressed，即使 synthetic payload 有很高压缩率。目标配置因此默认关闭 compression；LZ4/Zstd 保留为低带宽 placement 的显式 opt-in，启用前必须在目标网络重跑 paired A/B。

实验同时发现 SG-14 的“每 block 一个 wire tensor”会产生约 300 次 Gloo tensor 调度。实现已改为每个原 tensor 内部继续逐 block codec/raw fallback，但将所有 block payload 打包成一个 wire tensor；pre-pack 与最终结果都保存在 artifact。打包减少 tensor calls，但仍不足以让压缩在当前网络胜过 raw。

live-loop 2 ms probe：

| mode | p50 | p95 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| baseline | 0.561 ms | 1.012 ms | 1.053 ms | 1.062 ms |
| inline | 1702.777 ms | 3049.184 ms | 3168.859 ms | 3195.778 ms |
| async task | 1706.160 ms | 3040.567 ms | 3160.239 ms | 3187.162 ms |
| thread | 0.565 ms | 1.016 ms | 1.053 ms | 1.062 ms |
| independent process | 0.564 ms | 1.017 ms | 1.055 ms | 1.750 ms |

结论：inline 和只包一层 coroutine 的 async task 都会阻塞 event loop；thread 与独立 process 在本轮 p99 接近 baseline，但只有独立 `TrajectoryStorageWorker` 同时提供 ownership、failure、queue 和 memory isolation。当前 provisional gate 为 process probe p99 不高于 paired baseline `+0.5 ms`；本轮 delta 为 `+0.002 ms`。

性能回归规则：

- SG-16 及后续 trajectory change 必须在同 checkout/同机器重跑 paired raw/default A/B；不能比较不同运行的单步 Runner wall time。
- 默认模式必须是目标 placement 上实测最快模式；当前为 raw。
- deterministic gates 继续要求 bitwise round-trip、raw fallback payload 不膨胀和 artifact completeness。
- `process_time()` 累加进程内所有线程的 CPU 时间，因此可能大于 wall time；完整 CPU/RSS 数值保存在 raw JSON/report，不应误读为单核 latency。

保留风险：

- cross-host tool 使用与生产一致的 Gloo tensor primitive 和 compression layout，但不包含 Ray control-plane RPC；真实 Runner 短 E2E 仍只提供 correctness evidence。
- real-camera compression ratio、受限带宽 break-even、NUMA/CPU affinity 和 reusable packed-buffer pool 尚未测量。由于当前默认 raw，这些不是 SG-16 correctness blocker。
- packed compression 会额外复制 block payload 到连续 wire buffer；它减少 collective calls，但增加本地 memory traffic。只有低带宽环境可能获益。

### SG-15B：压缩流水线优化

SG-15B 消除了 SG-15 实现中已经定位的本地开销，但没有改变 wire
metadata、逐 block raw fallback 或 bitwise 语义：

- `CompressionPipeline` 为每条 codec lane 持有独立 LZ4/Zstd context 和按
  tensor key 复用的有界 workspace；第二次相同 schema 压缩不再分配 workspace。
- block 直接写入最终 packed workspace。不可压缩 block 在同一目标 offset
  写入 raw bytes，不再先生成 block tensor、再复制到另一个 packed tensor。
- `num_threads` 将 tensor leaves 确定性地分给独立 lane；每条 lane 串行使用
  自己的 codec context，避免并发共享 Zstd context。
- Storage 的同步 `send()` 返回时 collective 已完成，之后才允许复用发送
  workspace。Actor 解压 tensor 被训练 batch 持有，因此只做并行解压，不复用
  最终 destination。

目标 workload 的本机 codec round-trip p50：

| implementation | raw | LZ4 | Zstd |
|---|---:|---:|---:|
| SG-15 block-then-pack | 13.14 ms | 223.47 ms | 313.70 ms |
| direct-pack + reusable workspace，1 lane | 13.81 ms | 217.15 ms | 303.96 ms |
| direct-pack + reusable workspace，2 lanes | 14.51 ms | 116.01 ms | 173.00 ms |
| direct-pack + reusable workspace，4 lanes | 34.64 ms | 68.52 ms | 未测 |

同一 checkout 的 `bjd_dev → bjd_dev_2` Gloo round-trip：

| lanes | mode | p50 | p95 | p99 | effective raw throughput |
|---:|---|---:|---:|---:|---:|
| 1 | raw | 164.20 ms | 168.67 ms | 169.50 ms | 1.78 GiB/s |
| 1 | LZ4 | 234.21 ms | 257.88 ms | 271.97 ms | 1.25 GiB/s |
| 1 | Zstd | 317.70 ms | 320.71 ms | 320.78 ms | 0.92 GiB/s |
| 2 | raw | 167.76 ms | 180.71 ms | 186.79 ms | 1.75 GiB/s |
| 2 | LZ4 | 134.40 ms | 137.44 ms | 139.10 ms | 2.18 GiB/s |
| 2 | Zstd | 191.90 ms | 192.70 ms | 192.89 ms | 1.53 GiB/s |
| 4 | raw | 168.32 ms | 170.87 ms | 171.40 ms | 1.74 GiB/s |
| 4 | LZ4 | 84.99 ms | 87.66 ms | 88.56 ms | 3.45 GiB/s |
| 8 | LZ4 | 91.37 ms | 94.76 ms | 95.55 ms | 3.21 GiB/s |

四 lane LZ4 相对同轮 raw 的 p50 降低 `49.5%`，p99 降低 `48.3%`；8 lanes
开始回退，因此目标配置采用 4 lanes。双 lane LZ4 已经产生正收益，四 lane
进一步降低延迟；因此
优化后的压缩在这份 synthetic target workload 上已经产生端到端正收益。
真实 StorageWorker→Actor Worker 双线程 compressed E2E 也已通过。

目标示例仍保持 `compression.enabled: false`。当前 synthetic tiles 的 LZ4
压缩率为 12.64×，不能代表真实 LIBERO camera distribution；在真实 trajectory
样本完成 paired A/B 前，不能把 synthetic 正收益直接转化为生产默认值。
正式原始结果位于
`artifacts/trajectory_channel/20260720_sg15b_openpi_libero/`。

### SG-15C：Env→Rollout raw fixed-frame

用户已确认真实 LIBERO camera 的无损压缩率，本阶段跳过重复的 camera ratio
实验，先建立可比较的 raw live-path 基线。`PolicyInput` 的目标 OpenPI schema 在
Runner 初始化时由配置注册，不允许首帧动态注册：

- 固定 tensor payload：`main_images [B,256,256,3] uint8`、
  `wrist_images [B,256,256,3] uint8`、`states [B,8] float32`、两个可选
  bool flags；`extra_view_images` 在当前目标配置必须为 `None`。
- 固定 metadata payload：sequence、step/epoch/chunk、slot ids、UTF-8 task
  description 的长度与定长 byte buffer。发送端直接复用原 observation tensors，
  只预分配 metadata workspace；接收端预分配并复用全部 buffers。
- Env rank 直接向同 rank Rollout 发送，不再让大 observation 绕行
  `TrajectoryChannelWorker`。`PolicyOutput` 和稀疏 `ValueRequest` 仍复用原有
  channel，避免在本阶段扩大协议范围。
- 接收结果的 tensor 生命周期截止到同一 façade 的下一次
  `take_policy_input()`；Rollout 必须在下一次 receive 前完成模型消费。

同机 paired benchmark 使用目标每-rank `B=8`、两路 256×256 RGB、总 tensor
payload 3,145,984 bytes：

| path | sender p50/p99 | receiver p50/p99 |
|---|---:|---:|
| ChannelWorker two-hop | 5.28 / 5.90 ms | 5.10 / 6.34 ms |
| direct fixed-frame | 1.82 / 1.89 ms | 1.81 / 1.85 ms |

真实 `libero_spatial_ppo_openpi_trajectory_channel` 以 32 env、4 Env/Rollout/
Actor ranks 完成一次 10-step rollout 和 Actor/Critic update；没有使用 synthetic
model 代替。原始 paired benchmark 位于
`artifacts/trajectory_channel/20260720_sg15c_live_raw/local.json`。

SG-15C 当时的保留限制是尚未接入 XOR/LZ4、pinned receive buffers、H2D overlap
和非一一 rank mapping；其中 compression 与 routing 已由下述 SG-15E/G 完成。
extra camera views、变长 batch、pinned receive 和 H2D overlap 仍属于后续
subgoal，不应隐藏为自动 fallback。

### SG-15G：fixed-frame 非一一 rank routing

direct live path 现在直接消费既有 `RoutePlan.routes()`，不再假定 Env rank 与
Rollout rank 数量、rank id 或 slot range 相同：

- Env 按 route 的 `source_indices` 生成每个 destination fragment；完整 batch
  route 保留原 tensor 引用，不做无意义的切片 copy。
- Rollout 按静态 source rank 顺序接收 fragment，并用 `destination_indices`
  写入预分配的完整 destination batch；单 fragment 完整覆盖时直接返回 receive
  buffer，不再复制一次。
- sequence、workspace、compression reference 均按 source-destination stream
  独立持有，避免多 peer 时共享状态。

真实 Ray Worker E2E 覆盖 2 Env→1 Rollout 和 1 Env→2 Rollout，每个 case 连续
传输两帧，并同时启用 XOR+LZ4 做 bitwise 字段验证。SG-15G 当时仍要求各 Env
rank batch size 相等；该限制已由下述 SG-15D 的 capacity layout 移除。

### SG-15E：live image XOR+LZ4

`PolicyInputLayout.compress_images` 在初始化期决定 live image wire protocol：

- 每个 stream 的首帧独立 LZ4；后续帧使用 `current XOR previous` 后 LZ4。
- 两路 image 各自持有 LZ4 context、reference、XOR workspace 和
  `compressBound` capacity buffer；不生成 Python bytes。
- header 先传输每路 image 的 mode 和实际 encoded bytes，payload 只发送实际
  view，不发送完整 capacity。
- 如果 LZ4 结果不小于原图，立即使用 raw flattened view；接收端始终 bitwise
  恢复并更新 reference。state、flags 和 descriptions 保持 raw fixed tensors。

目标同机 paired A/B（`B=8`、双 256² RGB、warmup 3、20 samples）：

| path | sender p50/p99 | receiver p50/p99 |
|---|---:|---:|
| ChannelWorker two-hop | 5.09 / 6.98 ms | 5.18 / 6.87 ms |
| direct raw | 1.61 / 1.89 ms | 1.61 / 2.00 ms |
| direct XOR+LZ4 | 2.80 / 2.95 ms | 2.80 / 2.95 ms |

因此目标同机配置保持 `compress_images: false`；SG-15E 是显式 placement policy，
不能因为压缩率有效就忽略 codec CPU/memory cost。低带宽或跨机 placement 只有在
同一 benchmark 产生端到端正收益后才能开启。原始结果位于
`artifacts/trajectory_channel/20260720_sg15e_live_compression/local.json`。

保留风险：当前没有周期 keyframe 或 stream restart resynchronization；collective
正常有序传输不会丢帧，但单边 worker restart 必须重建双方 façade。两路 codec
当前串行执行；是否引入 shared codec lanes 必须先测跨机尾延迟和 CPU contention。
SG-15E 验收时 pinned receive/H2D overlap 尚未实现，现由下述 SG-15F 完成。

### SG-15D：observation capacity 与 extra camera

`PolicyInputLayout.batch_size` 现在表示初始化期 capacity，而不是要求每个 Env
rank 实际 batch 完全相等。runtime 使用 ceil division 注册 capacity，实际 route
fragment 仍使用精确 batch shape，因此支持 total slots 不能被 Env/Rollout ranks
整除。`extra_view_shape` 在初始化期声明额外 camera tensor shape；未声明时仍严格
要求 `extra_view_images=None`，不会动态注册或 pickle fallback。

Ray Worker E2E 使用 5 slots、2 Env、3 Rollout，覆盖不同 Env batch size、extra
camera、XOR+LZ4、pinned receive 和两帧 bitwise round-trip。

### SG-15F：pinned receive 与 OpenPI non-blocking H2D

- fixed-frame receive buffers 可在初始化期设为 pinned memory；Gloo 直接写入这些
  reusable buffers，E2E 明确验证收到的 image tensor `is_pinned()`。
- OpenPI 在 `input_transform` 后的真实 model-input 边界维护按 key/shape/dtype
  复用的 pinned staging buffers。CPU transform output copy 到 staging 后使用
  `to(device, non_blocking=True)`；同一 CUDA default stream 保证后续模型算子等待
  copy，无需 CPU synchronize 或额外 event。
- 这里的 overlap 是 image H2D 与 CPU 准备/提交后续 fields 的有限重叠；Env-policy
  是 action-dependent 闭环，不能声称与“下一帧 receive”重叠。

同机 transport paired A/B 中 direct raw p50 为 1.87 ms，direct pinned 为
1.42 ms（降低 24.4%）；因此目标配置启用 `pin_memory: true`。真实 32-env、
4-rank OpenPI/LIBERO 10-step rollout 和 Actor/Critic update 已通过，rollout
predict 为 1.011 s。原始 transport 结果位于
`artifacts/trajectory_channel/20260720_sg15f_pinned_h2d/local.json`。

保留风险：当前 staging 在首次遇到新 shape/dtype 时分配；固定 target schema 后会
复用。extra camera 目前 raw 传输，尚未纳入 XOR+LZ4 image set；是否压缩应基于
该 camera 的 paired latency，而不是自动继承 main/wrist 策略。

### SG-16：路径互斥、回归验证与实现收口

本阶段不删除或迁移普通 Channel。它仍供已有 Env/Rollout/Reward/Actor 路径、
evaluation、pipeline 和非 OpenPI 组件使用。SG-16 只保证启用 trajectory runtime
时执行路径唯一，并完成正确性与性能回归。

启动期配置验证在创建 Actor、Rollout、Env 等 component WorkerGroup 前执行。当前
trajectory runtime 明确拒绝：

- `runner.use_training_pipeline=true`；
- `rollout.pipeline_stage_num != 1`；
- `runner.only_eval=true`；
- `runner.overlap_env_bootstrap=true`；
- `reward.use_reward_model=true`；
- 非 `openpi` 模型。

这些能力不是理论上不能支持，而是当前尚未被 trajectory runtime 完整拥有。显式
失败优于静默退回普通 path 或同时运行两套协议。

reward/bootstrap 所有权审计结果：

- trajectory Env 只写原始 `env_rewards`、termination 和 truncation；不调用 legacy
  `compute_bootstrap_rewards()`；
- Rollout 只对 timeout terminal observation 和有效 segment tail 计算稀疏 value，
  然后直接 bypass 到 Storage；
- Storage 汇合 transition 与 boundary values；Actor pull 后只调用一次
  `compose_embodied_rewards()` 和 `compute_embodied_gae()`；
- 普通 Channel 保留 Env reward correction 与 legacy `T+1` GAE，但 Runner 的路径
  互斥，单次 run 不会进入两套逻辑。

验收：trajectory 配置边界、transition-aligned GAE、普通 Channel bootstrap 的
focused suite 为 29 tests；包含全部 trajectory unit tests 的 suite 为 153 tests，
均通过。目标配置完成一次 32-env、4-rank rollout→Storage→Actor/Critic update。
paired live transport 的 direct raw p50 为 1.57 ms，SG-15F 同机结果为
1.87 ms，没有观察到性能回归。pinned 与 raw 的短基准差异较小，pinned 的保留理由
是 OpenPI non-blocking H2D，而非宣称纯 CPU transport 必然更快。原始结果位于
`artifacts/trajectory_channel/20260720_sg16_final/`。

保留风险：external RewardWorker、evaluation、pipeline、多 rollout stage 和非 OpenPI
schema 尚未接入 trajectory runtime；普通 Channel 也尚未做等价迁移。首次 E2E 在
benchmark 遗留的 Ray runtime 中因 Worker 未继承仓库 `PYTHONPATH` 而在 NodeProbe
阶段失败；清理 runtime 并显式设置 `PYTHONPATH` 后通过。这是运行环境前置条件，
不是 trajectory 数据路径错误。

依赖顺序：

```text
SG-00
  -> SG-01 -> SG-02 -> SG-03 -> SG-04 -> SG-05
  -> SG-06 -> SG-07
  -> SG-08 + SG-09 -> SG-10 -> SG-11 -> SG-12
  -> SG-13 -> SG-14 -> SG-15 -> SG-16
```

## 11. 每轮验收记录模板

完成一个 subgoal 后，在本节顶部追加记录：

```text
### YYYY-MM-DD — SG-XX

Status: awaiting acceptance | accepted | rejected

Decisions:
- ...

Changed files:
- ...

Verification:
- command: ...
  result: ...

Known limitations:
- ...

Next gate:
- ...
```

### 2026-07-20 — SG-12

Status: completed, awaiting human acceptance

Decisions:

- live critical path 仅保留 `PolicyInput`（Env→Rollout）和 `PolicyOutput`（Rollout→Env）；Rollout 的 `forward_inputs`、old logprobs、state values、versions 以及 Env 的原始 transition 数据均直接 bypass 到 Storage。
- timeout terminal observation 与 segment-tail observation 只生成稀疏 `ValueRequest`；`ValueResult` 直接写入 Storage。Env 不接收 value，也不修改 reward。
- Actor 从 Storage pull transition-aligned `TrajectoryBatch`，在一个兼容边界完成 `[E,S,B]`→`[S,E*B]`、reward composition、GAE 和 shuffle；普通 channel 继续使用 legacy `T+1` 路径。
- 生产 endpoint 在 runtime 初始化时按语义和 source rank 注册；Writer 与所有 Storage ranks 使用同一组 immutable schema，steady-state 不存在注册 RPC。SG-12 使用 `INGESTED` ack，确保 Actor pull 前所有 records 已可见。
- 新 runtime 从实际 WorkerGroup placement 构造 Env/Rollout/Actor/Storage layouts、RoutePlan、writers 和 reader；不手写逻辑 data ranks。

Changed files:

- `examples/embodiment/train_embodied_agent.py`
- `examples/embodiment/config/libero_spatial_ppo_openpi_trajectory_channel.yaml`
- `rlinf/runners/embodied_runner.py`
- `rlinf/workers/actor/fsdp_actor_worker.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/trajectory/actor.py`
- `rlinf/workers/trajectory/bypass.py`
- `rlinf/workers/trajectory/live.py`
- `rlinf/workers/trajectory/records.py`
- `rlinf/workers/trajectory/runtime.py`
- `rlinf/workers/trajectory/workers.py`
- `tests/unit_tests/test_trajectory_actor.py`
- `tests/unit_tests/test_trajectory_records.py`
- `tests/unit_tests/test_trajectory_runtime.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python examples/embodiment/train_embodied_agent.py --config-name libero_spatial_ppo_openpi_trajectory_channel env.train.max_steps_per_rollout_epoch=10 actor.micro_batch_size=16 actor.global_batch_size=64 algorithm.update_epoch=1`
  result: passed；真实 checkpoint、32 env、4 Env/Rollout/Actor ranks 和 4 Storage shards 完成一次 rollout→bypass ingest→稀疏 value inference→Actor pull→transition-aligned GAE→optimizer update。`Global Step 1/1`，`actor/policy_loss=6.48e-04`，`critic/value_loss=5.26e-04`。
- command: SG-01～12 combined unit suite
  result: 123 passed。
- command: Ruff check/format check, Python py_compile and `git diff --check`
  result: passed。

Known limitations:

- 上述真实验收为 10 primitive steps 的小规模正确性流程；目标配置文件默认仍是 240 steps。240-step 流程在本轮调试中已完成 rollout、Storage ingest、value inference 和 Actor pull，但修正 Actor transition-aligned shuffle 后未再次运行完整 240-step optimizer update，因此不能将其表述为已验收性能基线。
- 当前 runtime 仅支持 `pipeline_stage_num=1`、Env/Rollout 等 world size、单 generation (`runner.max_steps=1`)。generation release、容量上限和多 generation 生命周期属于 SG-13。
- 目标配置未启用外部 RewardWorker，因此 per-step/history-buffer external reward 的真实组件接线尚未验收；当前实现明确拒绝该配置，不会静默回退或把 reward 绕回 critical path。开始支持外部 reward 前必须确定 Reward input 的非关键路径 ownership 与 history alignment。
- 同步 `INGESTED` ack 和每次独立 receive buffer 以正确性为先；它们不是性能结论。early ack、overlap、buffer pool、背压和 CPU 隔离由 SG-13/SG-15 实测决定。
- Storage/Channel workers 当前按 placement 启动但未隔离 GPU visibility；其计算路径为 CPU。SG-15 需要量化是否产生 CUDA runtime 或资源争用，再决定是否收紧 worker runtime env。

Next gate:

- 人工验收真实组件字段 ownership、短流程证据和上述边界；通过后开始 SG-13。

Acceptance correction:

- 人工审查发现首条真实 record 曾触发 `EndpointSchema.from_example()` 和 Storage `register_endpoint()`。该设计已在 SG-12 内撤销：runtime 现在为每个 source logical rank 显式构造 Env、OpenPI LIBERO Rollout、timeout Value 和 tail Value schemas，经 `StorageWorkerConfig.endpoints` 在 configure 阶段统一加载，并经 `schemas_by_rank` 交给 Writer。
- `TrajectoryStorageWorker.register_endpoint()` 已删除；`TrajectoryWriter._prepare()` 不再创建或注册 schema，缺失 `(source_rank, record kind)` 时直接失败。显式 schema 与真实 record layout 的一致性由 `test_trajectory_runtime.py` 覆盖。
- real-checkpoint correction 确认目标配置的 Env chunk reward/done/termination/truncation 为 `[B,5]`，环境转换后的 actions/forward action 为 `float64`，state value 为 `[B,1]`；初始化 schema 已按这些实际事实固定，而不是沿用 `[B,1]` reward 或 float32 action 的假设。
- command: `/opt/venv/openpi/bin/python examples/embodiment/train_embodied_agent.py --config-name libero_spatial_ppo_openpi_trajectory_channel env.train.max_steps_per_rollout_epoch=10 actor.micro_batch_size=16 actor.global_batch_size=64 algorithm.update_epoch=1`
  result: passed after static-schema correction；32 env、4 producer ranks、4 Storage shards 完成 `Global Step 1/1`，Actor 与 Critic 均完成 optimizer update。steady-state 没有 schema registration API/RPC。

### 2026-07-20 — SG-11

Status: completed, awaiting human acceptance

Decisions:

- Actor 通过 `TrajectoryReader` 主动 pull；不创建额外 ChannelWorker 或 completed mailbox。
- Storage 按 RoutePlan 只导出目标 Actor rank 拥有的 slots；Actor 从所有相交 Storage ranks 收集后按全局 slot order 合并。
- nested observations、普通 `[E,S,B,...]` 字段、tail `[E,B,...]` 字段和 flattened typed `ForwardInputs` 使用各自明确的 slot axis，不依赖 shape 猜测。
- Storage→Actor wire body 是纯 `dict[str, torch.Tensor]`，tensor-free skeleton 通过 piggyback 发送；使用 flatten/restore 函数，不增加新的数据 class，也不在生产热路径重复扫描 skeleton。
- 未引入 reward/bootstrap mutation；Actor 收到的仍是 raw advantage inputs。

Changed files:

- `rlinf/workers/trajectory/storage.py`
- `rlinf/workers/trajectory/output.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_output.py`
- `tests/e2e_tests/embodied/verify_trajectory_output.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_output.py tests/unit_tests/test_trajectory_storage.py tests/unit_tests/test_trajectory_workers.py`
  result: 23 passed；覆盖跨 Storage slot selection、乱序 Actor merge、nested observations、typed ForwardInputs、tensor-free skeleton、bitwise tensor round-trip、overlap/incomplete coverage failure、Storage ready export，以及 RECEIVED 后未 ingest 时 pull 失败、queue ingest 完成后 pull 成功的确定性时序。
- command: SG-01～11 combined unit suite
  result: 116 passed。
- command: `env -u RAY_ADDRESS PYTHONPATH=<repo> /opt/venv/openpi/bin/python tests/e2e_tests/embodied/verify_trajectory_output.py`
  result: passed；真实 `TrajectoryStorageWorker` 通过 tensor dict + piggyback skeleton 向真实 Actor Worker 发送 mixed observations 和 typed OpenPI ForwardInputs，Actor restore 后字段一致。
- command: Ruff check/format check and Python py_compile
  result: passed。

Known limitations:

- output tensor dict 仍携带每次 transfer 的路径/skeleton metadata，不是 SG-06 式预注册 fixed frame；SG-14 压缩设计需决定是否缓存 output schema，并用目标配置数据量量化 metadata 成本。
- Actor pull 与最后一条 default `RECEIVED` submit 可能竞争；确定性测试已覆盖未 ingest 时 pull 失败、queue.join 后成功。SG-12 仍需验证真实 Runner 只在 readiness 后 pull，SG-13 再定义 timeout/retry/backpressure。
- “generation release”指 Actor 已成功接收并 merge 某个 generation 后，Storage 删除该 generation 的 records、identities、assembled/export tensors 和 receive-buffer 引用，使对应 CPU 内存可以回收；它不表示 Actor 释放正在训练的 batch。当前 Storage 只有单 generation 且直到 shutdown 才释放。SG-13 需要按预期 Actor consumers 维护 ack/refcount，最后一个 consumer 完成且 send work 完成后才能 release，并设置 active generation/bytes 上限与 eviction policy。
- 真实 Storage Worker→Actor Worker collective round-trip 已覆盖 tensor observations、字符串 metadata 和 typed OpenPI ForwardInputs。接收 Actor 必须在 restore 前加载对应 ForwardInputs registry module；SG-12 应由 Actor 模型初始化统一保证，而不是依赖测试内 import。跨机与目标数据量性能仍属于 SG-15。
- RoutePlan 允许空 partition，但当前 `TrajectoryReader.pull()` 要求 Actor data-owning rank 至少拥有一个 slot。SG-12 placement wiring 必须排除空 Actor data ranks，或在需要支持 world_size > total_slots 时把 pull 返回类型扩展为 optional。

Next gate:

- 人工验收 Storage→Actor ownership、pull/merge API 和风险边界；通过后开始 SG-12。

### 2026-07-20 — SG-10

Status: completed, awaiting human acceptance

Decisions:

- 先修正 SG-09：默认 RECEIVED early ack + bounded background ingestion；INGESTED 成为同一 submit API 的可选强保证，drain 等待 queue 清空。
- 不提前做错误的 buffer reuse；Storage 成功写入后仍持有 receive tensors。
- ValueRequest 使用现有 typed `TrajectoryChannel`，按 sparse slots 从 Env 路由到 Rollout，并在 worker 内使用独立 lane。
- value inference 调用实际模型 `predict_action_batch(..., compute_values=True)`/value head；缺失 values 明确失败，不回退成 zeros。
- ValueResult 复用 fixed-frame bypass 直接进入 Storage，不返回 Env、不修改 reward。

Changed files:

- `rlinf/workers/trajectory/value.py`
- `rlinf/workers/trajectory/live.py`
- `rlinf/workers/trajectory/bypass.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `tests/unit_tests/test_trajectory_value.py`
- `tests/unit_tests/test_trajectory_bypass.py`
- `tests/e2e_tests/embodied/verify_openpi_forward_inputs.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_value.py tests/unit_tests/test_trajectory_bypass.py tests/unit_tests/test_trajectory_workers.py`
  result: 13 passed；覆盖 timeout/tail sparse route、terminal observation preservation、value-head call contract、no-head failure、ValueResult fixed-frame ingestion，以及 RECEIVED/INGESTED 后台 queue 语义。
- command: SG-01～10 combined unit suite
  result: 110 passed。
- command: real OpenPI checkpoint verification with batch size 2 on CUDA
  result: passed；`value_request_shape=[2,1]`，values finite，实际 checkpoint value head 被执行。

Known limitations:

- SG-10 不增加独立 value actor、WorkerGroup 或 placement。共享 actor event loop 是否产生 policy head-of-line blocking 尚未实测；SG-15 必须分别测量无 value 请求、稀疏 timeout 请求和 segment-tail 请求下 policy latency 的 p50/p95/p99/p99.9。只有观测到显著退化后才引入可配置物理隔离，且不能复制业务 façade。
- `TrajectoryChannelWorker.publish_value_request()` 内部同步 collective `recv` 可能占用 async actor event loop；需要在 SG-12 真实接线后确认 Ray `max_concurrency` 下 policy queue 是否仍能及时调度，不能仅用 via-Ray 单元测试推断无干扰。
- 当前 OpenPI value 语义沿用模型现有 `predict_action_batch(..., compute_values=True)`，它会同时完成 action sampling；若未来模型提供经过验证的专用 value-only forward，可在不改变协议的情况下优化计算成本。
- backend ingestion 有界队列已经建立，但 timeout、metrics、跨 generation buffer lease/pool 仍属于 SG-13。

Next gate:

- 人工验收 SG-09 early-ack 修正、TrajectoryChannel value lane、真实 value-head 调用和 direct-to-Storage 语义；通过后开始 SG-11。

### 2026-07-20 — SG-09

Status: completed, awaiting human acceptance

Decisions:

- producer 使用 `TrajectoryWriter` 直接 route/split 到 Storage actor，ChannelWorker 没有 record submit API。
- fixed-frame lane identity 是 `(schema_id, source WorkerAddress)`；wire sequence 不跨 producer 共享。
- Storage submit 默认在 fixed header/payload 进入自有 buffers 后返回 RECEIVED ack，并由有界后台 queue ingestion；可请求 INGESTED 强保证。logical retry 与 wire ack 都是幂等的，冲突 retry 明确失败。
- `trajectory_ready` 与 ack durability 分开；SG-11 最终选择 pull，因此 submit ack 不承担 Actor consumption 保证。
- ValueResult 不进入本阶段 submit allowlist，避免提前实现 SG-10。

Changed files:

- `rlinf/workers/trajectory/bypass.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_bypass.py`
- `tests/e2e_tests/embodied/verify_trajectory_bypass.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_bypass.py`
  result: 6 passed；覆盖三类 record 乱序、route/split、physical/logical Storage rank、per-source sequence lane、fixed-buffer submit、exact/conflicting retry、strong ack 和 Channel API 边界。
- command: `PYTHONPATH=/mnt/public/daibo/timeline/0703/RLinf /opt/venv/openpi/bin/python tests/e2e_tests/embodied/verify_trajectory_bypass.py`
  result: passed；真实启动三个 producer Ray Worker groups 与一个 StorageWorker，通过 `send_tensor/recv_tensor` 按 Reward→Rollout→Env 乱序提交，最终 trajectory ready，Env 重试返回 `inserted=False`。
- command: SG-01～09 combined unit suite
  result: 107 passed。
- command: Python 3.10 `py_compile`, full SG-01～09 Ruff check/format check, and `git diff --check`
  result: passed。

Known limitations:

- SG-09 每次 submit 独立分配 receive buffers，正确但尚未池化；成功 ingest buffers 被 Storage 持有，pool 必须等 SG-13 的 generation lease 设计。
- writer 默认等待 RECEIVED，允许 Storage.write 与 producer 后续工作 overlap；timeout、批量 ack 和 metrics 属于 SG-13。
- 真实 Env/Rollout/Reward 组件尚未迁移到 writer；属于 SG-12。当前 e2e 使用最小真实 Worker groups 验证生产 collective data plane。
- lossless compression 尚未接入 fixed tensors；属于 SG-14。

Next gate:

- 人工验收 bypass ownership、lane/ack 语义和真实 Ray fixed-frame 测试；通过后开始 SG-10。

### 2026-07-20 — SG-08

Status: completed, awaiting human acceptance

Decisions:

- 新增 `TrajectoryChannel` typed façade；live API 只有 policy input/output 两个方向，record/value/reward 不复用该入口。
- RoutePlan 在 Channel actor 内按 slot 切分，Env/Rollout 的 `WorkerLayout` 在 façade 把 physical rank 转为 logical data rank。
- collective wire 使用私有 tensor-extracted representation，避免 mixed nested observations 把图像序列化进 Python skeleton；task strings 作为小型无损 metadata 保留。
- actor queue 保存已验证、已路由的业务对象；drain 仍能统计同一 queue map，lifecycle 非 ready 时 typed live API fail fast。

Changed files:

- `rlinf/workers/trajectory/live.py`
- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_live.py`
- `tests/unit_tests/test_trajectory_workers.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_live.py tests/unit_tests/test_trajectory_workers.py`
  result: 9 passed；覆盖 slot route/split、双向 façade round-trip、OpenPI-style image/state/task strings、wire tensor extraction、非法 record type 和 lifecycle gate。
- command: `ruff check` and `ruff format --check` for SG-08 files
  result: passed。
- command: SG-01～08 combined unit suite
  result: 101 passed。
- command: Python 3.10 `py_compile`, full SG-01～08 Ruff check/format check, and `git diff --check`
  result: passed。

Known limitations:

- 当前 `TrajectoryChannel.from_worker_group()` 明确要求单个 Channel actor；多节点 Channel replica 的 key→replica locality 由 SG-12 结合真实 placement 接线并验证。
- Ray variant 会使用 Ray object transport，仅服务 driver tests/control；生产 Worker 方法已走 RLinf collective tensor fast path，但尚未用真实 Env/Rollout component 做端到端迁移，该验证属于 SG-12。
- SG-13 才加入 queue timeout、容量预算和 latency/bytes metrics；SG-08 的 bounded queue 使用现有 `asyncio.Queue` 语义。

Next gate:

- 人工验收 live façade、wire tensor extraction、route 和字段边界；通过后开始 SG-09。

### 2026-07-20 — SG-07

Status: completed, awaiting human acceptance

Decisions:

- Channel 与 Storage 使用两个独立 Ray actors，control plane 没有 Channel→Storage 引用或等待边。
- 使用显式有序 `data_ranks` 定义 physical→logical data-rank 映射；Storage configure 强制校验 route world size 和本地 slot ownership。
- configure 负责加载模型 registry、构造 transport endpoints 和纯 Storage；实际 frame submit/get 留给 SG-09/SG-11。
- health 包含 lifecycle、PID、physical/logical rank 和失败摘要；trajectory completeness 不混入 lifecycle ready。
- drain 对未清空 queue/in-flight frame fail fast；shutdown 在 ownership 已排空后释放本 worker 状态。

Changed files:

- `rlinf/workers/trajectory/workers.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_workers.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_workers.py`
  result: 4 passed；验证 configure/ready/drain/shutdown、失败状态、logical rank/slot 校验，以及真实独立 Ray actors。
- isolation method: 对 Storage actor PID 发送 `SIGSTOP`，Channel `health()` 在 1 秒 timeout 内仍返回；`SIGCONT` 后 kill Storage actor，Channel 仍返回 ready。测试 launch 使用 `catch_system_failure=False`，避免把故意 kill 误报为 RLinf 全局集群故障。
- command: `ruff check` and `ruff format --check` for SG-07 files
  result: passed。
- command: SG-01～07 combined unit suite
  result: 96 passed。
- command: Python 3.10 `py_compile`, full SG-01～07 Ruff check/format check, and `git diff --check`
  result: passed。

Known limitations:

- SG-07 只建立 lifecycle/control plane；fixed-frame `submit(src_addr, schema_id)`、receive buffer lease 和 strong submit ack 属于 SG-09。
- 继承的 generic Channel `put/get` 尚未替换或 lifecycle-gate；它不是目标协议 API，SG-08 必须用 typed live façade 收口并验证 draining 后拒绝新请求。
- `WorkerLayout.data_ranks` 已能正确表达 data-owning ranks，但从真实 placement/model-parallel topology 生成该 tuple 的组件迁移 wiring 属于 SG-12。
- drain 当前 fail fast，不提供 timeout/backpressure metrics；属于 SG-13。

Next gate:

- 人工验收 lifecycle、data-rank 映射和 Ray 隔离测试；通过后开始 SG-08。

### 2026-07-19 — SG-06

Status: awaiting acceptance

Decisions:

- `EndpointSchema` 固定 record type、tensor paths、batch-excluded shapes、dtypes/element sizes、constant fields、max batch 和可选 typed `ForwardInputs` identity；schema 可作为 control-plane metadata 序列化，但 tensor payload 不走 pickle。
- `TransportEndpoint.encode()` 只创建小型 `int64` header，payload 保持原 contiguous CPU tensor references；没有 tensor→NumPy/bytes 或隐藏 clone。
- receiver 为每个 tensor leaf 预分配 max-batch CPU buffer；header 到达后 `payload_views()` 返回 actual-batch prefix，decode 结果继续引用 receive storage。
- header v1 固定 magic/version/schema/sequence、trajectory coordinates、batch/tensor count、flags、slot capacity 与逐 lane raw bytes；corrupt magic/version/schema/flags/size/slot 明确失败。
- endpoint sequence 从 0 单调递增；future/gap frame 失败，任意更早 sequence 作为 retry 重新 decode。retry 不在 transport 层跳过 payload，因此 identity 相同但内容不同仍能由 Storage 检出冲突。
- `TransportAck` 对 sender-side sequence ownership 幂等；wrong schema 和 future ack 失败。它不假装表示 Storage `INGESTED` 或 Actor 已消费 output。
- Env/Rollout/Reward/Value 四类 storage results 均完成逐字段 exact round-trip；OpenPI ForwardInputs 通过 `(schema_name, schema_version)` registry 恢复具体类型。
- `TensorLayout` 保留逐 field lanes，为 SG-14 的 per-field compression/raw fallback 提供边界；是否按 dtype 聚合 lanes 由 SG-15 真实延迟实验决定。
- transport 业务 API 没有新增 `*Frame` 类型；`PreparedSend`/`ReceiveBuffers` 仅是 wire ownership 对象，不进入 Env/Rollout 数据协议。

Changed files:

- `rlinf/workers/trajectory/transport.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_transport.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_transport.py`
  result: 21 passed。
- command: `/opt/venv/openpi/bin/python -m pytest -q` for SG-01..06 unit tests
  result: 92 passed。
- command: `/opt/venv/openpi/bin/ruff check` for SG-01..06 Python files
  result: passed；仓库配置对未启用 preview 的 rules 输出已有 warning。
- command: `/opt/venv/openpi/bin/ruff format --check` for SG-01..06 Python files
  result: passed。
- command: Python 3.10 `py_compile` for SG-06 Python files
  result: passed。
- command: `git diff --no-index --check` for SG-06 new/untracked files and this document
  result: passed（exit 1 表示与 `/dev/null` 存在预期 diff；没有 whitespace diagnostic）。
- command: `rg 'pickle|numpy|np\.' rlinf/workers/trajectory/transport.py`
  result: no matches。

Known limitations:

- 本阶段用 `copy_` 模拟 data plane，尚未验证 RLinf collective 的 header-first 两阶段 send/recv；这是 SG-07/SG-09 的真实 Worker integration 验收项。
- decode 输出引用 receive buffers；buffer pool 在 Storage 不再引用前不得复用。pool lease、ack strength、drain 与 backpressure 属于 SG-07/SG-13。
- CPU staging/pinning 尚未实现；sender 会明确拒绝 CUDA 或 non-contiguous tensors，避免把隐式拷贝成本藏进 critical path。
- batch string lists 没有 fixed encoding；SG-08 必须根据真实 PolicyInput 决定在 Env tokenization、固定 UTF-8 lane 或其他明确方案之间选择。
- StorageWorker 进程必须在 configure 时加载对应 ForwardInputs registry；该 lifecycle/import wiring 属于 SG-07。
- 当前逐 field tensor calls 的网络 latency、header-first round trips 和 lane grouping 尚未测量，不能作为性能结论；SG-15 负责基于目标数据量实验。

Next gate:

- 人工验收 schema/header、zero-copy ownership、retry/ack 语义和测试；通过后把当前阶段切换为 SG-07。

### 2026-07-19 — SG-05

Status: accepted

Decisions:

- 新增独立 `compute_embodied_gae()`，只接受 transition-aligned `[E,S,B,...]` 数据，不通过 shape 猜测兼容 legacy `T+1` 时间轴。
- 每个 rollout epoch 独立反向递推；中间 alive transition 使用下一 `state_values`，最后 alive transition 使用 `tail_values`，truncation 使用同 transition 的 `timeout_values`。
- done 会同时切断 continuation value 与 GAE recursion，因此 auto-reset 后的 state value 不会泄漏到上一 episode；true termination 不消费 boundary value。
- timeout/tail validity masks 与 reduced truncation/final-alive masks 必须精确相等；done 必须等于 termination/truncation 的并集，且同一 macro transition 不允许同时 termination 与 truncation。
- `compose_embodied_rewards()` 分别归约 env/external action-chunk rewards，再按显式权重组合，避免 `[B,1]` external reward 广播到 5 个 primitive actions 后被重复求和。
- external validity mask 使用 `torch.where`，无效位置即使是 NaN 也不会污染 effective reward；raw inputs 与 Storage 内容不修改。
- advantages/returns 输出 `[E,S,B,1]`；normalization 支持 macro 或 action-chunk loss mask，并且不改变 returns。
- legacy registry、现有 Actor 调用与普通 channel 未修改；SG-12 接入新 API，
  SG-16 决定继续按 execution path 隔离，不删除旧 `T+1` GAE。

Changed files:

- `rlinf/algorithms/embodied_gae.py`
- `tests/unit_tests/test_embodied_gae.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_embodied_gae.py`
  result: 12 passed。
- command: `/opt/venv/openpi/bin/python -m pytest -q` for SG-01..05 unit tests
  result: 71 passed。
- command: `/opt/venv/openpi/bin/ruff check` for SG-01..05 Python files
  result: passed；仓库配置对未启用 preview 的 rules 输出已有 warning。
- command: `/opt/venv/openpi/bin/ruff format --check` for SG-01..05 Python files
  result: passed。
- command: Python 3.10 `py_compile` for SG-05 Python files
  result: passed。
- command: `git diff --no-index --check` for SG-05 new/untracked files and this document
  result: passed（exit 1 表示与 `/dev/null` 存在预期 diff；没有 whitespace diagnostic）。

Known limitations:

- 本阶段只实现目标 OpenPI PPO 所需的 chunk-level critic。action-level GAE 需要每个 primitive state 的 value，不能只靠当前每个 action chunk 一个 `state_value` 推断。
- external rewards 必须在调用前已经对齐到 transition coordinates；`history_buffer` 的跨 transition 分配策略需要在 SG-12 结合真实 RewardWorker 语义实现，不能由本函数猜测。
- 新 API 尚未接入 Actor/registry，这是为避免 SG-12 前同时改变普通 channel 行为；真实端到端等价性仍需 SG-12 验证。

Next gate:

- 已人工验收；SG-06 可以开始。

### 2026-07-19 — SG-04

Status: accepted

Decisions:

- `StorageConfig` 冻结一个 generation/local shard 的 epoch、transition、slot coverage，以及 Env/Rollout/Value optional fields 和 external reward coordinates。
- optional fields 使用 exact schema；目标 OpenPI PPO 明确要求 `forward_inputs`、`prev_logprobs` 和 `state_values`，没有把宽泛 optional 默认为合法缺失。
- `TrajectoryStorage.write()` 只接受 `EnvResult`、`RolloutResult`、`RewardResult` 和 `ValueResult`；返回 `False` 表示 exact duplicate retry，identity 内容冲突和 slot overlap 明确失败。
- readiness 先要求每个 transition 的完整 Env/Rollout coverage，再根据 truncation/final-done masks 动态要求 timeout/tail values。value 先于 Env 到达仍合法；多余 boundary value 会使 trajectory 不 ready。
- `TrajectoryBatch` 按 `[E,S,B,...]` 导出 raw tensors；OpenPI `ForwardInputs` 按 E→S→本地 slot 顺序折叠为 `[E*S*B,...]` 并通过具体 schema 重建。
- external rewards 使用独立 tensor 与 coverage mask；Storage 不组合 reward，也不应用 bootstrap。
- `ForwardInputs` 增加抽象 `from_model_inputs()`，使 Storage 不依赖 OpenPI concrete class 也能重建 typed batch。
- placement 到 logical data-owning ranks 的推导不属于 SG-03/SG-04，明确留给 SG-07 configure，并由 SG-12 的真实 placement 验证。

Changed files:

- `rlinf/data/forward_inputs.py`
- `rlinf/workers/trajectory/storage.py`
- `rlinf/workers/trajectory/__init__.py`
- `tests/unit_tests/test_trajectory_storage.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_storage.py`
  result: 13 passed。
- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_trajectory_storage.py tests/unit_tests/test_openpi_forward_inputs.py tests/unit_tests/test_trajectory_data.py tests/unit_tests/test_trajectory_route_plan.py`
  result: 59 passed。
- command: `/opt/venv/openpi/bin/ruff check` for SG-01..04 Python files
  result: passed；仓库配置对未启用 preview 的 rules 输出已有 warning。
- command: `/opt/venv/openpi/bin/ruff format --check` for SG-01..04 Python files
  result: passed。
- command: Python 3.10 `py_compile` for SG-04 Python files
  result: passed。
- command: `git diff --no-index --check` for SG-04 new/untracked files and this document
  result: passed（exit 1 表示与 `/dev/null` 存在预期 diff；没有 whitespace diagnostic）。

Known limitations:

- 本阶段是 correctness baseline；merge/export 会分配输出 tensors，尚未实现 buffer pool、zero-copy receive ownership 或性能优化。
- 成功写入后 Storage 按只读 ownership 保存对象引用，不 clone tensor；SG-06/SG-09 必须保证 ack 前后的 receive buffer 生命周期与不可变性。
- `reward_mode=terminal/history_buffer` 的实际 algorithm alignment 仍需迁移时提供明确 `reward_steps`；reward composition 属于 SG-05/SG-12，不在 Storage 内猜测。
- source address、sequence、ack、placement-derived ranks、Worker 生命周期和跨进程故障传播分别属于 SG-06、SG-07、SG-09 和 SG-12。

Next gate:

- 已人工验收；SG-05 可以开始。

### 2026-07-19 — SG-03

Status: accepted

Decisions:

- `RoutePlan` 只接收 `total_slots` 和参与方 rank 数；Env、Rollout、Reward、Storage 与 Actor 使用同一套 contiguous balanced partition。
- 不整除时前 `total_slots % world_size` 个 rank 各多一个 slot；该规则与 list batch 的常用 balanced split 一致，并在文档与测试中固定。
- `Route` 保存 destination rank、source batch indices、destination-rank local indices 和 global slot IDs，没有引入 transport 或 Worker 地址。
- `routes()` 处理 source rank 的完整连续 batch；`route_slots()` 处理稀疏 batch，并强制 source ownership。
- `slot_range()`、`owner()`、`local_index()` 和 `global_slot()` 提供 ownership 与 global/local 双向转换。
- world size 大于 slot 数时允许空 shard；空 source shard 返回空 routes，不产生空 record。
- ownership tables 在构造时预计算；当前配置的 slot 数很小，换取 steady-state 路由不做区间搜索。

Changed files:

- `rlinf/workers/trajectory/__init__.py`
- `rlinf/workers/trajectory/route_plan.py`
- `tests/unit_tests/test_trajectory_route_plan.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `python -m pytest -q tests/unit_tests/test_trajectory_route_plan.py`
  result: 12 passed。
- command: `python -m pytest -q tests/unit_tests/test_trajectory_route_plan.py tests/unit_tests/test_openpi_forward_inputs.py tests/unit_tests/test_trajectory_data.py`
  result: 46 passed。
- command: `ruff check rlinf/workers/trajectory tests/unit_tests/test_trajectory_route_plan.py`
  result: passed。
- command: `ruff format --check rlinf/workers/trajectory tests/unit_tests/test_trajectory_route_plan.py`
  result: passed。
- command: Python 3.10 `py_compile` for all three SG-03 Python files
  result: passed。
- command: `git diff --no-index --check` for SG-03 new/untracked files and this document
  result: passed。

Coverage:

- 不整除 partition、world size 大于 slot 数产生空 shard、source batch 跨部分或全部 Storage ranks。
- 稀疏乱序 slot batch 的 source-order 恢复、Storage→Actor 映射、global/local index 可逆。
- 性能参考规模 32 slots/4 Storage ranks，每 rank 稳定拥有 8 slots。
- 穷举 `total_slots=1..12`、source/destination world size `1..6` 的 slot coverage 与索引不变量。
- pickle round-trip 后 ownership 和 routes 不变，保证计划能够跨 Ray/process control plane 传递。
- 调用方修改 `world_sizes` 返回值不会改变已构建 plan 的 ownership tables。

Known limitations:

- `world_sizes` 表示拥有独立数据 shard 的逻辑 ranks。SG-07/SG-12 从实际 WorkerGroup 和 placement 构建计划时，必须区分 data-owning rank 与纯 model-parallel rank，不能直接无条件使用物理进程数。
- 本阶段只计算路线，不执行 tensor/data-object select，也不连接 Worker；分别属于 SG-04、SG-08/09。

Next gate:

- 已人工验收；SG-04 可以开始。

### 2026-07-19 — SG-02

Status: accepted

Decisions:

- `ForwardInputs` 基类只定义 `validate()`、`tensor_fields()`、`select()`、`to_model_kwargs()` 和 batch/schema identity，不承担 transport 或 storage 行为。
- registry 使用 `(schema_name, schema_version)`，当前注册 `openpi_libero` version `1`。
- `OpenPILiberoForwardInputs` 精确对应目标 PPO 配置产生的九个 tensor leaves；其他 OpenPI 环境、NFT 和 DSRL 不作为 optional leaves 混入 v1。
- `tensor_fields()` 使用模型现有 flat dict key，并保持固定顺序；`to_model_kwargs()` 直接恢复 `model(forward_inputs=...)` 所需调用形式。
- `RolloutResult.forward_inputs` 只接受 `ForwardInputs`，从数据协议层拒绝任意 nested dict。
- validation 和 `to_model_kwargs()` 不移动或复制 tensor；`select()` 是唯一有意产生所选 batch tensor 的操作。

Changed files:

- `rlinf/data/forward_inputs.py`
- `rlinf/data/trajectory.py`
- `rlinf/models/embodiment/openpi/forward_inputs.py`
- `tests/unit_tests/test_openpi_forward_inputs.py`
- `tests/unit_tests/test_trajectory_data.py`
- `tests/e2e_tests/embodied/verify_openpi_forward_inputs.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `/opt/venv/openpi/bin/python -m pytest -q tests/unit_tests/test_openpi_forward_inputs.py tests/unit_tests/test_trajectory_data.py`
  result: 34 passed。
- command: `ruff check` for all five SG-02 Python files
  result: passed。
- command: `ruff format --check` for all five SG-02 Python files
  result: passed。
- command: Python 3.10 `py_compile` for all five SG-02 Python files
  result: passed；该 Python 3.10 环境没有 PyTorch，因此 runtime tests 使用 `/opt/venv/openpi` 的 Python 3.11 环境。
- command: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/public/daibo/timeline/0703/RLinf /opt/venv/openpi/bin/python tests/e2e_tests/embodied/verify_openpi_forward_inputs.py --model-path '/mnt/public/daibo/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT' --device cuda:0 --batch-size 2`
  result: passed；真实 checkpoint 完成 `predict_action_batch()` → typed conversion → raw/typed `default_forward()` → select/reassembly。raw 与 typed model outputs 逐 tensor `torch.equal`，所有 selected fields 重组后逐 tensor `torch.equal`。
- command output: actions/logprobs `[2, 5, 7]`，values `[2]`；schema fields 为 chains `[2, 5, 50, 32]`、denoise indices `[2, 4]`、tokens/mask `[2, 48]`、action `[2, 35]`、model action `[2, 1600]`、两路 image `[2, 256, 256, 3]` 和 state `[2, 8]`。
- command: `git diff --no-index --check` for new/untracked implementation, tests and this document
  result: passed。

Known limitations:

- schema registration 在导入具体模型 schema module 时发生；SG-06 endpoint 注册需要显式导入配置选择的 schema，不能依赖隐式全局 discovery。
- 当前只覆盖 `libero_spatial_ppo_openpi`；其他 OpenPI 配置需要各自固定 schema，不得无校验地复用 v1。

Next gate:

- 已人工验收；SG-03 可以开始。

### 2026-07-19 — SG-01

Status: accepted

Decisions:

- 新增一个共享 `TrajectoryData`、两个 live-path 类型和五个 bypass/value 类型；不新增 key、metadata、observation、frame、message 或 envelope wrapper。
- `PolicyInput`/`PolicyOutput` 强制 live path 只传动作生成与环境推进必需字段；`EnvResult`、`RolloutResult` 和 `RewardResult` 直接进入 Storage，不经过 Env↔Rollout live path。
- rollout 坐标和 `slot_ids` 直接位于数据对象；source endpoint 和 wire sequence 不进入业务数据。
- timeout 与 alive-tail 共享 `ValueRequest`/`ValueResult`，用两个受校验的字符串值区分，不新增 enum class。
- observations 和 `forward_inputs` 继续使用 nested dict；模型级 `ForwardInputs` 属于 SG-02。
- 构造时执行结构校验，但不移动、复制或 canonicalize tensor。
- 基础 dataclass 校验字段内部不变量；配置决定的 required fields 在 SG-04 注册 storage schema 时强制。

Changed files:

- `rlinf/data/trajectory.py`
- `tests/unit_tests/test_trajectory_data.py`
- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `python -m pytest -q tests/unit_tests/test_trajectory_data.py`
  result: 24 passed。
- command: `ruff check rlinf/data/trajectory.py tests/unit_tests/test_trajectory_data.py`
  result: passed。
- command: `ruff format --check rlinf/data/trajectory.py tests/unit_tests/test_trajectory_data.py`
  result: passed。
- command: Python 3.10 `py_compile` for implementation and tests
  result: passed；该 Python 3.10 环境没有 PyTorch，因此 runtime tests 使用项目 Python 3.11 环境。
- command: `git diff --no-index --check` for all three new/untracked files
  result: passed。

Known limitations:

- 当前名称是最小实现提案，仍需人工确认是否符合项目语言；代码没有 transport-derived class 名。
- OpenPI `ForwardInputs` 的具体 required leaves 和 select/to-model behavior 尚未实现，属于 SG-02。
- 场景 schema 尚未实现，因此直接构造 `RolloutResult` 时允许省略 logprobs/value/forward inputs；目标 PPO 的必选约束将在 SG-04 从配置注册，不能由调用方自行决定。

Next gate:

- 已人工验收；SG-02 可以开始。

### 2026-07-19 — SG-00

Status: accepted

Decisions:

- Live path 只保留 policy inference 输入和 executable actions；training data 直接 bypass 到独立 StorageWorker。SG-01 最终采用 `PolicyInput`/`PolicyOutput` 表达该边界。
- Reward inference 是辅助非关键路径；RewardWorker 输出直接进入 Storage，effective reward 在 Actor advantage 准备阶段组合一次。
- Env 触发 timeout/segment-tail value request，Rollout 复用实际 `value_head` 路径，response 直接进入 Storage。
- 新协议不定义聚合式 Env/Rollout bootstrap；环境 transition、timeout value 和 alive-tail value 使用独立语义，具体类型名留给 SG-01。
- transition masks 使用 `[T]`；state values 使用 `[T]`；timeout values 按 transition 对齐；alive tail value 单独保存。
- 逻辑时间坐标为 `(generation_id, rollout_epoch, chunk_step)`；`stage_id` 只属于 source/route metadata。
- logical write identity 还需要 data role、source 和 `slot_ids`；wire sequence/schema 与逻辑身份分离。
- Python 3.10 使用 `str, Enum` 和 `typing_extensions.Self`。

Changed files:

- `docs/notes/trajectory_channel_reimplementation.md`

Verification:

- command: `EMBODIED_PATH=... python examples/embodiment/train_embodied_agent.py --config-name libero_spatial_ppo_openpi --cfg job --resolve`
  result: success；确认 64 envs、8 epochs、240 env steps、5-action chunks、48 transitions/epoch、OpenPI value head enabled、train auto-reset disabled。
- command: `git diff --check`
  result: success。

Known limitations:

- 目标仓库没有 trajectory-channel 配置；32-env/4-shard 性能规模来自参考仓库，最终配置在 SG-12 创建。
- 历史 3.512 GiB/shard 数字没有归档原始 artifact，只能作为 SG-15 的待复核估计。
- 用户主动删除了旧 trajectory 草案；本轮按 clean-slate 设计，没有恢复或修改运行时代码。

Next gate:

- SG-01 已获准开始；旧骨架和草案是用户主动删除的，不恢复也不兼容。

## 12. Benchmark Artifact Contract

每次正式 benchmark 使用独立目录：

```text
artifacts/trajectory_channel/<run_id>/
    manifest.yaml
    resolved_config.yaml
    git_status.txt
    environment.txt
    commands.txt
    raw/*.json
    raw/*.csv
    report.md
```

`manifest.yaml` 至少记录 commit、dirty diff digest、host、CPU/NUMA、GPU、NIC、Python/PyTorch/Ray、placement、payload schema、warmup/repeat 次数和限速配置。

历史实验只支持以下定性结论，精确数字必须重新执行后才能引用：

- 同 event loop 的大规模 assembly 会显著增加 live-path tail latency；
- thread compression 仍可能通过 CPU、allocator、GIL 和 memory bandwidth 干扰 live path；
- 独立 StorageWorker process 是当前首选隔离边界；
- pickle IPC 原型不能作为生产 transport；
- 小 action payload 应优先减少 hop，大 image payload才值得比较压缩。

## 13. 新对话启动指令

把下面内容交给新的 Codex 对话即可继续：

```text
在 /mnt/public/daibo/timeline/0703/RLinf 工作。
完整阅读 docs/notes/trajectory_channel_reimplementation.md。
检查 git status，保留已有修改。
只执行文档“当前阶段”标记的 subgoal，不实现后续 subgoal。
先把本 subgoal 的范围、设计决定和验收命令更新到文档，再修改代码。
完成测试后把验收记录写回文档并停止，等待人工验收。
若代码事实与文档冲突，以代码证据为准，先更新 decision log，不要静默改变协议。
```

## 14. Decision Log

### 2026-07-19 — 创建实施基线

- 使用单一 Markdown 保存跨对话上下文、设计决定、subgoal 和验收记录。
- 由当前对话先执行 SG-00；文档达到自洽后，后续 subgoal 可以安全转交新对话。
- live channel 与 record storage 使用独立 worker process。
- 第一版 transport 先完成 raw fixed-frame，再接入无损压缩。
- bootstrap/reward 语义以“raw reward 不变、Actor advantage 唯一应用”为目标。

### 2026-07-19 — 完成 SG-00 语义审计

- 当前 `T+1 values/dones` 来自每个 epoch 末额外一次完整 policy forward；目标改为 `[T]` transition data 加显式 tail value。
- 当前 Env 可对 auto-reset truncation 修改 reward；目标由 Actor advantage 使用 timeout value 一次性表达，非-auto-reset truncation 也使用 terminal observation value。
- Env 是 boundary observation 的原始所有者，因此由 Env 触发 ValueRequest；Storage 不理解 Rollout placement。
- `stage_id` 是 source batch partition，不是算法时间身份，因此不进入逻辑时间坐标。
- external reward 不需要返回 Env 参与 trajectory assembly。

### 2026-07-19 — 完成 SG-01 数据协议

- 数据对象只表达业务数据和 rollout 坐标；transport schema、sequence、buffer ownership 留在 transport 层。
- live policy 输入/输出与 Env、Rollout、Reward 的 storage 结果使用不同类型，禁止训练字段绕经 Env。
- 使用一个 value kind 字段复用 request/result 类型，避免 timeout/tail 各造一套 class。
- required/optional 分为结构层和配置 schema 层，避免基础数据类绑定某个算法。
- validation 不隐式进行 CPU copy 或 contiguous conversion。

## 15. 未决问题

这些问题必须在标注的 subgoal 内关闭：

| 问题 | 截止阶段 |
|---|---|
| Actor output 采用 pull、push 或 bounded mailbox | 已在 SG-11 关闭：Actor pull + Storage direct response，无 mailbox |
| optional field 使用多 schema 还是 validity mask | SG-01/SG-06 |
| Storage active trajectory 上限和过期策略 | SG-13 |
| LZ4/Zstd/block threshold/CPU affinity | SG-14/SG-15 |
| checksum 默认开启还是 debug-only | SG-06/SG-13 |

可以使用，但我建议先做 1–2 个 global step 的 canary，不要直接把正式 200-step 实验整体切过去。

## 如何配置

核心关系是：

```text
total_num_envs = 物理 EnvWorker/GPU 数 × 每卡 VectorEnv slot 数
```

如果仍然有 20 个物理 EnvWorker，每卡 4 个环境：

```yaml
env:
  train:
    total_num_envs: 80
    num_env_subprocess: 1
    skip_intermediate_obs_in_chunk: true
    subpool:
      outcome_group_size: 20
      dynamic_updates: false
      state_cache_size: 20

algorithm:
  outcome_dynamic_sampling:
    enabled: true
    parallel_groups: true
    group_size: 20
    groups_per_update: 20
```

含义是：

- 20 张环境 GPU。
- 每张 GPU 一个 OmniGibson 进程。
- 每个进程内部包含 4 个 vector scene。
- 80 个 slot 一轮并行采集 4 个 logical state group。
- 20 个 init states 最少需要 5 轮，而不是以前的 20 轮。
- 每个 state 仍有 20 条 trajectory。
- 每次更新仍是 `20 states × 20 trajectories = 400 trajectories`。
- 因此 40 chunks/trajectory 时，`GBS=16000` 不需要改变。

基于现有脚本可这样启动 canary：

```bash
export B1K_RL_REPO=/mnt/public/daibo/timeline/0914/RLinf-vector-env

bash toolkits/b1k_grounded/run_radio_pickup_canonical20_rl.sh \
  env.train.total_num_envs=80 \
  +env.train.num_env_subprocess=1 \
  env.train.subpool.outcome_group_size=20 \
  algorithm.outcome_dynamic_sampling.parallel_groups=true \
  algorithm.outcome_dynamic_sampling.group_size=20 \
  algorithm.outcome_dynamic_sampling.groups_per_update=20 \
  runner.max_steps=2 \
  runner.save_interval=1
```

当前正式脚本的 `MBS=100 / GBS=16000 / policy epoch=1 / critic epoch=5` 可以保持。

### Failure-state save

追加：

```bash
env.train.subpool.failure_state_capture.enabled=true \
env.train.subpool.failure_state_capture.output_dir=/path/to/failure_states \
env.train.subpool.failure_state_capture.run_id=vector4-canary
```

每个 slot 会独立保存 terminal state。跨机房 collector 保存到远端机器自己的 `/mnt/public`，实验结束后再 rsync 回来。

跨机房目前继续使用：

```yaml
dynamic_updates: false
```

因为远端 catalog 没有集中一致性协议。nxb 本地共享文件系统上的纯本地实验可以开启 dynamic updates。

### 远端 collector

collector 必须显式加载修复后的 OmniGibson，避免误用 `/opt/venv` 中的旧版本：

```bash
python toolkits/b1k_grounded/manage_remote_collectors.py start \
  --gpus 0,1,2,3 \
  --ports 46100,46101,46102,46103 \
  --python /opt/venv/openpi/bin/python3 \
  --repo /mnt/public/daibo/timeline/0914/RLinf-vector-env \
  --omnigibson-path \
    /mnt/public/daibo/timeline/0914/BEHAVIOR-1K-vector-env/OmniGibson \
  --log-dir /mnt/public/daibo/results/b1k_collectors/vector4
```

我刚补上了 `--omnigibson-path`，并完成 120 项回归测试。

## Dynamic batching 怎么开

第一轮 canary 建议先关闭 dynamic batching，只验证 VectorEnv：

```yaml
runner.enable_decoupled_mode: false
rollout.dynamic_batching.enabled: false
```

原因是以前一个 collector shard 只有 1 个 observation；现在一个 shard 包含 4 个。旧的：

```yaml
max_batch_size: 5
```

现在可能形成 20 张图像 observation 的推理 batch，显存和延迟特征完全不同。

VectorEnv 确认稳定后，再从下面开始：

```yaml
runner.enable_decoupled_mode: true
rollout.dynamic_batching:
  enabled: true
  max_batch_size: 1
  max_wait_seconds: 0.05
```

随后再测试 `max_batch_size=2`。不要直接沿用原来的 5。

## 像素灰度差异会不会影响训练

我的判断是：大概率不会成为主要问题，但不能说影响严格为零。

关键事实：

- 没有使用 JPEG 或有损压缩；当前 zlib 是无损的。
- 没有调整相机分辨率、画质或 renderer 参数。
- 单环境重复恢复同一个 snapshot，本身就会产生：
  - 主相机平均绝对差异：`5.57–7.14 / 255`
  - 腕部相机：`2.36–2.39 / 255`
- vec4 中：
  - 主相机约 `6.94–10.01 / 255`
  - 腕部约 `6.92–8.92 / 255`
- grounding bbox 仅偏移约 `1–4 / 1024` 个量化坐标。
- 排除 2025 数据已知无效的 `joint_qeffort` 后，同状态 proprio 的平均差异只有 `1e-4–2.4e-3`。

因此，大部分 RGB 差异属于 RTX/ray-tracing 的正常非确定性，会表现为轻微 observation noise，而不是视角错位或状态错误。它甚至可能有一点数据增强效果。

值得注意的是腕部相机在 vec4 下的差异比单环境自然波动更大，所以正式使用前应做一个 slot-conditioned AB：

- 固定同一个 checkpoint。
- 20 个 held-out states。
- 每个 vector slot 至少 100 条 trajectory，总计 400 条。
- 分别记录 slot 0/1/2/3 的成功率。
- 同时用 scalar env 做 400 条基线。

在 400 条规模下，我建议：

- vector 总成功率与 scalar 差距不超过约 7 个百分点；
- 任意两个 slot 的差距不超过约 10 个百分点；
- 不出现某个 slot 持续特异性失败；
- 最终 checkpoint 必须继续用 scalar held-out eval 作为真实指标。

如果 slot 3 持续明显更差，就先降为每卡 2 env，而不是修改画质或容忍系统性视觉偏差。

综合来看，VectorEnv 可以进入 canary。当前预期收益约 26%，像素噪声不是阻止使用它的理由，但 scalar held-out eval 必须始终作为最终裁判。

## 子集 step 的全局 PhysX 约束

`env_indices` 只能筛选 action 写入以及对应 slot 的 observation、reward 和
termination 计算，不能临时裁剪 `og.sim._scenes`。OmniGibson 的 articulation
和 gripper contact tensor view 是在完整共享 stage 上建立并按全局 scene
index 索引的；修改 scene registry 会让 controller callback 与这些 view 的
索引生命周期不一致。已完成 slot 仍会随共享 stage 发生物理推进并执行旧的
controller goal，因此它必须保持逻辑冻结，且在完整 restore 前不得再次启用。

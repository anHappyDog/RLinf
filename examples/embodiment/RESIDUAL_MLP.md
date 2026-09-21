# Residual MLP A / 通用残差 MLP A

Train a bounded, always-on residual around a frozen VLA with the existing TD3,
replay, target-network and FSDP infrastructure. Start from the clean committed
`279011e8ff3e52a71e91f58c61f3348feafd3786` code. The abandoned uncommitted
advantage experiments are not included. No PPO advantage implementation changes.

基于冻结 VLA 训练小型有界 residual，复用 TD3、replay、target network 和 FSDP。
代码基线为已提交的 `279011e8`，不包含 shapingv2 工作区未提交的 advantage 实验。

## Policy And Data / 策略与数据

`action = reference + epsilon * tanh(MLP(pooled_VLM, proprio, reference))`.
The output layer starts at zero. All action dimensions and chunk positions are
trainable; the example retains 23 dimensions and 32 steps. The raw 2048-dimensional
masked VLM mean and normalized padded proprio are frozen replay features. The MLP's
first layer supplies the trainable feature projection; there is no cached trainable
projection and no dependency on learned RLT weights.

输出层零初始化，全部动作维度与 chunk 位置均可训练。缓存冻结 VLM 的带 mask 均值
和归一化 proprio；MLP 输入层承担可训练投影，不缓存其训练中变化的输出。

The reference uses the existing eval ODE sampler with 10 iterations, including its
initial Gaussian noise. Evaluation disables residual exploration; it does not remove
VLA sampling randomness. Initial-action equality requires the same observation and
VLA initial noise. Reference actions have already passed through the VLA output
transform. Epsilon is in **environment command units**, not normalized model units;
there is no second unnormalization and no absolute `[-1, 1]` action clipping.

参考动作沿用原 ODE eval sampler（10 次迭代），保留 VLA 初始采样噪声。零残差一致性
要求 observation 和 VLA 初始噪声相同。epsilon 使用环境控制输入单位；不重复反归一化。

The example's explicit 23-value epsilon is `max(0.1 * action_std, 0.001)`, using
`rollout.rlt_feature_model.openpi_data.norm_stats_path`. This is an initial experiment
setting, not a tuned physical tolerance. The model's actual normalization uses
quantiles; the standard deviation here only sets the residual bound. In BEHAVIOR,
commands include base velocity-controller inputs, absolute joint positions and
gripper commands. Review bounds for a different robot/task.

示例 epsilon 来自 action std 的 10%，下限为 0.001，仅为初始实验设置。VLA 本身使用
quantile normalization；这里 std 只用于设置残差范围。更换机器人时需要重新设置各维边界。

Replay records full commands, per-primitive rewards, actual execution masks,
terminations/truncations, and current/next frozen features. Padded nonexecuted rewards
are excluded. The target uses `sum(gamma^i * r_i) + gamma^k * Q_target`, with actual
executed length `k`. Terminations stop bootstrap. `algorithm.residual.bootstrap_truncation`
explicitly controls timeout bootstrap (default true, matching the Sunday standard
bootstrap convention); timeout targets use the pre-reset final observation.
Auto-reset observations are used for the next episode's actions, not for the previous
transition's target. The rollout's final unexecuted policy output is not stored as an action.

Replay 按真实执行长度折扣；terminal 不 bootstrap，timeout 由显式配置控制，默认开启。
Auto-reset 前的 final observation 用于上一条 transition；reset 后 observation 用于下一回合。
保留原任务 completion、manifest reward 与 termination，不新增稳定抓持或碰倒判据。

No teleoperation/correction dataset is required. Existing logs/videos/checkpoints
cannot replace transitions. Reuse the initial-state manifests and frozen checkpoint;
collect fresh replay when training starts. Uniform sampling covers recent 50,000
transitions without pickup-specific windows. Replay is saved automatically; this
window limits sampling, not total files retained on disk. Keep the frozen checkpoint,
feature extraction, action transforms and epsilon unchanged when resuming replay.
Other environments must supply accurate execution masks and final observations.

无需遥操数据。训练启动后收集新 replay，复用原 checkpoint 和初始状态池。均匀采样最近
50,000 条 transition，不使用 pickup 专属窗口。自动保存 replay；采样窗口不是磁盘容量上限。
恢复训练时保持冻结模型、特征提取、动作变换与 epsilon 不变。

## Training Schedule / 训练顺序

The first rollout version uses only the frozen reference, without consuming residual
exploration noise. Subsequent rollouts use bounded exploration (sigma 0.1 in units of
epsilon). Once replay has 512 rows per actor rank, run 100 critic-only updates. Then
update the actor every two critic updates with `-min(Q1,Q2) + 0.01 * mean((delta/epsilon)^2)`.
Target smoothing has sigma 0.1, noise clip 0.25 and a final residual bound of ±epsilon.
The example caps each training call at 100 updates. These are initial settings, not
claims of improved success rate. No gate, expert takeover, BC data or VLA update.

首轮仅 base；之后有界探索。每个 actor rank 的 replay 满 512 条后预热 critic 100 次，
然后每两次 critic 更新一次 residual actor。参数为待验证的初始设置，不代表已获得成功率提升。

## Run It / 运行

Use the existing `behavior_openpi` environment. No dependency installation is needed.
The launcher sets `TMPDIR=/mnt/public/daibo/tmp` and host-qualified
`XDG_CACHE_HOME=/mnt/public/daibo/cache/<hostname>/residual_mlp_a`, with TorchInductor
and Triton caches beneath it. Set the same TMPDIR before starting Ray and remote
collectors; the collector manager explicitly assigns each daemon its own XDG cache
under its host/session/GPU appdata directory, including when tmux already exists.

启动脚本将临时文件放在 `/mnt/public/daibo/tmp`，XDG、TorchInductor、Triton 缓存按主机隔离。
启动 Ray 和 remote collector 前也必须设置相同 TMPDIR；collector 的 XDG 缓存额外按会话和 GPU 隔离。

The launch script resolves the verified checkpoint:

```text
/mnt/public/daibo/results/b1k_grounded_control_v01/rl/radio_pickup_canonical20_resume_rlstep10_gaelambda1_h1280_sde_dapo_groups20x20_traj400_mbs100_gbs16000_actionratio_kl001_fp32master_stateattention_p1_c5_crossdc20_steps15_v1/behavior_subpool_ppo_pi05/checkpoints/global_step_15/actor/model_state_dict/full_weights.pt
```

This initializes only the frozen VLA. `runner.resume_dir` is null for a new residual
experiment; do not set it to the PPO checkpoint. Missing base-model keys are fatal;
unused PPO value-head keys are expected.

该 checkpoint 仅初始化冻结 VLA。新 residual 实验的 `runner.resume_dir` 保持 null，
不恢复 PPO optimizer；加载缺失基础模型权重会报错，额外 value-head 权重不参与训练。

From this worktree, inspect the resolved configuration without launching workers:

```bash
bash examples/embodiment/run_residual_mlp_a.sh --config-only
```

Run CPU tests with GPU visibility disabled:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" \
  /mnt/public/daibo/venv/behavior_openpi/bin/python -m pytest \
  tests/unit_tests/test_residual_mlp.py \
  tests/unit_tests/test_embodied_rollout_batch_alignment.py -q
```

When you choose to run training, use a fresh output directory and an appropriately
configured Ray cluster. The default placement is one node: actor/rollout GPU 0,
environment GPU 1, one single-scene environment. Override `cluster` for your actual
hardware; the script does not start/stop Ray or collectors. Simulator assets and the
audited OmniGibson source must already be available to the workers.

实际训练前选择新的结果目录，按硬件修改 placement 并准备好 Ray、模拟器资产和审计过的
OmniGibson 环境。默认单节点：actor/rollout 使用 GPU 0，单场景 env 使用 GPU 1。
脚本不管理 Ray 或 collector 生命周期。

```bash
export B1K_SUBPOOL_RESULT_DIR=/mnt/public/daibo/results/b1k_grounded_control_v01/residual_mlp_a_v1
bash examples/embodiment/run_residual_mlp_a.sh
```

For evaluation after training, set `runner.only_eval=true` and `runner.ckpt_path`
to the residual actor's saved full weights. The example eval manifest uses the
held-out initial-state pool. Report that separately from train-pool success; compare
to zero residual with the same VLA sampler and state/noise seeds.

评估时加载 residual actor 权重，使用同一冻结 VLA。held-out 结果与训练池成功率分开报告；
与零残差基准使用相同状态池和 VLA 采样设置。

## Verification / 验证范围

FSDP must wrap `BoundedResidualActor` and `TwinQCritic` separately. A critic-only
backward with an unwrapped actor at the root can leave its original parameters
temporarily unregistered, breaking target EMA. The GPU regression test executes
100 critic-only updates followed by 20 delayed-actor updates (10 actor steps),
including target EMA throughout. Run it with
`python -m pytest tests/unit_tests/test_residual_mlp_fsdp.py -q -s` on one free GPU.

FSDP 必须分别包装 actor 和 twin-Q。GPU 回归测试覆盖 100 次 critic-only 更新及后续
20 次延迟 actor 更新（actor 实际更新 10 次），每步都检查 target EMA。

CPU tests cover full-action zero parity/bounds, masked feature pooling, synthetic
reference sampler parity, losses and parameter updates, timeout/final-observation
replay, storage sampling, strict checkpoint rejection and config/model factory loading.
The real step-15 checkpoint has 685 tensors; meta-device inspection matched all 667
base VLA keys/shapes, leaving 18 PPO value-head tensors. A subsequent A100 GPU check
loaded those real weights and passed exact reference-sampler and zero-residual action
equality with the same initial noise and synthetic blank camera inputs. Peak allocated
memory was 6.615 GiB. No simulator or training experiment was run; distributed FSDP and
real-observation VLA parity remain runtime checks.

真实权重的 A100 推理检查通过：相同噪声和合成空白图像下，参考 sampler 与零残差动作
均逐元素完全一致，峰值分配显存 6.615 GiB。尚未验证 FSDP 或模拟器联调；未启动训练实验，
也不对退化根因或 residual 的实际成功率作结论。

Validation on 2026-09-21: **69 CPU tests passed** (11 residual tests plus 58 existing
batch-alignment, Behavior subpool/reward and bootstrap tests). Ruff lint/format,
`git diff --check`, shell syntax and Hydra config resolution passed. All configured
checkpoint, normalization, token-map and train/held-out manifest paths exist. Both
original workspaces retain their pre-task Git status. The implementation is isolated on branch `feat/residual-mlp-a`.

2026-09-21 验证：69 项 CPU 测试通过；Ruff、shell 语法、Hydra 配置解析通过。
原始两份工作区保持不变；实现位于独立分支 `feat/residual-mlp-a`。

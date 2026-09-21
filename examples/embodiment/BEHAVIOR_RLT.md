# BEHAVIOR RLT baseline / 基线

This is a BEHAVIOR adaptation of the repository's two-stage RLT TD3 workflow,
not a claim of exact reproduction of the PI paper or a successful trained policy.
The residual experiment is separate. No residual correction or task-phase gate is
used here. All 23 action dimensions and 32 primitive steps per chunk are retained.

## Stage 1 / 第一阶段

`examples/sft/config/behavior_rlt_stage1.yaml` initializes the complete VLA from
PPO global_step_15 and initializes the RLT module from scratch. It trains both the
VLA flow-matching objective and the token reconstruction objective with alpha=1.
Prefix features are detached for reconstruction, matching the existing RLT SFT
implementation; the VLA receives its own action-loss gradients. The original
checkpoint is never overwritten. Joint SFT may change base success and must be
evaluated before attributing Stage 2 improvements to RL.

Reuse the existing train480 grounded sidecar (273228 rows, 480 episodes, three
tasks), three RGB views, full state/actions and padding masks. The RL heldout pool is held out from online RL, not necessarily SFT: 20/20 episode IDs overlap the existing SFT train set. Do not label it unseen-demonstration evaluation.
No new teleoperation
collection is required for Stage 1. The loader keeps the P2 grounded prompts and
explicitly loads the same norm_stats.json as rollout. Token budget is 512;
RLT prefix capacity is 1280 (768 image tokens + 512 prompt tokens), with masked
reconstruction. This differs from the prior residual run's 200-token prompt budget.

```bash
bash examples/sft/run_behavior_rlt_stage1.sh --config-only
bash examples/sft/run_behavior_rlt_stage1.sh
```

Default: local four GPUs, MBS4, GBS32, accumulation2, FSDP full-shard, FP32 masters /
BF16 compute, gradient checkpointing, 2000 optimizer steps (64000 samples), save250.
The wrapper sets TMPDIR and host-qualified XDG/Triton/TorchInductor caches.
On an existing Ray cluster, node runtime PYTHONPATH and cache variables must also
point at this checkout; use the archived launch bundle for the verified placement.

## Stage 2 / 第二阶段

```bash
export B1K_RLT_STAGE1_CHECKPOINT=/absolute/path/to/trained/stage1/checkpoints/global_step_N/actor
bash examples/embodiment/run_behavior_rlt_stage2.sh --config-only
bash examples/embodiment/run_behavior_rlt_stage2.sh
```

Stage 2 rejects a full-wrapper checkpoint without all RLT weights. It freezes the
Stage 1 VLA/token module, caches learned z_rl (2048), processed proprio (32), and
reference chunks, and trains a direct actor plus twin Q. Base actions fill replay;
the actor takes over the entire configured task only after 2000 learner updates.
There are no right-arm, pickup-phase, or expert-intervention rules in the policy.
The radio task selector lives in the environment config. Keep original completion,
reward and scene rendering; an unsuccessful horizon ends at 1280 primitive steps.

Reference BC, reference dropout (0.5), twin Q, delayed actor updates and the BC/Q
weight schedule follow the repository TD3 example. Adaptations: full-task routing,
1024-transition replay warmup, 2000-update control warmup, 5000-update BC/Q ramp,
100-update cap per collection, primitive gamma=.999686. Replay uses executed
prefix masks, pre-reset final observations and gamma^actual_length. Both terminal
and task timeout stop bootstrap in this baseline (unlike the residual run's timeout
bootstrap). No terminal-condition changes are made.

BEHAVIOR commands mix units and some pose commands exceed [-1,1]. The actor
therefore predicts standardized absolute commands, converted with saved action
mean/std (std floor .001); reference inputs, exploration sigma=.1, Q inputs and BC
errors use the same standardized units. The actor has no artificial [-1,1] clip.
The simulator's controller limits still apply. Replay always stores actual commands
sent to the env. Separate DirectGaussianActor/TwinQCritic FSDP units avoid unused-root
parameter registration during critic-only updates.

Stage 2 defaults to one local rollout GPU and one simulator GPU (one scene).
Override placement and env counts together for eight-scene local+nxb deployment.
Evaluate base-only Stage 1 before takeover and monitor success/timeout, BC error,
actor_switch, reconstruction validation and action magnitudes. Finite losses alone
are not evidence of useful learning. RLT replay schedule counters are not saved by
the inherited checkpoint backend, so Stage 2 resume is not exact schedule replay.

## 中文说明

本配置是仓库 RLT TD3 两阶段流程的 BEHAVIOR 适配，不宣称已复现论文结果。
第一阶段从 PPO step15 的副本初始化，联合训练 VLA 动作损失和 RL token 重建；
重建输入 detach，VLA 由动作损失更新。使用已有 train480 数据，无需先重采。
三路图像、P2 prompt、动作和 padding 保留；prompt 从旧实验的200改为512，
因此第二阶段前必须重新测 Stage1 base 成功率，不能继承原 checkpoint 的80%。

第二阶段冻结 Stage1，使用真实学习到的 RL token；直接预测完整23维、32步动作，
保留 BC、reference dropout 和 twin-Q。预热期间由 VLA 控制，预热后全任务由
small actor 控制，不引入抓取阶段 gate 或仅右臂规则。动作按统计量标准化，
不把物理命令直接裁剪到[-1,1]。回放保留执行长度和 reset 前末观测；本基线
terminal 与任务 timeout 均不 bootstrap，reward/completion 条件保持不变。
默认 Stage1 为4卡/MBS4/GBS32/2000步，Stage2需指定完整训练后的RLT checkpoint；
上述命令、warmup与更新预算均属于实验配置，尚需实际训练和评估确定效果。

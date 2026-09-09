训练指标
========

RLinf 通过 :doc:`MetricLogger <../guides/logger>` 在若干命名空间下记录指标——``train/``、``rollout/``、
``env/`` 与 ``time/``。本页统一给出它们的定义；示例页面直接链接到此处，而不再重复说明。

.. tip::

   对具身任务而言，最有用的单一指标是 **``env/success_once``** —— 未归一化的回合成功率。在稀疏
   奖励下，其他 ``env/*`` 指标往往难以直接解读（见下文）。

训练指标 —— ``train/``
----------------------

策略与价值优化的统计量，每次 actor 更新时记录。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``train/actor/approx_kl``
     - 新旧策略之间的近似 KL 散度。
   * - ``train/actor/clip_fraction``
     - 概率比被裁剪的更新比例。
   * - ``train/actor/clipped_ratio``
     - 被裁剪后概率比的均值。
   * - ``train/actor/grad_norm_before_clip``
     - embodied FSDP actor 记录的裁剪前梯度范数。使用 actor-critic 联合优化器时，该值覆盖全部被优化参数，而不只是策略分支。
   * - ``train/actor/grad_norm_after_clip``
     - 裁剪后的有效总梯度范数。配置分支独立阈值时，该值是分别裁剪后的 policy 与 value 范数的 L2 合成值。
   * - ``train/actor/grad_clip_coef``
     - 裁剪后有效总范数与裁剪前总范数的比值；``1`` 表示无需裁剪。
   * - ``train/actor/policy_grad_norm_before_clip``
     - policy 分支的全局裁剪前梯度范数；FSDP 分片仅在其分片进程组内归约一次。
   * - ``train/actor/policy_grad_norm_after_clip``
     - 应用 ``actor.optim.policy_clip_grad`` 后的 policy 梯度范数；未开启独立裁剪时使用统一的 ``actor.optim.clip_grad``。
   * - ``train/actor/policy_grad_clip_coef``
     - 仅作用于 policy 梯度的裁剪系数。
   * - ``train/critic/value_grad_norm_before_clip``
     - value head 的全局裁剪前梯度范数；可与 policy norm 对比，以判断哪个分支主导总梯度裁剪。
   * - ``train/critic/value_grad_norm_after_clip``
     - 应用 ``actor.optim.value_clip_grad`` 后的 value-head 梯度范数；未开启独立裁剪时使用统一的 ``actor.optim.clip_grad``。
   * - ``train/critic/value_grad_clip_coef``
     - 仅作用于 value-head 梯度的裁剪系数。
   * - ``train/actor/grad_norm``
     - 其他 actor worker 仍会记录的旧版裁剪前指标；如果存在上述显式指标，应优先使用它们。
   * - ``train/actor/lr``
     - 当前学习率。
   * - ``train/actor/policy_loss``
     - PPO / GRPO 策略损失。
   * - ``train/critic/value_loss``
     - 价值函数损失。
   * - ``train/critic/value_clip_ratio``
     - 价值目标更新被裁剪的比例。
   * - ``train/critic/explained_variance``
     - 价值预测的可解释方差（越接近 1 越好）。
   * - ``train/entropy_loss``
     - 策略熵。
   * - ``train/loss``
     - 训练总损失（actor + critic + 熵正则）。

Rollout 指标 —— ``rollout/``
----------------------------

rollout 阶段收集的优势与奖励统计量。

启用 outcome-dynamic sampling 时，``algorithm.outcome_dynamic_sampling.groups_per_update``
指定每次 actor 更新要拼接多少个独立通过配额筛选的 group。同质 group 会持续重采样，
直到满足配额；``attempt_warning_interval`` 控制长时间采样时的告警频率。
``env.train.rollout_epoch`` 应保持为 ``1``，以便逐个初始状态检查正负样本配额。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``rollout/advantages_max``
     - 该批次中优势的最大值。
   * - ``rollout/advantages_mean``
     - 该批次中优势的均值。
   * - ``rollout/advantages_min``
     - 该批次中优势的最小值。
   * - ``rollout/rewards``
     - 一个 rollout chunk 的奖励。
   * - ``rollout/dynamic_sampling/groups_per_update``
     - 本次 actor 更新中独立通过正负样本筛选的 group 数量。
   * - ``rollout/dynamic_sampling/attempts``
     - 为填满本次更新而采样的候选 group 总数。
   * - ``rollout/dynamic_sampling/rejected_groups``
     - 因未满足成功/失败配额而被拒绝的候选 group 数量。
   * - ``rollout/dynamic_sampling/successes`` / ``failures``
     - 所有已接受 group 中保留的成功与失败 trajectory 数量。
   * - ``rollout/dynamic_sampling/candidate_successes`` / ``candidate_failures``
     - 所有已接受及被拒绝候选 group 的 outcome 计数。

环境指标 —— ``env/``
--------------------

来自模拟器的任务级信号。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``env/success_once``
     - **推荐。** 未归一化的回合成功率——最能反映真实任务表现。
   * - ``env/episode_len``
     - 回合中实际经历的环境步数。
   * - ``env/return``
     - 回合总回报。稀疏奖励下在成功结束前几乎为 0，训练过程中参考价值有限。
   * - ``env/reward``
     - step 级奖励（中间步为 ``0``，成功时为 ``1``）。日志值按回合步数归一化，难以直接反映真实表现。

如何选择日志后端（TensorBoard、Weights & Biases、SwanLab）以及配置 ``runner.logger``，参见
:doc:`日志 <../guides/logger>` 教程。

Training Metrics
================

RLinf reports metrics through the :doc:`MetricLogger <../guides/logger>` under a few namespaces —
``train/``, ``rollout/``, ``env/``, and ``time/``. This page defines them once; example
pages link here instead of repeating the definitions.

.. tip::

   For embodied tasks the single most useful signal is **``env/success_once``** — the
   unnormalized episodic success rate. Most other ``env/*`` values are hard to read
   directly under sparse rewards (see below).

Training metrics — ``train/``
-----------------------------

Policy- and value-optimization statistics, logged every actor update.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``train/actor/approx_kl``
     - Approximate KL divergence between the old and new policies.
   * - ``train/actor/clip_fraction``
     - Fraction of updates where the probability ratio was clipped.
   * - ``train/actor/clipped_ratio``
     - Mean of the clipped probability ratios.
   * - ``train/actor/grad_norm_before_clip``
     - Gradient norm before clipping, reported by the embodied FSDP actor. For a joint actor-critic optimizer, this covers all optimized parameters rather than only the policy branch.
   * - ``train/actor/grad_norm_after_clip``
     - Effective total gradient norm after clipping. When branch-specific limits are configured, this is the L2 combination of the independently clipped policy and value norms.
   * - ``train/actor/grad_clip_coef``
     - Ratio of the effective total norm after clipping to the total norm before clipping; ``1`` means that no clipping was needed.
   * - ``train/actor/policy_grad_norm_before_clip``
     - Global pre-clipping norm of policy-branch gradients. FSDP shards are reduced once across their sharding group.
   * - ``train/actor/policy_grad_norm_after_clip``
     - Policy gradient norm after applying ``actor.optim.policy_clip_grad`` (or the shared ``actor.optim.clip_grad`` when independent clipping is disabled).
   * - ``train/actor/policy_grad_clip_coef``
     - Coefficient applied only to policy gradients.
   * - ``train/critic/value_grad_norm_before_clip``
     - Global pre-clipping norm of value-head gradients. Compare it with the policy norm to identify which branch drives total-gradient clipping.
   * - ``train/critic/value_grad_norm_after_clip``
     - Value-head gradient norm after applying ``actor.optim.value_clip_grad`` (or the shared ``actor.optim.clip_grad`` when independent clipping is disabled).
   * - ``train/critic/value_grad_clip_coef``
     - Coefficient applied only to value-head gradients.
   * - ``train/actor/grad_norm``
     - Legacy pre-clipping metric still emitted by other actor workers. Prefer the explicit metrics above when available.
   * - ``train/actor/lr``
     - Current learning rate.
   * - ``train/actor/policy_loss``
     - PPO / GRPO policy loss.
   * - ``train/critic/value_loss``
     - Value-function loss.
   * - ``train/critic/value_clip_ratio``
     - Fraction of value targets whose update was clipped.
   * - ``train/critic/explained_variance``
     - Explained variance of the value predictions (closer to 1 is better).
   * - ``train/entropy_loss``
     - Policy entropy.
   * - ``train/loss``
     - Total training loss (actor + critic + entropy regularization).
   * - ``train/diagnostics/policy_gradient/pairwise_negative_fraction``
     - Fraction of off-diagonal initial-state PPO gradient pairs with a negative cosine. Available only when policy-gradient diagnostics are enabled.
   * - ``train/diagnostics/policy_gradient/state_to_aggregate_negative_fraction``
     - Fraction of initial-state PPO gradients opposed to the aggregate PPO gradient.
   * - ``train/diagnostics/policy_gradient/kl_to_ppo_norm_ratio``
     - Norm of the reference-KL gradient contribution divided by the PPO gradient norm.
   * - ``train/diagnostics/policy_gradient/state_sum_additivity_error_ratio``
     - Relative error between the sum of per-state PPO gradients and a directly computed full-batch PPO gradient; this should be close to zero.

Rollout metrics — ``rollout/``
------------------------------

Statistics of the advantages and rewards collected during rollout.

For outcome-dynamic sampling, ``algorithm.outcome_dynamic_sampling.groups_per_update``
sets how many independently quota-filtered groups are concatenated into one actor
update. Rejected homogeneous groups are resampled until the quota is met, and
``attempt_warning_interval`` controls how often prolonged sampling emits a warning.
Keep ``env.train.rollout_epoch`` at ``1`` so quotas are checked per initial state.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``rollout/advantages_max``
     - Maximum advantage in the batch.
   * - ``rollout/advantages_mean``
     - Mean advantage in the batch.
   * - ``rollout/advantages_min``
     - Minimum advantage in the batch.
   * - ``rollout/rewards``
     - Reward of a rollout chunk.
   * - ``rollout/dynamic_sampling/groups_per_update``
     - Number of independently accepted positive/negative groups in this actor update.
   * - ``rollout/dynamic_sampling/attempts``
     - Total candidate groups sampled while filling the update.
   * - ``rollout/dynamic_sampling/rejected_groups``
     - Candidate groups rejected for not satisfying the success/failure quotas.
   * - ``rollout/dynamic_sampling/successes`` / ``failures``
     - Successful and failed trajectories retained across all accepted groups.
   * - ``rollout/dynamic_sampling/candidate_successes`` / ``candidate_failures``
     - Outcomes across accepted and rejected candidate groups.
   * - ``rollout/dynamic_sampling/actor_trainable_groups`` / ``actor_no_signal_groups``
     - Accepted logical states that did or did not satisfy the actor-only outcome quota.

Environment metrics — ``env/``
------------------------------

Task-level signals from the simulator.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``env/success_once``
     - **Recommended.** Unnormalized episodic success rate — the truest measure of task performance.
   * - ``env/episode_len``
     - Number of environment steps elapsed in the episode.
   * - ``env/return``
     - Episode return. Under sparse rewards this is near-zero until the terminal success step, so it is not very informative during training.
   * - ``env/reward``
     - Step-level reward (``0`` on intermediate steps, ``1`` on success). The logged value is normalized by episode length, which makes it hard to read as real performance.
   * - ``env/time/remote_collector_rpc``
     - End-to-end duration of the latest remote environment RPC, including transport and remote simulator handling.
   * - ``env/time/remote_collector_handler``
     - Time spent handling the request inside the remote collector daemon.
   * - ``env/time/remote_collector_transport``
     - Estimated network and serialization time, computed as RPC duration minus remote handler duration.
   * - ``env/remote_collector/request_mib`` / ``response_mib``
     - Encoded request and response sizes for the latest remote environment RPC, in MiB.
   * - ``env/remote_collector/response_raw_blob_mib``
     - Total tensor and byte payload size before lossless response compression, in MiB.
   * - ``env/remote_collector/response_compression_ratio``
     - Encoded blob bytes divided by original blob bytes. ``1`` means no size reduction.
   * - ``env/remote_collector/reconnects``
     - Number of SSH-tunnel or socket reconnects needed by the latest RPC. Persistent nonzero values indicate an unstable connection.

See also the :doc:`Logger <../guides/logger>` tutorial for choosing backends (TensorBoard,
Weights & Biases, SwanLab) and configuring ``runner.logger``.

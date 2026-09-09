# Correctness-first BEHAVIOR subpool RL

This pipeline trains one shared policy on independently reset BEHAVIOR skills.
It is intentionally serial at the simulator boundary until its behavior matches
the official BEHAVIOR-1K evaluator.

## Safety invariants

`BehaviorSubpoolEnv` rejects configurations that enable environment subprocess
sharding, intermediate-observation skipping, environment offload, rollout
pipeline stages, streaming training-pipeline normalization, or RLinf's reduced
texture-streaming budget. `renderer_mode: official` leaves the
Kit renderer settings untouched, matching the official B1K evaluator. Each
environment worker owns exactly one simulator.

Every action chunk records an `executed_action_mask`. A successful or timed-out
skill stops immediately; remaining actions in that chunk are not sent to the
simulator and do not contribute to PPO log-probability. With `auto_reset: true`,
the next chunk restores another audited snapshot and packs another episode into
the same fixed-length rollout. Done flags cut GAE and bootstrapping at every
packed episode boundary. Primitive rewards are collapsed as

```text
R = r[0] + gamma*r[1] + ... + gamma^(m-1)*r[m-1]
discount = gamma^m
```

where `m` is the number of actions actually executed.

## State pools

The JSONL manifest contains one line per complete OmniGibson state. Every
subtask must have a `canonical` state. Optional `predecessor_success` and
`recovery` records are sampled with configurable weights after first selecting
a subtask uniformly. The state file checksum, activity, scene, and asset
fingerprint are validated before restore.

States use format-v2 `.pt` checkpoints produced by
`og.sim.dump_state(serialized=False)`. Do not use the flat `.npy` representation:
OmniGibson currently omits assisted-grasp constraints from that serialization,
which silently corrupts press and place starts that follow pickup.

The adapter also builds its scene with the official B1K evaluator generator.
The generic RLinf BEHAVIOR example has different robot spawn, self-collision,
camera, action/render frequencies, and `scene.include_robots` settings, so its
states and action timing are not interchangeable with challenge-evaluator states.

A persistent simulator is locked to one `(activity, scene)` signature inferred
from the manifest. A subtask's canonical pool may contain audited starts from
multiple demo episodes and activity instances; every reset samples one of those
starts uniformly after selecting the subtask and pool. Mixed-activity or
mixed-scene manifests remain invalid because they require reconstructing the
simulator.

The persistent scene is bootstrapped from the official seed template (instance
0), exactly like the B1K evaluator, while the sampled challenge instance is
restored from its complete state and assigned as the task's logical instance.
These are distinct identifiers; attempting to construct
`*_0_<challenge_instance>_template.json` is incorrect because only the seed
template exists.

Dynamic state-pool updates are disabled for the init-only control experiment.
When enabled in a later recovery experiment, successful terminal states are
appended to the next subtask's `predecessor_success` pool. A timeout appends an
earlier state from the time-sampled ring buffer to the current `recovery` pool;
the direct terminal failure state is never used.

Each manifest record also stores the grounded control template and a common
reward specification. P2 object and part boxes are recomputed from local
instance masks after every simulator step and serialized with the same token
mapping used for SFT.

The init-only control manifest uses a uniform 1280-primitive-step timeout. The
longest audited radio pickup suffix is 989 steps, so this keeps 29% slack while
remaining below 90 seconds at 15-fps video playback. Forty 32-action chunks
across eight streams form one actor GBS 320 update. The step penalty is rescaled
to `-1 / 1280`, preserving a total timeout budget of -1.

Manipulation shaping is role- and predicate-specific: pickup measures the right
end effector to the radio, press measures the left end effector to the toggle,
and place measures the held radio to the support surface. It must not minimize
over both arms, because that rewards the free gripper for approaching an object
that the other gripper is already holding.

For a fixed-state policy-improvement diagnostic, use a terminal-time reward
with no potential terms: terminal success is `+10`, timeout is `-2`, and each
executed primitive action receives `-1 / max_steps`. The primitive rewards are
still collapsed into one duration-aware reward per executed action chunk. This
definition keeps the optimization target aligned with success and completion
time; a failed rollout cannot outrank a successful one merely by approaching or
pushing the target object.

## Advantage and loss

`subtask_gae` uses duration-aware discounts and stops recursion at termination
or a subtask boundary. Advantages are normalized separately for each subtask.
Per-transition weights make every represented subtask contribute equal total
actor and critic weight, independent of trajectory count or length. The scalar
critic is conditioned on the P2 prompt through the VLM representation, i.e.
`V(s, z)`.

For OpenPI-RLinf, `actor.model.openpi.value_vlm_mode: state_fusion` applies
LayerNorm to the masked-mean VLM prefix, encodes the current proprioceptive
state with a two-layer MLP, and feeds their concatenation to the value MLP.
`state_attention` instead uses the encoded state as a query over the valid VLM
prefix tokens, then predicts value from the attended tokens, masked-mean prefix,
and state features. This matches the attention critic used by the offline
fixed-batch capacity diagnostic.
Use FP32 actor master parameters with BF16 FSDP compute and FP32 gradient
reduction:

```yaml
actor:
  model:
    precision: fp32
    openpi:
      value_vlm_mode: state_fusion
  fsdp_config:
    mixed_precision:
      param_dtype: bf16
      reduce_dtype: fp32
      buffer_dtype: bf16
```

To fit the critic more often without repeatedly moving the policy on one
rollout batch, set `actor.policy_update_epochs` and
`actor.critic_update_epochs`. For example, values `1` and `5` perform one
policy-only optimizer step followed by five critic-only steps. Critic-only
passes skip OpenPI's action-suffix/log-probability computation. Leaving either
fields unset preserves the legacy joint `algorithm.update_epoch` behavior.

TensorBoard reports aggregate success plus `env/subtask/<id>/success` and
`env/subtask/<id>/timeout` for every represented skill. It also reports
`env/subtask/<id>/pool/<pool_id>/*`, where pool ids 0, 1, and 2 mean canonical,
predecessor-success, and recovery. Reward filtering is rejected because it can
silently remove the hardest skills. Online recovery states are sampled from a
lag window rather than from the direct failure state, and each dynamic
`(subtask, pool_type)` bucket is capped to bound storage.

## Outcome dynamic sampling

The production configuration uses DAPO-style dynamic sampling to prevent a PPO
update from containing only failures or only successes. The four environment
ranks restore the same sampled snapshot, while rank-offset rollout seeds create
four independent stochastic action streams. A group is trainable only when it
contains at least `min_successes` successful trajectories and `min_failures`
failed trajectories. Homogeneous groups are discarded before advantage
calculation, and a new snapshot is sampled with the same synchronized policy.
No rejected trajectory is replayed after a parameter update.

Before every candidate rollout, the runner explicitly resets all group members
with a shared collection index. Snapshot sampling is derived from
`(seed, outcome_group_id, collection_index)`, so independent auto-resets inside
the preceding rollout cannot desynchronize the next candidate's initial state.
The runner checks the snapshot id, episode, subtask, and pool reported by every
member and aborts instead of comparing trajectories from different starts.

When auto-reset packs multiple episodes into a stream, quota filtering uses the
outcome of the first complete episode only. Otherwise `any(success)` would favor
streams that happened to complete more short episodes. Later packed episodes
remain fresh on-policy training data, with their own done boundaries.

```yaml
algorithm:
  outcome_dynamic_sampling:
    enabled: true
    group_size: 4
    groups_per_update: 2
    parallel_groups: true
    min_successes: 1
    min_failures: 1
    attempt_warning_interval: 4
env:
  train:
    subpool:
      outcome_group_size: 4
rollout:
  seed: 1234
```

With `parallel_groups: false`, `group_size` must equal
`env.train.total_num_envs` and groups are collected serially. With
`parallel_groups: true`, set `total_num_envs` to
`group_size * groups_per_update`, use one EnvWorker per environment, and set the
actor world size to `group_size`. Each actor rank then receives one trajectory
from every concurrent group. Groups pass the quota independently: an accepted
group is retained while failed groups continue sampling in later rounds.
`rollout_epoch` must be one. OpenPI-RLinf also requires positive `flow_sde`
noise. Sampling continues until every group is mixed; every
`attempt_warning_interval` rejected candidates, the runner emits a warning
without training on an all-negative or all-positive batch. TensorBoard reports accepted counts under
`rollout/dynamic_sampling/{successes,failures}`, plus attempt, rejection, and
all-candidate counts. Parallel runs also report
`rollout/dynamic_sampling/sampling_rounds`. Per-snapshot tags under
`rollout/dynamic_sampling/snapshot/<snapshot_id>/` record the episode index,
candidate group and outcome counts, candidate success rate, and accepted group
count.

This filter cannot manufacture a positive trajectory when the policy has zero
success probability. If warnings repeat for a long time, stop the run and first
bootstrap the policy with easier canonical/recovery states or additional SFT.

This v1 intentionally uses fixed equal subtask allocation. Success-rate-aware
allocation, gradient surgery, and a learned successor/handoff value are not
enabled until the equal-weight baseline passes closed-loop gates; adding them
before that would make simulator, reward, and optimization failures harder to
separate.

Set `actor.optim.critic_only: true` for the critic diagnostic. This permanently
freezes every non-value parameter, suppresses policy loss, and builds the
optimizer from the value head only. The normal and critic-only controls should
start from the same SFT checkpoint and use the same init manifest, reward,
horizon, and rollout seeds. `critic/only_mode=1`, branch-local gradient norms,
and `critic/explained_variance` make the comparison explicit.

The production control uses `actor.global_batch_size: 320`, equal to the full
`40 chunks x 8 streams` rollout. GBS 64 would apply five sequential optimizer
steps to one accepted DAPO batch, so later minibatches would already see a
changed policy and would not individually preserve the accepted outcome mix.
The full-batch setting keeps `micro_batch_size: 1`; it increases gradient
accumulation per actor rank without increasing per-forward activation memory.

Use one simulator per env rank and an env world size that is an integer multiple
of the FSDP actor world size. The production topology uses eight env ranks and
four actor ranks. With `outcome_group_size: 1`, rank-offset round-robin sampling makes four
env ranks cover four radio skills once per reset cycle. With dynamic sampling's
`outcome_group_size: 4`, all ranks instead share one subtask and snapshot, and
the group cursor advances round-robin across reset attempts. Random subtask
sampling remains disabled in correctness mode.

Set the OmniGibson paths and `TMPDIR` before starting Ray on every environment
node. Ray captures its worker environment at startup, so exporting them only in
the launcher shell is insufficient for an already-running cluster. Each nested
`BehaviorProcess` is hard-pinned to its parent env worker's node; otherwise Ray
could move the simulator onto a non-rendering trainer node in a heterogeneous
cluster. A standalone local-Ray smoke may additionally use a short path such as
`RAY_TMPDIR=/mnt/public/daibo/tmp/r6`; long subdirectory names can exceed Ray's
107-byte Unix-socket path limit.

Use `examples/embodiment/config/behavior_subpool_ppo_openpi_pi05.yaml` only for
a one-node smoke. The production config is
`examples/embodiment/config/behavior_subpool_ppo_openpi_pi05_hetero.yaml`: actor
and rollout occupy the four GPUs on node rank 0, while two groups of four B1K
simulators occupy the RTX 4090 GPUs on node ranks 1 and 2. Required environment
variables are:

```bash
export TMPDIR=/mnt/public/daibo/tmp
export B1K_SUBPOOL_MANIFEST=/path/to/manifest.jsonl
export B1K_GROUNDED_TOKEN_MAPPING=/path/to/structural_token_mapping.json
export B1K_ASSET_FINGERPRINT=the-version-used-to-create-the-snapshots
export B1K_SUBPOOL_MODEL_PATH=/path/to/global_step_8000
export B1K_SUBPOOL_RESULT_DIR=/path/to/results
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data
export OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets
export OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key
export OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
export OMNI_KIT_ACCEPT_EULA=YES
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export HTTP_PROXY=http://10.10.20.100:1089
export HTTPS_PROXY=http://10.10.20.100:1089
```

Do not start PPO until demo replay has shown that every canonical snapshot
round-trips and that its selected reward stage succeeds under the corresponding
ground-truth action segment.

Canonical radio pools can be exported with the grounded evaluator entrypoint.
Run it in the BEHAVIOR/OpenPI environment and pass the same sidecar and token
mapping used for SFT:

```bash
export TMPDIR=/mnt/public/daibo/tmp
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data
export OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets
export OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key
export OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
python toolkits/b1k_grounded/eval_grounded_subtasks.py \
  policy=websocket \
  task.name=turning_on_radio \
  eval_level=subtask \
  keep_running_after_success=false \
  instance_reward_mode=task \
  demo_data_dir=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  'run_episode_indices=[10]' \
  +grounded_control_sidecar=/path/to/part-00000.parquet \
  +grounded_control_profile=p2_ground_sg \
  +grounded_eval_view_dir=/mnt/public/daibo/tmp/radio_subpool_eval_view \
  +subpool_export_manifest=/path/to/radio_subpool/manifest.jsonl \
  +subpool_reward_specs=toolkits/b1k_grounded/radio_subpool_reward_specs.json \
  +subpool_token_mapping=/path/to/structural_token_mapping.json \
  +subpool_asset_fingerprint=behavior-assets-2025 \
  subtask_index=null subtask_end_index=null
```

The exporter restores every dumped state and replays only that skill's GT suffix.
It aborts before writing a bad canonical record when the direct predicate fails.
Use `run_episode_indices` for explicit episode IDs. `run_episode_idx` is a position
in the resolved episode list and can silently select a different activity instance.
Use `official_subtask` for the init-state pool used by B1K subtask evaluation;
`demo_replay` instead reconstructs predecessor effects and is useful for later
stages whose starting state depends on earlier manipulation.

Open-loop GT actions are not always replayable after simulator reconstruction. For
an official subtask init-state pool, set
`subpool_require_gt_suffix_success=false` only when a subsequent real-env smoke is
required. The manifest retains `metadata.gt_validation.success=false`; this keeps
the failed attainability check visible instead of treating it as a verified demo
state. The default remains strict.

When episodes were exported independently, combine the validated catalogs before
training:

```bash
python toolkits/b1k_grounded/merge_subpool_manifests.py \
  --input-manifest /path/to/episode_10/manifest.jsonl \
  --input-manifest /path/to/episode_20/manifest.jsonl \
  --output-manifest /path/to/radio_pickup_pool/manifest.jsonl
```

The merger revalidates every checksum, rejects duplicate snapshot IDs and mixed
activity/scene catalogs, and copies the states into a standalone output directory.

After all canonical records pass, exercise the actual RL environment adapter before
starting PPO. The smoke command derives a separate one-step-timeout catalog; it
does not modify the source manifest:

```bash
python toolkits/b1k_grounded/smoke_behavior_subpool_env.py \
  --manifest /path/to/radio_subpool/manifest.jsonl \
  --token-mapping /path/to/structural_token_mapping.json \
  --output-dir /path/to/radio_subpool_env_smoke \
  --subtask-id 1 \
  --reset-count 100 \
  --require-all-snapshots
```

The smoke catalog retains every canonical snapshot for the selected subtask. The
resulting `report.json` requires every snapshot to be sampled across resets, a
one-action executed prefix, a completely masked chunk suffix, a completely frozen
next chunk, and an online P2 prompt.

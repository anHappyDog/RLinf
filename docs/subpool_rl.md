# Correctness-first BEHAVIOR subpool RL

This pipeline trains one shared policy on independently reset BEHAVIOR skills.
It is intentionally serial at the simulator boundary until its behavior matches
the official BEHAVIOR-1K evaluator.

## Safety invariants

`BehaviorSubpoolEnv` rejects configurations that enable environment subprocess
sharding, intermediate-observation skipping, environment offload, automatic
reset, rollout pipeline stages, streaming training-pipeline normalization, or
RLinf's reduced texture-streaming budget. `renderer_mode: official` leaves the
Kit renderer settings untouched, matching the official B1K evaluator. Each
environment worker owns exactly one simulator.

Every action chunk records an `executed_action_mask`. A successful or timed-out
skill stops immediately; remaining actions are not sent to the simulator and do
not contribute to PPO log-probability. Once a skill terminates, later chunks in
the fixed-length rollout return the cached terminal observation without stepping
the simulator. Primitive rewards are collapsed as

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

A persistent simulator is locked to one `(activity, scene, activity instance)`
signature inferred from the manifest. Mixed-instance manifests are rejected
instead of attempting an unsafe cross-instance `load_state`. For the first radio
pilot, export all four skill starts from one audited demo episode.

The persistent scene is bootstrapped from the official seed template (instance
0), exactly like the B1K evaluator, while the manifest's challenge instance is
restored from the complete state. These are distinct identifiers; attempting to
construct `*_0_<challenge_instance>_template.json` is incorrect because only the
seed template exists.

Successful terminal states are appended to the next subtask's
`predecessor_success` pool. A timeout appends an earlier state from the
time-sampled ring buffer to the current `recovery` pool; the direct terminal
failure state is never used.

Each manifest record also stores the grounded control template and a common
reward specification. P2 object and part boxes are recomputed from local
instance masks after every simulator step and serialized with the same token
mapping used for SFT.

Every timeout must be at least as long as its audited GT suffix; the exporter
rejects shorter horizons. The radio pilot uses 384 / 1152 / 384 / 512 primitive
steps for move, pickup, press, and place. Their per-step penalties are scaled so
`step_penalty * max_steps == -1` for every skill. The shared fixed rollout is
1152 primitive steps, matching the longest task-specific timeout; completed
skills are frozen and masked for its remainder.

Manipulation shaping is role- and predicate-specific: pickup measures the right
end effector to the radio, press measures the left end effector to the toggle,
and place measures the held radio to the support surface. It must not minimize
over both arms, because that rewards the free gripper for approaching an object
that the other gripper is already holding.

## Advantage and loss

`subtask_gae` uses duration-aware discounts and stops recursion at termination
or a subtask boundary. Advantages are normalized separately for each subtask.
Per-transition weights make every represented subtask contribute equal total
actor and critic weight, independent of trajectory count or length. The scalar
critic is conditioned on the P2 prompt through the VLM representation, i.e.
`V(s, z)`.

TensorBoard reports aggregate success plus `env/subtask/<id>/success` and
`env/subtask/<id>/timeout` for every represented skill. It also reports
`env/subtask/<id>/pool/<pool_id>/*`, where pool ids 0, 1, and 2 mean canonical,
predecessor-success, and recovery. Reward filtering is rejected because it can
silently remove the hardest skills. Online recovery states are sampled from a
lag window rather than from the direct failure state, and each dynamic
`(subtask, pool_type)` bucket is capped to bound storage.

This v1 intentionally uses fixed equal subtask allocation. Success-rate-aware
allocation, gradient surgery, and a learned successor/handoff value are not
enabled until the equal-weight baseline passes closed-loop gates; adding them
before that would make simulator, reward, and optimization failures harder to
separate.

Use one simulator per env rank and an env world size that is an integer multiple
of the FSDP actor world size. The default uses four env ranks and four actor
ranks. Rank-offset round-robin subtask sampling makes each group of four env
ranks cover the four radio skills once per reset cycle; random sampling remains
disabled in correctness mode. The runtime rejects a manifest whose subtask count
does not equal the env-rank count (except an explicit single-subtask overfit
run), because e.g. three tasks on four ranks would silently create a 2:1:1
gradient mixture.

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
and rollout occupy the four GPUs on node rank 0, while four independent B1K
simulators occupy the RTX 4090 GPUs on node rank 1. Required environment
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

After all canonical records pass, exercise the actual RL environment adapter before
starting PPO. The smoke command derives a separate one-step-timeout catalog; it
does not modify the source manifest:

```bash
python toolkits/b1k_grounded/smoke_behavior_subpool_env.py \
  --manifest /path/to/radio_subpool/manifest.jsonl \
  --token-mapping /path/to/structural_token_mapping.json \
  --output-dir /path/to/radio_subpool_env_smoke \
  --subtask-id 1
```

The resulting `report.json` requires a one-action executed prefix, a completely
masked chunk suffix, a completely frozen next chunk, and an online P2 prompt.

# BEHAVIOR-1K Grounded-Control Annotation Coverage

This toolkit validates raw BEHAVIOR-1K skill annotations against the frozen
v0.1 signatures in `rlinf.data.b1k_grounded`.

Run the full coverage scan without OpenPI or OmniGibson:

```bash
python -m toolkits.b1k_grounded.annotation_coverage \
  --dataset-root /path/to/2025-challenge-demos \
  --output /path/to/annotation_coverage.json
```

Each annotation has one explicit outcome:

- `valid`: converted into a typed `ParsedSkillSegment`.
- `ambiguous`: known skill with malformed or conflicting annotation data.
- `unsupported`: canonical skill missing from the registry.

The parser never repairs spelling errors or malformed object lists silently.
It normalizes harmless whitespace and capitalization in the symbolic hand
labels used by `hand over`.

For the audited 2025 challenge demonstrations (50 tasks, 10,000 episodes), the
expected result is 235,468 valid records, 24 ambiguous records, and no
unsupported skills. The ambiguous records comprise 14 duplicate object lists,
three invalid frame ranges, six misspelled hand labels, and one null
`skill_type`.

## Recorded segmentation grounding probe

The challenge demos already contain synchronized RGB and `seg_instance_id`
videos plus the per-episode mapping from mesh IDs to OmniGibson prim paths.
Run the five-case feasibility probe without launching the simulator:

```bash
python -m toolkits.b1k_grounded.recorded_grounding_probe \
  --dataset-root /path/to/2025-challenge-demos \
  --output-dir /path/to/grounding_probe
```

The probe covers radio pressing, trash placement, mousetrap placement, fridge
door opening, and cabbage chopping. It decodes three temporal candidates per
skill, selects the frame with the best entity visibility, and writes a JSON
report plus one three-camera debug image per case. Because the segmentation
streams use lossy MP4 encoding, the probe explicitly removes tiny disconnected
components while always preserving the largest component. It retains additional
components with at least 64 pixels and 0.5% of the largest component's area.

Before a full 18 GB metadata scan, audit deterministic samples from all 50
tasks:

```bash
python -m toolkits.b1k_grounded.mapping_coverage \
  --dataset-root /path/to/2025-challenge-demos \
  --episodes-per-task 1 \
  --output /path/to/mapping_coverage_1_per_task.json
```

This measures symbolic object/part to mesh-ID resolution. It does not claim
that an object is visible at every annotated timestep; visibility requires
decoding the corresponding segmentation frames.

## Grounded SFT pilot sidecar

Build one midpoint sample per valid skill interval for the first episode of
every task:

```bash
python -m toolkits.b1k_grounded.build_pilot_dataset \
  --dataset-root /path/to/2025-challenge-demos \
  --output-dir /path/to/b1k_grounded_pilot_50ep \
  --episodes-per-task 1 \
  --sample-fractions 0.5 \
  --action-horizon 32
```

The sidecar does not duplicate RGB videos. Each Parquet row references the
three source videos and stores the frame index, 256-dimensional state,
episode-clamped `32 x 23` action target, typed `control_json`, and explicit
object/part grounding eligibility. Version `v0.4` also records the rendered
`subgoal`, its `unique` / `composite` / `missing` status, and the source
`primitive_idx` values. The builder intentionally stores structured control
rather than committing to model-specific location tokens.

### Primitive-derived subgoals

The builder derives macro subgoals deterministically from each episode's
`primitive_annotation`. It uses `skill_idxes` to assign one primitive to all of
its child skills, renders primitive operations with fixed templates, and joins
multi-operation primitives with `then`. If one atomic skill is shared by
several primitives, their rendered text is ordered by `primitive_idx` and
joined with `; also `. When a B1K primitive has empty operation fields, the
builder renders its ordered `skill_idxes` with the same operation templates and
uses that atomic sequence as the primitive text. A skill is `missing` only when
no primitive references it.

The `<arg>` records and P2 boxes remain scoped to the current atomic skill.
The subgoal supplies macro context; it does not add future objects to the
current skill's grounding set.

Do not rebuild a composed production sidecar to populate these fields. Rebuilding
would repeat frame selection, grounding, composition, and action-boundary logic,
which can change its training population. Instead, enrich the exact existing
sidecar in a new directory:

```bash
python -m toolkits.b1k_grounded.enrich_sidecar_subgoals \
  --input-dir /path/to/oracle_1000ep_radio_microwave_lunchbox_dense_stride8_boundarysafe_v2 \
  --output-dir /path/to/oracle_1000ep_radio_microwave_lunchbox_dense_stride8_boundarysafe_subgoal_v3
```

The command preserves row order and every source column except
`control_json`; within `control_json`, only `subgoal` changes. It adds the three
subgoal metadata columns, copies the frozen structural-token mapping, preserves
the source manifest's composition and counts, and records its source lineage.
It verifies all preserved columns again after writing. The source directory is
never modified. Rerun the tokenizer audit on the enriched sidecar because the
historical v0.2 report measured prompts with `subgoal: null`.

To evaluate temporal frame selection without increasing the number of training
rows, decode three deterministic candidates and retain the best-grounded frame
for each skill interval:

```bash
python -m toolkits.b1k_grounded.build_pilot_dataset \
  --dataset-root /path/to/2025-challenge-demos \
  --output-dir /path/to/b1k_grounded_pilot_best3 \
  --episodes-per-task 1 \
  --sample-fractions 0.1,0.5,0.9 \
  --selection-mode best_visibility \
  --action-horizon 32
```

`best_visibility` ranks candidates by object and part completeness, visible
argument count, visible image fraction, and then proximity to the interval
midpoint. The manifest records the selection rule and reports part completeness
only over rows that actually contain a part argument.

On the audited 50-episode pilot, midpoint sampling produced 1,160/1,186
object-complete and 1,146/1,186 fully grounded rows. Selecting the best of the
10%, 50%, and 90% temporal candidates increased those counts to 1,183 and
1,169, respectively, without regressing any midpoint success. The remaining 17
rows comprise 14 unresolved opaque part links and three objects absent from all
three sampled views. Part grounding is complete for 71/85 rows containing a
part argument.

This best-visible shard is a feasibility and annotation-audit artifact, not the
final action-SFT sampling policy. Oracle profile comparisons must use the same
frame population for all profiles; a production builder should therefore shard
a frame-index policy selected independently of grounding success and represent
missing geometry explicitly rather than dropping those rows.

## P0/P1/P2 tokenizer audit

The serializer binds the 23 frozen v0.1 structural markers to existing
PaliGemma `<unusedN>` pieces and uses the pretrained `<loc0000>` through
`<loc1023>` pieces for geometry. Validate the real tokenizer and measure the
complete pi0.5 prompt length, including its discretized 23-dimensional state:

```bash
python -m toolkits.b1k_grounded.tokenizer_audit \
  --sidecar /path/to/pilot/data/part-00000.parquet \
  --tokenizer-model /path/to/paligemma_tokenizer.model \
  --norm-stats /path/to/norm_stats.json \
  --output /path/to/tokenizer_audit.json \
  --max-token-len 200
```

The command also writes `structural_token_mapping.json` next to the report.
That exact mapping must be copied into every trained checkpoint and validated on
resume; it must never be allocated again from a different tokenizer.

The audited 1,186-row fixed-midpoint pilot has complete pi0.5 prefix lengths of
112--208 tokens for P0, 126--313 for P1, and 136--445 for P2. Consequently, the
Oracle ablations use one shared 512-token budget; no audited profile is
truncated. A 200-token pi0.5 default is not a valid comparison.

## OpenPI grounded-control loader pilot

`GroundedBehaviorSftDataset` joins the structured sidecar to the original three
RGB videos and feeds the normal BEHAVIOR OpenPI transforms. It retains the
existing state normalization, 32-step flow-matching action target, and
`(Observation, actions)` training boundary; only the serialized condition is
switched between P0, P1, and P2. `EpisodeShardedSampler` assigns whole episodes
to one distributed rank, shuffles episode order between epochs, and keeps frame
order within an episode so the video decoders advance instead of seeking for
every row. Smaller rank partitions repeat a few local samples to keep all ranks
at the same step count without dropping source rows.

For long-horizon training where evaluation averages task success, configure the
hierarchical sampler explicitly:

```yaml
data:
  grounded_sampling:
    strategy: task_episode_stage
    stage_exponent: 0.5
    samples_per_epoch: 273216
```

It samples tasks uniformly, episodes uniformly within each task, stages in an
episode proportional to `num_stage_rows ** stage_exponent`, and rows uniformly
inside the selected stage. `stage_exponent: 0` is stage-uniform and `1` is
frame-uniform within an episode; `0.5` retains stage-duration information with
diminishing weight. `samples_per_epoch` must be divisible by
`actor.global_batch_size`. Each global batch differs by at most one sample per
task, and the extra sample rotates between tasks. Skill names do not enter the
sampling probability.

The checked-in `behavior_pi05_grounded_oracle_train480_sqrt_stage_smoke` config
runs a 200-step pipeline check on the cleaned 480-episode, three-task train
split. It starts from the official BEHAVIOR π0.5 checkpoint and keeps the
224-pixel OpenPI transform, 32-step action horizon, and P2 Oracle condition
fixed.

The corresponding formal run uses 20,000 optimizer steps at global batch size
64: 1,280,000 sampled examples, or about 4.68 sampling epochs. It uses a
1,000-step warmup, peak learning rate `1e-5`, cosine decay, and saves every
2,000 steps for rollout-based selection; the final step count is a training
budget, not an assumption that the last checkpoint must be best. A staged
4-GPU to 8-GPU run must keep global batch size 64 and
`actor.optim.total_training_steps: 20000`. Stop phase one by overriding only
`runner.max_steps`, then set `runner.resume_dir` to its DCP checkpoint and
restore `runner.max_steps: 20000`. The grounded loader derives its sampling
epoch and completed-global-batch offset from the resumed global step, so
changing the world size does not replay the beginning of the sampling epoch.

The ready-to-run pilot config defaults to P2:

```bash
bash examples/sft/run_vla_sft.sh behavior_pi05_grounded_oracle_pilot
```

This is full-parameter action SFT: `actor.model.is_lora` is false and the SFT
model-building path does not freeze the vision encoder, language model, or
action expert. The three profiles differ only in the serialized condition:

- P0 contains the overall goal.
- P1 adds the primitive-derived subgoal, GT skill, typed argument roles, object
  names, and available part names.
- P2 adds an Oracle object bbox from the recorded instance-segmentation stream.

The target remains the 32-step action chunk. This dataset is not a VQA dataset
and does not add question-answer, caption, or external web examples; those
objectives would confound the first Oracle action-policy comparison.

Select P0 or P1 with a Hydra override, while keeping every other budget fixed:

```bash
EMBODIED_PATH=$PWD/examples/sft python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config \
  --config-name behavior_pi05_grounded_oracle_pilot \
  data.grounded_control_profile=p1_simple_sg \
  runner.logger.log_path=/mnt/public/daibo/results/b1k_grounded_control_v01/training/pi05_grounded_oracle_pilot_p1
```

The checked-in pilot config uses exactly one temporal midpoint from every valid
skill interval in one episode per task. Grounding success never changes frame
selection or row retention: absent geometry is serialized as `<no_grounding>`.
This makes the 50-episode shard suitable for loader validation and controlled
overfit experiments, but its 1,186 rows are still too small for the formal
P0/P1/P2 result. Increase `--episodes-per-task` while preserving the midpoint
policy before launching the full Oracle comparison.

## Reproducible episode splits

Create an explicit split before a formal build. Episodes listed by an older
sidecar manifest, or supplied manually after an evaluation or visual audit,
are confined to train. Validation and test are selected from the remaining
episodes by a stable SHA-256 ranking:

```bash
python -m toolkits.b1k_grounded.episode_split \
  --dataset-root /path/to/2025-challenge-demos \
  --output /path/to/episode_split.json \
  --task-indices 0 12 40 \
  --train-per-task 160 \
  --validation-per-task 20 \
  --test-per-task 20 \
  --seed 20260905 \
  --exclude-manifest /path/to/old_sidecar/manifest.json \
  --exclude-episode 12:120700
```

Pass the split to each task build instead of relying on the legacy first-N
selection:

```bash
python -m toolkits.b1k_grounded.build_pilot_dataset \
  --dataset-root /path/to/2025-challenge-demos \
  --output-dir /path/to/turning_on_radio_train \
  --episodes-per-task 160 \
  --task-indices 0 \
  --episode-split /path/to/episode_split.json \
  --split-name train \
  --frame-stride 8 \
  --selection-mode all \
  --action-horizon 32
```

The sidecar manifest records the split path, split name, and split-manifest
hash. This makes the exact source episode set auditable even if the original
split file is later replaced.

The builder also checks every annotation interval against the recorded
metadata and Parquet length. If only an interval tail exceeds an otherwise
consistent episode, it clips that tail and records both boundaries under
`annotation_interval_repairs` in the manifest. An interval whose start is
outside the episode remains a hard error because it has no aligned sample to
recover.

## Dense single-task overfit control

Before scaling the Oracle comparison, build a dense ten-demo control for one
task. `--frame-stride` samples every Nth frame and always retains the first and
last frame of each half-open skill interval:

```bash
python -m toolkits.b1k_grounded.build_pilot_dataset \
  --dataset-root /path/to/2025-challenge-demos \
  --output-dir /path/to/turning_on_radio_10ep_stride8 \
  --episodes-per-task 10 \
  --task-indices 0 \
  --frame-stride 8 \
  --selection-mode all \
  --action-horizon 32 \
  --num-workers 8
```

`--num-workers` parallelizes independent episodes while preserving their
deterministic output order. Each worker may retain decoded frames from all
three segmentation cameras, so choose the worker count from available memory
rather than CPU count alone. The default remains one worker.

Run `tokenizer_audit` on the resulting sidecar to create its frozen structural
token mapping. Then launch the existing SFT config with the dense sidecar and a
single Oracle profile:

```bash
EMBODIED_PATH=$PWD/examples/embodiment python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config \
  --config-name behavior_pi05_grounded_oracle_pilot \
  runner.logger.log_path=/path/to/pi05_radio_dense_p1_2000 \
  runner.max_epochs=100 \
  runner.max_steps=2000 \
  runner.save_interval=250 \
  actor.optim.total_training_steps=2000 \
  actor.optim.lr_warmup_steps=200 \
  data.grounded_sidecar_path=/path/to/turning_on_radio_10ep_stride8/data/part-00000.parquet \
  data.grounded_token_mapping_path=/path/to/turning_on_radio_10ep_stride8/structural_token_mapping.json \
  data.grounded_control_profile=p1_simple_sg
```

This is an intentional same-demo overfit test. It answers whether the action
pipeline and low-level primitives can solve skill-reset states before spending
compute on a larger multi-task run.

To densify only one primitive while keeping all other skills at their temporal
midpoint, add `--frame-stride-skills`. For example, this converts every radio
`press` interval in the first 20 episodes to stride-8 samples without changing
the sampling policy for `move to`, `pick up from`, or `place on`:

```bash
python -m toolkits.b1k_grounded.build_pilot_dataset \
  --dataset-root /path/to/2025-challenge-demos \
  --output-dir /path/to/turning_on_radio_20ep_press_stride8 \
  --episodes-per-task 20 \
  --task-indices 0 \
  --frame-stride 8 \
  --frame-stride-skills press \
  --selection-mode all \
  --action-horizon 32
```

When replacing a midpoint primitive with dense rows, compose with
`--replacement-scope interval`. This removes every earlier sample from each
represented half-open skill interval before adding the later input, preventing
legacy midpoint and dense controls from coexisting for one segment.

For formal training, compose the broad midpoint dataset with the button-aware
dense task dataset. The composer deterministically replaces duplicate sample
IDs with the later input and repairs every action chunk at its half-open skill
boundary: the final valid action is repeated to retain the fixed horizon and
the repeated tail is excluded from flow loss through `action_is_pad`.

```bash
python -m toolkits.b1k_grounded.compose_pilot_datasets \
  --input-dir /path/to/1000ep_midpoint \
  --input-dir /path/to/turning_on_radio_10ep_stride8_button \
  --output-dir /path/to/1000ep_midpoint_radio_dense_boundary_safe \
  --replacement-scope interval
```

The second input must contain the inferred radio button part and its recorded
part grounding. Use the resulting sidecar for both P1 and P2 so their sample
set and optimization budget remain identical; only the serializer profile
changes.

After auditing the resulting sidecar, launch the paired four-rank FSDP jobs
independently:

```bash
bash toolkits/b1k_grounded/run_formal_mixed_sft.sh \
  p1_simple_sg /path/to/pi05_mixed_p1_2000
bash toolkits/b1k_grounded/run_formal_mixed_sft.sh \
  p2_ground_sg /path/to/pi05_mixed_p2_2000
```

Each invocation creates one FSDP group over the four GPUs in
`CUDA_VISIBLE_DEVICES` (defaults to `0,1,2,3`). Run the two commands on separate
four-GPU nodes for the paired experiment; the SFT runner does not create two
independent FSDP groups inside one process.

For one eight-rank training job on two four-GPU nodes, start one Ray cluster
across the nodes and pass `hybrid` as the third launcher argument:

```bash
bash toolkits/b1k_grounded/run_formal_mixed_sft.sh \
  p2_ground_sg /path/to/pi05_mixed_p2_hybrid_2000 hybrid
```

This selects classic FSDP with a `(replicate=2, shard=4)` device mesh:
parameters are sharded within each node, gradients are replicated across nodes,
and the gradient norm is reduced once per sharding group. `NODE_LOCAL_WORLD_SIZE`
must be four on every worker, which RLinf derives from its placement metadata.
FSDP2 is rejected for this mode until its two-dimensional mesh path is audited
separately.

When the training nodes do not share the B1K dataset filesystem, sync only the
videos referenced by the formal sidecar and launch after a size-complete
preflight:

```bash
bash toolkits/b1k_grounded/sync_required_videos_and_launch.sh \
  training-host p1_simple_sg b1k-p1 /remote/path/to/pi05-mixed-p1
```

If multiple training nodes share the same remote dataset filesystem, use the
default `sync` mode only once. Pass `wait` as the fifth argument for the other
nodes so they preflight the shared copy and launch without retransferring it.
Pass `hybrid` as the sixth argument to launch one two-node job after preflight:

```bash
bash toolkits/b1k_grounded/sync_required_videos_and_launch.sh \
  training-host p1_simple_sg b1k-p1-hybrid \
  /remote/path/to/pi05-mixed-p1-hybrid wait hybrid
```

## Calibrated skill-reset evaluation

The grounded evaluator primes the task reward to the selected skill after GT
demo warmup. Earlier sequential stages are therefore marked complete and do
not gate the selected skill. Manipulation skills use their task-specific direct
predicates. `move to` uses the selected demo segment's terminal base region,
with default tolerances of 0.5 m and 45 degrees, instead of end-effector
distance.

Before evaluating a checkpoint, run the same entrypoint in demo-only
calibration mode:

```bash
python toolkits/b1k_grounded/eval_grounded_subtasks.py \
  policy=websocket \
  task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  eval_level=subtask \
  keep_running_after_success=false \
  run_episode_indices=[10] \
  demo_data_dir=/path/to/2025-challenge-demos \
  instance_reward_mode=task \
  log_path=/path/to/demo_predicate_calibration \
  +grounded_control_sidecar=/path/to/sidecar.parquet \
  +grounded_eval_view_dir=/path/to/subtask_eval_view \
  +grounded_control_profile=p1_simple_sg \
  +grounded_demo_calibration=true
```

The command writes `turning_on_radio_demo_predicate_calibration.json` and exits
with an error if any selected GT segment fails its own predicate. Only after
this check passes should checkpoint success rates be interpreted. Grounded
policy serving uses a 512-token prefix budget by default, matching SFT. Demo
calibration always replays an annotated segment to its end, even if the
whole-task predicate fires earlier; this is needed to validate post-goal skills
such as placing a manipulated object back on its support.

## Failure-state capture and recovery eligibility

BEHAVIOR subpool training can persist exact
`og.sim.dump_state(serialized=False)` states for later DAgger or recovery-data
collection. Every timeout is retained as an audit artifact. For supported
skills, the simulator also captures the first stable recovery event before a
later action can turn a recoverable state into an unrecoverable one. Enable it
only on the training environment and use an experiment-specific output
directory:

```yaml
env:
  train:
    subpool:
      failure_state_capture:
        enabled: true
        output_dir: /path/to/experiment/failure_terminal_states
        run_id: radio-pickup-step20
        # Set this for standalone checkpoint evaluation. Training supplies it
        # through the runner automatically.
        policy_global_step: 20
        tipped_angle_deg: 45.0
        stable_steps: 8
        max_linear_speed: 0.05
        max_angular_speed: 0.2
```

The simulator process writes the state locally instead of returning it through
Ray. Artifacts are grouped as
`<output_dir>/<hostname>/global_step_XXXXXX/`; each failure has a `.pt` state
and a `.json` sidecar with its checksum, source snapshot, DAPO collection and
sampling group, policy global step, reward components, termination reason,
simulator facts, failure tags, and recovery eligibility. The JSON `state_path`
is relative to its own directory. Metadata is published only after the state
file has been atomically installed.

The initial pickup analyzer distinguishes three cases without a VLM:

- `not_needed`: the gripper is empty, but the target remains upright on its
  original support; the existing pickup policy can retry.
- `eligible`: the target is tipped by at least `tipped_angle_deg`, remains on
  its original support, and has stayed below both speed thresholds for
  `stable_steps` consecutive simulator steps.
- `ineligible`: the target has left its audited original support, for example
  a radio that has fallen from the table.

Unsupported skills are marked `unknown`, not admitted optimistically. Extend
the skill-relative analyzer with task predicates before using those states for
recovery training. `capture_kind=stable_recovery_event` identifies the early
eligible capture; `capture_kind=terminal` identifies the exact timeout state.

For SSH-backed cross-datacenter collectors, the same configured path is on the
collector's filesystem. Sync those host directories back after collection,
preserving their hostname directory:

```bash
rsync -avzP remote-host:/path/to/experiment/failure_terminal_states/ \
  /path/to/experiment/failure_terminal_states/
```

When a collector needs a patched BEHAVIOR-1K checkout, pass its OmniGibson
source explicitly. This prevents the daemon from silently importing an older
copy from the virtual environment:

```bash
python toolkits/b1k_grounded/manage_remote_collectors.py start \
  --gpus 0,1,2,3 --ports 46100,46101,46102,46103 \
  --python /opt/venv/openpi/bin/python3 \
  --repo /mnt/public/daibo/timeline/0914/RLinf-vector-env \
  --omnigibson-path \
    /mnt/public/daibo/timeline/0914/BEHAVIOR-1K-vector-env/OmniGibson \
  --log-dir /mnt/public/daibo/results/b1k_collectors/vector4
```

Captured states are deliberately not inserted into the recovery pool
automatically. Only simulator-certified `eligible` records should be converted
to recovery snapshots, followed by a restore smoke test and a short empirical
recoverability rollout.

To build a catalog containing only certified terminal failures in its recovery
pool (plus the canonical records required for catalog validation), run:

```bash
python toolkits/b1k_grounded/build_failure_recovery_catalog.py \
  --failure-root /path/to/failure_states \
  --canonical-manifest /path/to/canonical/manifest.jsonl \
  --output-manifest /new/path/recovery_catalog/manifest.jsonl
```

Recovery records preserve the canonical target orientation in
`metadata.recovery_provenance`. This is important: a tipped recovery reset must
not redefine the tipped pose as the object's new upright reference.

## Canonical train and held-out state pools

Build canonical snapshots with `export_radio_pickup_init_states.sh`. The
launcher accepts `B1K_SUBPOOL_SIDECAR`, `B1K_SUBPOOL_EPISODES`, and
`B1K_SUBPOOL_GPUS`, so a larger audited sidecar can be exported over all local
simulator GPUs. Export more than the final pool size because the official demo
suffix must reproduce the selected skill predicate before a snapshot is
eligible.

Merge successful exports into a self-contained terminal-time catalog:

```bash
python toolkits/b1k_grounded/prepare_canonical_pool.py assemble \
  --input-manifest /path/to/episodes/*/manifest.jsonl \
  --output-manifest /new/path/candidates/manifest.jsonl \
  --subtask-id 1 \
  --horizon 1280
```

If the validated catalog is larger than the evaluation budget, select a fixed
candidate set across the demonstrated suffix-duration range before measuring
policy success:

```bash
python toolkits/b1k_grounded/prepare_canonical_pool.py select \
  --source-manifest /path/to/all_valid/manifest.jsonl \
  --output-manifest /new/path/candidates40/manifest.jsonl \
  --count 40 \
  --seed 20260911
```

Evaluate every candidate with the same frozen SFT checkpoint and the same
number of stochastic trajectories. Store counts as
`{"snapshots": {"<snapshot-id>": {"successes": N, "attempts": M}}}`. Then
create paired train and held-out catalogs across the empirical success range:

```bash
python toolkits/b1k_grounded/prepare_canonical_pool.py scores \
  --source-manifest /path/to/candidates40/manifest.jsonl \
  --metrics /path/to/batch_*/eval_metrics.json \
  --output /path/to/sft20k_scores.json \
  --expected-attempts 20
```

```bash
python toolkits/b1k_grounded/prepare_canonical_pool.py split \
  --source-manifest /path/to/candidates/manifest.jsonl \
  --scores /path/to/sft20k_scores.json \
  --output-dir /new/path/success_stratified_split \
  --split-size 20 \
  --seed 20260911
```

For formal multi-state DAPO, set
`subpool.outcome_snapshot_schedule=shuffled_round_robin` and
`subpool.sticky_outcome_snapshot=true`. Every logical group is then assigned
one state from a per-update shuffled traversal. Quota retries and auto-resets
reuse that exact state. Set `outcome_dynamic_sampling.max_attempts_per_group`
to a positive limit to fail explicitly with a `no-signal` state instead of
silently replacing a repeatedly all-positive or all-negative group.

`run_radio_pickup_canonical_baseline.sh` evaluates two self-contained 20-state
partitions for 20 SFT-20K trials per state, writes exact per-state JSON metrics,
and invokes the score and split commands above. The resulting train manifest is
accepted by `run_radio_pickup_canonical20_rl.sh`, which uses action-level PPO
ratios, a frozen SFT-20K KL reference, FP32 master weights with BF16 compute,
and one shuffled logical group per canonical state. Its four-A100 actor uses
`SHARD_GRAD_OP`, no CPU offload, no gradient checkpointing, forward/backward
prefetch, and micro batch size 100 by default. The latter was the largest
configuration with practical memory headroom in the matching 80-GiB profile;
override `B1K_RL_MICRO_BATCH_SIZE` when the actor hardware changes.

## Subpool rollout performance

Canonical snapshot files are immutable during one run, so repeated resets can
cache their deserialized state dictionaries inside each EnvWorker. Size the LRU
to the number of canonical states assigned to one worker; zero keeps the legacy
uncached behavior:

```yaml
env:
  train:
    subpool:
      state_cache_size: 20
```

`benchmark_behavior_chunk_step.py --num-resets 8 --state-cache-size 20`
measures reset and primitive-step throughput while also recording simulator
state, reward, termination, and camera hashes for correctness comparison. The
cache changes neither rendering nor observation pixels.

### One simulator process with multiple vector scenes

BEHAVIOR 3.7.2 places all `VectorEnvironment` scenes in one PhysX stage. The
`env_indices` argument limits action writes, rewards, termination checks, and
observations, but simulator callbacks and physics still advance for every scene.
The global scene registry must remain intact because articulation and contact
tensor views are indexed across the complete shared stage.
Subpool vectorization therefore uses a synchronized lifecycle:

- one `BehaviorProcess` and one `VectorEnvironment` per EnvWorker / GPU;
- all vector slots reset and restore together at a collection boundary;
- a completed slot is frozen logically and never resumed before the next full
  restore;
- terminal observations and canonical scene-zero states are captured at the
  exact transition where the slot completed;
- partial asynchronous resets are rejected for subpool execution.

Each simulator process also receives a private appdata directory keyed by the
stable `RLINF_NODE_RANK` (or hostname), EnvWorker rank, CUDA device, and process
index. This prevents concurrent Isaac Sim writers from corrupting one cache
while preserving shader-cache reuse across Ray restarts.

Canonical scene states are translated into each vector scene's coordinate
frame before loading. Active particle-system states are currently rejected
because copying them without an equivalent frame transform would be silently
incorrect. This restriction does not affect the radio pickup experiment, whose
snapshots have no active particle systems.

B1K's R1Pro proprioception exposes `robot_pos` in the global stage frame. RLinf
subtracts each vector scene's translation before sending that state to the
policy; otherwise slots 1--3 receive artificial offsets even when all four
slots load the same canonical snapshot. Vector scenes with a non-identity
rotation are rejected until every orientation and velocity field has an audited
frame conversion.

Set `env.train.total_num_envs` to the number of logical slots, not the number of
EnvWorkers. For example, 20 environment GPUs with four slots each use 80 total
environments. Keep `num_env_subprocess: 1`; the four scenes live inside that
single process. An outcome group must contain a whole number of EnvWorkers, so
`outcome_dynamic_sampling.group_size` must be divisible by four. Logical groups
may outnumber physical groups: with group size 20, 80 slots collect four groups
at once and a 20-state update takes five collection rounds.

Failure-state capture and same-filesystem dynamic pool updates operate per
vector slot. SSH collectors still require `dynamic_updates: false`, since their
filesystems are not shared; failure artifacts can be rsynced back as described
above. Cross-datacenter request / response payloads carry the complete local
vector batch.

Use the real simulator smoke test before increasing the slot count:

```bash
python toolkits/b1k_grounded/smoke_behavior_subpool_env.py \
  --manifest /path/to/canonical/manifest.jsonl \
  --token-mapping /path/to/structural_token_mapping.json \
  --output-dir /new/path/vector-smoke \
  --num-envs 4 --chunk-size 8 --reset-count 1 \
  --skip-intermediate-obs --skip-official-task-termination \
  --verify-failure-state-save --verify-dynamic-updates
```

Run a separate subset-step regression with `--staggered-timeouts`. It assigns
slots distinct one-to-N timeouts so one chunk exercises active subsets of size
N, N-1, ..., 1 while R1Pro's controller and gripper-contact callbacks remain
backed by the complete simulator scene registry.

With `--verify-dynamic-updates`, the smoke catalog uses a four-step timeout so
each slot has an auditable lagged state. The test then verifies that one recovery
snapshot per slot was appended, can be loaded, and matches its recorded checksum.

Then compare scalar and vector throughput with identical actions and snapshots
using `benchmark_behavior_chunk_step.py --num-envs 1` and `--num-envs 4`.
The reports include per-slot camera hashes, canonical state hashes, reward and
termination traces, reset time, physics-step rate, logical environment-step
rate, and max/mean per-slot differences for policy state and camera pixels. Do
not accept a speedup unless the smoke test passes and same-state vector slots
remain equivalent within the audited simulator tolerances.

Finite dynamic rollout batching removes the all-environment barrier at every
action-chunk boundary. EnvWorkers place ready observations in a shared route;
each rollout worker waits for one shard and coalesces additional ready shards
for a small bounded window. The epoch still ends only after every configured
trajectory is collected, so actor training receives the same complete on-policy
population:

```yaml
runner:
  enable_decoupled_mode: true
  val_check_interval: -1
rollout:
  pipeline_stage_num: 1
  collect_final_values: true
  dynamic_batching:
    enabled: true
    # Maximum incoming EnvWorker shards per inference call.
    max_batch_size: 5
    # Starts after the first shard arrives.
    max_wait_seconds: 0.1
```

The initial implementation is training-only and requires the EnvWorker count to
be divisible by the rollout-worker count. Bootstrap-value requests use a
separate route so a fast environment cannot be mistaken for another
environment's regular action request. Use
`benchmark_openpi_rollout_batch.py` to choose `max_batch_size` from measured
policy latency rather than assuming the largest batch is fastest. The rollout
log reports a batch-size histogram; reduce the wait only when the histogram
shows that requests still coalesce near `max_batch_size`.

`max_batch_size` counts incoming EnvWorker shards, not logical environments. If
each EnvWorker hosts `V` vector slots, one shard contains `V` observations and
the largest model batch contains `V * max_batch_size` rows. For example, with
80 logical environments on 20 EnvWorkers and four rollout workers, each shard
contains four rows and each rollout worker can receive at most five shards.
`max_batch_size: 1`, `2`, and `5` therefore produce model batches of at most 4,
8, and 20 rows, respectively. Re-benchmark this setting whenever the per-worker
vector width changes; actor and critic batch sizes need not change when the
total number of trajectories per update stays constant.

For VectorEnvironment correctness canaries, enable
`algorithm.outcome_dynamic_sampling.log_actor_shard_metrics`. In the standard
parallel routing layout, each actor shard receives a stable vector-slot index,
so `dynamic_sampling/actor_shard/*/success_rate` reveals persistent slot bias.
Treat these as routing diagnostics rather than replacement task metrics.

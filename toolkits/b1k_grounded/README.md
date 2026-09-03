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
object/part grounding eligibility. The builder intentionally stores structured
control rather than committing to model-specific location tokens.

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

The ready-to-run pilot config defaults to P2:

```bash
bash examples/sft/run_vla_sft.sh behavior_pi05_grounded_oracle_pilot
```

This is full-parameter action SFT: `actor.model.is_lora` is false and the SFT
model-building path does not freeze the vision encoder, language model, or
action expert. The three profiles differ only in the serialized condition:

- P0 contains the overall goal.
- P1 adds GT skill, typed argument roles, object names, and available part names.
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
  --action-horizon 32
```

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

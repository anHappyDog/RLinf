# π0.5 Memory Experiments

This directory contains standalone development tools for the π0.5-based memory
reproduction. They validate model capabilities before those capabilities are
integrated into distributed SFT or RL runners.

## R0: high-level text micro-overfit

`r0_overfit.py` checks that the base π0.5 checkpoint can learn and greedily
generate canonical B1K primitive text from current images and a full-task prompt.
It deliberately bypasses Ray and FSDP.

The default task-0000 microset contains four frames from each primitive:

- `pick up radio from coffee table`;
- `press radio`;
- `place radio on coffee table`.

All samples have the same task prompt. R0 omits state from the text prefix so the
three labels must be distinguished from images. Normalized state is added when
the high-level path is integrated into the production B1K pipeline in R1.

Use the OpenPI environment and a base checkpoint containing `model.safetensors`.
The loader accepts both the upstream OpenPI PyTorch layout and the converted
OpenPI_RLinf layout:

```bash
/opt/venv/openpi/bin/python toolkits/mem/r0_overfit.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --model-path /mnt/public/daibo/models/pi05_base_pytorch \
  --output-dir /path/to/results/r0_turning_on_radio
```

For the simpler fixed-label R0-A plumbing test:

```bash
/opt/venv/openpi/bin/python toolkits/mem/r0_overfit.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --model-path /mnt/public/daibo/models/pi05_base_pytorch \
  --output-dir /path/to/results/r0_fixed_label \
  --fixed-target "press radio" \
  --steps 200
```

The tool writes:

- `r0_manifest.jsonl`: fixed episode/frame/primitive targets;
- `metrics.json`: initial, final, and shuffled-image metrics;
- `predictions.jsonl`: decoded text for all evaluations;
- `trainable_tail.pt`: only the expert-0 tail parameters changed by R0.

The visual R0 gate requires at least 95% loss reduction, final CE below 0.1,
100% token and exact-match accuracy, and at most 50% exact match after images are
rotated across primitive groups. These thresholds test plumbing and conditional
overfitting; they do not measure held-out generalization.

### Validated run (2026-08-29)

The downloaded `lerobot/pi05_base` checkpoint has SHA-256
`0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`.
Both gates passed on an A100 80GB:

| Gate | Steps | Initial CE | Final CE | Final exact match | Shuffled-image exact match |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0-A fixed label | 200 | 8.9568 | 6.62e-9 | 100% | 100% (expected) |
| R0-B visual subtasks | 500 | 5.5021 | 1.15e-7 | 100% | 0% |

Artifacts are stored under `/mnt/public/daibo/experiments/pi_mem/`.

## R1: episode-held-out high-level training

Build the full deterministic R1 manifest before training:

```bash
/opt/venv/openpi/bin/python toolkits/mem/build_r1_manifest.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --output-dir /mnt/public/daibo/experiments/pi_mem/r1_manifest \
  --samples-per-primitive 2 \
  --seed 42
```

The builder canonicalizes compound primitives, falls back to their associated
skills when a primitive description is empty, samples manipulation intervals,
and splits complete episodes within every task. With the current B1K snapshot it
produces 104,720 train, 13,044 validation, and 13,118 test examples over
8,000/1,000/1,000 disjoint episodes. All 65,441 primitives were converted; none
were skipped.

Run a bounded held-out pilot before scaling to the full mixed-task SFT:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/venv/openpi/bin/python toolkits/mem/r1_train.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --manifest-dir /mnt/public/daibo/experiments/pi_mem/r1_manifest \
  --model-path /mnt/public/daibo/models/pi05_base_pytorch \
  --output-dir /mnt/public/daibo/experiments/pi_mem/r1_task0_pilot \
  --task-index 0 \
  --max-train-samples 96 \
  --max-val-samples 24 \
  --input-mode image \
  --batch-size 4 \
  --steps 300 \
  --max-new-tokens 12
```

The task-0000 pilot uses 78 train and 15 disjoint validation episodes. It
improved held-out exact match from 0% to 83.33%, validation CE from 5.1341 to
0.0959, and validation token accuracy from 38.62% to 97.24%. This is an R1
pipeline/pilot result, not the final multi-task high-level model. The test split
remains untouched.

### Current-state ablations

R1 supports three matched input modes:

- `image`: three current RGB views and no state tokens;
- `state`: an image-masked control with normalized 23D state tokens;
- `image_state`: three current RGB views plus normalized 23D state tokens.

State extraction reuses the existing BEHAVIOR policy's 256D-proprio-to-23D
mapping. Quantile statistics are computed from the selected training frames
only and saved as `state_norm_stats.json`; validation and test frames never
contribute to them. Multi-task runs use task-balanced sampling by default and
report micro/macro task exact match plus verb, object, destination, and step
count diagnostics.

The matched 96/24 task-0000 pilot produced 83.33% image-only, 91.67% state-only,
and 75.00% image+state exact match. The subsequent 960/120, three-seed study
selected state-only, the last two Gemma layers, and 500 updates as the safer R1
recipe:

| Recipe | Validation exact match | Action relative RMSE | Gate pass rate |
| --- | ---: | ---: | ---: |
| state, 1,000 steps | 91.11% ± 2.19% | 12.04% ± 3.76% | 1/3 |
| state, 500 steps | 88.33% ± 2.04% | 5.97% ± 1.84% | 3/3 |
| image+state, 500 steps | 88.33% ± 2.04% | 8.10% ± 4.11% | 2/3 |

The 1,000-step recipe was rejected despite its higher text accuracy because its
action drift was unstable. An evaluation-loader bug that advanced the training
sampler epoch was fixed before these summaries; all reported seeds now use
independent training sequences and an identical validation selection.

The image+state seed-43 counterfactual scored 90.83% with aligned inputs, 39.17%
with mismatched images, and 20.00% with mismatched state. Thus the model uses
both modalities, but RGB does not improve aggregate clean task-0000 accuracy at
this budget. This motivates explicit correct/repeated/shuffled history controls
when short memory is evaluated.

A seven-task task-balanced pilot (100 train and 20 validation examples per
task) reached 86.43% micro/macro exact match. Verb, object, and destination
accuracy were 95.81%, 98.95%, and 89.63%. Its 32-sample action-preservation run
had 3.13% relative RMSE and 0.9994 mean cosine similarity.

A harder four-task pilot used `turning_on_radio`, `picking_up_trash`,
`putting_dishes_away_after_cleaning`, and `preparing_lunch_box`. With 100 train
and 20 validation examples per task, macro exact match was 50.00%; per-task
exact match was 65%, 70%, 30%, and 35%, respectively. This is a useful R2 entry
point but not evidence that long-horizon high-level prediction is solved. Its
32-sample action-preservation run passed with 5.33% relative RMSE and 0.9985
mean cosine similarity.

The same 500-step recipes were then transferred to the action-capable
OpenPI-Comet `pi05-b1kpt50-cs32` checkpoint. The seven-task pilot reached 80.00%
exact match with 1.05% action relative RMSE. The harder four-task pilot reached
56.25% macro exact match (70% radio, 80% trash, 30% dishes, 45% lunch-box) with
2.26% action relative RMSE. Both pass the provisional action gate. This B1K
checkpoint is the main initialization going forward; generic `pi05_base` remains
an ablation and R0 plumbing reference.

### Action-preservation gate

Check that text-tail training has not materially changed the original action
policy by sampling actions with identical observations, initial noise, and
Euler steps before and after loading the tail:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/venv/openpi/bin/python \
  toolkits/mem/action_preservation.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --calibration-manifest /path/to/r1_run/val_selection.jsonl \
  --state-norm-stats /path/to/r1_run/state_norm_stats.json \
  --model-path /mnt/public/daibo/models/pi05_base_pytorch \
  --tail-checkpoint /path/to/r1_run/trainable_tail.pt \
  --output-dir /path/to/r1_run/action_preservation
```

The 10% threshold is a provisional regression guard, not an environment-success
metric. Run it across seeds and on a larger calibration selection before
accepting a text-tail recipe. `summarize_r1.py` verifies identical validation
selection hashes and aggregates both text and action metrics; use
`r1_counterfactual.py` to compare aligned, mismatched-image, and mismatched-state
generation.

## R2: task/primitive-conditioned action SFT

The production BEHAVIOR SFT loader accepts four prompt sources through
`data.prompt_source`: `task`, `skill`, `primitive`, and `mixed`. Primitive mode
derives a canonical label from the native annotation. Mixed mode trains the same
policy on full-task and primitive prompts;
`data.primitive_prompt_probability` sets the primitive fraction (0.5 by
default). The per-frame choice is deterministic across ranks and workers.

Primitive and mixed modes drop any sample whose 32-step action chunk crosses a
primitive boundary. Both prompt types in mixed mode therefore see the same
boundary-safe action distribution. The alignment uses the union of referenced
skill ranges instead of the primitive's outer duration envelope; the latter is
nested or overlapping in many long tasks. Residual frames covered by multiple
primitives are also excluded rather than assigned an arbitrary label.

Train the comparison checkpoint with both interfaces:

```bash
export EMBODIED_PATH="$(pwd)/examples/sft"
export REPO_PATH="$(pwd)"
/mnt/public/daibo/venv/behavior_openpi/bin/python examples/sft/train_vla_sft.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name behavior_pi05_vla \
  data.train_data_paths=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  data.behavior_dataset_root=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  data.prompt_source=mixed \
  data.primitive_prompt_probability=0.5 \
  data.use_skill=false \
  actor.model.model_path=/mnt/public/daibo/models/pi05_b1kpt50_pt \
  actor.model.openpi.assets_dir=/mnt/public/daibo/models/pi05_b1kpt50_pt \
  actor.model.openpi.asset_id=physical-intelligence/behavior
```

The important configuration contract is the prompt source, checkpoint, and
inherited norm stats. `pi05_b1kpt50_pt` is the local PyTorch conversion of
OpenPI-Comet's 50-task B1K checkpoint, not an official Physical Intelligence
B1K release. For the Oracle-HL gate, use this one mixed checkpoint twice: first
with the full-task prompt and then with the annotation-derived primitive. A
separate `prompt_source=primitive` checkpoint remains useful as a deployment
ablation, but cannot by itself isolate the benefit of the prompt interface.

The required closed-loop comparison is:

1. original B1Kpt50 + full task (pre-SFT baseline);
2. mixed checkpoint + full task (isolates extra action SFT);
3. the same mixed checkpoint + Oracle primitive (isolates hierarchical prompting).

### Conservative R2 recovery recipe

Use `behavior_pi05_vla_conservative` when full-parameter mixed SFT damages the
base policy. It makes three changes while keeping the task/primitive interface:

- freezes SigLIP, the token embedder, and Gemma expert-0 with
  `actor.model.openpi.train_expert_only: true`;
- reduces the learning rate to `2.5e-6` and saves steps 25/50/75/100;
- sets `data.mixed_boundary_fallback_to_task: true`, so a chunk that crosses a
  primitive boundary retains full-task action supervision instead of being
  discarded. Primitive-only mode remains boundary-safe and strict.

Launch it with the same path overrides as the original recipe:

```bash
export EMBODIED_PATH="$(pwd)/examples/sft"
export REPO_PATH="$(pwd)"
/mnt/public/daibo/venv/behavior_openpi/bin/python examples/sft/train_vla_sft.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name behavior_pi05_vla_conservative \
  data.train_data_paths=/path/to/2025-challenge-demos \
  data.behavior_dataset_root=/path/to/2025-challenge-demos \
  actor.model.model_path=/path/to/pi05_b1kpt50_pt \
  actor.model.openpi.assets_dir=/path/to/pi05_b1kpt50_pt \
  actor.model.openpi.asset_id=physical-intelligence/behavior
```

Do not select this checkpoint by flow loss alone. Evaluate each saved step with
the original task prompt and corrected Oracle primitive, and require the task
path to retain the base policy's closed-loop capability.

If expert-only training retains task behavior but cannot use the primitive
prompt, run the R2c causal ablation with the same conservative config plus:

```text
actor.model.openpi.train_expert_only=false
actor.model.openpi.freeze_vision_encoder=true
```

This keeps SigLIP fixed but lets the token embedder and Gemma expert-0 adapt to
the task/primitive routing interface. Keep the 100-step budget, learning rate,
mixture, and boundary fallback unchanged so the freeze scope is the only
experimental variable.

### Short-memory causal evaluation

The paper-aligned v2 recipe uses six observations (five past plus current).
B1K training samples use a 30-frame stride at 30 Hz. BEHAVIOR evaluation keeps
every 32-action policy decision at the official 30 Hz cadence, giving a
1.067-second interval and a 5.33-second oldest-to-current window. All six
normalized proprioceptive states are continuous prefix tokens; the duplicate
discrete current-state text input is disabled. Older K=4 checkpoints predate
the corrected temporal position-encoding order and are diagnostic artifacts,
not v2 initializations.

`short_memory_paired_eval.py` compares one memoryless checkpoint against the
matching short-memory checkpoint under three history conditions:

1. correctly ordered history (`short_none`);
2. valid past slots replaced by the current observation
   (`short_repeat_current`);
3. valid past slots deterministically reversed while the current observation,
   padding mask, and temporal offsets stay fixed (`short_shuffle_past`).

The controls use the same checkpoint and rollout seed. A short-memory gain is
credible only if correctly ordered history beats both the memoryless policy and
both content/order counterfactuals; training loss alone is insufficient.
The canonical `turning_on_radio` protocol matches the official B1K environment:
30 Hz action/rendering, 120 Hz physics, and a 2,048 simulator-step horizon
(64 policy decisions with 32-action chunks, about 68 seconds). It keeps
`skip_intermediate_obs_in_chunk=false`. The latter still executes physics,
reward, and termination logic when enabled, but changes OmniGibson's render and
observation schedule, so it is reserved for throughput experiments rather than
acceptance evaluation. Full-frame videos are encoded at 30 fps to match the
configured action/observation frequency. For K=6 short memory, consecutive
policy decisions are used (`history_decision_stride=1`), giving an approximately
5.33-second oldest-to-current context.

Before launching the closed-loop matrix, run the paired offline loss gate on a
saved v2 checkpoint. It holds demonstration actions, flow noise, and flow time
fixed while replacing or shuffling only the history content:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/public/daibo/venv/behavior_openpi/bin/python \
  toolkits/mem/short_memory_offline_eval.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --checkpoint /path/to/global_step_100 \
  --assets-dir /mnt/public/daibo/models/pi05_b1kpt50_pt \
  --output-dir /path/to/offline_step_100
```

`directional_gate=true` means the mean correct-history loss is below both
controls. It is a cheap sensitivity screen, not a substitute for closed-loop
success and progress comparisons.

When a closed-loop trace reaches the pickup stage but never commands gripper
closure, run the demonstration grasp-event generation gate. It selects balanced
open-control, close-onset, and closed-hold windows for one canonical primitive,
then samples actions with several fixed flow-noise draws:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/public/daibo/venv/behavior_openpi/bin/python \
  toolkits/mem/grasp_event_offline_eval.py \
  --dataset-root /mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
  --checkpoint /path/to/global_step_1100 \
  --assets-dir /mnt/public/daibo/models/pi05_b1kpt50_pt \
  --output-dir /path/to/grasp_event_step_1100 \
  --samples-per-phase 8 \
  --max-samples-per-episode 2 \
  --open-control-margin 32 \
  --noise-draws 4 \
  --history-conditions correct repeat_current shuffle_past
```

The open-control margin keeps negative windows away from a closure that falls
just beyond the 32-action target horizon. The tool writes a readable selection
manifest, generated-action metrics, and `selection_samples.pt`. Reuse the same
observations for another checkpoint with `--selection-cache <path>` and
`--reuse-selection-cache`, so checkpoint comparisons differ only in weights.
The gate requires event closure, correct-hand closure, and closed-timestep
recall to reach 50%, while far-open false closure must stay at or below 25%.
With multiple history conditions, every sample uses identical flow-noise draws.
`metrics.json` reports per-condition action MAE for base, trunk, both arms, and
both grippers, non-gripper cosine similarity, and paired deltas against correct
history. Passing the event gate is not enough to claim temporal memory: correct
history must also improve over shuffled past, which preserves content but
destroys order.

The rollout observation schema must forward all four `history_*` tensors and
masks. Runs made before this transport was fixed were effectively current-only,
even when the launcher selected correct, repeated, or shuffled history. Do not
use their condition deltas as memory evidence.

The original streaming loader independently partitioned 250-frame chunks and
cleared history at every chunk load. A K=6, stride-30 observation needs 151
consecutive frames, so up to the first 150 frames of every chunk had partial
history. The fixed short-memory path partitions complete episodes across
rank/worker consumers, shuffles only episode order, and streams each episode's
chunks contiguously. History is retained across same-episode chunk boundaries
and reset only when the episode changes; memoryless chunk sampling is unchanged.
Checkpoints trained before this fix remain diagnostic artifacts.

The fixed path passed a production smoke on four A100 ranks with two spawned
DataLoader workers per rank, FULL_SHARD, gradient checkpointing, and a global
batch of 32. Two optimizer steps completed with finite loss and gradients; the
steady second step took 2.03 seconds. A real task-0000 stream retained a 6/6
valid K=6 mask across frame 249 to the next chunk's frame 250.

### K=6 v3 memory-critical curriculum

`behavior_pi05_short_memory_v3.yaml` adds two training-only changes while
keeping the v2 model shape and initialization:

- annotation-aligned pickup windows whose 32-step target contains a gripper
  closure are always retained and marked critical; 20% of noncritical windows
  remain as open/navigation controls;
- critical samples with a complete K=6 context use a paired ranking loss against
  `repeat_current` and `shuffle_past`. Correct and controlled histories share
  actions, flow noise, and flow time. The control loss is stop-gradient, so the
  auxiliary objective improves the correct-history prediction without training
  the policy to intentionally fail when history is corrupted.

The ranking margin is 0.01 and its weight is 0.25. If any FSDP rank has a
critical sample, a distributed max reduction makes every rank execute the same
two control forwards; ranks without a local critical sample contribute a
differentiable zero. This prevents mismatched FSDP collectives.

On a real 256-sample task-0000 stream, 17.19% of yielded windows were critical,
88.28% had complete history, and the first critical window was episode 10 frame
1064 (31 frames before the demonstrated closure). A forced-critical four-A100
smoke exercised both controls with finite backward: repeat/shuffle loss deltas
were +0.0414/+0.00346, auxiliary loss was 0.0088, total loss was 0.120, and the
gradient norm was 4.28.

A 50-step four-A100 pilot completed without distributed errors and saved full
checkpoints at steps 25 and 50. The first event cache used for evaluation was
created before the episode-contiguous stream fix and must not be reused: it
stores observation tensors, not only frame indices, and only 7/24 samples had a
complete K=6 history. The replacement cache is
`/mnt/public/daibo/results/mem_short_k6_streamfix_grasp_event_selection/selection_samples.pt`;
all 24 samples have six valid frames. On that cache, step 50 correct / repeat /
shuffle event rates were 73.44% / 43.75% / 75.00%, recalls were 59.34% / 38.18%
/ 59.71%, and open false-positive rates were 6.25% / 15.625% / 3.125%.
Correct history clearly beats repeat-current but does not beat shuffled past,
so this recipe passes the history-content gate and fails the temporal-order
gate. Do not extend it solely because the training ranking deltas are positive.

Launch v3 with the same path overrides as v2:

```bash
export EMBODIED_PATH="$(pwd)/examples/sft"
export REPO_PATH="$(pwd)"
/mnt/public/daibo/venv/behavior_openpi/bin/python \
  examples/sft/train_vla_sft.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name behavior_pi05_short_memory_v3 \
  data.train_data_paths=/path/to/2025-challenge-demos \
  data.behavior_dataset_root=/path/to/2025-challenge-demos \
  actor.model.model_path=/path/to/pi05_b1kpt50_pt \
  actor.model.openpi.assets_dir=/path/to/pi05_b1kpt50_pt \
  actor.model.openpi.asset_id=physical-intelligence/behavior
```

Select an early checkpoint with the exact-observation three-condition action
fidelity gate. Do not accept v3 from training loss alone.

For simulator-state diagnostics, `midstage_pickup_eval.py` restores each raw
HDF5 episode's `scene_file`, loads the sparse UUID-tagged state sequence through
the requested frame, and can replace the first live policy input with a cached
demonstration observation. `--oracle-actions` additionally replaces the first
normalized model-action chunk with the matching demonstration actions:

```bash
/mnt/public/daibo/venv/behavior_openpi/bin/python \
  toolkits/mem/midstage_pickup_eval.py \
  --checkpoint /path/to/global_step_1100 \
  --norm-stats-path /path/to/norm_stats.json \
  --instance-dir /path/to/turning_on_radio_instances \
  --raw-data-dir /path/to/2025-challenge-rawdata \
  --instance-ids 0,0,0,0 \
  --episode-indices 10,20,30,40 \
  --frame-indices 1080,1149,914,1046 \
  --selection-cache /path/to/selection_samples.pt \
  --oracle-actions \
  --output-dir /path/to/midstage_results \
  --run
```

This is a takeover consistency test, not a policy acceptance metric. Contact
trajectories can diverge after sparse-state restoration even when the first
observation and action chunk match the demonstration exactly.

```bash
/mnt/public/daibo/venv/behavior_openpi/bin/python \
  toolkits/mem/short_memory_paired_eval.py \
  --memoryless-checkpoint /path/to/memoryless/global_step_500 \
  --short-memory-checkpoint /path/to/short_memory/global_step_500 \
  --norm-stats-path /path/to/pi05_b1kpt50_pt/physical-intelligence/behavior/norm_stats.json \
  --instance-dir /path/to/turning_on_radio_instances \
  --instance-ids 242,295,211,203 \
  --max-episode-steps 2048 \
  --output-dir /path/to/paired_results \
  --run
```

### Validated Behavior environment

Use `/mnt/public/daibo/venv/behavior_openpi/bin/python` for production B1K
streaming and simulator work. It contains OmniGibson 3.7.2, OpenPI, PyTorch
2.5.1+cu124, Ray, and sees all four A100 GPUs. A real primitive-conditioned
batch passed the full streaming and transform pipeline:

- raw RGB: three `[3, 720, 720]` views;
- raw action chunk: `[32, 23]`;
- transformed RGB: three `[B, 224, 224, 3]` tensors;
- transformed state/actions: `[B, 32]` and `[B, 32, 32]`;
- tokenized prompt: `[B, 200]`;
- one `pi05_b1kpt50_pt` flow-loss forward: finite `[1, 32]`, mean `0.00507`.

Mixed-mode streaming was additionally checked on real task-0000 data: eight
consecutive boundary-safe chunks produced four task prompts and four primitive
prompts, each with an unpadded `[32, 23]` action target. A transformed two-sample
mixed batch also passed the production SFT wrapper on `pi05_b1kpt50_pt` with a
finite scalar flow loss of `0.03032`.

A four-A100 FSDP benchmark also completed 20 optimizer steps with the production
mixed data path and a global batch of 256 (`micro_batch_size=32` per rank and two
gradient-accumulation micro-batches). The validated settings are full sharding,
bf16 parameter compute with fp32 reductions, non-reentrant gradient
checkpointing, forward prefetch, backward prefetch (`pre`), limited all-gathers,
and `no_sync` gradient accumulation. Excluding the first step's 32-worker data
loader cold start, the 19 steady-state steps averaged 7.655 seconds (7.613-second
median, 7.559--8.077-second range). Peak sampled memory was approximately
66--67 GiB per A100. CPU offload remains disabled, and Liger is unavailable for
the OpenPI model type.

Set `PYTHONPATH` to this RLinf checkout when invoking the environment so it does
not import an older checkout. `pip check` reports several declared-version
mismatches in the prebuilt environment, but the exact data/model path above is
validated. Treat checkpoint save/resume and simulator rollout as the remaining
compatibility tests.

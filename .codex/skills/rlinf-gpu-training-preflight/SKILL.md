---
name: rlinf-gpu-training-preflight
description: Preflight, tune, launch, or resume RLinf SFT/RL GPU training jobs by choosing an efficient micro-batch/global-batch plan, preserving the sample budget, and validating memory, throughput, storage, and checkpoint invariants. Use before long training runs or when changing GPU hardware or batch sizes; do not use for evaluation-only jobs.
---

# RLinf GPU Training Preflight

Optimize steady-state global samples per second while preserving training semantics and leaving enough memory for a reliable run. High GPU utilization or high allocated memory is evidence, not the objective.

## Scope and authorization

- Treat read-only inspection, batch planning, and short smoke tests requested as part of a training launch as normal implementation steps.
- Do not stop an unrelated live job, delete outputs, or overwrite a checkpoint without the user's authorization.
- Preserve user-selected models, data, seeds, and experimental comparisons. Surface any change required for feasibility.

## Choose the first MBS by estimation

Do not scan upward from `MBS=1`. Inspect the effective model, precision, FSDP world size, trainable parameters, gradient checkpointing, image count/resolution, history length, action horizon, and prior peak-memory logs. Prefer measurements from the same model and input shape.

When two comparable measurements exist, estimate peak memory as `static_memory + MBS * activation_memory` and select the largest practical power-of-two MBS below the memory target. With one or no comparable measurement, use the closest known workload profile and make the uncertainty explicit.

For the common RLinf OpenPI/pi05 workload (about 3.35B parameters, BF16 FSDP full-shard over 8 GPUs, gradient checkpointing, full SFT, two images plus short visual history), use these first probes:

| Per-GPU VRAM | First MBS probe |
| --- | ---: |
| 80 GB or more | 4 |
| 40-48 GB | 2 |
| 24 GB or less | 1 |

On an 80GB GPU, never start this profile at MBS 1 or 2. Start at MBS 4. Deviate only when a materially larger model/input shape or prior peak-memory evidence predicts that MBS 4 is unsafe, and record that evidence before launching.

After the first probe:

- If every rank uses at most 85% of VRAM and retains at least 10 GiB, probe double the MBS next.
- If the job is stable and fixed-shape with at least 4 GiB or 5% free on every rank, accept the current MBS without probing smaller values.
- If it OOMs or leaves less than the required reserve, halve MBS once and retry after cleaning up failed workers. Do not restart the search at 1.
- For variable sequence/image shapes, reserve at least 10% or 8 GiB, whichever is larger.

## Make the batch plan valid

Compute the data-parallel degree from the effective placement, then require:

```text
gradient_accumulation = GBS / (MBS * data_parallel_degree)
```

`GBS` must be divisible by `MBS * data_parallel_degree`. Prefer accumulation between 1 and 4; treat accumulation above 8 as a throughput warning that requires measured justification. For 8-way data parallelism with MBS 4, GBS 128 gives accumulation 4 and is the default comparison point.

When changing GBS, preserve the intended total samples or tokens rather than the raw optimizer-step count. Recompute training steps, warmup, decay boundaries, save/eval intervals, and resume offsets in sample units. Report any rounding difference. Keep GBS and the sample budget identical across baseline and experiment comparisons unless the user explicitly chooses otherwise.

## Emit a launch manifest

Every training launch or resume must produce a user-visible launch manifest. Prepare the intended values before launch, then read them back from the effective Hydra configuration, Ray placement, dataset metadata, and first stable training steps. Do not treat CLI overrides or the launch script alone as proof of the effective configuration.

Immediately after launch:

1. Save the effective manifest as `launch_manifest.md` or `launch_manifest.json` in the result directory.
2. Print the important fields to the user in the current conversation; a log path by itself is insufficient.
3. Compare intended and effective values. If a material field differs or cannot be verified, flag it immediately and do not claim the launch is successful.

The manifest must include, as applicable:

- **Identity:** host/node, start time, tmux or job identifier, repository path and revision, result directory, experiment name, seed, base checkpoint, and whether the run starts fresh or resumes.
- **Hardware/parallelism:** node count, GPU count/model/VRAM, world size, data-parallel degree, training backend, sharding strategy, CPU/offload settings, and gradient checkpointing.
- **Model/input:** model type and size, trainable scope, frozen components, LoRA settings, precision/mixed precision, image count/resolution, history settings, state/action dimensions, and action horizon.
- **Batch/budget:** MBS, GBS, derived gradient accumulation, max/remaining steps, total samples or tokens, sample-budget rounding, and expected epochs when meaningful.
- **Optimizer/schedule:** optimizer, initial and minimum learning rates, scheduler, warmup steps or samples, weight decay, beta values, epsilon, gradient clipping, and any loss weights.
- **Data composition:** every dataset path/repository, split, task IDs and human-readable task names, episode/frame counts, mixture weights or sampling ratios, FPS, worker count, and preprocessing that changes the examples.
- **Checkpoint/logging:** save/eval intervals, whether optimizer/training state is saved, expected checkpoint size/count, log backends, and Ray/output filesystems.
- **Observed health:** after the warmup and at least 10 stable steps, current step, median/P90 step time, global samples per second, peak memory/headroom on every rank, GPU utilization, estimated completion time, and absence or presence of OOM, NaN/Inf, and exceptions.

Write `N/A` or `disabled` for inapplicable fields rather than silently omitting them. Include derived values such as accumulation and total samples explicitly so arithmetic mistakes are visible.

## Run a bounded representative probe

Use the real training entrypoint, representative maximum input shapes, and the intended precision/FSDP/checkpointing settings. Run one compile/warmup step followed by at least 10 stable optimizer steps. Exclude startup and compile time from throughput.

Record:

- median and P90 optimizer-step time;
- global samples per second;
- peak allocated/reserved memory on every rank;
- GPU utilization, dataloader stalls, OOM/retry events, NaN/Inf, loss, and gradient norm.

Choose the candidate with the best stable global samples per second that satisfies the memory reserve. Do not spend time benchmarking lower MBS values after a higher candidate has already passed unless a fairness/debugging question specifically needs that comparison.

## Validate checkpoint and storage invariants

Before the long run:

- Check free space and quota on the actual output and Ray spill filesystems. Do not stage large checkpoints in `/tmp` without verifying its capacity.
- Estimate concurrent peak storage for temporary shards, distributed training state, consolidated weights, logs, and the previous checkpoint.
- Require a positive reachable `runner.save_interval` when the run or wrapper expects a final checkpoint. Reject contradictions such as `save_interval: -1` combined with a postcondition that waits for `full_weights.pt`.
- Use `save_training_state: true` when exact optimizer/scheduler resume is required. State explicitly when only model weights are being preserved.
- Verify the saved artifact exists, is nontrivial in size, and can be loaded before removing or replacing any source checkpoint.

## Launch and early monitoring gate

Launch the long job only after the selected probe passes. Use a unique output directory and Ray temporary directory, preserve the exact command/config, and verify the effective logged MBS, GBS, world size, accumulation count, LR schedule, data paths, and save settings.

Inspect the first 10 stable long-run steps. If throughput is materially below the probe, GPU utilization is persistently low while memory headroom is large, or the effective configuration differs from the plan, stop early when authorized, diagnose, and retune instead of allowing an hours-long inefficient run.

Report the effective launch manifest, selected batch plan, measured throughput, peak memory/headroom, preserved sample budget, estimated completion time, output path, and checkpoint policy to the user.

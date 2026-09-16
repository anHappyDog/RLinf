# B1K RL experiment archive

Use two Markdown records for every production run in the dedicated local Git
repository `/mnt/public/daibo/timeline/0831/b1k-rl-experiments`:

- a central append-only `EXPERIMENTS.md` index;
- a self-contained `experiments/<experiment-name>.md` record.

Create and commit both records before launch. Commit subsequent result updates
to this documentation repository, not the RLinf or BEHAVIOR-1K repositories.
Do not create a new code commit merely because a new experiment reuses an
existing clean source revision. Do not reuse a result directory for a different
source revision or hypothesis. Failed startup attempts remain events inside the
original record, with their preserved logs.

## Central index entry

Append one row when reserving the run. Keep the row concise and link to the
run-local record.

```markdown
| Start (UTC) | Experiment | Status | RLinf / B1K commits | Parent checkpoint | Primary result | Record |
| --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD HH:MM | `<name>` | prepared/running/completed/failed/stopped | `<rlinf12>` / `<b1k12>` | `<checkpoint>` | pending or concise result | [record](experiments/<name>.md) |
```

Update the existing row's status and primary result; never delete an old row or
repurpose it for another run.

## Run-local record template

```markdown
# <experiment name>

## Status

- Lifecycle: prepared/running/completed/failed/stopped
- Created / started / ended (UTC):
- Owner or launcher:
- Hypothesis and one primary comparison:

## Immutable identity

- RLinf repository / branch / commit / tree / remote-push status:
- BEHAVIOR-1K repository / branch / commit / tree / remote-push status:
- Source manifests and per-host verification result:
- Base or resume checkpoint and artifact identity:
- Result directory and experiment name:

## Effective configuration

- Task, subtask, train/eval state manifests, and logical-state schedule:
- Reward, termination, gamma/lambda/bootstrap, and advantage normalization:
- DAPO/resampling and actor signal gate:
- PPO ratio level, clip, KL/reference, policy/critic epochs and clipping:
- Trajectories, groups, MBS, policy/critic GBS, and derived accumulation:
- Precision, FSDP, value head/cache/warmup, and trainable parameters:
- Simulator mode, cameras, horizon/FPS, dynamic batching, and compression:
- Hardware placement and collector endpoint mapping:
- Max steps/epochs, seed, save/eval intervals, and failure-state capture:

## Operations and artifacts

- Preflight, collector start, launch, observe, stop, and resume commands:
- Driver, metrics, effective config, TensorBoard, checkpoint, collector, video,
  and failure-state paths:
- Tmux/job identities and expected first-useful-metric time:

## Lifecycle events

| UTC time | Event | Evidence / artifact |
| --- | --- | --- |

## Results

Record step-indexed success, return/reward components, per-state results,
advantages, actor ratio/KL/clipping/gradient metrics, critic loss/explained
variance, timing, failures, and evaluation confidence intervals when available.

## Conclusion

- Outcome relative to the stated hypothesis:
- Known confounders and what this run does not establish:
- Best checkpoint and selection criterion:
- Next action:
```

The result summary must distinguish online rollout success from fixed-seed or
held-out evaluation. Record sample counts; do not compare percentages without
their denominators and state composition.

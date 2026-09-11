---
name: b1k-eval
description: Launch reproducible BEHAVIOR-1K closed-loop evaluations for COMET-native or RLinf/FSDP pi05 checkpoints, including cross-datacenter remote environment collectors. Use when asked to start, compare, rerun, monitor, or locate a B1K evaluation; do not use for training or dataset generation.
---

# B1K Evaluation

Launch the requested evaluation and hand it back once the processes are healthy. Do not wait for completion unless the user explicitly asks for monitoring or results.

## Required inputs

Resolve these from the request, nearby experiment artifacts, and prior matching evaluations:

- checkpoint format and exact checkpoint directory
- structural-token mapping and normalization statistics
- task, episode indices, subtask range, and control profile
- grounded sidecar when using grounded control
- unique run name, ports, GPU assignment, and output directory

Do not silently substitute a different checkpoint, episode, sidecar, skill, sampler, noise level, or seed. A COMET websocket server requires `model.safetensors` and matching normalization assets. The RLinf evaluator can load a complete RLinf/FSDP checkpoint directly; do not convert it merely to satisfy the websocket workflow.

Read [references/environment.md](references/environment.md) for the authoritative local paths and required environment variables. Then choose exactly one launch mode:

- Read [references/launch.md](references/launch.md) for a COMET-native policy served over websocket to standalone B1K evaluators.
- Read [references/rlinf-crossdc.md](references/rlinf-crossdc.md) for an RLinf/FSDP checkpoint evaluated through RLinf Ray workers, including remote collectors across datacenters.

## Preflight

1. Work from `/mnt/public/daibo/timeline/0831/RLinf` and use the fixed Python and dataset paths in `environment.md`.
2. Verify the checkpoint, normalization stats, token mapping, sidecar, demo parquet, task-instance metadata, and output parent exist. Check that the OmniGibson asset version matches the installed environment; do not fall back to similarly named old asset roots.
3. Inspect `nxb_4090` GPU use, existing tmux sessions, target ports, and exact command lines. Never kill unrelated jobs. Reuse a process only after proving that its checkpoint, profile, task, and port match the request.
4. On `nxb_4090`, set `TMPDIR=/mnt/public/daibo/tmp` inside every B1K evaluator process before Python starts. This is mandatory: B1K otherwise fills the host's small `/tmp` filesystem. Put all other caches below the same persistent root, with a run-specific TorchInductor cache for every policy server and a run/slot-specific OmniGibson app-data directory for every evaluator. Never broadly clear `/tmp` or shared cache roots.
5. Create persistent, inspectable tmux sessions for the selected topology. Preserve a self-contained launch script in the result root, including checkpoint validation, seeds, sampler settings, output paths, and restart/skip behavior.

## Choose placement

For COMET websocket evaluation, default to serving pi05 and running B1K on `nxb_4090`. Since one B1K process and one pi05 server each occupy a GPU, use explicit non-overlapping GPU assignments; the established four-GPU pattern is policy on GPUs 0/2 and B1K on GPUs 1/3.

Prefer split placement when an available pi05 host is in the same datacenter/network as `nxb_4090`, has the same checkpoint files, and can spare inference GPUs. Launch pi05 there and use all four 4090 GPUs for B1K. The policy server binds `0.0.0.0`; pass its reachable private IP as `model.host`. Before launching the evaluator, prove from `nxb_4090` that the exact policy port is reachable. Fall back to colocated placement if routing, latency, filesystem visibility, or server health is uncertain.

Do not assume SSH reachability implies websocket reachability. Do not expose the policy server outside the trusted private network.

For RLinf/FSDP evaluation, keep model inference on the trainer's local GPUs. Use same-datacenter Ray nodes for local B1K workers and SSH-forwarded collector daemons for remote-datacenter B1K workers. Remote collector hosts must not join the trainer Ray cluster.

## Launch and verify

Use the template for the selected mode. For websocket mode, start the pi05 server first and verify its checkpoint and listener before starting B1K. For RLinf cross-datacenter mode, verify or start the Ray cluster and collector daemons before starting the evaluation driver.

Perform only a bounded startup check:

- both tmux sessions exist on their stated hosts;
- the expected Python processes and GPU assignments are present;
- `/proc/<evaluator-pid>/environ` contains `TMPDIR=/mnt/public/daibo/tmp`;
- the policy log has no immediate traceback and the port is reachable;
- the evaluator log exists and has begun environment/task initialization, or clearly states that cold initialization is still in progress.

TorchInductor compilation on the first policy request may take several minutes. Treat autotuning messages about rejected kernels as non-fatal unless the process exits or a traceback follows.

## Handoff

Return immediately after the bounded startup check. Include:

- checkpoint, task, episodes, subtask range, profile, and sidecar
- policy-server host/GPU/port/tmux name and server log
- B1K host/GPU/tmux name and evaluator log
- run root, metrics directory, video directory, and expected video filename pattern
- current startup state and the exact commands to attach to both tmux sessions

Say that metrics/videos may still be growing. Do not promise success, keep polling, or stop the sessions unless the user asks.

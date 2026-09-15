---
name: b1k-rl
description: Preflight, deploy, launch, resume, or diagnose BEHAVIOR-1K RLinf training with local and cross-datacenter remote environment collectors. Use for B1K RL jobs that require reproducible RLinf/OmniGibson source synchronization, environment validation, collector lifecycle management, effective-configuration checks, and log/checkpoint handoff; do not use for SFT-only jobs.
---

# B1K RL

Make every run reproducible from two immutable source identities: one RLinf
revision and one BEHAVIOR-1K revision. Treat an editable Python installation,
an SSH-accessible checkout, or a launch script as unverified until the running
process reports the expected source paths and deployed files pass a manifest.

## Required companion preflight

Before selecting batch sizes or launching training, read
`../rlinf-gpu-training-preflight/SKILL.md` completely and apply its batch,
checkpoint, storage, probe, and launch-manifest requirements. This skill adds
B1K-specific checks; it does not replace those requirements.

## Choose the execution mode

- Default to one OmniGibson scene per EnvWorker. Require
  `total_num_envs / env_world_size == 1` and `num_env_subprocess == 1`.
- Do not enable RLinf multi-scene execution, BEHAVIOR `VectorEnvironment` with
  more than one scene, or `num_env_subprocess > 1` unless the user explicitly
  requests a new correctness audit. Throughput is not parity evidence.
- Remote collectors are independent single-scene simulators. They do not join
  the training Ray cluster.
- Preserve image resolution, camera modalities, physics stepping, reward,
  termination, snapshot schedule, and action horizon unless the experiment is
  explicitly testing one of them.

## Freeze and deploy both source trees

1. Work from clean branches. Record the full RLinf and BEHAVIOR-1K commit IDs.
2. For patched OmniGibson, pass its `OmniGibson` directory through
   `manage_remote_collectors.py --omnigibson-path`; never rely on whichever
   editable checkout happens to be installed in the venv.
3. Create manifests with `scripts/source_manifest.py` for runtime source
   prefixes, copy trees and manifests, then verify them on every host.
4. On a host without `.git`, a copied commit ID is only metadata. Manifest
   verification is the authority.
5. Start collectors only after all hosts pass. Never modify a deployed source
   tree while its Python processes are alive.

Read [references/inventory.md](references/inventory.md) for current hosts and
paths. Revalidate it because machines may be restarted or remounted.

## Preflight the B1K runtime

On every simulator host, verify without printing secrets:

- SSH reachability, hostname, GPU count/model, free memory, and intended GPU IDs;
- Python executable and `torch` version;
- `omnigibson.__file__` resolves below the audited deployed source after setting
  `PYTHONPATH`;
- B1K data, dataset, key, robot assets, subpool manifest/states, grounded token
  mapping, checkpoint, and norm-stat paths exist;
- required environment-variable names are set: `TMPDIR`,
  `OMNIGIBSON_DATA_PATH`, `OMNIGIBSON_DATASET_PATH`, `OMNIGIBSON_KEY_PATH`,
  `OMNIGIBSON_ASSET_PATH`, `OMNI_KIT_ACCEPT_EULA`, and
  `RLINF_REMOTE_COLLECTOR_TOKEN`;
- intended ports are free before startup and listening afterward;
- old collector tmux sessions and Ray processes do not own selected resources.
  Stop or replace only collectors belonging to the run in scope.

Use a unique run ID for tmux, logs, Ray temp data, OmniGibson appdata,
TorchInductor, and Triton caches. Hosts sharing `/mnt/public` must still use
host-qualified cache directories.

## Start and validate remote collectors

Read [references/remote-collectors.md](references/remote-collectors.md) before a
cross-datacenter launch. Start one daemon per remote GPU in persistent tmux.
Use unique global `env_rank`, remote port, and trainer-side forwarded port for
every endpoint.

Run a bounded initialize → canonical reset → observation → chunk step → reset
smoke test against every endpoint. Check snapshot/episode/subtask identity,
observation keys/shapes/dtypes, reward fields, termination flags, and action
shape. A running tmux session or occupied GPU is not proof of health.

At the start of a new run, replace its collector daemons so no simulator state
survives from an interrupted driver. Do not silently reconnect a new training
run to an already-mutated environment.

## Launch and hand off

Save these files in the result directory before calling the job healthy:

- `launch.sh`: exact command and overrides;
- `launch_manifest.json` or `.md`: intended and effective configuration;
- `source_manifest.rlinf.json` and `source_manifest.b1k.json`;
- `launcher.log` and `metrics.log` locations;
- `remote_collectors.json`: endpoint-to-host/GPU/port/log mapping;
- effective Hydra config, TensorBoard directory, tmux names, and observation commands.

The launch manifest must include all fields required by the companion training
preflight plus: both source commits and per-host verification, actual
`omnigibson.__file__`, local/remote EnvWorker mapping, one-scene invariant,
subpool train/eval manifests and logical states, group size/count, accepted and
candidate trajectory counts, DAPO/signal gate, reward definition,
gamma/lambda/bootstrap, advantage normalization, ratio level, KL/reference,
policy/critic epochs and GBS, failure-state capture, remote compression, dynamic
rollout batching, and collector logs.

After launch, tell the user the result directory, launcher/metrics/TensorBoard
paths, training and collector tmux names, checkpoint path/save interval, exact
tail/TensorBoard/status commands, and key effective parameters. Never return
only “started successfully.”

## Early health gate and failures

Verify liveness immediately, then inspect the first complete rollout and update.
Check per-state candidate/accepted success, trajectory count, reward components,
advantages, ratio/KL/clipping, policy/critic gradients, value targets and
explained variance, timing, collector reconnects, and checkpoint writes.

If a process exits, surface the first causal traceback with host, rank, endpoint,
and source revisions. Do not hide it behind repeated restarts. Fix an in-scope
deterministic issue, rerun preflight, and restart when the user requested a
persistent training outcome.

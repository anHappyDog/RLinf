---
name: b1k-rl
description: Prepare, hand off, launch, resume, or diagnose BEHAVIOR-1K RLinf training with local and cross-datacenter remote environment collectors. Use for B1K RL jobs that require reproducible RLinf/OmniGibson source synchronization, environment validation, collector lifecycle management, effective-configuration checks, operator scripts, and log/checkpoint handoff; do not use for SFT-only jobs.
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

## Enforce a commit gate and experiment archive

Do not launch a production experiment from uncommitted RLinf or BEHAVIOR-1K
code. A new experiment does not require a new code commit when its source is
unchanged: reuse the same clean commit and put experiment-specific records in
the documentation repository. Before allocating GPUs:

1. require clean `git status --porcelain` output in both repositories;
2. commit every experiment-affecting code change and record the branch, full
   commit ID, and tree ID for both repositories;
3. never use blanket `git add` or commit unrelated pre-existing edits merely to
   make the gate pass; separate the changes or ask the user to resolve ownership;
4. push newly created code revisions to the configured shared remote when one is
   available, and record explicitly when a revision exists only locally;
5. generate source manifests from those exact commits, deploy them, and verify
   every local and remote runtime against the manifests;
6. copy the exact launch/configuration files into a new result directory and
   create its `experiment.md` before starting the production process.

Maintain experiment records in the separate local Git repository
`/mnt/public/daibo/timeline/0831/b1k-rl-experiments`. Do not commit per-run
records to the RLinf or BEHAVIOR-1K source repositories. Its central
`EXPERIMENTS.md` links to one record below `experiments/` per run. Commit the
prepared record before launch, then commit result/lifecycle updates after the
first complete update, at important decisions/checkpoints, and when the run
completes or stops. A run-local `experiment.md` may be kept as an artifact copy,
but the documentation-repository version is authoritative. Preserve failed
starts and their first causal error instead of silently replacing history.

Read [references/experiment-archive.md](references/experiment-archive.md) for
the required fields and template. A clean commit makes code identity
reconstructable; it does not replace the effective Hydra config, source
manifests, runtime placement, logs, or results archive.

## Default responsibility boundary

Default to an operator-handoff workflow:

1. inspect or implement the requested code change;
2. run bounded unit, smoke, config, and endpoint tests;
3. prepare a complete, syntax-checked operator bundle;
4. teach the user the exact start, observe, stop, and resume commands;
5. leave the formal long-running experiment for the user to start.

Do not start or babysit a production B1K RL run unless the user explicitly asks
for that launch. A request to modify code, prepare an experiment, or explain how
to run it does not by itself authorize a long run. Short representative smoke
tests remain part of implementation and preflight. When the user explicitly asks
for a persistent outcome, launch it and apply the early health gate below.

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

The collector daemon starts without an environment. The trainer sends the
`initialize` RPC containing the effective environment configuration. Repeating
the identical payload is idempotent; a different payload must fail instead of
silently reusing stale state. Treat this as reconnect safety, not permission to
reuse a mutated simulator across experiments. On hosts that may contain more
than one Ray cluster, pass the collector's explicit host-local Ray address
(normally `127.0.0.1:<port>`); do not use `auto` as deployment identity.

At the start of a new run, replace its collector daemons so no simulator state
survives from an interrupted driver. Do not silently reconnect a new training
run to an already-mutated environment.

## Build the operator bundle

Save these files in the result directory before calling the job healthy:

- `preflight.sh`: read-only, fail-fast checks for sources, paths, ports, GPUs,
  endpoint health, checkpoint, storage, and derived batch arithmetic;
- `start_collectors.sh`: exact collector deployment and status commands, when
  remote collectors are used;
- `launch.sh`: exact command and overrides;
- `observe.sh`: one command that shows driver/tmux liveness, collector health,
  the latest launcher/metrics lines, checkpoints, and GPU ownership;
- `stop.sh`: stop only this run's named processes, sessions, and collectors;
- `resume.sh`, or an explicit documented resume command when the runner cannot
  use a fixed reusable script;
- `experiment.md`: objective, immutable source identity, configuration,
  lifecycle events, results, conclusion, and follow-up;
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

Before handoff, run `bash -n` on every shell script, run config-only validation
when the entrypoint supports it, and execute `preflight.sh`. Do not describe a
script as ready when these checks have not passed.

At handoff, give the user one ordered command sequence and state which commands
are optional. Include the result directory, launcher/metrics/TensorBoard paths,
training and collector tmux names, checkpoint path/save interval, expected first
startup milestones, approximate time to the first useful metric, exact
tail/TensorBoard/status commands, and key effective parameters. Distinguish
"prepared", "collectors healthy", and "training healthy"; never report a stage
that has not been observed. Read
[references/operator-handoff.md](references/operator-handoff.md) for the
required user-facing format.

## Early health gate and failures

For an explicitly requested launch, verify liveness immediately, then inspect
the first complete rollout and update.
Check per-state candidate/accepted success, trajectory count, reward components,
advantages, ratio/KL/clipping, policy/critic gradients, value targets and
explained variance, timing, collector reconnects, and checkpoint writes.

If a process exits, surface the first causal traceback with host, rank, endpoint,
and source revisions. Do not hide it behind repeated restarts. Fix an in-scope
deterministic issue, rerun preflight, and restart when the user requested a
persistent training outcome.

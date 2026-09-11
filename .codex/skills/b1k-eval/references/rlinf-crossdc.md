# RLinf cross-datacenter B1K evaluation

Use this mode when the policy is an RLinf/FSDP checkpoint and remote GPU hosts cannot safely join the trainer's Ray cluster. The trainer runs model inference; each remote host runs an independent local Ray instance and persistent B1K collector daemons. RLinf reaches those daemons through per-endpoint SSH forwarding.

## Topology and invariants

The established 20-environment topology is:

- trainer node rank 0: four rollout workers and twelve lightweight EnvWorkers that proxy remote collectors;
- `nxb_4090` and `nxb_4090_2`, node ranks 1 and 2: eight local B1K workers;
- `gdb_4090_1`: eight collector daemons;
- `gdb_4090_2`: four collector daemons.

The gdb hosts do not join the trainer Ray cluster and do not share `/mnt/public` with it. Before launch, require matching hashes for `rlinf/envs/remote_collector.py`, `rlinf/envs/behavior/remote_collector.py`, and the collector manager. Also verify that every manifest-relative state file and required B1K asset exists on each collector host. Model checkpoints are needed only on trainer-side rollout nodes.

Each endpoint needs a unique `env_rank`, remote port, and trainer-side `local_port`. Require `total_num_envs == number of remote endpoints + number of local B1K GPUs`, and one environment per EnvWorker. Use the same non-empty `RLINF_REMOTE_COLLECTOR_TOKEN` in trainer workers and collector daemons. Never put the token literal in logs.

## Start the three Ray clusters

Start a fresh trainer Ray cluster only after proving the target nodes are idle. Set `RLINF_NODE_RANK` before `ray start`, because Ray captures it at startup. Run every command in a named tmux session with `--block`; on managed development machines, a daemon started by a short-lived SSH or command-execution session may be reclaimed when that session exits:

```bash
# trainer
RLINF_NODE_RANK=0 ray start --head --port=6379 --block \
  --node-ip-address=<trainer-private-ip> --temp-dir=<run-specific-ray-dir> \
  --include-dashboard=false

# same-datacenter B1K nodes
RLINF_NODE_RANK=1 ray start --address=<trainer-private-ip>:6379 --block \
  --node-ip-address=<nxb1-private-ip> --temp-dir=<run-specific-ray-dir>
RLINF_NODE_RANK=2 ray start --address=<trainer-private-ip>:6379 --block \
  --node-ip-address=<nxb2-private-ip> --temp-dir=<run-specific-ray-dir>
```

Each gdb host uses its own single-node Ray head. It must never point at the trainer head. Existing healthy gdb Ray heads may be reused. Set the gdb Ray `--temp-dir` below `/mnt/public/daibo/tmp`; setting collector `TMPDIR` alone does not relocate Ray's session and object-spill directories, and root-backed `/tmp` may fill during long evaluations.

The dashboard is unnecessary for this workflow and may prevent Ray startup when an optional dashboard module is unavailable. Prefer `--include-dashboard=false`; diagnose GCS/raylet health from `ray status` and the run-specific Ray logs.

## Manage remote collectors

Use `toolkits/b1k_grounded/manage_remote_collectors.py` on each gdb host. Set the complete B1K environment before invoking it: `TMPDIR`, OmniGibson data/asset/key paths, `OMNI_KIT_ACCEPT_EULA`, and `RLINF_REMOTE_COLLECTOR_TOKEN`. Use one GPU and port per daemon, a run-identifying `--session-prefix`, and a persistent `--log-dir`.

Inspect existing sessions and listeners before using `--replace`; replace only sessions known to belong to the requested evaluation. After start, require every tmux session, Python process, listener, and log to exist. The collector manager assigns separate OmniGibson app-data and compilation-cache roots per GPU.

The client establishes SSH forwards from `ssh_host`, `port`, and `local_port` in `env.eval.remote_collector.endpoints`. A successful prior SSH login does not prove forwarding health. Verify an actual collector RPC during bounded startup. If a collector reports a stale owner after a crashed driver, first confirm the reset-on-close hook is present on both sides; restart only the named collector if state was not released.

## Reproducible checkpoint comparisons

Keep all non-policy inputs fixed across comparison cells:

- exact manifest and snapshot IDs;
- `env.eval.seed` and `rollout.seed`;
- environment count, reset count, horizon, action horizon, and `num_steps`;
- code revision, assets, token mapping, normalization stats, renderer, reward, and observation-skip mode.

For 20 environments and 320 trials, set `env.eval.rollout_epoch: 16`. Store each checkpoint/sampler pair under its own output directory. A useful sampler matrix is `flow_sde` at the training noise, `flow_sde` at a lower noise, and deterministic `flow_ode` with `noise_level: 0.0`.

Do not compare a DAPO training `success_once` directly with standalone evaluation: training collection is pre-update, adaptively resampled, and stochastic. Compare checkpoint outputs produced by this fixed evaluation matrix.

## Persistent launch artifact

Put `eval_config.yaml`, `launch.sh`, and `cluster_start.sh` in the result root. The launch script must:

1. validate every checkpoint and manifest row before starting;
2. enumerate the checkpoint/sampler matrix explicitly;
3. write one log and metrics directory per cell;
4. skip only cells with an explicit completion marker, not merely a non-empty partial log;
5. run from the intended RLinf checkout and export `TMPDIR=/mnt/public/daibo/tmp`;
6. run in a uniquely named tmux session so interruption of the controlling shell does not stop evaluation.

Record the resolved config for every cell. Leave collector daemons running for reuse only when their code, token scope, assets, and ports remain valid; otherwise stop their exact tmux sessions. Never broadly kill Ray, tmux, Python, or GPU processes belonging to other runs.

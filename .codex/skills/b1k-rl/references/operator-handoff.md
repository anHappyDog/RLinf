# B1K RL operator handoff

Use this compact structure when transferring a prepared or running experiment
to the user. Resolve every placeholder; do not make the user infer paths or
search logs.

## Status

State exactly one:

- **Prepared:** scripts and preflight passed; no production process was started.
- **Collectors healthy:** every endpoint passed initialize/reset/step/reset; the
  trainer was not started.
- **Training healthy:** the trainer is alive and the stated health milestone was
  actually observed.

## Run identity and effective configuration

Report the repository path and commit, BEHAVIOR-1K path and commit or manifest,
base/resume checkpoint, result directory, train/eval subpool manifests, logical
states, reward and termination definition, gamma/lambda/bootstrap, advantage
normalization, action- or chunk-level ratio, KL/reference settings, trajectory
and group counts, policy/critic MBS and GBS, derived accumulation, update epochs,
GPU placement, EnvWorker count, and collector endpoint count.

## Commands

Give one copyable ordered block, normally:

```bash
cd <audited-repository>
bash <result-dir>/preflight.sh
bash <result-dir>/start_collectors.sh   # only when collectors are not healthy
bash <result-dir>/launch.sh
bash <result-dir>/observe.sh
```

Also give exact scoped stop and resume commands. Never recommend broad `pkill`,
`ray stop`, cache deletion, or deletion of a result tree as a routine stop path.

## Observation locations

Always name:

- launcher log;
- metrics log;
- effective Hydra config;
- TensorBoard directory and exact `tensorboard --logdir ...` command;
- checkpoint directory and save interval;
- local and remote collector logs;
- video/failure-state directories when enabled;
- tmux names and exact attach/status commands.

Also link the central `EXPERIMENTS.md` index and the run-local `experiment.md`.
State whether both RLinf and BEHAVIOR-1K launch revisions were committed,
clean, pushed when possible, and verified on every execution host.

State the expected startup milestones and the approximate time to first rollout,
first optimizer update, and first checkpoint using prior comparable evidence or
mark the timing unknown. Include failure criteria such as a missing endpoint,
source-manifest mismatch, rank/trajectory-count mismatch, reconnects, NaN/Inf,
or an unexpected effective configuration.

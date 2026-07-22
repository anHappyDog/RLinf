# SG-16 final verification

Date: 2026-07-20

Scope: execution-path mutual exclusion, reward/bootstrap ownership audit,
ordinary-Channel regression, target-config E2E, and live transport regression.
Ordinary Channel implementation and call sites were intentionally retained.

## Correctness

Command:

```text
/opt/venv/openpi/bin/python -m pytest -q \
  tests/unit_tests/test_trajectory_runtime.py \
  tests/unit_tests/test_trajectory_actor.py \
  tests/unit_tests/test_embodied_gae.py \
  tests/unit_tests/test_overlap_env_bootstrap.py
```

Result: 29 passed in 6.10 s.

The expanded suite covering every `test_trajectory*.py` module plus embodied
GAE and ordinary-Channel bootstrap passed 153 tests in 30.90 s.

The trajectory entrypoint now rejects unsupported mixed execution modes before
launching component worker groups. The disabled trajectory configuration remains
unaffected.

## Target configuration E2E

Command:

```text
PYTHONPATH=/mnt/public/daibo/timeline/0703/RLinf \
/opt/venv/openpi/bin/python examples/embodiment/train_embodied_agent.py \
  --config-name libero_spatial_ppo_openpi_trajectory_channel \
  env.train.max_steps_per_rollout_epoch=10 \
  actor.micro_batch_size=16 actor.global_batch_size=64 \
  algorithm.update_epoch=1 runner.max_epochs=1
```

Result: passed one full 32-env, four-rank rollout, Storage assembly, Actor pull,
transition-aligned GAE, and Actor/Critic optimizer update. Global Step reached
1/1; step time was 20.884 s, policy loss -0.0055, and value loss 0.0023.

The first attempt reused a Ray runtime created by the transport benchmark and
failed during NodeProbe creation with `No module named 'rlinf'`. It did not enter
the trajectory runtime. Explicit repository `PYTHONPATH` fixed the environment.

## Live transport regression

Raw samples: `live_local.json`.

| mode | sender p50 (ms) | sender p95 (ms) | receiver p50 (ms) |
|---|---:|---:|---:|
| ordinary Channel | 5.600 | 6.723 | 5.693 |
| direct raw | 1.572 | 1.640 | 1.577 |
| direct pinned | 1.549 | 1.579 | 1.549 |
| direct XOR+LZ4 | 2.818 | 3.052 | 2.836 |

The SG-15F direct-raw p50 was 1.873 ms, so SG-16 introduced no observed live
transport regression. These short local runs are not a claim about cross-node
tail latency. Live compression remains disabled in the target config.

## Remaining scope

- External RewardWorker, evaluation-only execution, training pipeline, multiple
  rollout pipeline stages, and non-OpenPI schemas are rejected explicitly.
- Ordinary Channel remains supported and was not migrated or removed.
- Cross-node paired E2E and longer statistical performance gates remain useful
  before production rollout, but are not SG-16 correctness blockers.

# Validated B1K environment

Use these paths for the current shared-storage installation. Verify them before each launch because a missing path is a hard error, not a reason to select an older dataset copy.

| Purpose | Path |
| --- | --- |
| RLinf checkout | `/mnt/public/daibo/timeline/0831/RLinf` |
| Python | `/mnt/public/daibo/venv/behavior_openpi/bin/python` |
| BEHAVIOR-1K checkout | `/mnt/public/daibo/venv/behavior_openpi/BEHAVIOR-1K` |
| COMET OpenPI checkout | `/mnt/public/daibo/repos/comet/openpi-comet` |
| COMET source | `/mnt/public/daibo/repos/comet/openpi-comet/src` |
| 2025 challenge demos | `/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos` |
| OmniGibson data root | `/mnt/public/daibo/datasets/omni_data` |
| scene/object assets | `/mnt/public/daibo/datasets/omni_data/behavior-1k-assets` |
| robot assets | `/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets` |
| OmniGibson key | `/mnt/public/daibo/datasets/omni_data/omnigibson.key` |
| task-instance metadata | `/mnt/public/daibo/datasets/omni_data/2025-challenge-task-instances/metadata/episodes.jsonl` |
| persistent temporary/cache root | `/mnt/public/daibo/tmp` |
| evaluation results root | `/mnt/public/daibo/results/b1k_grounded_control_v01/eval` |

The validated `behavior-1k-assets/VERSION` is `3.7.2rc1`, corresponding to the installed BEHAVIOR/OmniGibson stack. In particular, do not use `/mnt/public/daibo/datasets/omini_data`; that misspelled directory previously selected incompatible assets.

## Shared Python environment

Both the pi05 server and B1K evaluator use:

```bash
PYTHON=/mnt/public/daibo/venv/behavior_openpi/bin/python
PYTHONPATH=/mnt/public/daibo/timeline/0831/RLinf:/mnt/public/daibo/repos/comet/openpi-comet/src
```

The pi05 server must run with the COMET checkout as its working directory because `serve_grounded_policy.py` reads `scripts/task_mapping.json` from that checkout.

## B1K evaluator environment

Set these explicitly inside every evaluator tmux session:

```bash
TMPDIR=/mnt/public/daibo/tmp
OMNI_KIT_ACCEPT_EULA=YES
OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data
OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets
OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key
OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
OMNIGIBSON_APPDATA_PATH=/mnt/public/daibo/tmp/b1k_<run>_<slot>_appdata
```

`TMPDIR` is a hard requirement on `nxb_4090`; export it in the `env` command that directly launches Python, rather than relying on an interactive shell profile. After launch, verify it through `/proc/<evaluator-pid>/environ`. Without it, B1K can exhaust the host's root-backed `/tmp`.

Give every concurrent evaluator a different `OMNIGIBSON_APPDATA_PATH`. A warm, known-compatible app-data directory may be reused only for the same installed stack.

## pi05 server environment

Set a persistent, run-specific compilation cache:

```bash
HF_HOME=/opt/.cache/huggingface
TORCHINDUCTOR_CACHE_DIR=/mnt/public/daibo/tmp/torchinductor_b1k_<run>_<slot>
```

Do not place these caches in `/tmp` and do not delete another run's cache.

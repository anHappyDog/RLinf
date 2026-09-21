#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export TMPDIR=/mnt/public/daibo/tmp
export XDG_CACHE_HOME="/mnt/public/daibo/cache/$(hostname)/rlt_stage1"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export EMBODIED_PATH="$repo/examples/sft"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export B1K_RLT_RESULT_DIR=${B1K_RLT_RESULT_DIR:-/mnt/public/daibo/results/b1k_grounded_control_v01/rlt/stage1_v1}
venv=${B1K_RL_VENV:-/mnt/public/daibo/venv/behavior_openpi}
cd "$repo"
if [[ ${1:-} == --config-only ]]; then
    shift
    exec "$venv/bin/python" examples/sft/train_vla_sft.py --config-name behavior_rlt_stage1 --cfg job --resolve "$@"
fi
exec "$venv/bin/python" examples/sft/train_vla_sft.py --config-name behavior_rlt_stage1 "$@"

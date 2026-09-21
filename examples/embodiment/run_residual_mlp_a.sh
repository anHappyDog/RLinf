#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/mnt/public/daibo/tmp
export XDG_CACHE_HOME="/mnt/public/daibo/cache/$(hostname)/residual_mlp_a"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export EMBODIED_PATH="$repo/examples/embodiment"
export B1K_SUBPOOL_MANIFEST=${B1K_SUBPOOL_MANIFEST:-/mnt/public/daibo/results/b1k_grounded_control_v01/subpool/radio_pickup_canonical40_formal_v1/success_stratified_20train_20heldout/train/manifest.jsonl}
export B1K_EVAL_MANIFEST=${B1K_EVAL_MANIFEST:-/mnt/public/daibo/results/b1k_grounded_control_v01/subpool/radio_pickup_canonical40_formal_v1/success_stratified_20train_20heldout/heldout_eval/manifest.jsonl}
export B1K_GROUNDED_TOKEN_MAPPING=${B1K_GROUNDED_TOKEN_MAPPING:-/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_2025_radio_lunchbox_microwave_train480_stride8_subgoal_v04_split_v1/structural_token_mapping.json}
export B1K_ASSET_FINGERPRINT=${B1K_ASSET_FINGERPRINT:-b1k-2fd66d5c-radio-bc82d211}
export B1K_SUBPOOL_RESULT_DIR=${B1K_SUBPOOL_RESULT_DIR:-/mnt/public/daibo/results/b1k_grounded_control_v01/residual_mlp_a_v1}
venv=${B1K_RL_VENV:-/mnt/public/daibo/venv/behavior_openpi}
og=${B1K_RL_OMNIGIBSON_PATH:-/mnt/public/daibo/timeline/0831/BEHAVIOR-1K-b1k-rl-singleenv/OmniGibson}
export PYTHONPATH="$repo:$og${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
if [[ ${1:-} == --config-only ]]; then
    shift
    exec "$venv/bin/python" examples/embodiment/train_embodied_agent.py --config-name behavior_residual_mlp_a --cfg job --resolve "$@"
fi
# This script never creates/stops Ray or collectors. Placement is user configurable.
exec "$venv/bin/python" examples/embodiment/train_embodied_agent.py --config-name behavior_residual_mlp_a "$@"

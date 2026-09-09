#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=/mnt/public/daibo/timeline/0831/RLinf
VENV=/mnt/public/daibo/venv/behavior_openpi
MODEL=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_dense_comet_native_v1/model
TOKEN_MAPPING=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_dense_comet_native_v1/structural_token_mapping.json

: "${B1K_SUBPOOL_MANIFEST:?Set B1K_SUBPOOL_MANIFEST to the merged init-state manifest.}"
: "${B1K_SUBPOOL_RESULT_DIR:?Set B1K_SUBPOOL_RESULT_DIR to a fresh result directory.}"

MAX_STEPS=${B1K_RL_MAX_STEPS:-1}
SAVE_INTERVAL=${B1K_RL_SAVE_INTERVAL:-$MAX_STEPS}

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export B1K_SUBPOOL_MODEL_PATH="$MODEL"
export B1K_GROUNDED_TOKEN_MAPPING="$TOKEN_MAPPING"
export B1K_ASSET_FINGERPRINT=b1k-2fd66d5c-radio-bc82d211
export TMPDIR=/mnt/public/daibo/tmp

cd "$RLINF_ROOT"
python examples/embodiment/train_embodied_agent.py \
    --config-name behavior_subpool_ppo_openpi_pi05_hetero \
    "runner.max_steps=$MAX_STEPS" \
    "runner.save_interval=$SAVE_INTERVAL" \
    env.train.subpool.fixed_subtask_id=1 \
    env.train.subpool.pool_weights.canonical=1.0 \
    env.train.subpool.pool_weights.predecessor_success=0.0 \
    env.train.subpool.pool_weights.recovery=0.0 \
    env.train.subpool.dynamic_updates=false \
    env.train.video_cfg.save_video=false \
    env.eval.video_cfg.save_video=false \
    "$@"

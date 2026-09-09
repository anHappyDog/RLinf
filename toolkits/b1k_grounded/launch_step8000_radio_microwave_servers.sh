#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=/mnt/public/daibo/timeline/0831/RLinf
OPENPI_ROOT=/mnt/public/daibo/repos/comet/openpi-comet
VENV=/mnt/public/daibo/venv/behavior_openpi
CHECKPOINT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_dense_comet_native_v1
DATASET_ROOT=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos
OUTPUT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_microwave_10ep_formal_v1/local_servers
BASE_PORT=${BASE_PORT:-18070}

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT:$OPENPI_ROOT/src"
export TMPDIR=/mnt/public/daibo/tmp

mkdir -p "$OUTPUT_ROOT" "$TMPDIR"

server_pids=()
for gpu in 0 1 2 3; do
    if ((gpu < 2)); then
        task_name=turning_on_radio
    else
        task_name=make_microwave_popcorn
    fi
    port=$((BASE_PORT + gpu))
    log="$OUTPUT_ROOT/${task_name}_gpu${gpu}_port${port}.log"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_b1k_s8000_10ep_g${gpu}"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
        cd "$OPENPI_ROOT"
        exec python "$RLINF_ROOT/toolkits/b1k_grounded/serve_grounded_policy.py" \
            --checkpoint-dir "$CHECKPOINT_ROOT/model" \
            --token-mapping-path "$CHECKPOINT_ROOT/structural_token_mapping.json" \
            --control-profile P2_GROUND_SG \
            --task-name "$task_name" \
            --dataset-root "$DATASET_ROOT" \
            --port "$port"
    ) >"$log" 2>&1 &
    server_pids+=("$!")
    echo "$!" >"$OUTPUT_ROOT/gpu${gpu}.pid"
done

trap 'kill "${server_pids[@]}" 2>/dev/null || true' EXIT
wait "${server_pids[@]}"

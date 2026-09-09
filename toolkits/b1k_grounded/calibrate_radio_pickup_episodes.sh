#!/usr/bin/env bash

set -euo pipefail

VENV=/mnt/public/daibo/venv/behavior_openpi
RLINF_ROOT=/mnt/public/daibo/timeline/0831/RLinf
DATASET_ROOT=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos
SIDECAR=/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_1000ep_radio_microwave_lunchbox_dense_stride8_boundarysafe_v2/data/part-00000.parquet
OUTPUT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/subpool/radio_pickup_calibration_40ep_v2

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT"
export TMPDIR=/mnt/public/daibo/tmp
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data
export OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets
export OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key
export OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
export OMNI_KIT_ACCEPT_EULA=YES

mkdir -p "$OUTPUT_ROOT/supervisor"

calibrate_lane() {
    local gpu=$1
    local episode_list=$2
    local output_dir="$OUTPUT_ROOT/lane${gpu}"
    local appdata="$TMPDIR/b1k_radio_pickup_calibration_g${gpu}_appdata"

    mkdir -p \
        "$output_dir" \
        "$appdata/local" \
        "$appdata/global/cache" \
        "$appdata/global/data"
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$RLINF_ROOT"
    python toolkits/b1k_grounded/eval_grounded_subtasks.py \
        policy=websocket \
        task.name=turning_on_radio \
        eval_level=subtask \
        headless=true \
        keep_running_after_success=false \
        write_video=false \
        "run_episode_indices=$episode_list" \
        subtask_index=1 \
        subtask_end_index=1 \
        demo_data_dir="$DATASET_ROOT" \
        instance_reward_mode=task \
        log_path="$output_dir" \
        +grounded_control_sidecar="$SIDECAR" \
        +grounded_control_profile=p2_ground_sg \
        +grounded_infer_missing_parts=true \
        +grounded_eval_view_dir="$output_dir/subtask_eval_view" \
        +grounded_demo_calibration=true \
        --portable-root "$appdata/local" \
        --/app/tokens/omni_global_cache="$appdata/global/cache" \
        --/app/tokens/omni_global_data="$appdata/global/data"
}

episode_lists=(
    '[10,50,90,130,170,210,250,290,330,370]'
    '[20,60,100,140,180,220,260,300,340,380]'
    '[30,70,110,150,190,230,270,310,350,390]'
    '[40,80,120,160,200,240,280,320,360,400]'
)

lane_pids=()
for gpu in 0 1 2 3; do
    calibrate_lane "$gpu" "${episode_lists[$gpu]}" \
        >"$OUTPUT_ROOT/supervisor/lane${gpu}.log" 2>&1 &
    lane_pids+=("$!")
done

failed=0
for pid in "${lane_pids[@]}"; do
    wait "$pid" || failed=1
done
date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/calibration.done"
exit "$failed"

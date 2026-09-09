#!/usr/bin/env bash

set -euo pipefail

TASK_NAME=${1:?Usage: $0 <task_name> <gpu> <stage> <port>}
GPU=${2:?Usage: $0 <task_name> <gpu> <stage> <port>}
STAGE=${3:?Usage: $0 <task_name> <gpu> <stage> <port>}
PORT=${4:?Usage: $0 <task_name> <gpu> <stage> <port>}

VENV=/mnt/public/daibo/venv/behavior_openpi
RLINF_ROOT=/mnt/public/daibo/timeline/0831/RLinf
DATASET_ROOT=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos
SIDECAR=/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_1000ep_radio_microwave_lunchbox_dense_stride8_boundarysafe_v2/data/part-00000.parquet
OUTPUT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_microwave_10ep_formal_v1
export TMPDIR=/mnt/public/daibo/tmp

case "$TASK_NAME" in
    turning_on_radio)
        EPISODES='[30,40,50,60,70,80,100,110,120,130]'
        ;;
    make_microwave_popcorn)
        EPISODES='[400010,400020,400040,400050,400060,400070,400080,400090,400100,400110]'
        ;;
    *)
        echo "Unsupported task: $TASK_NAME" >&2
        exit 2
        ;;
esac

stage_name=$(printf '%02d' "$STAGE")
output_dir="$OUTPUT_ROOT/$TASK_NAME/stage_${stage_name}"
appdata="$TMPDIR/b1k_s8000_radio_microwave_10ep_g${GPU}_appdata"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data

mkdir -p \
    "$output_dir" \
    "$appdata/local" \
    "$appdata/global/cache" \
    "$appdata/global/data"

cd "$RLINF_ROOT"
python toolkits/b1k_grounded/eval_grounded_subtasks.py \
    policy=websocket \
    task.name="$TASK_NAME" \
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
    eval_level=subtask \
    headless=true \
    keep_running_after_success=false \
    write_video=true \
    "run_episode_indices=$EPISODES" \
    subtask_index="$STAGE" \
    subtask_end_index="$STAGE" \
    demo_data_dir="$DATASET_ROOT" \
    instance_reward_mode=task \
    log_path="$output_dir" \
    model.host=127.0.0.1 \
    model.port="$PORT" \
    +grounded_control_sidecar="$SIDECAR" \
    +grounded_eval_view_dir="$output_dir/subtask_eval_view" \
    +grounded_control_profile=p2_ground_sg \
    +grounded_infer_missing_parts=true \
    +grounded_demo_calibration=false \
    --portable-root "$appdata/local" \
    --/app/tokens/omni_global_cache="$appdata/global/cache" \
    --/app/tokens/omni_global_data="$appdata/global/data"

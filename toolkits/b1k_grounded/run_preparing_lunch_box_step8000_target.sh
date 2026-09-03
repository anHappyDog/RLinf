#!/usr/bin/env bash

set -euo pipefail

MODE=${1:?Usage: $0 <serve|calibrate|evaluate> <gpu> [stage] [port]}
GPU=${2:?Usage: $0 <serve|calibrate|evaluate> <gpu> [stage] [port]}
STAGE=${3:-0}
PORT=${4:-8070}

VENV=/mnt/public/daibo/venv/behavior_openpi
RLINF_ROOT=/mnt/public/daibo/tmp/RLinf_grounded_eval
OPENPI_ROOT=/mnt/public/daibo/tmp/openpi-comet-grounded-eval
CHECKPOINT_ROOT=/mnt/public/daibo/models/b1k_grounded_step8000_lunch_eval
DATASET_ROOT=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos
OUTPUT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/preparing_lunch_box_step8000_ep120010

export TMPDIR=/mnt/public/daibo/tmp
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT:$OPENPI_ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data

COMMON_ARGS=(
    policy=websocket
    task.name=preparing_lunch_box
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
    eval_level=subtask
    headless=true
    keep_running_after_success=false
    "run_episode_indices=[120010]"
    demo_data_dir="$DATASET_ROOT"
    instance_reward_mode=task
    subtask_index="$STAGE"
    subtask_end_index="$STAGE"
    +grounded_control_sidecar="$CHECKPOINT_ROOT/lunch_ep120010_sidecar.parquet"
    +grounded_control_profile=p2_ground_sg
    +grounded_infer_missing_parts=true
)

run_evaluator() {
    local run_mode=$1
    local write_video=$2
    local output_dir="$OUTPUT_ROOT/$run_mode/stage_$(printf '%02d' "$STAGE")"
    local appdata
    if [[ $run_mode == calibration ]]; then
        appdata="$TMPDIR/b1k_lunch_s8000_${run_mode}_g${GPU}_st${STAGE}_appdata"
    else
        appdata="$TMPDIR/b1k_lunch_s8000_${run_mode}_g${GPU}_appdata"
    fi
    mkdir -p "$output_dir" "$appdata/local" "$appdata/global/cache" "$appdata/global/data"
    cd "$RLINF_ROOT"
    python toolkits/b1k_grounded/eval_grounded_subtasks.py \
        "${COMMON_ARGS[@]}" \
        write_video="$write_video" \
        log_path="$output_dir" \
        +grounded_eval_view_dir="$output_dir/subtask_eval_view" \
        --portable-root "$appdata/local" \
        --/app/tokens/omni_global_cache="$appdata/global/cache" \
        --/app/tokens/omni_global_data="$appdata/global/data"
}

case "$MODE" in
    serve)
        export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_b1k_lunch_s8000_g${GPU}"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$OUTPUT_ROOT/servers"
        cd "$OPENPI_ROOT"
        python "$RLINF_ROOT/toolkits/b1k_grounded/serve_grounded_policy.py" \
            --checkpoint-dir "$CHECKPOINT_ROOT/model" \
            --token-mapping-path "$CHECKPOINT_ROOT/structural_token_mapping.json" \
            --control-profile p2_ground_sg \
            --task-name preparing_lunch_box \
            --dataset-root "$DATASET_ROOT" \
            --port "$PORT"
        ;;
    calibrate)
        COMMON_ARGS+=(+grounded_demo_calibration=true)
        run_evaluator calibration false
        calibration_report="$OUTPUT_ROOT/calibration/stage_$(printf '%02d' "$STAGE")/metrics/preparing_lunch_box_demo_predicate_calibration.json"
        if ! grep -q '"all_passed": true' "$calibration_report"; then
            echo "Calibration report did not pass: $calibration_report" >&2
            exit 1
        fi
        ;;
    evaluate)
        COMMON_ARGS+=(model.host=127.0.0.1 model.port="$PORT" +grounded_demo_calibration=false)
        run_evaluator evaluation true
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

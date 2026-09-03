#!/usr/bin/env bash

set -euo pipefail

MODE=${1:?Usage: $0 <calibrate|evaluate> [first_gpu] [last_gpu]}
TARGET_SCRIPT=/mnt/public/daibo/tmp/RLinf_grounded_eval/toolkits/b1k_grounded/run_preparing_lunch_box_step8000_target.sh
OUTPUT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/preparing_lunch_box_step8000_ep120010
NUM_GPUS=8
NUM_STAGES=26
BASE_PORT=8070
NUM_EVAL_LANES=4
FIRST_GPU=${2:-0}
LAST_GPU=${3:-$((NUM_GPUS - 1))}

if ((FIRST_GPU < 0 || LAST_GPU >= NUM_GPUS || FIRST_GPU > LAST_GPU)); then
    echo "Invalid GPU range: $FIRST_GPU..$LAST_GPU" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT/supervisor" "$OUTPUT_ROOT/servers"

run_calibration_lanes() {
    local lane_pids=()
    rm -f \
        "$OUTPUT_ROOT/supervisor/calibration.done" \
        "$OUTPUT_ROOT/supervisor/calibration.failed"
    for ((gpu = FIRST_GPU; gpu <= LAST_GPU; gpu++)); do
        (
            lane_failed=0
            for ((stage = gpu; stage < NUM_STAGES; stage += NUM_GPUS)); do
                stage_name=$(printf '%02d' "$stage")
                calibration_report="$OUTPUT_ROOT/calibration/stage_${stage_name}/metrics/preparing_lunch_box_demo_predicate_calibration.json"
                if grep -q '"all_passed": true' "$calibration_report" 2>/dev/null; then
                    echo "Skipping calibrated stage $stage."
                    continue
                fi
                port=$((BASE_PORT + gpu))
                log="$OUTPUT_ROOT/supervisor/${MODE}_stage_${stage_name}.log"
                if ! "$TARGET_SCRIPT" "$MODE" "$gpu" "$stage" "$port" >"$log" 2>&1; then
                    lane_failed=1
                fi
            done
            exit "$lane_failed"
        ) &
        lane_pids+=("$!")
    done

    local failed=0
    for pid in "${lane_pids[@]}"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

run_evaluation_lanes() {
    local lane_pids=()
    for ((lane = 0; lane < NUM_EVAL_LANES; lane++)); do
        (
            eval_gpu=$((lane + NUM_EVAL_LANES))
            port=$((BASE_PORT + lane))
            for ((stage = lane; stage < NUM_STAGES; stage += NUM_EVAL_LANES)); do
                stage_name=$(printf '%02d' "$stage")
                log="$OUTPUT_ROOT/supervisor/evaluate_stage_${stage_name}.log"
                "$TARGET_SCRIPT" evaluate "$eval_gpu" "$stage" "$port" >"$log" 2>&1
            done
        ) &
        lane_pids+=("$!")
    done

    local failed=0
    for pid in "${lane_pids[@]}"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

wait_for_servers() {
    for ((lane = 0; lane < NUM_EVAL_LANES; lane++)); do
        port=$((BASE_PORT + lane))
        for _ in {1..180}; do
            if (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
                break
            fi
            if ! kill -0 "${SERVER_PIDS[$lane]}" 2>/dev/null; then
                echo "Policy server on GPU $lane exited before opening port $port." >&2
                return 1
            fi
            sleep 2
        done
        if ! (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            echo "Timed out waiting for policy server on port $port." >&2
            return 1
        fi
    done
}

case "$MODE" in
    calibrate)
        if run_calibration_lanes; then
            date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/calibration.done"
        else
            date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/calibration.failed"
            exit 1
        fi
        ;;
    evaluate)
        SERVER_PIDS=()
        for ((lane = 0; lane < NUM_EVAL_LANES; lane++)); do
            port=$((BASE_PORT + lane))
            "$TARGET_SCRIPT" serve "$lane" 0 "$port" \
                >"$OUTPUT_ROOT/servers/gpu_${lane}.log" 2>&1 &
            SERVER_PIDS+=("$!")
        done
        trap 'kill "${SERVER_PIDS[@]}" 2>/dev/null || true' EXIT
        wait_for_servers
        run_evaluation_lanes
        date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/evaluation.done"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

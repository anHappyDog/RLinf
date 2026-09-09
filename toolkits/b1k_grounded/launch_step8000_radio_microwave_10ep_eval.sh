#!/usr/bin/env bash

set -euo pipefail

TARGET_SCRIPT=/mnt/public/daibo/timeline/0831/RLinf/toolkits/b1k_grounded/run_step8000_radio_microwave_10ep_target.sh
OUTPUT_ROOT=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_microwave_10ep_formal_v1
BASE_PORT=${BASE_PORT:-18070}

mkdir -p "$OUTPUT_ROOT/supervisor"

expected_metrics_exist() {
    local task_name=$1
    local stage=$2
    local stage_name
    stage_name=$(printf '%02d' "$stage")
    local metrics_dir="$OUTPUT_ROOT/$task_name/stage_${stage_name}/metrics"
    local episodes
    if [[ $task_name == turning_on_radio ]]; then
        episodes=(30 40 50 60 70 80 100 110 120 130)
    else
        episodes=(400010 400020 400040 400050 400060 400070 400080 400090 400100 400110)
    fi
    for episode in "${episodes[@]}"; do
        [[ -s "$metrics_dir/${task_name}_ep${episode}_st${stage}.json" ]] || return 1
    done
}

run_stage() {
    local task_name=$1
    local gpu=$2
    local stage=$3
    local port=$4
    local stage_name
    stage_name=$(printf '%02d' "$stage")
    local log="$OUTPUT_ROOT/supervisor/${task_name}_stage_${stage_name}.log"
    if expected_metrics_exist "$task_name" "$stage"; then
        echo "Skipping complete task=$task_name stage=$stage"
        return
    fi
    "$TARGET_SCRIPT" "$task_name" "$gpu" "$stage" "$port" >"$log" 2>&1
}

wait_for_port() {
    local port=$1
    for _ in {1..90}; do
        if (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            return
        fi
        sleep 2
    done
    echo "Timed out waiting for reverse-tunneled policy server on port $port" >&2
    return 1
}

for lane in 0 1 2 3; do
    wait_for_port "$((BASE_PORT + lane))"
done

lane_pids=()
(
    run_stage turning_on_radio 0 0 "$((BASE_PORT + 0))"
    run_stage turning_on_radio 0 2 "$((BASE_PORT + 0))"
) &
lane_pids+=("$!")
(
    run_stage turning_on_radio 1 1 "$((BASE_PORT + 1))"
    run_stage turning_on_radio 1 3 "$((BASE_PORT + 1))"
) &
lane_pids+=("$!")
(
    for stage in 0 2 4 6; do
        run_stage make_microwave_popcorn 2 "$stage" "$((BASE_PORT + 2))"
    done
) &
lane_pids+=("$!")
(
    for stage in 1 3 5 7; do
        run_stage make_microwave_popcorn 3 "$stage" "$((BASE_PORT + 3))"
    done
) &
lane_pids+=("$!")

failed=0
for pid in "${lane_pids[@]}"; do
    wait "$pid" || failed=1
done

if ((failed)); then
    echo "At least one evaluation lane failed." >&2
    exit 1
fi
date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/evaluation.done"

#!/usr/bin/env bash

set -euo pipefail

VENV=/mnt/public/daibo/venv/behavior_openpi
RLINF_ROOT=/mnt/public/daibo/timeline/0831/RLinf
DATASET_ROOT=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos
SIDECAR=/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_1000ep_radio_microwave_lunchbox_dense_stride8_boundarysafe_v2/data/part-00000.parquet
TOKEN_MAPPING=/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_step8000_radio_dense_comet_native_v1/structural_token_mapping.json
REWARD_SPECS="$RLINF_ROOT/toolkits/b1k_grounded/radio_subpool_reward_specs.json"
OUTPUT_ROOT=${B1K_SUBPOOL_OUTPUT_ROOT:-/mnt/public/daibo/results/b1k_grounded_control_v01/subpool/radio_pickup_init_10ep_v1}
if [[ -n ${B1K_SUBPOOL_EPISODES:-} ]]; then
    read -r -a EPISODES <<<"$B1K_SUBPOOL_EPISODES"
else
    EPISODES=(30 40 50 60 70 80 100 110 120 130)
fi
if ((${#EPISODES[@]} == 0)); then
    echo "B1K_SUBPOOL_EPISODES did not contain any episode IDs." >&2
    exit 2
fi

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT"
export TMPDIR=/mnt/public/daibo/tmp
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data
export OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets
export OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key
export OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
export OMNI_KIT_ACCEPT_EULA=YES

mkdir -p "$OUTPUT_ROOT/supervisor"

export_episode() {
    local gpu=$1
    local episode=$2
    local output_dir="$OUTPUT_ROOT/episodes/ep${episode}"
    local manifest="$output_dir/manifest.jsonl"
    local appdata="$TMPDIR/b1k_radio_pickup_pool_g${gpu}_appdata"

    if [[ -s $manifest ]]; then
        echo "Skipping existing manifest for episode $episode"
        return
    fi
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
        "run_episode_indices=[$episode]" \
        subtask_index=1 \
        subtask_end_index=1 \
        demo_data_dir="$DATASET_ROOT" \
        instance_reward_mode=task \
        log_path="$output_dir" \
        +grounded_control_sidecar="$SIDECAR" \
        +grounded_control_profile=p2_ground_sg \
        +grounded_infer_missing_parts=true \
        +grounded_eval_view_dir="$output_dir/subtask_eval_view" \
        +subpool_export_manifest="$manifest" \
        +subpool_reward_specs="$REWARD_SPECS" \
        +subpool_token_mapping="$TOKEN_MAPPING" \
        +subpool_asset_fingerprint=b1k-2fd66d5c-radio-bc82d211 \
        +subpool_state_init_mode=official_subtask \
        +subpool_require_gt_suffix_success=false \
        --portable-root "$appdata/local" \
        --/app/tokens/omni_global_cache="$appdata/global/cache" \
        --/app/tokens/omni_global_data="$appdata/global/data"
}

lane_pids=()
for gpu in 0 1 2 3; do
    (
        failed=0
        for ((index = gpu; index < ${#EPISODES[@]}; index += 4)); do
            episode=${EPISODES[$index]}
            log="$OUTPUT_ROOT/supervisor/ep${episode}.log"
            if ! export_episode "$gpu" "$episode" >"$log" 2>&1; then
                echo "Episode $episode export failed; see $log" >&2
                failed=1
            fi
        done
        exit "$failed"
    ) &
    lane_pids+=("$!")
done

failed=0
for pid in "${lane_pids[@]}"; do
    wait "$pid" || failed=1
done
date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/attempts.done"
if ((failed)); then
    exit 1
fi
date -u +%FT%TZ >"$OUTPUT_ROOT/supervisor/export.done"

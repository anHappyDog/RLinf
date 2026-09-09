#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: $0 <host> <profile> <tmux_session> <remote_log_path> [sync|wait] [single|hybrid]" >&2
    exit 2
fi

TARGET_HOST=$1
PROFILE=$2
TMUX_SESSION=$3
REMOTE_LOG_PATH=$4
SYNC_MODE=${5:-sync}
FSDP_MODE=${6:-single}
case "$PROFILE" in
    p1_simple_sg | p2_ground_sg) ;;
    *)
        echo "Unsupported grounded-control profile: $PROFILE" >&2
        exit 2
        ;;
esac
if [[ ! "$TMUX_SESSION" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "tmux session contains unsupported characters: $TMUX_SESSION" >&2
    exit 2
fi
case "$SYNC_MODE" in
    sync | wait) ;;
    *)
        echo "Unsupported video sync mode: $SYNC_MODE" >&2
        exit 2
        ;;
esac
case "$FSDP_MODE" in
    single | hybrid) ;;
    *)
        echo "Unsupported FSDP mode: $FSDP_MODE" >&2
        exit 2
        ;;
esac

REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${B1K_SFT_PYTHON:-/mnt/public/daibo/venv/behavior_openpi/bin/python}
DATASET_ROOT=${B1K_DATASET_ROOT:-/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos}
SIDECAR_PATH=${B1K_SIDECAR_PATH:-/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_1000ep_press_stride8_radio_dense_button_boundarysafe_v3/data/part-00000.parquet}
REMOTE_SCRIPT="$REPO_PATH/toolkits/b1k_grounded/run_formal_mixed_sft.sh"

list_required_videos() {
    "$PYTHON_BIN" - "$SIDECAR_PATH" <<'PY'
import sys

import pyarrow.parquet as pq

columns = ["rgb_head_path", "rgb_left_wrist_path", "rgb_right_wrist_path"]
table = pq.read_table(sys.argv[1], columns=columns)
paths = sorted({path for column in columns for path in table[column].to_pylist()})
print("\n".join(paths))
PY
}

video_differences() {
    list_required_videos |
        rsync \
        -ani \
        --size-only \
        --files-from=- \
        "$DATASET_ROOT/" \
        "$TARGET_HOST:$DATASET_ROOT/" |
        sed -n '/^[<>]f/p'
}

if [[ "$SYNC_MODE" == "sync" ]]; then
    echo "Syncing sidecar-required videos to $TARGET_HOST ..."
    list_required_videos | rsync \
        -a \
        --partial \
        --stats \
        --files-from=- \
        "$DATASET_ROOT/" \
        "$TARGET_HOST:$DATASET_ROOT/"
else
    echo "Waiting for sidecar-required videos on $TARGET_HOST ..."
fi

while true; do
    if ! remaining=$(video_differences); then
        echo "Remote video preflight command failed." >&2
        exit 1
    fi
    if [[ -z "$remaining" ]]; then
        break
    fi
    if [[ "$SYNC_MODE" == "sync" ]]; then
        echo "Remote video preflight failed; rsync still reports differences:" >&2
        echo "$remaining" >&2
        exit 1
    fi
    echo "Video preflight still has $(wc -l <<<"$remaining") missing files."
    sleep 30
done

echo "Video preflight passed; launching $PROFILE as $TMUX_SESSION on $TARGET_HOST."
ssh "$TARGET_HOST" \
    tmux new-session -d -s "$TMUX_SESSION" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 RLINF_NODE_RANK=0 \
    bash "$REMOTE_SCRIPT" "$PROFILE" "$REMOTE_LOG_PATH" "$FSDP_MODE"

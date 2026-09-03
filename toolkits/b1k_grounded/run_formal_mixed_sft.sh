#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <profile> <log_path> [single|hybrid] [hydra_override ...]" >&2
    exit 2
fi

PROFILE=$1
LOG_PATH=$2
FSDP_MODE=${3:-single}
EXTRA_OVERRIDES=("${@:4}")
case "$PROFILE" in
    p1_simple_sg | p2_ground_sg) ;;
    *)
        echo "Unsupported grounded-control profile: $PROFILE" >&2
        exit 2
        ;;
esac
case "$FSDP_MODE" in
    single)
        FSDP_OVERRIDES=()
        ;;
    hybrid)
        FSDP_OVERRIDES=(
            cluster.num_nodes=2
            actor.fsdp_config.strategy=fsdp
            actor.fsdp_config.sharding_strategy=hybrid_shard
            actor.fsdp_config.hybrid_shard_size=4
        )
        ;;
    *)
        echo "Unsupported FSDP mode: $FSDP_MODE" >&2
        exit 2
        ;;
esac

REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${B1K_SFT_PYTHON:-/mnt/public/daibo/venv/behavior_openpi/bin/python}

export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export RLINF_NODE_RANK=${RLINF_NODE_RANK:-0}

# Keep torch.compile artifacts local to this experiment. Reusing the global
# per-user cache across concurrent distributed jobs can expose a stale or
# partially written Inductor autotuning result to every rank.
CACHE_KEY=$(basename "$LOG_PATH")
CACHE_KEY=${CACHE_KEY//[^[:alnum:]_.-]/_}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_${USER:-rlinf}_${CACHE_KEY}}

mkdir -p "$LOG_PATH"
cd "$REPO_PATH"

"$PYTHON_BIN" examples/sft/train_vla_sft.py \
    --config-path "$REPO_PATH/examples/sft/config" \
    --config-name behavior_pi05_grounded_oracle_mixed \
    runner.logger.log_path="$LOG_PATH" \
    data.grounded_control_profile="$PROFILE" \
    "${FSDP_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee "$LOG_PATH/launcher.log"

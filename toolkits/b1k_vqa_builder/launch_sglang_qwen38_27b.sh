#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-Qwen/Qwen3.8-27B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-4}

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}"

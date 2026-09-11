#!/usr/bin/env bash

set -euo pipefail

repo=${B1K_EVAL_REPO:-/mnt/public/daibo/timeline/0831/RLinf}
venv=${B1K_EVAL_VENV:-/mnt/public/daibo/venv/behavior_openpi}
model=${B1K_SUBPOOL_EVAL_CHECKPOINT:-/mnt/public/daibo/models/b1k_grounded_control_v01/pi05_official_train480_p2_sqrt_stage_step20000}
norm_stats_input=${B1K_SUBPOOL_NORM_STATS:-/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_train480_step6000_comet_native_v1/model/assets/behavior-1k/2025-challenge-demos/norm_stats.json}
if [[ -d $norm_stats_input ]]; then
  norm_stats_dir=$norm_stats_input
  norm_stats=$norm_stats_dir/norm_stats.json
else
  norm_stats=$norm_stats_input
  norm_stats_dir=$(dirname "$norm_stats")
fi

: "${B1K_CANONICAL_CANDIDATE_MANIFEST:?Set the exact 40-state candidate manifest.}"
: "${B1K_CANONICAL_BATCH_ROOT:?Set the directory containing batch_*/manifest.jsonl.}"
: "${B1K_CANONICAL_BASELINE_RESULT:?Set a fresh baseline evaluation directory.}"
: "${B1K_CANONICAL_SPLIT_OUTPUT:?Set a fresh 20/20 split output directory.}"

export PATH="$venv/bin:$PATH"
export PYTHONPATH="$repo"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="$repo/examples/embodiment"
export RAY_ADDRESS=${RAY_ADDRESS:-10.208.38.172:6379}
export TMPDIR=${TMPDIR:-/mnt/public/daibo/tmp}
export B1K_SUBPOOL_EVAL_CHECKPOINT="$model"
export B1K_SUBPOOL_NORM_STATS="$norm_stats_dir"
export B1K_GROUNDED_TOKEN_MAPPING=${B1K_GROUNDED_TOKEN_MAPPING:-/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_2025_radio_lunchbox_microwave_train480_stride8_subgoal_v04_split_v1/structural_token_mapping.json}
export B1K_ASSET_FINGERPRINT=b1k-2fd66d5c-radio-bc82d211
export RLINF_REMOTE_COLLECTOR_TOKEN=${RLINF_REMOTE_COLLECTOR_TOKEN:-B1K_CROSSDC_RADIO_0909_V1}

mapfile -t batches < <(find "$B1K_CANONICAL_BATCH_ROOT" -mindepth 2 -maxdepth 2 -name manifest.jsonl | sort)
if ((${#batches[@]} != 2)); then
  echo "Expected exactly two canonical evaluation batches, got ${#batches[@]}." >&2
  exit 2
fi

"$venv/bin/python" - "$B1K_CANONICAL_CANDIDATE_MANIFEST" "$norm_stats" "${batches[@]}" <<'PY'
import json
import sys
from pathlib import Path

candidate = Path(sys.argv[1])
norm_stats = Path(sys.argv[2])
batches = [Path(value) for value in sys.argv[3:]]
candidate_rows = [json.loads(line) for line in candidate.read_text().splitlines() if line]
batch_rows = [
    json.loads(line)
    for batch in batches
    for line in batch.read_text().splitlines()
    if line
]
assert len(candidate_rows) == 40, len(candidate_rows)
assert all(len([line for line in batch.read_text().splitlines() if line]) == 20 for batch in batches)
assert {row["snapshot_id"] for row in candidate_rows} == {row["snapshot_id"] for row in batch_rows}
assert all(row["metadata"]["reward"]["max_steps"] == 1280 for row in batch_rows)
assert all(json.loads(row["control_json"])["subgoal"] for row in batch_rows)
assert norm_stats.is_file(), norm_stats
print("Validated two disjoint 20-state SFT baseline batches.")
PY

mkdir -p "$B1K_CANONICAL_BASELINE_RESULT"
metrics_paths=()
max_eval_attempts=${B1K_CANONICAL_BASELINE_MAX_ATTEMPTS:-2}
cd "$repo"
for manifest in "${batches[@]}"; do
  batch=$(basename "$(dirname "$manifest")")
  output="$B1K_CANONICAL_BASELINE_RESULT/$batch"
  metrics="$output/eval_metrics.json"
  metrics_paths+=("$metrics")
  if [[ -s "$metrics" && -f "$output/evaluation.done" ]]; then
    echo "Skipping completed baseline batch $batch."
    continue
  fi
  mkdir -p "$output"
  export B1K_SUBPOOL_MANIFEST="$manifest"
  export B1K_SUBPOOL_EVAL_OUTPUT="$output"
  export B1K_SUBPOOL_EVAL_CELL="sft20k_${batch}_sde04"
  export B1K_SUBPOOL_EVAL_METRICS="$metrics"
  completed=false
  for attempt in $(seq 1 "$max_eval_attempts"); do
    if "$venv/bin/python" evaluations/eval_embodied_agent.py \
      --config-path "$repo/evaluations/behavior" \
      --config-name behavior_radio_canonical20_sft_eval \
      2>&1 | tee "$output/eval.log"; then
      completed=true
      break
    fi
    mv "$output/eval.log" "$output/eval.attempt_${attempt}.failed.log"
    echo "Baseline batch $batch failed on attempt $attempt/$max_eval_attempts."
  done
  if [[ $completed != true ]]; then
    echo "Baseline batch $batch exhausted $max_eval_attempts attempts." >&2
    exit 1
  fi
  date -u +%FT%TZ >"$output/evaluation.done"
done

scores="$B1K_CANONICAL_BASELINE_RESULT/sft20k_scores.json"
"$venv/bin/python" toolkits/b1k_grounded/prepare_canonical_pool.py scores \
  --source-manifest "$B1K_CANONICAL_CANDIDATE_MANIFEST" \
  --metrics "${metrics_paths[@]}" \
  --output "$scores" \
  --expected-attempts 20
"$venv/bin/python" toolkits/b1k_grounded/prepare_canonical_pool.py split \
  --source-manifest "$B1K_CANONICAL_CANDIDATE_MANIFEST" \
  --scores "$scores" \
  --output-dir "$B1K_CANONICAL_SPLIT_OUTPUT" \
  --split-size 20 \
  --seed 20260911
date -u +%FT%TZ >"$B1K_CANONICAL_BASELINE_RESULT/baseline_and_split.done"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${1:-"${SCRIPT_DIR}/configs/b1k_vqa_qwen38_27b.yaml"}
OUTPUT_ROOT=${2:-$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "${CONFIG}")}

if [[ ! -f "${OUTPUT_ROOT}/intermediate/candidates.jsonl" ]]; then
  python "${SCRIPT_DIR}/build_candidates.py" \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}"
fi

python "${SCRIPT_DIR}/run_sglang.py" \
  --config "${CONFIG}" \
  --mode generate \
  --output-root "${OUTPUT_ROOT}"

python "${SCRIPT_DIR}/run_sglang.py" \
  --config "${CONFIG}" \
  --mode judge \
  --output-root "${OUTPUT_ROOT}"

python "${SCRIPT_DIR}/finalize_dataset.py" \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}"

python "${SCRIPT_DIR}/validate_dataset.py" \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}"

python "${SCRIPT_DIR}/make_review_sheet.py" \
  --dataset-dir "${OUTPUT_ROOT}/dataset" \
  --output "${OUTPUT_ROOT}/review.html"

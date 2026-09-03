#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
CONFIG_NAME=libero_spatial_ppo_openpi
CONFIG_DIR=${SCRIPT_DIR}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openpi/bin/python}
RAY_BIN=${RAY_BIN:-/opt/venv/openpi/bin/ray}
PEER_HOST=${PEER_HOST:-bjd_dev_2}
HEAD_IP=${HEAD_IP:-10.204.5.212}
PEER_IP=${PEER_IP:-10.204.34.200}
NET_DEVICE=${NET_DEVICE:-eth0}
OUTPUT_ROOT=${1:-/mnt/public/daibo/tensor_compression_e2e_0824/100mbps/provider_comparison_$(date +%Y%m%d_%H%M%S)}
BENCHMARK_CASES=${BENCHMARK_CASES:-baseline,lz4,zstd}
BENCHMARK_EXCLUDED_DTYPES=${BENCHMARK_EXCLUDED_DTYPES:-'[float32]'}
export BENCHMARK_EXCLUDED_DTYPES

export EMBODIED_PATH=${REPO_ROOT}/examples/embodiment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export RLINF_CODE_WORKING_DIR=${REPO_ROOT}
export PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/supervisor.log") 2>&1

stop_ray() {
    "${RAY_BIN}" stop --force || true
    ssh "${PEER_HOST}" "${RAY_BIN} stop --force" || true
}

clear_tc() {
    tc qdisc del dev "${NET_DEVICE}" root 2>/dev/null || true
    ssh "${PEER_HOST}" "tc qdisc del dev '${NET_DEVICE}' root 2>/dev/null || true"
}

cleanup() {
    stop_ray
    clear_tc
}
trap cleanup EXIT

set_tc() {
    tc qdisc replace dev "${NET_DEVICE}" root tbf \
        rate 100mbit burst 1mb latency 400ms
    ssh "${PEER_HOST}" \
        "tc qdisc replace dev '${NET_DEVICE}' root tbf rate 100mbit burst 1mb latency 400ms"
}

capture_tc() {
    local destination=$1
    {
        echo "head"
        tc -s qdisc show dev "${NET_DEVICE}"
        echo "peer"
        ssh "${PEER_HOST}" "tc -s qdisc show dev '${NET_DEVICE}'"
    } >"${destination}"
}

start_ray() {
    local log_file=$1
    RLINF_NODE_RANK=0 RLINF_COMM_NET_DEVICES=${NET_DEVICE} \
        "${RAY_BIN}" start --head --port=6379 \
        --node-ip-address="${HEAD_IP}" --include-dashboard=false \
        --disable-usage-stats >>"${log_file}" 2>&1
    ssh "${PEER_HOST}" \
        "RLINF_NODE_RANK=1 RLINF_COMM_NET_DEVICES='${NET_DEVICE}' '${RAY_BIN}' start --address='${HEAD_IP}:6379' --node-ip-address='${PEER_IP}'" \
        >>"${log_file}" 2>&1
    "${RAY_BIN}" status >>"${log_file}" 2>&1
}

record_source_manifest() {
    local destination=$1
    sha256sum \
        "${REPO_ROOT}/rlinf/scheduler/cluster/cluster.py" \
        "${REPO_ROOT}/rlinf/scheduler/collective/collective_group.py" \
        "${REPO_ROOT}/rlinf/scheduler/collective/tensor_buffer_pool.py" \
        "${REPO_ROOT}/rlinf/scheduler/collective/tensor_compression.py" \
        "${REPO_ROOT}/rlinf/scheduler/worker/worker.py" \
        "${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
        >"${destination}"
}

run_case() {
    local case_name=$1
    local compression_enabled=$2
    local codec=$3
    local case_dir=${OUTPUT_ROOT}/${case_name}
    local exit_code

    echo "[$(date --iso-8601=seconds)] starting ${case_name}"
    mkdir -p "${case_dir}"
    stop_ray
    set_tc
    capture_tc "${case_dir}/tc_before_ray.txt"
    start_ray "${case_dir}/ray_start.log"
    capture_tc "${case_dir}/tc_before_job.txt"
    record_source_manifest "${case_dir}/manifest.sha256"
    cp "${CONFIG_DIR}/${CONFIG_NAME}.yaml" "${case_dir}/input_config.yaml"

    export BENCHMARK_CASE=${case_name}
    export BENCHMARK_RESULT_DIR=${case_dir}
    export BENCHMARK_COMPRESSION_ENABLED=${compression_enabled}
    export BENCHMARK_CODEC=${codec}

    "${PYTHON_BIN}" "${EMBODIED_PATH}/train_embodied_agent.py" \
        --config-path "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
        --cfg job --resolve >"${case_dir}/resolved_config.yaml"

    date +%s.%N >"${case_dir}/e2e_start"
    set +e
    "${PYTHON_BIN}" "${EMBODIED_PATH}/train_embodied_agent.py" \
        --config-path "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
        >"${case_dir}/train.log" 2>&1
    exit_code=$?
    set -e
    date +%s.%N >"${case_dir}/e2e_end"
    printf '%s\n' "${exit_code}" >"${case_dir}/exit_code"
    capture_tc "${case_dir}/tc_final.txt"
    stop_ray

    if ((exit_code != 0)); then
        echo "${case_name} failed with exit code ${exit_code}; stopping comparison"
        return "${exit_code}"
    fi
    echo "[$(date --iso-8601=seconds)] completed ${case_name}"
}

printf 'case\tcompression\tcodec\tstatus\n' >"${OUTPUT_ROOT}/cases.tsv"
IFS=',' read -r -a case_names <<<"${BENCHMARK_CASES}"
for case_name in "${case_names[@]}"; do
    case "${case_name}" in
        baseline) compression_enabled=false; codec=lz4 ;;
        lz4) compression_enabled=true; codec=lz4 ;;
        zstd) compression_enabled=true; codec=zstd ;;
        *) echo "Unsupported benchmark case: ${case_name}"; exit 2 ;;
    esac
    run_case "${case_name}" "${compression_enabled}" "${codec}"
    printf '%s\t%s\t%s\tcomplete\n' \
        "${case_name}" "${compression_enabled}" "${codec}" \
        >>"${OUTPUT_ROOT}/cases.tsv"
done

touch "${OUTPUT_ROOT}/complete"
echo "[$(date --iso-8601=seconds)] all cases completed"

#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_placement_rate_matrix_pair12_g3/control
RESULTS=/mnt/public/daibo/trajectory_benchmarks/20260720_placement_rate_matrix_pair12_g3
TRAJECTORY_REPO=/mnt/public/daibo/sg16_trajectory_run
MAIN_REPO=/mnt/public/daibo/sg16_main_baseline
BASELINE_CONFIG=libero_spatial_ppo_openpi_channel_bandwidth_baseline
TRAJECTORY_CONFIG=libero_spatial_ppo_openpi_trajectory_channel_2node_250mbit

mkdir -p "${CONTROL}" "${RESULTS}"
exec > >(tee -a "${RESULTS}/head-controller.log") 2>&1

cleanup() {
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc del dev eth0 root 2>/dev/null || true
}
trap cleanup EXIT

wait_for_file() {
  local path=$1 timeout=$2 started=$SECONDS
  while [[ ! -e "${path}" ]]; do
    if (( SECONDS - started >= timeout )); then return 1; fi
    sleep 1
  done
}

run_phase() {
  local phase=$1 repo=$2 config=$3
  shift 3
  local overrides=("$@") log=${RESULTS}/${phase}.log
  rm -f "${CONTROL}/${phase}."{start,ready,stop,stopped,status}
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc replace dev eth0 root tbf rate 100mbit burst 1mbit latency 400ms
  PYTHONPATH="${repo}" RLINF_NODE_RANK=0 RLINF_COMM_NET_DEVICES=eth0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
    /opt/venv/openpi/bin/ray start --head --node-ip-address="${HEAD_IP}" \
      --port=6379 --num-cpus=50 --num-gpus=4 --disable-usage-stats
  touch "${CONTROL}/${phase}.start"
  wait_for_file "${CONTROL}/${phase}.ready" 300 || return 124
  local started=$SECONDS
  set +e
  (cd "${repo}" && PYTHONPATH="${repo}" EMBODIED_PATH="${repo}/examples/embodiment" \
    MUJOCO_GL=egl /opt/venv/openpi/bin/python \
      examples/embodiment/train_embodied_agent.py --config-name "${config}" \
      "${overrides[@]}") 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${status}" >"${CONTROL}/${phase}.status"
  tc -s qdisc show dev eth0 >"${RESULTS}/${phase}.head-tc.txt"
  echo "$(date -Is) ${phase} status=${status} wall_seconds=$((SECONDS - started))"
  touch "${CONTROL}/${phase}.stop"
  wait_for_file "${CONTROL}/${phase}.stopped" 300 || true
  /opt/venv/openpi/bin/ray stop --force || true
}

run_phase g3_100_ar_baseline "${MAIN_REPO}" "${BASELINE_CONFIG}" \
  cluster.component_placement.rollout.placement=0-3 || true
run_phase g3_100_ar_trajectory "${TRAJECTORY_REPO}" "${TRAJECTORY_CONFIG}" \
  cluster.component_placement.rollout.placement=0-3 \
  trajectory.live.compress_images=true trajectory.compression.enabled=true || true
touch "${CONTROL}/all.done"

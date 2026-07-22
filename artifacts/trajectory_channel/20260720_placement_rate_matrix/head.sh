#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_placement_rate_matrix/control
RESULTS=/mnt/public/daibo/trajectory_benchmarks/20260720_placement_rate_matrix
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
  local path=$1
  local timeout=$2
  local started=$SECONDS
  while [[ ! -e "${path}" ]]; do
    if (( SECONDS - started >= timeout )); then
      echo "Timed out waiting for ${path}" >&2
      return 1
    fi
    sleep 1
  done
}

run_phase() {
  local phase=$1
  local rate=$2
  local repo=$3
  local config=$4
  shift 4
  local overrides=("$@")
  local log=${RESULTS}/${phase}.log

  rm -f "${CONTROL}/${phase}."{start,ready,stop,stopped,status}
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc replace dev eth0 root tbf rate "${rate}mbit" burst 1mbit latency 400ms
  echo "$(date -Is) ${phase} head qdisc: $(tc qdisc show dev eth0)"

  PYTHONPATH="${repo}" RLINF_NODE_RANK=0 RLINF_COMM_NET_DEVICES=eth0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
    /opt/venv/openpi/bin/ray start --head \
      --node-ip-address="${HEAD_IP}" --port=6379 \
      --num-cpus=50 --num-gpus=4 --disable-usage-stats
  touch "${CONTROL}/${phase}.start"
  wait_for_file "${CONTROL}/${phase}.ready" 300 || return 124
  /opt/venv/openpi/bin/ray status

  echo "$(date -Is) starting ${phase}: ${config} ${overrides[*]}"
  local started=$SECONDS
  set +e
  (
    cd "${repo}" || exit 125
    PYTHONPATH="${repo}" \
      EMBODIED_PATH="${repo}/examples/embodiment" \
      MUJOCO_GL=egl \
      /opt/venv/openpi/bin/python \
        examples/embodiment/train_embodied_agent.py \
        --config-name "${config}" "${overrides[@]}"
  ) 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${status}" >"${CONTROL}/${phase}.status"
  tc -s qdisc show dev eth0 >"${RESULTS}/${phase}.head-tc.txt"
  echo "$(date -Is) ${phase} status=${status} wall_seconds=$((SECONDS - started))"

  touch "${CONTROL}/${phase}.stop"
  wait_for_file "${CONTROL}/${phase}.stopped" 300 || true
  /opt/venv/openpi/bin/ray stop --force || true
  return 0
}

printf '%s\n' \
  'g1_250_ar_baseline:   250 Mbit, Actor+Rollout, ordinary Channel' \
  'g1_250_ar_trajectory: 250 Mbit, Actor+Rollout, live XOR+LZ4 + trajectory LZ4' \
  'g2_100_re_baseline:   100 Mbit, Rollout+Env, ordinary Channel' \
  'g2_100_re_trajectory: 100 Mbit, Rollout+Env, trajectory LZ4 only' \
  'g3_100_ar_baseline:   100 Mbit, Actor+Rollout, ordinary Channel' \
  'g3_100_ar_trajectory: 100 Mbit, Actor+Rollout, live XOR+LZ4 + trajectory LZ4' \
  >"${RESULTS}/matrix.txt"

run_phase g1_250_ar_baseline 250 "${MAIN_REPO}" "${BASELINE_CONFIG}" \
  cluster.component_placement.rollout.placement=0-3 || true
run_phase g1_250_ar_trajectory 250 "${TRAJECTORY_REPO}" "${TRAJECTORY_CONFIG}" \
  cluster.component_placement.rollout.placement=0-3 \
  trajectory.live.compress_images=true trajectory.compression.enabled=true || true

run_phase g2_100_re_baseline 100 "${MAIN_REPO}" "${BASELINE_CONFIG}" || true
run_phase g2_100_re_trajectory 100 "${TRAJECTORY_REPO}" "${TRAJECTORY_CONFIG}" \
  trajectory.live.compress_images=false trajectory.compression.enabled=true || true

run_phase g3_100_ar_baseline 100 "${MAIN_REPO}" "${BASELINE_CONFIG}" \
  cluster.component_placement.rollout.placement=0-3 || true
run_phase g3_100_ar_trajectory 100 "${TRAJECTORY_REPO}" "${TRAJECTORY_CONFIG}" \
  cluster.component_placement.rollout.placement=0-3 \
  trajectory.live.compress_images=true trajectory.compression.enabled=true || true

touch "${CONTROL}/all.done"
echo "$(date -Is) matrix finished"

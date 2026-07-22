#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_sg16_250mbit/control
RESULTS=/mnt/public/daibo/trajectory_benchmarks/20260720_sg16_250mbit
TRAJECTORY_REPO=/mnt/public/daibo/sg16_trajectory_run
MAIN_REPO=/mnt/public/daibo/sg16_main_baseline

mkdir -p "${CONTROL}" "${RESULTS}"
exec > >(tee -a "${RESULTS}/head-controller.log") 2>&1

cleanup() {
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc del dev eth0 root 2>/dev/null || true
}
trap cleanup EXIT

tc qdisc replace dev eth0 root tbf rate 250mbit burst 1mbit latency 400ms
echo "$(date -Is) head qdisc: $(tc qdisc show dev eth0)"

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
  local repo=$2
  local config=$3
  local log=${RESULTS}/${phase}.log

  rm -f "${CONTROL}/${phase}."{start,ready,stop,stopped,status}
  /opt/venv/openpi/bin/ray stop --force || true
  PYTHONPATH="${repo}" RLINF_NODE_RANK=0 RLINF_COMM_NET_DEVICES=eth0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
    /opt/venv/openpi/bin/ray start --head \
      --node-ip-address="${HEAD_IP}" --port=6379 \
      --num-cpus=50 --num-gpus=4 --disable-usage-stats
  touch "${CONTROL}/${phase}.start"
  wait_for_file "${CONTROL}/${phase}.ready" 300 || return 124
  /opt/venv/openpi/bin/ray status

  echo "$(date -Is) starting ${phase}"
  local started=$SECONDS
  set +e
  (
    cd "${repo}" || exit 125
    PYTHONPATH="${repo}" \
      EMBODIED_PATH="${repo}/examples/embodiment" \
      MUJOCO_GL=egl \
      /opt/venv/openpi/bin/python \
        examples/embodiment/train_embodied_agent.py \
        --config-name "${config}"
  ) 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${status}" >"${CONTROL}/${phase}.status"
  echo "$(date -Is) ${phase} status=${status} wall_seconds=$((SECONDS - started))"

  touch "${CONTROL}/${phase}.stop"
  wait_for_file "${CONTROL}/${phase}.stopped" 300 || true
  /opt/venv/openpi/bin/ray stop --force || true
  return 0
}

# Always enqueue main after trajectory, including when the first training command
# fails. Each phase records its own exit status under control/.
run_phase trajectory "${TRAJECTORY_REPO}" \
  libero_spatial_ppo_openpi_trajectory_channel_2node_250mbit || true
run_phase main "${MAIN_REPO}" \
  libero_spatial_ppo_openpi_channel_bandwidth_baseline || true

touch "${CONTROL}/all.done"
echo "$(date -Is) both phases finished"

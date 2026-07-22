#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
WORKER_IP=10.204.17.231
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_placement_rate_matrix_pair12_g3/control
RESULTS=/mnt/public/daibo/trajectory_benchmarks/20260720_placement_rate_matrix_pair12_g3
mkdir -p "${CONTROL}" "${RESULTS}"
exec > >(tee -a "${RESULTS}/worker-controller.log") 2>&1
cleanup() { /opt/venv/openpi/bin/ray stop --force || true; tc qdisc del dev eth0 root 2>/dev/null || true; }
trap cleanup EXIT

for phase in g3_100_ar_baseline g3_100_ar_trajectory; do
  while [[ ! -e "${CONTROL}/${phase}.start" ]]; do sleep 1; done
  if [[ "${phase}" == *_trajectory ]]; then
    repo=/mnt/public/daibo/sg16_trajectory_run
  else
    repo=/mnt/public/daibo/sg16_main_baseline
  fi
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc replace dev eth0 root tbf rate 100mbit burst 1mbit latency 400ms
  PYTHONPATH="${repo}" RLINF_NODE_RANK=1 RLINF_COMM_NET_DEVICES=eth0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
    /opt/venv/openpi/bin/ray start --address="${HEAD_IP}:6379" \
      --node-ip-address="${WORKER_IP}" --num-cpus=50 --num-gpus=4 \
      --disable-usage-stats
  touch "${CONTROL}/${phase}.ready"
  while [[ ! -e "${CONTROL}/${phase}.stop" ]]; do sleep 1; done
  tc -s qdisc show dev eth0 >"${RESULTS}/${phase}.worker-tc.txt"
  /opt/venv/openpi/bin/ray stop --force || true
  touch "${CONTROL}/${phase}.stopped"
done

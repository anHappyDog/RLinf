#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
WORKER_IP=10.204.17.231
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_correctness_1000_env64/control
REPO=/mnt/public/daibo/sg16_trajectory_run

mkdir -p "${CONTROL}"
cleanup() {
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc del dev eth0 root 2>/dev/null || true
}
trap cleanup EXIT

while [[ ! -e "${CONTROL}/start" ]]; do sleep 1; done
/opt/venv/openpi/bin/ray stop --force || true
tc qdisc del dev eth0 root 2>/dev/null || true
PYTHONPATH="${REPO}" RLINF_NODE_RANK=1 RLINF_COMM_NET_DEVICES=eth0 \
  GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
  /opt/venv/openpi/bin/ray start --address="${HEAD_IP}:6379" \
    --node-ip-address="${WORKER_IP}" --num-cpus=50 --num-gpus=4 \
    --disable-usage-stats
touch "${CONTROL}/ready"
while [[ ! -e "${CONTROL}/stop" ]]; do sleep 1; done
touch "${CONTROL}/stopped"

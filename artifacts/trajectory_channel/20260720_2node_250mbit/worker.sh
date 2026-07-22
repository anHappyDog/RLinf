#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
WORKER_IP=10.204.17.231
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_sg16_250mbit/control
RESULTS=/mnt/public/daibo/trajectory_benchmarks/20260720_sg16_250mbit

mkdir -p "${CONTROL}" "${RESULTS}"
exec > >(tee -a "${RESULTS}/worker-controller.log") 2>&1

cleanup() {
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc del dev eth0 root 2>/dev/null || true
}
trap cleanup EXIT

tc qdisc replace dev eth0 root tbf rate 250mbit burst 1mbit latency 400ms
echo "$(date -Is) worker qdisc: $(tc qdisc show dev eth0)"

wait_for_file() {
  local path=$1
  while [[ ! -e "${path}" ]]; do sleep 1; done
}

for phase in trajectory main; do
  if [[ "${phase}" == trajectory ]]; then
    repo=/mnt/public/daibo/sg16_trajectory_run
  else
    repo=/mnt/public/daibo/sg16_main_baseline
  fi
  wait_for_file "${CONTROL}/${phase}.start"
  /opt/venv/openpi/bin/ray stop --force || true
  PYTHONPATH="${repo}" RLINF_NODE_RANK=1 RLINF_COMM_NET_DEVICES=eth0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
    /opt/venv/openpi/bin/ray start \
      --address="${HEAD_IP}:6379" --node-ip-address="${WORKER_IP}" \
      --num-cpus=50 --num-gpus=4 --disable-usage-stats
  touch "${CONTROL}/${phase}.ready"
  echo "$(date -Is) ${phase} worker joined"
  wait_for_file "${CONTROL}/${phase}.stop"
  /opt/venv/openpi/bin/ray stop --force || true
  touch "${CONTROL}/${phase}.stopped"
  echo "$(date -Is) ${phase} worker stopped"
done

echo "$(date -Is) worker finished"

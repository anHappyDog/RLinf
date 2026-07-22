#!/usr/bin/env bash
set -uo pipefail

HEAD_IP=10.204.17.213
CONTROL=/mnt/public/daibo/trajectory_benchmarks/20260720_correctness_1000_env64/control
RESULTS=/mnt/public/daibo/trajectory_benchmarks/20260720_correctness_1000_env64
REPO=/mnt/public/daibo/sg16_trajectory_run
CONFIG=libero_spatial_ppo_openpi_trajectory_channel_2node_250mbit

mkdir -p "${CONTROL}" "${RESULTS}"
exec > >(tee -a "${RESULTS}/head-controller.log") 2>&1

cleanup() {
  /opt/venv/openpi/bin/ray stop --force || true
  tc qdisc del dev eth0 root 2>/dev/null || true
}
trap cleanup EXIT

/opt/venv/openpi/bin/ray stop --force || true
tc qdisc del dev eth0 root 2>/dev/null || true
PYTHONPATH="${REPO}" RLINF_NODE_RANK=0 RLINF_COMM_NET_DEVICES=eth0 \
  GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
  /opt/venv/openpi/bin/ray start --head --node-ip-address="${HEAD_IP}" \
    --port=6379 --num-cpus=50 --num-gpus=4 --disable-usage-stats
touch "${CONTROL}/start"

started=$SECONDS
while [[ ! -e "${CONTROL}/ready" ]]; do
  if (( SECONDS - started >= 300 )); then exit 124; fi
  sleep 1
done

set +e
(
  cd "${REPO}" || exit 125
  PYTHONPATH="${REPO}" EMBODIED_PATH="${REPO}/examples/embodiment" MUJOCO_GL=egl \
    /opt/venv/openpi/bin/python examples/embodiment/train_embodied_agent.py \
      --config-name "${CONFIG}" \
      cluster.component_placement.rollout.placement=0-3 \
      runner.max_epochs=1000 runner.max_steps=1000 \
      runner.logger.experiment_name=trajectory_correctness_1000_env64 \
      env.train.total_num_envs=64 \
      trajectory.live.compress_images=true \
      trajectory.compression.enabled=true
) 2>&1 | tee "${RESULTS}/run.log"
status=${PIPESTATUS[0]}
set -e
echo "${status}" >"${CONTROL}/status"
echo "$(date -Is) status=${status} wall_seconds=$((SECONDS - started))"
touch "${CONTROL}/stop"
stopping=$SECONDS
while [[ ! -e "${CONTROL}/stopped" && $((SECONDS - stopping)) -lt 300 ]]; do sleep 1; done
exit "${status}"

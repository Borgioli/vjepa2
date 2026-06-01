#!/usr/bin/env bash
set -euo pipefail

cd /path/to/data_root/vjepa2_1

export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export TORCH_DIST_TIMEOUT_SECONDS=300

PYTORCHRUN=/path/to/data_root/vjepa2_1/.venv/bin/torchrun
CONFIG_DIR=/path/to/data_root/vjepa2_1/configs/heads/single_tool
MASTER_ADDR=10.200.1.1

PORTS=(
  29721
  29722
  29723
  29724
  29725
  29726
)

CONFIGS=(
  clipper_tool_unconditioned.yaml
  clipper_action_unconditioned.yaml
  clipper_target_unconditioned.yaml
  clipper_tool_conditioned.yaml
  clipper_action_conditioned.yaml
  clipper_target_conditioned.yaml
)

run_head() {
  local port="$1"
  local config="$2"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] spark1 training ${config} on port ${port}"

  "${PYTORCHRUN}" \
    --nnodes=2 \
    --nproc-per-node=1 \
    --node-rank=1 \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${port}" \
    -m evals.main \
    --fname "${CONFIG_DIR}/${config}" \
    --devices cuda:0 \
    --debugmode 1 \
    --use_fsdp
}

for index in "${!CONFIGS[@]}"; do
  run_head "${PORTS[$index]}" "${CONFIGS[$index]}"
done

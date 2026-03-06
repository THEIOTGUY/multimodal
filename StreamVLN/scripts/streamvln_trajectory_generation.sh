#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

set -e
set -x
umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve Python from the RL micromamba env or active env
RL_PYTHON="${REPO_ROOT}/../.micromamba/envs/RL/bin/python"
if [ -x "${RL_PYTHON}" ]; then
  PYTHON="${RL_PYTHON}"
else
  PYTHON="$(command -v python)"
fi

# Absolute data root so habitat finds scenes regardless of cwd
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

# EGL/Habitat GPU binding (works on Colab and bare Linux with NVIDIA drivers)
if [ -d /usr/lib64-nvidia ]; then
  export LD_LIBRARY_PATH="/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
fi
cat >/tmp/10_nvidia_egl.json <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
JSON
export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/10_nvidia_egl.json
export HABITAT_GPU_DEVICE_ID="${HABITAT_GPU_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"

DATASET="${DATASET:-R2R}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/vln_r2r.yaml}"
if [ -z "${OUTPUT_PATH:-}" ]; then
  LATEST_EXISTING=$(ls -1dt "${DATA_ROOT}/trajectory_data/${DATASET}"/* 2>/dev/null | head -n 1 || true)
  OUTPUT_PATH="${LATEST_EXISTING:-${DATA_ROOT}/trajectory_data/${DATASET}/current}"
fi
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/datasets/envdrop/envdrop.json.gz}"

mkdir -p "${OUTPUT_PATH}" "${REPO_ROOT}/logs"

echo "==========================================================="
echo "  StreamVLN Trajectory Generation"
echo "  Dataset  : ${DATASET}"
echo "  Config   : ${CONFIG_PATH}"
echo "  Output   : ${OUTPUT_PATH}"
echo "  Data     : ${DATA_PATH}"
echo "  Python   : ${PYTHON}"
echo "==========================================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    streamvln/streamvln_trajectory_generation.py \
        --dataset "${DATASET}" \
        --config_path "${CONFIG_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --data_path "${DATA_PATH}" \
    2>&1 | tee "${OUTPUT_PATH}/log.log"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YAML="${ENV_YAML:-${REPO_ROOT}/multimodal_env_colab.yml}"
ENV_NAME="${ENV_NAME:-RL}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${REPO_ROOT}/.micromamba}"
MAMBA_BIN="${MAMBA_BIN:-${REPO_ROOT}/.local/bin/micromamba}"

USE_CONDA=0
USE_MICROMAMBA=0

if command -v conda >/dev/null 2>&1; then
  USE_CONDA=1
else
  mkdir -p "$(dirname "${MAMBA_BIN}")"
  if [ ! -x "${MAMBA_BIN}" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$(dirname "${MAMBA_BIN}")" --strip-components=1 bin/micromamba
  fi
  if [ ! -x "${MAMBA_BIN}" ]; then
    echo "Failed to install micromamba at ${MAMBA_BIN}"
    exit 1
  fi
  USE_MICROMAMBA=1
fi

if [ ! -f "$ENV_YAML" ]; then
  echo "Missing env yaml: $ENV_YAML"
  exit 1
fi

if [ "$USE_CONDA" -eq 1 ]; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$ENV_YAML" --prune
  else
    conda env create -n "$ENV_NAME" -f "$ENV_YAML"
  fi
  conda activate "$ENV_NAME"
else
  export MAMBA_ROOT_PREFIX
  if "${MAMBA_BIN}" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    "${MAMBA_BIN}" env update -n "$ENV_NAME" -f "$ENV_YAML" --prune
  else
    "${MAMBA_BIN}" env create -n "$ENV_NAME" -f "$ENV_YAML"
  fi
fi

# Ensure Habitat-Lab/Baselines are the expected editable v0.2.4 sources.
if [ ! -d "${REPO_ROOT}/habitat-lab" ]; then
  git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git "${REPO_ROOT}/habitat-lab"
fi
if [ "$USE_CONDA" -eq 1 ]; then
  pip install -e "${REPO_ROOT}/habitat-lab/habitat-lab" -e "${REPO_ROOT}/habitat-lab/habitat-baselines"
else
  "${MAMBA_BIN}" run -n "$ENV_NAME" pip install -e "${REPO_ROOT}/habitat-lab/habitat-lab" -e "${REPO_ROOT}/habitat-lab/habitat-baselines"
fi

# Required by StreamVLN rollout/eval scripts.
if [ "$USE_CONDA" -eq 1 ]; then
  pip install "depth-camera-filtering @ git+https://github.com/naokiyokoyama/depth_camera_filtering@39d6e2f391c8b2198a67ad96f94bf6da0acd48a0"
else
  "${MAMBA_BIN}" run -n "$ENV_NAME" pip install "depth-camera-filtering @ git+https://github.com/naokiyokoyama/depth_camera_filtering@39d6e2f391c8b2198a67ad96f94bf6da0acd48a0"
fi

# Optional vision utility dependency.
if [ "$USE_CONDA" -eq 1 ]; then
  pip install "open-clip-torch==2.32.0"
else
  "${MAMBA_BIN}" run -n "$ENV_NAME" pip install "open-clip-torch==2.32.0"
fi

cat <<EOF
Environment ready in conda env: ${ENV_NAME}

To run RL script in this repo:
  cd ${REPO_ROOT}/StreamVLN
  conda activate ${ENV_NAME}  # or micromamba run -n ${ENV_NAME}
  bash scripts/streamvln_rl_train.sh

If you need bitsandbytes CUDA compatibility, export:
  export BNB_CUDA_VERSION=126
EOF

#!/bin/bash
# =============================================================================
# StreamVLN RL Training — MindDrive Framework Launch Script
#
# This script launches the PPO-based RL fine-tuning of StreamVLN using the
# MindDrive architecture:
#
#   Block 1: Actor-Critic Brain (StreamVLN + Value Head)
#   Block 2: LoRA + KL-Penalty (Frozen Reference Model)
#   Block 3: Online Rollouts in Habitat Simulator
#   Block 4: Dense Rewards + GAE Advantage Estimation
#
# Usage:
#   bash scripts/streamvln_rl_train.sh
#
# Prerequisites:
#   - Pre-trained StreamVLN checkpoint at checkpoints/StreamVLN_Qwen3VL_4B_r2r_rxr
#   - Habitat-sim installed with MP3D scene datasets
#   - R2R VLN dataset at data/datasets/r2r/
#   - PEFT, trl, depth-camera-filtering installed
# =============================================================================

set -e

# ── Resolve repo root (one level above StreamVLN/scripts/) ──
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

MAMBA_BIN="${MAMBA_BIN:-${REPO_ROOT}/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${REPO_ROOT}/.micromamba}"
ENV_NAME="${ENV_NAME:-RL}"
export MAMBA_ROOT_PREFIX

if [ "${STREAMVLN_IN_MAMBA:-0}" != "1" ] && [ -x "${MAMBA_BIN}" ]; then
  exec "$MAMBA_BIN" run -n "$ENV_NAME" env STREAMVLN_IN_MAMBA=1 bash "$0" "$@"
fi

# bitsandbytes CUDA backend hints.
ENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [ -n "${ENV_SITE}" ]; then
  CUDA_LIBS=$(ls -d "${ENV_SITE}"/nvidia/*/lib 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"
fi
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"

# Colab EGL/Habitat GPU binding.
if [ -d /usr/lib64-nvidia ]; then
  export LD_LIBRARY_PATH="/usr/lib64-nvidia:${LD_LIBRARY_PATH}"
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
export HABITAT_GPU_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0,1
export EGL_DEVICE_ID=0

# ── Parse overrides from command-line (e.g. --output_dir passed by slurm wrapper) ──
for _arg in "$@"; do
  case "${_prev_arg}" in
    --output_dir) OUTPUT_DIR_OVERRIDE="${_arg}" ;;
    --model_path) MODEL_PATH_OVERRIDE="${_arg}" ;;
  esac
  _prev_arg="${_arg}"
done

# ── Config ──
MODEL_PATH="${MODEL_PATH_OVERRIDE:-checkpoints/streamvln_official_v1_3}"
HABITAT_CONFIG="config/vln_r2r.yaml"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-checkpoints/rl_minddrive}"
export TRAIN_LOG_BASENAME="${TRAIN_LOG_BASENAME:-train_rl.log}"
export TRAIN_LOG_MODE="${TRAIN_LOG_MODE:-a}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# ── Training Hyperparameters ──
# Scaled up to better utilise the 49 GB A6000 (baseline peak ~24 GB on original settings).
NUM_ITERATIONS=1000
EPISODES_PER_UPDATE=32        # 16 → 32: larger rollout buffer, richer PPO batches
LEARNING_RATE=1e-5
PPO_EPOCHS=4
MINI_BATCH_SIZE=8             # 4 → 8: doubles PPO update batch; ~40-44 GB peak with 4-bit
GRADIENT_ACCUMULATION_STEPS=2 # effective batch = MINI_BATCH_SIZE × ACCUM = 16

# ── LoRA Config (Block 2) ──
LORA_R=64
LORA_ALPHA=64   # alpha=r → scaling=1.0 (was 0.25, too conservative)
LORA_DROPOUT=0.05
LORA_TARGET="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# ── PPO Config (Block 4) ──
GAMMA=0.99
GAE_LAMBDA=0.95
CLIPRANGE=0.2
VF_COEF=0.5
ENTROPY_COEF=0.01
INIT_KL_COEF=0.1
TARGET_KL=6.0
MAX_GRAD_NORM=0.5

# ── Reward Config (Block 4: Dense Rewards) ──
# Dense rewards provide a per-step learning signal for long-horizon navigation:
#   progress shaping:   +0.1 × (d_prev - d_curr)  ← guides toward goal each step
#   collision penalty:  −0.05 per collision        ← discourages unsafe navigation
#   terminal:           +1.0 success / −1.0 failure
SUCCESS_REWARD=1.0
FAILURE_REWARD=-1.0
STEP_REWARD=0.0
USE_PROGRESS_REWARD=1  # enable DTG-based per-step shaping
PROGRESS_SCALE=0.1     # reward = 0.1 × (d_prev - d_curr) per step
USE_COLLISION_PENALTY=1
COLLISION_PENALTY=-0.05
SUCCESS_DISTANCE=3.0
EARLY_STOP_PENALTY=-0.5   # penalty for STOP within first MIN_STEPS_BEFORE_STOP steps
MIN_STEPS_BEFORE_STOP=5

# ── Rollout Config (Block 3) ──
NUM_ENVS=4         # Parallel Habitat envs: ≈4× faster rollout collection
MAX_EPISODE_STEPS=500
NUM_FRAMES=32
NUM_HISTORY=8
TEMPERATURE=0.6

# ── Checkpointing ──
SAVE_EVERY=5
EVAL_EVERY=25
MAX_CHECKPOINTS=5
SEED=42

echo "============================================================"
echo "  StreamVLN RL Training — MindDrive Framework"
echo "============================================================"
echo "  Model:     ${MODEL_PATH}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  LoRA:      r=${LORA_R}, α=${LORA_ALPHA}"
echo "  PPO:       γ=${GAMMA}, λ=${GAE_LAMBDA}, ε=${CLIPRANGE}"
echo "  Reward:    success=${SUCCESS_REWARD}, fail=${FAILURE_REWARD}"
echo "  Episodes:  ${EPISODES_PER_UPDATE} per update"
echo "  Iterations: ${NUM_ITERATIONS}"
echo "============================================================"

cd "$(dirname "$0")/.."

RESUME_ARGS=()
if [ "${AUTO_RESUME}" = "1" ]; then
  # Prefer highest numeric rl_iter_<N>; fall back to rl_iter_best / rl_iter_final.
  # -L: follow symlinks so symlinked checkpoint dirs are traversed.
  LATEST_NUMERIC=$(find -L "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'rl_iter_[0-9]*' 2>/dev/null \
    | sed -E 's#.*rl_iter_([0-9]+)$#\1 #g' \
    | awk '{print $1}' \
    | sort -n \
    | tail -n 1)
  if [ -n "${LATEST_NUMERIC}" ]; then
    RESUME_PATH="${OUTPUT_DIR}/rl_iter_${LATEST_NUMERIC}"
    RESUME_ARGS=(--resume_from "${RESUME_PATH}")
    echo "  Auto-resume: ${RESUME_PATH}"
  elif [ -d "${OUTPUT_DIR}/rl_iter_best" ]; then
    RESUME_PATH="${OUTPUT_DIR}/rl_iter_best"
    RESUME_ARGS=(--resume_from "${RESUME_PATH}")
    echo "  Auto-resume: ${RESUME_PATH}"
  elif [ -d "${OUTPUT_DIR}/rl_iter_final" ]; then
    RESUME_PATH="${OUTPUT_DIR}/rl_iter_final"
    RESUME_ARGS=(--resume_from "${RESUME_PATH}")
    echo "  Auto-resume: ${RESUME_PATH}"
  else
    echo "  Auto-resume: no checkpoint found, starting fresh"
  fi
fi

python streamvln/rl/train_rl.py \
    --model_path "${MODEL_PATH}" \
    --habitat_config_path "${HABITAT_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_iterations ${NUM_ITERATIONS} \
    --episodes_per_update ${EPISODES_PER_UPDATE} \
    --learning_rate ${LEARNING_RATE} \
    --ppo_epochs ${PPO_EPOCHS} \
    --mini_batch_size ${MINI_BATCH_SIZE} \
    --gamma ${GAMMA} \
    --gae_lambda ${GAE_LAMBDA} \
    --cliprange ${CLIPRANGE} \
    --vf_coef ${VF_COEF} \
    --entropy_coef ${ENTROPY_COEF} \
    --init_kl_coef ${INIT_KL_COEF} \
    --target_kl ${TARGET_KL} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lora_enable \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules "${LORA_TARGET}" \
    --success_reward ${SUCCESS_REWARD} \
    --failure_reward ${FAILURE_REWARD} \
    --step_reward ${STEP_REWARD} \
    ${USE_PROGRESS_REWARD:+--use_progress_reward} \
    --progress_scale ${PROGRESS_SCALE} \
    ${USE_COLLISION_PENALTY:+--use_collision_penalty} \
    --collision_penalty ${COLLISION_PENALTY} \
    --success_distance ${SUCCESS_DISTANCE} \
    --early_stop_penalty ${EARLY_STOP_PENALTY} \
    --min_steps_before_stop ${MIN_STEPS_BEFORE_STOP} \
    --max_episode_steps ${MAX_EPISODE_STEPS} \
    --num_frames ${NUM_FRAMES} \
    --num_history ${NUM_HISTORY} \
    --temperature ${TEMPERATURE} \
    --num_envs ${NUM_ENVS} \
    --save_every ${SAVE_EVERY} \
    --eval_every ${EVAL_EVERY} \
    --max_checkpoints ${MAX_CHECKPOINTS} \
    --seed ${SEED} \
    "${RESUME_ARGS[@]}" \
    "$@"

echo "============================================================"
echo "  Training complete. Checkpoints at: ${OUTPUT_DIR}"
echo "============================================================"

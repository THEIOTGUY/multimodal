"""
StreamVLN RL Training Script — MindDrive Framework

Main entry point for reinforcement learning fine-tuning of StreamVLN
using Proximal Policy Optimization (PPO) in the Habitat simulator.

This script orchestrates all four blocks of the MindDrive architecture:

  Block 1: Load pre-trained IL weights → Wrap with Value Head (Actor-Critic)
  Block 2: Apply LoRA adapters → Create frozen reference model (KL penalty)
  Block 3: Collect rollout episodes in Habitat (online simulation loop)
  Block 4: Compute dense rewards → PPO optimization with GAE advantages

Usage:
    python streamvln/rl/train_rl.py \\
        --model_path checkpoints/streamvln_official_v1_3 \\
        --habitat_config_path config/vln_rl_train.yaml \\
        --output_dir checkpoints/rl_minddrive \\
        --num_iterations 1000 \\
        --episodes_per_update 16 \\
        --lora_r 64 \\
        --learning_rate 1e-5

See ``--help`` for all options.
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import time
import shutil
import random
import logging
import warnings
import argparse
import glob
from datetime import datetime
from typing import Dict

import torch
import numpy as np

# ── Path setup (so imports work from any CWD) ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
_streamvln_dir = os.path.abspath(os.path.join(_script_dir, ".."))
_root_dir = os.path.abspath(os.path.join(_script_dir, "../.."))
for _p in (_streamvln_dir, _root_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import transformers
from transformers import AutoConfig

from model.stream_video_vln import StreamVLNForCausalLM
from rl.streamvln_value_head import StreamVLNWithValueHead, create_reference_model
from rl.reward import VLNRewardFunction, RewardConfig
from rl.rollout_collector import (
    RolloutCollector,
    RolloutConfig,
    trajectories_to_transition_batch,
)
from rl.streamvln_ppo_trainer import StreamVLNPPOTrainer, StreamVLNPPOConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="StreamVLN RL Training — MindDrive Framework"
    )

    # ── Model (Block 1: Pre-trained IL Weights) ──
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to pre-trained StreamVLN checkpoint (IL weights)"
    )
    parser.add_argument("--model_max_length", type=int, default=4096)

    # ── Habitat (Block 3: Simulation Loop) ──
    parser.add_argument(
        "--habitat_config_path", type=str, default="config/vln_rl_train.yaml",
        help="Path to Habitat config YAML"
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="val_unseen")

    # ── RL Training ──
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/rl_minddrive",
        help="Output directory for RL checkpoints"
    )
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--episodes_per_update", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # ── PPO Hyperparameters (Block 4) ──
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient (helps exploration; e.g., 0.01).",
    )
    parser.add_argument("--init_kl_coef", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=6.0)
    parser.add_argument(
        "--target_kl_stop",
        type=float,
        default=0.0,
        help=(
            "Early-stop PPO epochs in an iteration when minibatch approx_kl exceeds "
            "this value (>0 enables)."
        ),
    )
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument(
        "--use_kl_loss",
        action="store_true",
        default=False,
        help="Add explicit KL regularization loss against reference policy.",
    )
    parser.add_argument(
        "--kl_loss_coef",
        type=float,
        default=0.001,
        help="Coefficient for explicit KL loss term.",
    )
    parser.add_argument(
        "--kl_loss_type",
        type=str,
        default="mse",
        choices=["mse", "abs", "k1"],
        help="KL loss proxy type for explicit KL regularization.",
    )

    # ── LoRA (Block 2) ──
    parser.add_argument(
        "--lora_enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA adapters (use --no-lora_enable to disable)",
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules"
    )
    parser.add_argument(
        "--train_vision_lora",
        action="store_true",
        default=False,
        help=(
            "If set, keep LoRA parameters inside vision/multimodal towers trainable. "
            "Default is False to reduce GPU memory usage for RL updates."
        ),
    )

    # ── Reward (Block 4: Dense Rewards) ──
    parser.add_argument("--success_reward", type=float, default=1.0)
    parser.add_argument("--failure_reward", type=float, default=-1.0)
    parser.add_argument("--step_reward", type=float, default=0.0)
    parser.add_argument(
        "--use_progress_reward", action="store_true", default=True,
        help="Enable dense progress-based reward shaping (default: True)"
    )
    parser.add_argument("--progress_scale", type=float, default=0.1)
    parser.add_argument("--use_collision_penalty", action="store_true", default=True)
    parser.add_argument("--collision_penalty", type=float, default=-0.05)
    parser.add_argument(
        "--success_distance", type=float, default=3.0,
        help="Distance threshold in meters for episode success (must match Habitat config)"
    )
    parser.add_argument(
        "--early_stop_penalty", type=float, default=-0.5,
        help="Penalty applied when STOP is taken before min_steps (discourages STOP bias)"
    )
    parser.add_argument(
        "--min_steps_before_stop", type=int, default=5,
        help="Minimum steps before STOP is allowed without penalty"
    )

    # ── Parallel Rollout Collection ──
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of parallel Habitat environments for rollout collection (>1 uses VectorEnv)",
    )

    # ── Rollout (Block 3) ──
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--num_future_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--ppo_max_views",
        type=int,
        default=1,
        help=(
            "Maximum number of visual views per transition during PPO update. "
            "Lower values reduce GPU memory spikes (default: 1)."
        ),
    )

    # ── Checkpointing ──
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--max_checkpoints", type=int, default=5,
        help="Maximum number of numeric rl_iter_N checkpoints to keep (0=unlimited)",
    )

    # ── Gradient Accumulation ──
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2,
        help="Number of PPO minibatches to accumulate before optimizer.step()",
    )

    # ── LR Schedule ──
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Fraction of total iterations for linear LR warm-up (0 disables)",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine",
        choices=["cosine", "linear", "constant"],
        help="LR decay schedule after warm-up",
    )

    # ── Logging ──
    parser.add_argument(
        "--log_with", type=str, default=None,
        choices=["wandb", "tensorboard", None],
    )
    parser.add_argument("--wandb_project", type=str, default="streamvln-rl")
    parser.add_argument("--log_every", type=int, default=1)

    return parser.parse_args()


# ======================================================================
# Setup helpers
# ======================================================================

def _rotate_checkpoints(output_dir: str, max_keep: int = 5):
    """Delete oldest numeric rl_iter_N checkpoints when count exceeds max_keep."""
    if max_keep <= 0:
        return

    ckpt_dirs = []
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        m = re.match(r"rl_iter_(\d+)$", name)
        if m:
            ckpt_dirs.append((int(m.group(1)), os.path.join(output_dir, name)))
    ckpt_dirs.sort(key=lambda x: x[0])
    while len(ckpt_dirs) > max_keep:
        _, path = ckpt_dirs.pop(0)
        logger.info(f"Rotating checkpoint: removing {path}")
        shutil.rmtree(path, ignore_errors=True)


def setup_logging(output_dir: str):
    """Set up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_basename = os.environ.get("TRAIN_LOG_BASENAME", "train_rl.log").strip()
    if log_basename:
        log_file = os.path.join(output_dir, log_basename)
    else:
        log_file = os.path.join(
            output_dir, f"train_rl_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
    log_mode = os.environ.get("TRAIN_LOG_MODE", "a")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode=log_mode),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Logging to {log_file}")


def load_model_and_tokenizer(args, quantize=True, device_map=None):
    """
    Block 1 (part 1): Load the pre-trained StreamVLN model (IL weights).

    Parameters
    ----------
    quantize : bool
        If True (default), use 4-bit NF4 quantization when available.
        Set False to load in plain bfloat16 (e.g. for a frozen ref model
        on a separate GPU where accelerate hooks cause issues).
    device_map : dict, optional
        Explicit device_map passed to from_pretrained (e.g. {"":"cuda:1"}).
    """
    logger.info(f"Loading pre-trained IL model from {args.model_path}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
    )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # FlashAttention2 is optional in Colab; fall back to SDPA when unavailable.
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"
        logger.info("flash_attn not found; using attn_implementation=sdpa")

    model_kwargs = dict(
        attn_implementation=attn_impl,
        config=config,
        trust_remote_code=True,
    )

    # Always use 4-bit NF4 QLoRA – halves model weight memory, freeing VRAM for
    # rollout buffers and larger minibatches regardless of GPU size.
    enable_4bit = False
    if quantize:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            bnb_libs = glob.glob(
                os.path.join(
                    os.path.dirname(bitsandbytes.__file__),
                    "libbitsandbytes_cuda*.so",
                )
            )
            if torch.cuda.is_available() and len(bnb_libs) > 0:
                enable_4bit = True
            if enable_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is faster on Ampere+
                )
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("4-bit NF4 QLoRA enabled (double quant, bfloat16 compute)")
            else:
                logger.info("bitsandbytes CUDA backend not found; using bfloat16 loading")
                model_kwargs["torch_dtype"] = torch.bfloat16
        except Exception as e:
            logger.info(f"4-bit path unavailable ({e}); using bf16 loading")
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        logger.info("Loading model in bfloat16 (no quantization)")
        model_kwargs["torch_dtype"] = torch.bfloat16

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    # Suppress "copying from a non-meta parameter … to a meta parameter" warnings.
    # These fire whenever from_pretrained uses init_empty_weights (4-bit quantization
    # OR device_map triggers low_cpu_mem_usage=True).  The outer loader's
    # set_module_tensor_to_device properly materialises the weights; for the 4-bit
    # policy model our post-load vision tower reload below fixes them explicitly.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*non-meta parameter.*meta parameter.*",
        )
        base_model = StreamVLNForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs,
        )
    base_model.model.num_history = args.num_history
    if enable_4bit:
        # Keep lm_head in bfloat16 for consistency with compute dtype.
        base_model.lm_head = base_model.lm_head.to(torch.bfloat16)
        logger.info("4-bit path: lm_head in bfloat16 for dtype consistency")

    # ── Fix: 4-bit / accelerate loading creates vision tower on meta device ──
    # BitsAndBytes forces init_empty_weights(), so the nested
    # SigLipVisionModel.from_pretrained() inside __init__ silently fails —
    # the entire vision tower ends up with uninitialized weights after
    # dispatch.  We fix this by always reloading when 4-bit is active:
    #   1. Reloading the SigLip pretrained weights (real, non-meta)
    #   2. Overwriting with StreamVLN fine-tuned vision tower weights
    vision_tower = base_model.model.get_vision_tower()
    if vision_tower is not None and enable_4bit:
        logger.info(
            "Reloading vision tower to fix uninitialized weights from "
            "4-bit quantized loading (init_empty_weights issue)..."
        )
        # Step 1: Reload base SigLip pretrained weights
        vision_tower.is_loaded = False
        vision_tower.load_model()

        # Step 2: Overwrite with StreamVLN fine-tuned vision tower weights
        from safetensors.torch import load_file as safe_load_file
        vt_prefix = "model.vision_tower.vision_tower."
        n_loaded = 0
        for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
            shard_tensors = safe_load_file(shard, device="cpu")
            vt_state = {}
            for key, tensor in shard_tensors.items():
                if key.startswith(vt_prefix):
                    local_key = key[len(vt_prefix):]
                    vt_state[local_key] = tensor
            if vt_state:
                vision_tower.vision_tower.load_state_dict(vt_state, strict=False)
                n_loaded += len(vt_state)
            del shard_tensors, vt_state

        vision_tower.vision_tower.requires_grad_(False)
        vt_device = next(base_model.parameters()).device
        vision_tower.vision_tower = vision_tower.vision_tower.to(
            device=vt_device, dtype=torch.bfloat16
        )
        if n_loaded == 0:
            logger.warning(
                "Vision tower reload found 0 fine-tuned weight tensors in "
                "checkpoint shards — model may be using base SigLip weights!"
            )
        else:
            logger.info(
                f"Vision tower reloaded: {n_loaded} fine-tuned weight tensors applied"
            )

    total_params = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Loaded StreamVLNForCausalLM: {total_params / 1e6:.1f}M params")

    return base_model, tokenizer


def apply_lora(model, args):
    """
    Block 2 (part 1): Apply LoRA adapters to the base model.

    Only ~1% of parameters become trainable — the rest are frozen
    (the "Frozen Model (99% Weights)" from the diagram).
    """
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Default memory-safe behavior for single-GPU RL:
    # keep language-side LoRA trainable but freeze vision/multimodal LoRA.
    if not args.train_vision_lora:
        frozen_prefixes = (
            "vision_tower",
            "mm_projector",
            "vision_resampler",
            "multimodal_resampler",
        )
        frozen_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and any(prefix in name for prefix in frozen_prefixes):
                param.requires_grad_(False)
                frozen_params += param.numel()
        if frozen_params > 0:
            logger.info(
                "Froze vision/multimodal LoRA params for RL memory safety: "
                f"{frozen_params / 1e6:.2f}M params"
            )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied: {trainable / 1e6:.2f}M trainable / "
        f"{total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)"
    )

    return model


def setup_habitat_env(args, split="train"):
    """
    Block 3 (part 1): Initialize Habitat environment for rollout collection.
    """
    import habitat
    from habitat import Env
    from habitat_baselines.config.default import get_config as get_habitat_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
    )

    config = get_habitat_config(args.habitat_config_path)

    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.task.measurements.update(
            {
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = Env(config=config)
    logger.info(
        f"Habitat env initialized: {len(env.episodes)} episodes ({split})"
    )
    return env


class VLNEnvForRL:
    """
    Thin wrapper around ``habitat.Env`` that exposes extra methods needed
    by VectorEnv for parallel RL rollout collection.

    VectorEnv workers call methods by name via IPC; subclassing directly
    avoids pickling issues with lambda/partial functions.
    """

    def __init__(self, config):
        import habitat
        from habitat import Env
        self._env = Env(config=config)

    # ── Standard Env interface (delegated) ──
    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def original_action_space(self):
        """Required by Habitat VectorEnv's EnvObsDictWrapper via CALL_COMMAND."""
        return self._env.action_space

    @property
    def episodes(self):
        return self._env.episodes

    @property
    def current_episode(self):
        return self._env.current_episode

    @current_episode.setter
    def current_episode(self, value):
        self._env.current_episode = value

    @property
    def episode_over(self):
        return self._env.episode_over

    @property
    def number_of_episodes(self):
        return self._env.number_of_episodes

    def reset(self, **kwargs):
        return self._env.reset()

    def step(self, action):
        """Return (obs, reward, done, info) 4-tuple required by VectorEnv worker."""
        obs = self._env.step(action)
        done = bool(self._env.episode_over)
        return obs, 0.0, done, {}

    def get_metrics(self):
        return self._env.get_metrics()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        pass

    # Gym-style render (required by Habitat wrappers)
    def render(self, mode="human"):
        return self._env.render(mode)

    # ── Extra methods needed by collect_batch_parallel ──

    def set_episode_by_id(self, episode_id: str) -> None:
        """Set the episode matching episode_id as the current episode."""
        ep_id = str(episode_id)
        for ep in self._env.episodes:
            if str(ep.episode_id) == ep_id:
                self._env.current_episode = ep
                return
        raise ValueError(f"Episode ID {ep_id!r} not found in env episode list")

    def get_agent_height(self) -> float:
        """Return the Y-axis height of the agent in the current sim state."""
        return float(self._env.sim.get_agent_state().position[1])

    def get_instruction(self) -> str:
        """Return the navigation instruction for the current episode."""
        ep = self._env.current_episode
        return ep.instruction.instruction_text


def _make_vln_env(config):
    """Factory function used by VectorEnv as ``make_env_fn``."""
    return VLNEnvForRL(config)


def setup_habitat_vec_env(args, num_envs: int, split: str = "train"):
    """
    Create a Habitat VectorEnv of ``num_envs`` parallel VLNEnvForRL instances.

    Returns
    -------
    vec_env : habitat.core.vector_env.VectorEnv
    sim_sensors_config : OmegaConf
        Sensor config (camera_height, depth range, intrinsics) needed by
        ``collect_batch_parallel``.
    """
    import habitat
    from habitat.core.vector_env import VectorEnv
    from habitat_baselines.config.default import get_config as get_habitat_config
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig

    config = get_habitat_config(args.habitat_config_path)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.task.measurements.update(
            {"collisions": CollisionsMeasurementConfig()}
        )

    # Pull out sensor config before spinning up workers
    sim_sensors_config = (
        config.habitat.simulator.agents.main_agent.sim_sensors
    )

    env_fn_args = tuple(
        (config,) for _ in range(num_envs)
    )
    vec_env = VectorEnv(
        make_env_fn=_make_vln_env,
        env_fn_args=env_fn_args,
        auto_reset_done=False,  # we manage episode resets manually
    )
    logger.info(
        f"VectorEnv initialised: {num_envs} parallel Habitat envs ({split})"
    )
    return vec_env, sim_sensors_config


# ======================================================================
# Evaluation helper
# ======================================================================

def run_evaluation(model, tokenizer, args, iteration: int) -> Dict:
    """
    Run evaluation on val split and return metrics.

    Uses the existing VLNEvaluator from streamvln_eval.py.
    """
    from streamvln_eval import VLNEvaluator, evaluate

    logger.info(f"Running evaluation at iteration {iteration}")

    model.eval()
    model.reset(1)

    # Create a minimal args namespace for evaluator
    eval_args = argparse.Namespace(
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        num_history=args.num_history,
        save_video=False,
        model_max_length=args.model_max_length,
    )

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=1,
        output_path=os.path.join(args.output_dir, f"eval_iter_{iteration}"),
        model=model.pretrained_model,
        tokenizer=tokenizer,
        epoch=iteration,
        args=eval_args,
    )

    sucs, spls, oss, nav_errors, ep_num = evaluator.eval_action(0)
    metrics = {
        "eval/success_rate": sucs.float().mean().item(),
        "eval/spl": spls.float().mean().item(),
        "eval/oracle_success": oss.float().mean().item(),
        "eval/nav_error": nav_errors.float().mean().item(),
        "eval/num_episodes": int(ep_num.item()) if hasattr(ep_num, 'item') else int(ep_num),
    }

    logger.info(
        f"Eval [{iteration}]: SR={metrics['eval/success_rate']:.2%}, "
        f"SPL={metrics['eval/spl']:.4f}, "
        f"OS={metrics['eval/oracle_success']:.2%}, "
        f"NE={metrics['eval/nav_error']:.2f}m"
    )

    return metrics


# ======================================================================
# LR Scheduler
# ======================================================================

def _create_lr_scheduler(optimizer, num_iterations, warmup_ratio=0.05, scheduler_type="cosine"):
    """
    Create a learning rate scheduler with linear warm-up.

    Stabilizes early RL training when the value head is randomly initialized
    and LoRA adapters have not yet adapted to the reward signal.
    """
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = max(1, int(num_iterations * warmup_ratio)) if warmup_ratio > 0 else 0

    if scheduler_type == "constant" and warmup_steps == 0:
        return None

    def lr_lambda(current_step):
        # Linear warm-up
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # After warm-up
        progress = float(current_step - warmup_steps) / float(max(1, num_iterations - warmup_steps))
        if scheduler_type == "cosine":
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif scheduler_type == "linear":
            return max(0.05, 1.0 - progress)
        else:  # constant
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Logging backend (WandB / TensorBoard)
# ======================================================================

def _init_logging_backend(args):
    """Initialize WandB or TensorBoard if requested via --log_with."""
    if args.log_with == "wandb":
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"rl_{datetime.now():%Y%m%d_%H%M%S}",
                config=vars(args),
                reinit=True,
            )
            logger.info(f"WandB initialized — project: {args.wandb_project}")
            return wandb
        except Exception as e:
            logger.warning(f"WandB init failed ({e}); falling back to file logging")
            return None
    elif args.log_with == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.output_dir, "tensorboard")
            writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard initialized — log_dir: {tb_dir}")
            return writer
        except Exception as e:
            logger.warning(f"TensorBoard init failed ({e}); falling back to file logging")
            return None
    return None


def _log_to_backend(summary_writer, stats: Dict, step: int, args):
    """Push stats to WandB or TensorBoard."""
    if summary_writer is None:
        return
    try:
        if args.log_with == "wandb":
            summary_writer.log(stats, step=step)
        elif args.log_with == "tensorboard":
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    summary_writer.add_scalar(k, v, global_step=step)
    except Exception:
        pass  # Non-critical — don't crash training for logging failures


# ======================================================================
# Main training function
# ======================================================================

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Block 1: The Actor-Critic Brain
    # AutoModelForCausalLMWithValueHead pattern:
    #   StreamVLN Base (Actor) + Value Head (Critic)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 60)
    logger.info("Block 1: Loading Actor-Critic (StreamVLN + Value Head)")
    logger.info("=" * 60)

    base_model, tokenizer = load_model_and_tokenizer(args)

    # Wrap with Value Head (Critic)
    model = StreamVLNWithValueHead(base_model, summary_dropout_prob=0.0)
    logger.info("Value head attached — Actor-Critic brain ready")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Block 2: Low-Rank Adaptation (LoRA) & KL-Penalty
    #   - LoRA Adapter (Trained Weights) on q_proj, v_proj
    #   - Frozen Model (99% Weights) = base model without LoRA
    #   - Reference Model (Frozen StreamVLN Copy) for KL-Penalty
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 60)
    logger.info("Block 2: Applying LoRA & Creating Reference Model")
    logger.info("=" * 60)

    if args.lora_enable:
        model.pretrained_model = apply_lora(model.pretrained_model, args)
        model.is_peft_model = True

    # Create frozen reference model for KL penalty.
    # With 2+ GPUs we load a separate copy from disk onto cuda:1.
    # deepcopy does NOT work with 4-bit models (accelerate hooks break).
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        ref_device = torch.device("cuda:1")
        logger.info("2-GPU mode: policy on cuda:0, reference model on cuda:1")
        logger.info("Loading reference model from disk onto cuda:1 (bf16, no quantization) ...")
        ref_base, _ = load_model_and_tokenizer(args, quantize=False, device_map={"":"cuda:1"})
        ref_model = StreamVLNWithValueHead(ref_base, summary_dropout_prob=0.0)
        ref_model.requires_grad_(False)
        ref_model.eval()
        logger.info("Reference model loaded and frozen on cuda:1")
    else:
        ref_device = torch.device("cuda:0")
        ref_model = create_reference_model(model)
        if ref_model is not None:
            logger.info("Created explicit frozen reference model")
        else:
            logger.info(
                "Using PEFT disable_adapter() for reference logits (memory efficient)"
            )

    # Move to GPUs
    model = model.to(torch.device("cuda:0"))
    if ref_model is not None:
        ref_model = ref_model.to(ref_device)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Block 3: Online Rollouts (Simulation Loop)
    #   Habitat Simulator → Observation → Action Generation (LLM + Logit Mask)
    #   → Selected Action Token → Environment Step → Buffer Storage
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 60)
    logger.info("Block 3: Initializing Rollout Collector (Habitat Simulation)")
    logger.info("=" * 60)

    # Get image processor from the vision tower
    image_processor = model.pretrained_model.get_vision_tower().image_processor

    # Reward function (Block 4)
    reward_config = RewardConfig(
        success_reward=args.success_reward,
        failure_reward=args.failure_reward,
        step_reward=args.step_reward,
        use_progress_reward=args.use_progress_reward,
        progress_scale=args.progress_scale,
        use_collision_penalty=args.use_collision_penalty,
        collision_penalty=args.collision_penalty,
        max_episode_steps=args.max_episode_steps,
        success_distance=args.success_distance,
        early_stop_penalty=args.early_stop_penalty,
        min_steps_before_stop=args.min_steps_before_stop,
    )
    reward_fn = VLNRewardFunction(reward_config)

    # Rollout config
    rollout_config = RolloutConfig(
        max_episode_steps=args.max_episode_steps,
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        num_history=args.num_history,
        temperature=args.temperature,
        buffer_max_views=args.ppo_max_views,
    )

    rollout_collector = RolloutCollector(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        reward_fn=reward_fn,
        config=rollout_config,
        device=torch.device("cuda:0"),
        ref_model=ref_model,
        ref_device=ref_device,
    )

    # Habitat environment
    env = setup_habitat_env(args, split=args.train_split)

    # Parallel VectorEnv (only when --num_envs > 1)
    vec_env          = None
    sim_sensors_config = None
    if args.num_envs > 1:
        logger.info(
            f"  Parallel rollout: {args.num_envs} envs "
            f"(expected ≈{args.num_envs}× speedup over sequential)"
        )
        vec_env, sim_sensors_config = setup_habitat_vec_env(
            args, args.num_envs, split=args.train_split
        )
    else:
        # Extract sim_sensors_config from single env for pass-through
        env_cfg = env._config
        if hasattr(env_cfg, "habitat"):
            sim_sensors_config = (
                env_cfg.habitat.simulator.agents.main_agent.sim_sensors
            )
        else:
            sim_sensors_config = (
                env_cfg.simulator.agents.main_agent.sim_sensors
            )

    # Initialize streaming caches (one slot per parallel env)
    _cache_slots = max(args.num_envs, 1)
    model.reset(_cache_slots)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Block 4: Dense Rewards & Advantage Estimation
    #   PPO Trainer with:
    #     - Reward Signal: progress shaping + collision penalty + terminal
    #     - Advantage = Actual Reward - Critic's Prediction
    #     - Update LoRA Weights (Actor) + Value Head (Critic)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 60)
    logger.info("Block 4: Initializing PPO Trainer")
    logger.info("=" * 60)

    ppo_config = StreamVLNPPOConfig(
        batch_size=args.episodes_per_update,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lam=args.gae_lambda,
        init_kl_coef=args.init_kl_coef,
        target=args.target_kl,
        target_kl_stop=args.target_kl_stop,
        cliprange=args.cliprange,
        vf_coef=args.vf_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        use_kl_loss=args.use_kl_loss,
        kl_loss_coef=args.kl_loss_coef,
        kl_loss_type=args.kl_loss_type,
        episodes_per_update=args.episodes_per_update,
        max_episode_steps=args.max_episode_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )

    ppo_trainer = StreamVLNPPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        action_token_ids=rollout_collector.valid_token_ids,
        ppo_max_views=args.ppo_max_views,
        rollout_temperature=args.temperature,
    )

    # ── LR Scheduler with warm-up ──
    lr_scheduler = _create_lr_scheduler(
        optimizer=ppo_trainer.optimizer,
        num_iterations=args.num_iterations,
        warmup_ratio=args.warmup_ratio,
        scheduler_type=args.lr_scheduler_type,
    )
    ppo_trainer.lr_scheduler = lr_scheduler
    if lr_scheduler is not None:
        logger.info(
            f"LR scheduler: {args.lr_scheduler_type} with "
            f"{args.warmup_ratio:.0%} warm-up "
            f"({int(args.warmup_ratio * args.num_iterations)} iters)"
        )

    # ── Logging backend (WandB / TensorBoard) ──
    summary_writer = _init_logging_backend(args)

    # Resume from checkpoint (loads value head, optimizer, LoRA adapters, trainer state)
    if args.resume_from is not None:
        ppo_trainer.load_checkpoint(args.resume_from)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training Loop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 60)
    logger.info("Starting MindDrive RL Training Loop")
    logger.info(f"  Iterations: {args.num_iterations}")
    logger.info(f"  Episodes per update: {args.episodes_per_update}")
    logger.info(f"  PPO epochs: {args.ppo_epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(
        f"  PPO reg: entropy_coef={args.entropy_coef}, "
        f"use_kl_loss={args.use_kl_loss}, kl_loss_coef={args.kl_loss_coef}, "
        f"target_kl_stop={args.target_kl_stop}"
    )
    logger.info(f"  LoRA: r={args.lora_r}, α={args.lora_alpha}")
    logger.info(f"  Reward: success={args.success_reward}, fail={args.failure_reward}, step={args.step_reward}")
    logger.info("=" * 60)

    # Episode sampling — shuffle episodes each full pass for diversity
    random.seed(args.seed)
    all_episodes = list(env.episodes)
    random.shuffle(all_episodes)
    episode_idx = 0
    # Build episode-ID → episode map for vec_env (uses IDs not objects)
    _ep_id_list = [str(ep.episode_id) for ep in all_episodes]
    best_success_rate = 0.0
    best_eval_sr = 0.0  # Track eval SR separately (preferred for best-model)
    metrics_log = []

    for iteration in range(ppo_trainer.current_step, args.num_iterations):
        iter_start = time.time()

        # Re-shuffle episodes at the start of each full pass through the dataset
        if episode_idx >= len(all_episodes):
            random.shuffle(all_episodes)
            _ep_id_list = [str(ep.episode_id) for ep in all_episodes]
            episode_idx = 0
            logger.info("Episode pool exhausted — reshuffled for next pass")

        # ── Block 3: Collect rollout episodes ──
        logger.info("")
        logger.info("━" * 62)
        logger.info(
            f"  ITERATION {iteration + 1} / {args.num_iterations}"
            f"  —  Collecting {args.episodes_per_update} episodes"
        )
        logger.info("━" * 62)

        model.eval()
        trajectories = []

        if vec_env is not None:
            # ── Parallel collection via VectorEnv ──
            # Build a pool of episode IDs starting from episode_idx
            need = args.episodes_per_update
            pool_size = need * 3          # generous pool to handle skipped scenes
            pool_ids = [
                _ep_id_list[(episode_idx + k) % len(_ep_id_list)]
                for k in range(pool_size)
            ]
            episode_idx = (episode_idx + pool_size) % len(_ep_id_list)
            try:
                trajectories = rollout_collector.collect_batch_parallel(
                    vec_env, pool_ids, need, sim_sensors_config
                )
            except Exception as _e:
                logger.error(f"Parallel collection failed: {_e}", exc_info=True)
                trajectories = []
        else:
            # ── Sequential fallback (single env) ──
            skipped_missing_scene = 0
            attempts = 0
            max_attempts = max(args.episodes_per_update * 5, args.episodes_per_update)

            while len(trajectories) < args.episodes_per_update and attempts < max_attempts:
                attempts += 1
                episode = all_episodes[episode_idx % len(all_episodes)]
                episode_idx += 1

                try:
                    traj = rollout_collector.collect_episode(
                        env, episode, env_idx=0
                    )
                    trajectories.append(traj)
                    n_done = len(trajectories)
                    n_total = args.episodes_per_update
                    ok = "✓" if traj.success else "✗"
                    logger.info(
                        f"  Episode {n_done}/{n_total} {ok}  "
                        f"steps={traj.num_steps:3d}  "
                        f"reward={traj.total_reward:+.3f}  "
                        f"dtg={traj.distance_to_goal:.1f}m"
                    )
                except AssertionError as e:
                    err = str(e)
                    if "Likely an invalid scene name" in err or "Missing (at least) one of scene dataset attributes" in err:
                        skipped_missing_scene += 1
                        logger.warning(
                            "Skipping episode due to missing scene asset: "
                            f"episode_id={getattr(episode, 'episode_id', 'unknown')} "
                            f"scene_id={getattr(episode, 'scene_id', 'unknown')}"
                        )
                        continue
                    raise

            if skipped_missing_scene > 0:
                logger.warning(
                    f"Skipped {skipped_missing_scene} episode(s) with missing scene assets "
                    f"during rollout collection for iteration {iteration + 1}"
                )

        if len(trajectories) == 0:
            logger.warning(
                "No valid trajectories collected after retries; skipping iteration"
            )
            continue

        # Log rollout stats
        successes = [t.success for t in trajectories]
        rewards = [t.total_reward for t in trajectories]
        steps = [t.num_steps for t in trajectories]
        dtgs = [t.distance_to_goal for t in trajectories]

        logger.info(
            f"  Rollout done: {len(trajectories)} episodes | "
            f"SR={np.mean(successes):.1%} | "
            f"Reward={np.mean(rewards):+.3f} | "
            f"Steps={np.mean(steps):.0f} | "
            f"DTG={np.mean(dtgs):.2f}m"
        )
        logger.info("  Starting PPO optimization ...")

        # ── Convert to per-step transition batch format ──
        ppo_batch = trajectories_to_transition_batch(trajectories)

        if len(ppo_batch["transitions"]) == 0:
            logger.warning("Transition batch is empty after conversion, skipping iteration")
            continue

        logger.info(
            f"  PPO batch: {len(ppo_batch['transitions'])} transitions from "
            f"{len(trajectories)} episodes"
        )
        for h in logging.getLogger().handlers:
            h.flush()

        # ── Block 4: PPO optimization step ──
        model.train()
        try:
            stats = ppo_trainer.step(
                transitions=ppo_batch["transitions"],
            )
        except Exception as ppo_exc:
            logger.error(f"PPO step FAILED: {ppo_exc}", exc_info=True)
            for h in logging.getLogger().handlers:
                h.flush()
            raise

        # Step the LR scheduler once per training iteration (NOT per minibatch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        iter_time = time.time() - iter_start

        # ── Log stats ──
        stats["rollout/success_rate"] = np.mean(successes)
        stats["rollout/mean_reward"] = np.mean(rewards)
        stats["rollout/mean_steps"] = np.mean(steps)
        stats["rollout/mean_dtg"] = np.mean(dtgs)
        stats["timing/iter_seconds"] = iter_time
        # Include current learning rate for tracking
        if lr_scheduler is not None:
            stats["optim/lr"] = lr_scheduler.get_last_lr()[0]

        if (iteration + 1) % args.log_every == 0:
            ppo_trainer.log_stats(stats)
            _log_to_backend(summary_writer, stats, step=iteration + 1, args=args)

        metrics_log.append(
            {
                "iteration": iteration + 1,
                **{
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in stats.items()
                },
            }
        )

        # ── Save checkpoint ──
        if (iteration + 1) % args.save_every == 0:
            ppo_trainer.save_checkpoint(args.output_dir, iteration + 1)
            # Rotate old checkpoints to save disk space
            _rotate_checkpoints(args.output_dir, max_keep=getattr(args, 'max_checkpoints', 5))
            with open(
                os.path.join(args.output_dir, "metrics.jsonl"), "w"
            ) as f:
                for m in metrics_log:
                    f.write(json.dumps(m) + "\n")

        # ── Optional periodic evaluation (run BEFORE best-model tracking) ──
        eval_sr = None
        if args.eval_every > 0 and (iteration + 1) % args.eval_every == 0:
            try:
                eval_metrics = run_evaluation(
                    model, tokenizer, args, iteration + 1
                )
                stats.update(eval_metrics)
                eval_sr = eval_metrics.get("eval/success_rate", None)
                # Log eval metrics to backend
                _log_to_backend(summary_writer, eval_metrics, step=iteration + 1, args=args)

                # Best model from eval SR (preferred, noise-free metric)
                if eval_sr is not None and eval_sr > best_eval_sr:
                    best_eval_sr = eval_sr
                    ppo_trainer.save_checkpoint(args.output_dir, "best")
                    logger.info(
                        f"New best eval success rate: {best_eval_sr:.2%}"
                    )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
            finally:
                # run_evaluation resets the streaming cache to 1 slot.
                # Restore to the correct number of slots for continued training.
                model.reset(_cache_slots)

        # ── Fallback: track rollout SR when eval is not run this iter ──
        if eval_sr is None:
            current_sr = np.mean(successes)
            if current_sr > best_success_rate and best_eval_sr == 0.0:
                # Only save rollout-based "best" if we've never had an eval best
                best_success_rate = current_sr
                ppo_trainer.save_checkpoint(args.output_dir, "best")
                logger.info(
                    f"New best rollout success rate: {best_success_rate:.2%}"
                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training Complete
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    overall_best = max(best_success_rate, best_eval_sr)
    logger.info("=" * 60)
    logger.info("MindDrive RL Training Complete!")
    logger.info(f"Best eval   SR: {best_eval_sr:.2%}")
    logger.info(f"Best rollout SR: {best_success_rate:.2%}")
    logger.info(f"Overall best SR: {overall_best:.2%}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("=" * 60)

    # Final save
    ppo_trainer.save_checkpoint(args.output_dir, "final")

    with open(os.path.join(args.output_dir, "metrics.jsonl"), "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    # Close logging backend
    if summary_writer is not None:
        try:
            import wandb
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass
        if hasattr(summary_writer, 'close'):
            summary_writer.close()

    env.close()
    if vec_env is not None:
        try:
            vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

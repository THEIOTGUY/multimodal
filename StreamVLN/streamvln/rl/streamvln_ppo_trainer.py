"""
StreamVLN PPO Trainer — Adapted PPO for multimodal VLN.

Implements the core optimization of the MindDrive architecture (Block 4):

    ┌────────────────────────────────────────────────────────────────┐
    │  4. Dense Rewards & Advantage Estimation                      │
    │                                                               │
    │  Advantage = Actual Reward - Critic's Prediction              │
    │                                                               │
    │  ├──► Update LoRA Weights (Actor)                             │
    │  └──► Update Value Head (Critic)                              │
    └────────────────────────────────────────────────────────────────┘

This trainer handles:
    1. Episode-based rollouts in Habitat simulator
    2. Per-step advantage estimation from dense episode rewards
    3. KL-penalty against a frozen reference model (Block 2)
    4. LoRA (Actor) + Value Head (Critic) weight updates
"""

from __future__ import annotations

import gc
import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW

logger = logging.getLogger(__name__)


# ======================================================================
# Utility functions (self-contained, no trl dependency)
# ======================================================================

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    """Compute mean of tensor where mask is 1."""
    mask = mask.float()
    if dim is not None:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)
    return (tensor * mask).sum() / mask.sum().clamp(min=1e-8)


def masked_var(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute variance of tensor where mask is 1."""
    mean = masked_mean(tensor, mask)
    return masked_mean((tensor - mean) ** 2, mask)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def clip_by_value(tensor: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Element-wise clip tensor between lo and hi."""
    return torch.max(torch.min(tensor, hi), lo)


# ======================================================================
# KL Controllers
# ======================================================================

class AdaptiveKLController:
    """Adaptive KL penalty coefficient controller (from TRL)."""

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL penalty coefficient controller."""

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


# ======================================================================
# Config
# ======================================================================

@dataclass
class StreamVLNPPOConfig:
    """
    Extended PPO configuration for VLN reinforcement learning.

    Defaults tuned for the MindDrive VLN domain:
    - Moderate discount (0.99) since episodes are ~50-200 steps
    - Higher value function coefficient (0.5)
    - Lower initial KL penalty (0.1) for stability with vision models
    """

    # ── PPO Hyperparameters ──
    gamma: float = 0.99
    """Discount factor (episodic VLN, 50-200 steps)."""

    lam: float = 0.95
    """GAE lambda."""

    init_kl_coef: float = 0.1
    """Initial KL penalty coefficient."""

    adap_kl_ctrl: bool = True
    """Use adaptive KL controller."""

    target: float = 6.0
    """Target KL for adaptive controller."""

    horizon: int = 10000
    """Horizon for adaptive KL controller."""

    vf_coef: float = 0.5
    """Value function loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy bonus coefficient (encourages exploration)."""

    cliprange: float = 0.2
    """PPO clipping range for policy ratio."""

    cliprange_value: float = 0.2
    """PPO clipping range for value function."""

    ppo_epochs: int = 4
    """Number of optimization epochs per PPO update."""

    batch_size: int = 16
    """Number of episodes per PPO update."""

    mini_batch_size: int = 8
    """Mini-batch size for forward/backward passes."""

    learning_rate: float = 1e-5
    """Learning rate for AdamW optimizer."""

    max_grad_norm: float = 0.5
    """Gradient clipping norm."""

    use_kl_loss: bool = False
    """If True, add explicit KL loss term against reference policy."""

    kl_loss_coef: float = 0.001
    """Coefficient for KL loss term when ``use_kl_loss`` is enabled."""

    kl_loss_type: str = "mse"
    """KL loss proxy type: ``mse`` (recommended), ``abs``, or ``k1``."""

    target_kl_stop: float = 0.0
    """Early-stop PPO epochs if minibatch approx_kl exceeds this (>0 to enable)."""

    seed: int = 42
    """Random seed."""

    # ── VLN-specific additions ──
    episodes_per_update: int = 16
    """Number of rollout episodes to collect before each PPO update."""

    max_episode_steps: int = 500
    """Maximum steps per episode in Habitat."""

    save_every: int = 50
    """Save checkpoint every N PPO iterations."""

    eval_every: int = 25
    """Evaluate every N PPO iterations."""

    # ── Logging ──
    log_with: Optional[str] = None
    """Logging backend: 'wandb', 'tensorboard', or None."""

    # ── Gradient Accumulation ──
    gradient_accumulation_steps: int = 1
    """Number of minibatches to accumulate gradients before optimizer.step()."""


# ======================================================================
# Trainer
# ======================================================================

class StreamVLNPPOTrainer:
    """
    PPO Trainer adapted for StreamVLN multimodal navigation.

    Unlike standard PPO which expects text-only query/response pairs,
    this trainer handles:

    1. **Episode-based rollouts** in Habitat simulator
    2. **Per-step advantage estimation** from dense episode rewards
    3. **KL penalty** against frozen reference model (Block 2)
    4. **LoRA + Value Head** weight updates (Block 4)

    Parameters
    ----------
    config : StreamVLNPPOConfig
        PPO hyperparameters.
    model : StreamVLNWithValueHead
        Actor-critic model with value head.
    ref_model : StreamVLNWithValueHead or None
        Frozen reference model for KL penalty.
        If None and model uses PEFT, KL is computed via disable_adapter().
    tokenizer : PreTrainedTokenizer
        Tokenizer.
    optimizer : torch.optim.Optimizer, optional
        If None, AdamW is created internally.
    """

    def __init__(
        self,
        config: StreamVLNPPOConfig,
        model,
        ref_model=None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
        action_token_ids: Optional[List[int]] = None,
        ppo_max_views: int = 1,
        rollout_temperature: float = 1.0,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.device = model.device
        # Reference model may live on a different GPU (2-GPU mode)
        if ref_model is not None:
            self.ref_device = next(ref_model.parameters()).device
        else:
            self.ref_device = self.device

        torch.manual_seed(config.seed)

        # ── Optimizer ──
        # Two param groups: value head and LoRA/trainable base params.
        # Avoids creating O(N) groups (one per parameter) which adds
        # significant AdamW overhead at large LoRA rank.
        if optimizer is None:
            lora_params = [
                p for p in model.pretrained_model.parameters() if p.requires_grad
            ]
            self.optimizer = AdamW(
                [
                    {"params": list(model.v_head.parameters()), "lr": config.learning_rate},
                    {"params": lora_params, "lr": config.learning_rate},
                ],
                lr=config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        # ── LR Scheduler ──
        self.lr_scheduler = lr_scheduler

        # ── KL Controller (Adaptive or Fixed) ──
        if config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                config.init_kl_coef, config.target, config.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.init_kl_coef)

        # ── Running stats ──
        self.current_step = 0

        # Action vocabulary used by rollout logit masking.
        # PPO ratios/KL must be computed in the same action space.
        self.action_token_ids = (
            [int(t) for t in action_token_ids]
            if action_token_ids
            else self._infer_action_token_ids()
        )
        self.action_token_ids = list(dict.fromkeys(self.action_token_ids))
        if not self.action_token_ids:
            raise ValueError("No action token IDs provided or inferred for PPO.")
        self._action_token_to_idx = {
            tid: idx for idx, tid in enumerate(self.action_token_ids)
        }
        self.ppo_max_views = max(int(ppo_max_views), 1)
        self.rollout_temperature = float(rollout_temperature)

    def _infer_action_token_ids(self) -> List[int]:
        """Infer action-token IDs from tokenizer as a fallback."""
        if self.tokenizer is None:
            return []
        action_ids = []
        for token in ("STOP", "↑", "←", "→"):
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if ids:
                action_ids.append(int(ids[0]))
        return action_ids

    # ------------------------------------------------------------------
    # PPO Step: the main optimization method
    # ------------------------------------------------------------------

    def step(
        self,
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        """
        Perform a single PPO optimization step over per-step transitions.

        Parameters
        ----------
        transitions : list[dict]
            Per-step transitions produced by ``trajectories_to_transition_batch``.
            Each transition contains the multimodal state used for action sampling,
            the sampled action token, old policy logprob/value, reward, and done flag.
        """
        if transitions is None:
            raise ValueError(
                "Per-step transition data is required. "
                "Use trajectories_to_transition_batch(...) before trainer.step()."
            )

        if len(transitions) == 0:
            return {}

        n_steps = len(transitions)
        cfg = self.config
        timing = {}
        t0 = time.time()
        skipped_oom_minibatches = 0
        early_stop_triggered = False
        completed_epochs = 0

        # ── 1. Prepare tensors from rollout buffer ──
        old_logprobs = torch.tensor(
            [float(t["old_logprob"]) for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            [float(t["old_value"]) for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        env_rewards = torch.tensor(
            [float(t["reward"]) for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [bool(t["done"]) for t in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        timing["prepare"] = time.time() - t0

        logger.info(
            f"  PPO step: {n_steps} transitions, "
            f"mean_old_lp={old_logprobs.mean():.4f}, "
            f"mean_reward={env_rewards.mean():.4f}"
        )

        # Validate episode-boundary invariant: wherever the episode_id
        # changes (or at the buffer end), done must be True.  This prevents
        # GAE from bootstrapping across episode boundaries.
        for i in range(n_steps):
            is_last = (i == n_steps - 1)
            is_boundary = is_last or (
                transitions[i].get("episode_id") != transitions[i + 1].get("episode_id")
            )
            if is_boundary and not dones[i]:
                logger.warning(
                    f"Episode-boundary at transition {i} "
                    f"(ep={transitions[i].get('episode_id')}) has done=False; "
                    "forcing to True to prevent GAE leakage."
                )
                dones[i] = True

        # ── 2. Reference-policy logprobs for KL penalty ──
        t_ref = time.time()
        _has_cached_ref = all(t.get("ref_logprob") is not None for t in transitions)
        if _has_cached_ref:
            logger.info("  Using cached ref logprobs (skipping ref-model forward)")
            ref_logprobs = torch.tensor(
                [float(t["ref_logprob"]) for t in transitions],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            with torch.no_grad():
                ref_logprobs, _, _ = self._evaluate_transitions(
                    transitions, use_ref=True, with_grad=False
                )
        timing["forward_ref"] = time.time() - t_ref

        # ── 3. Reward shaping + GAE (per-step) ──
        rewards, non_score_rewards, kls = self.compute_rewards(
            env_rewards, old_logprobs, ref_logprobs
        )

        t_adv = time.time()
        values, advantages, returns = self.compute_advantages(
            old_values, rewards, dones
        )
        timing["advantages"] = time.time() - t_adv

        logger.info(
            f"  GAE done: mean_adv={advantages.mean():.4f}, "
            f"mean_ret={returns.mean():.4f}, mean_val={values.mean():.4f}"
        )

        # ── 4. PPO optimization epochs ──
        t_opt = time.time()
        all_stats = []
        accum_steps = max(1, cfg.gradient_accumulation_steps)
        for epoch_idx in range(cfg.ppo_epochs):
            logger.info(f"  PPO epoch {epoch_idx + 1}/{cfg.ppo_epochs} ...")
            perm = torch.randperm(n_steps, device=self.device)
            self.optimizer.zero_grad()
            accum_count = 0
            for mb_start in range(0, n_steps, cfg.mini_batch_size):
                mb_end = min(mb_start + cfg.mini_batch_size, n_steps)
                mb_idx = perm[mb_start:mb_end]
                mb_transitions = [transitions[i] for i in mb_idx.tolist()]
                try:
                    mb_logprobs, mb_vpreds, mb_entropies = self._evaluate_transitions(
                        mb_transitions, use_ref=False, with_grad=True
                    )

                    stats = self.train_minibatch(
                        old_logprobs=old_logprobs[mb_idx],
                        old_values=values[mb_idx],
                        ref_logprobs=ref_logprobs[mb_idx],
                        logprobs=mb_logprobs,
                        vpreds=mb_vpreds,
                        entropies=mb_entropies,
                        advantages=advantages[mb_idx],
                        returns=returns[mb_idx],
                        scale_loss=1.0 / accum_steps,
                        do_optimizer_step=False,
                    )
                    all_stats.append(stats)
                    accum_count += 1

                    # Optimizer step after accumulating enough minibatches
                    if accum_count % accum_steps == 0:
                        if cfg.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                cfg.max_grad_norm,
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if cfg.target_kl_stop > 0.0 and stats.get("approx_kl", 0.0) > cfg.target_kl_stop:
                        early_stop_triggered = True
                        logger.info(
                            "Early stop PPO update: minibatch approx_kl "
                            f"{stats.get('approx_kl', 0.0):.4f} exceeded "
                            f"target_kl_stop={cfg.target_kl_stop:.4f}."
                        )
                        break
                except (torch.OutOfMemoryError, RuntimeError) as _oom_exc:
                    # Catch both torch OOM and CUBLAS allocation failures
                    _msg = str(_oom_exc)
                    if not ("out of memory" in _msg.lower() or
                            "CUBLAS" in _msg or
                            "ALLOC_FAILED" in _msg or
                            "cudaMalloc" in _msg):
                        raise  # re-raise unrelated RuntimeErrors
                    skipped_oom_minibatches += 1
                    logger.warning(
                        "CUDA OOM during PPO minibatch; skipping minibatch "
                        f"({skipped_oom_minibatches} skipped so far in iter {self.current_step + 1})."
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
            # Flush remaining accumulated gradients at end of epoch
            if accum_count % accum_steps != 0:
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
            completed_epochs = epoch_idx + 1
            if early_stop_triggered:
                break
        timing["optimization"] = time.time() - t_opt

        # ── 5. Aggregate and log ──
        train_stats = self._aggregate_stats(all_stats)

        mean_ref_kl = kls.mean().item()
        self.kl_ctl.update(mean_ref_kl, n_steps)

        returns_var = returns.var(unbiased=False).item() if returns.numel() > 0 else 0.0
        val_error = ((values - returns) ** 2).mean().item() if returns.numel() > 0 else 0.0
        stats = {
            "ppo/loss/policy": train_stats.get("policy_loss", 0.0),
            "ppo/loss/value": train_stats.get("value_loss", 0.0),
            "ppo/loss/kl": train_stats.get("kl_loss", 0.0),
            "ppo/loss/entropy_bonus": train_stats.get("entropy_bonus", 0.0),
            "ppo/loss/total": train_stats.get("total_loss", 0.0),
            "ppo/policy/entropy": train_stats.get("entropy", 0.0),
            "ppo/policy/approx_kl": train_stats.get("approx_kl", 0.0),
            "ppo/policy/ref_kl": mean_ref_kl,
            "ppo/policy/clipfrac": train_stats.get("clipfrac", 0.0),
            "ppo/returns/mean": returns.mean().item(),
            "ppo/returns/var": returns_var,
            "ppo/val/mean": values.mean().item(),
            "ppo/val/error": val_error,
            "ppo/rewards/mean": env_rewards.mean().item(),
            "ppo/rewards/non_score_mean": non_score_rewards.mean().item(),
            "ppo/rewards/total_mean": rewards.mean().item(),
            "ppo/skipped_oom_minibatches": float(skipped_oom_minibatches),
            "ppo/optim/early_stop": float(early_stop_triggered),
            "ppo/optim/completed_epochs": float(completed_epochs),
            "ppo/kl_coef": self.kl_ctl.value,
            "ppo/timing/total": time.time() - t0,
        }
        stats.update({f"ppo/timing/{k}": v for k, v in timing.items()})

        self.current_step += 1
        return stats

    # ------------------------------------------------------------------
    # Transition forward passes
    # ------------------------------------------------------------------

    def _prepare_transition_inputs(
        self, transition: Dict[str, Any], target_device: torch.device = None
    ) -> Dict[str, Any]:
        """Convert one rollout transition into model forward inputs."""
        dev = target_device if target_device is not None else self.device
        required_keys = (
            "input_ids",
            "images",
            "depths",
            "poses",
            "intrinsics",
            "action_token_id",
        )
        for key in required_keys:
            if key not in transition:
                raise KeyError(f"Transition missing required key: {key}")

        input_ids = transition["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids.to(dev, dtype=torch.long, non_blocking=True),
        }

        for key in ("images", "depths", "poses", "intrinsics"):
            tensor = transition[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Transition field '{key}' must be a torch.Tensor")
            # History-expanded states can spike memory on PPO update.
            # Keep only the most recent views for stable single-GPU training.
            if tensor.dim() >= 2 and tensor.shape[1] > self.ppo_max_views:
                tensor = tensor[:, -self.ppo_max_views :]
            model_inputs[key] = tensor.to(dev, non_blocking=True)

        time_ids = transition.get("time_ids", [])
        if isinstance(time_ids, torch.Tensor):
            time_ids = time_ids.tolist()
        if time_ids is None:
            time_ids = []
        if len(time_ids) > self.ppo_max_views:
            time_ids = time_ids[-self.ppo_max_views :]
        if len(time_ids) == 0:
            time_ids = [0]
        model_inputs["time_ids"] = [[int(t) for t in time_ids]]

        task_type = transition.get("task_type", 0)
        if isinstance(task_type, torch.Tensor):
            task_type = int(task_type.item())
        model_inputs["task_type"] = [int(task_type)]

        return model_inputs

    def _select_action_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Select logits over the rollout action vocabulary only."""
        token_index = torch.tensor(
            self.action_token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        return logits.index_select(dim=-1, index=token_index)

    def _compute_action_stats_from_logits(
        self,
        last_logits: torch.Tensor,
        action_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action log-prob and entropy over the constrained action space.
        """
        action_idx = self._action_token_to_idx.get(int(action_token_id))
        if action_idx is None:
            raise ValueError(
                f"Action token {action_token_id} not in action vocabulary "
                f"{self.action_token_ids}"
            )

        action_logits = self._select_action_logits(last_logits)
        if self.rollout_temperature != 1.0:
            action_logits = action_logits / self.rollout_temperature
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_index = torch.full(
            (action_logits.shape[0], 1),
            action_idx,
            dtype=torch.long,
            device=action_logits.device,
        )
        action_logprob = log_probs.gather(dim=-1, index=action_index).squeeze(-1)
        entropy = entropy_from_logits(action_logits)
        return action_logprob, entropy

    def _forward_transition(
        self,
        transition: Dict[str, Any],
        use_ref: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one transition through policy/reference and return action stats."""

        # output_hidden_states=True is set internally by StreamVLNWithValueHead.forward()
        if use_ref and self.ref_model is not None:
            model_inputs = self._prepare_transition_inputs(
                transition, target_device=self.ref_device
            )
            logits, _, values = self.ref_model(**model_inputs)
            logits = logits.to(self.device, non_blocking=True)
            values = values.to(self.device, non_blocking=True)
        elif use_ref:
            model_inputs = self._prepare_transition_inputs(transition)
            if not hasattr(self.model.pretrained_model, "disable_adapter"):
                raise RuntimeError(
                    "Reference model unavailable and model has no disable_adapter()."
                )
            with self.model.pretrained_model.disable_adapter():
                logits, _, values = self.model(**model_inputs)
        else:
            model_inputs = self._prepare_transition_inputs(transition)
            logits, _, values = self.model(**model_inputs)

        action_token_id = int(transition["action_token_id"])
        action_logprob, entropy = self._compute_action_stats_from_logits(
            logits[:, -1, :], action_token_id
        )
        value = values[:, -1]
        return action_logprob, value, entropy

    def _evaluate_transitions(
        self,
        transitions: List[Dict[str, Any]],
        use_ref: bool = False,
        with_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a list of transitions and return stacked
        (action_logprobs, values, entropies).
        """
        if len(transitions) == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty, empty, empty

        logprobs = []
        values = []
        entropies = []
        context = torch.enable_grad if with_grad else torch.no_grad
        with context():
            for i, transition in enumerate(transitions):
                lp, val, ent = self._forward_transition(
                    transition,
                    use_ref=use_ref,
                )
                logprobs.append(lp.squeeze(0))
                values.append(val.squeeze(0))
                entropies.append(ent.squeeze(0))
                # Periodic CUDA cache cleanup to reduce peak memory usage during
                # long transition evaluation loops (ref or policy).
                if (i + 1) % 64 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return (
            torch.stack(logprobs),
            torch.stack(values),
            torch.stack(entropies),
        )

    # ------------------------------------------------------------------
    # Reward computation (KL penalty + environment reward)
    # ------------------------------------------------------------------

    def compute_rewards(
        self,
        env_rewards: torch.FloatTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute per-step shaped rewards:
            reward_t = env_reward_t - kl_coef * (logπ_old(a_t|s_t) - logπ_ref(a_t|s_t))
        """
        kls = old_logprobs - ref_logprobs
        non_score_rewards = -self.kl_ctl.value * kls
        rewards = env_rewards + non_score_rewards
        return rewards, non_score_rewards, kls

    # ------------------------------------------------------------------
    # GAE Advantage Estimation (Block 4)
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        dones: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Generalized Advantage Estimation (GAE-λ) over flattened transitions.

        Transitions from multiple episodes are concatenated.  Episode boundaries
        are detected via ``dones``: when ``dones[t]`` is True, step ``t`` is
        terminal and step ``t+1`` (if it exists) belongs to a **different**
        episode.  We must NOT bootstrap across episode boundaries — the next
        value after a terminal step is always 0, regardless of ``values[t+1]``.
        """
        cfg = self.config
        values = values.detach()
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

        n_steps = rewards.shape[0]
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                # Last transition in the buffer — no next step exists.
                next_nonterminal = 0.0
                next_values = torch.zeros_like(values[t])
            else:
                # dones[t] == True  →  step t is terminal, step t+1 is a new
                # episode.  We must NOT use values[t+1] as bootstrap.
                next_nonterminal = 1.0 - dones[t].float()
                next_values = values[t + 1] * next_nonterminal

            delta = rewards[t] + cfg.gamma * next_values - values[t]
            # Reset the GAE accumulator at episode boundaries so advantages
            # from one episode never leak into another.
            lastgaelam = (
                delta + cfg.gamma * cfg.lam * next_nonterminal * lastgaelam
            )
            advantages[t] = lastgaelam

        returns = advantages + values

        # Normalize advantages for stability.
        if advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        return values, advantages, returns

    # ------------------------------------------------------------------
    # PPO mini-batch optimization (Block 4: Update Weights)
    # ------------------------------------------------------------------

    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        old_values: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        entropies: torch.FloatTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        scale_loss: float = 1.0,
        do_optimizer_step: bool = True,
    ) -> Dict:
        """
        Compute PPO loss and accumulate gradients for one transition mini-batch.

        Parameters
        ----------
        scale_loss : float
            Loss scaling factor for gradient accumulation (1/accum_steps).
        do_optimizer_step : bool
            If True, run optimizer.step() + grad clipping inside this method
            (legacy behavior). If False, the caller is responsible for stepping.
        """
        cfg = self.config

        ratio = torch.exp(logprobs - old_logprobs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - cfg.cliprange, 1.0 + cfg.cliprange
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        vpredclipped = clip_by_value(
            vpreds,
            old_values - cfg.cliprange_value,
            old_values + cfg.cliprange_value,
        )
        vf_loss1 = (vpreds - returns) ** 2
        vf_loss2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        kl_loss = torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
        if cfg.use_kl_loss and cfg.kl_loss_coef > 0:
            kl_delta = logprobs - ref_logprobs
            if cfg.kl_loss_type == "abs":
                kl_loss = kl_delta.abs().mean()
            elif cfg.kl_loss_type == "k1":
                kl_loss = kl_delta.mean().clamp(min=0.0)
            else:
                # Default to low-variance positive penalty.
                kl_loss = 0.5 * (kl_delta ** 2).mean()

        entropy_bonus = entropies.mean()
        loss = pg_loss + cfg.vf_coef * vf_loss
        if cfg.use_kl_loss and cfg.kl_loss_coef > 0:
            loss = loss + cfg.kl_loss_coef * kl_loss
        if cfg.entropy_coef > 0:
            loss = loss - cfg.entropy_coef * entropy_bonus

        # Scale loss for gradient accumulation
        scaled_loss = loss * scale_loss
        scaled_loss.backward()

        if do_optimizer_step:
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            clipfrac = (torch.abs(ratio - 1.0) > cfg.cliprange).float().mean().item()
            approx_kl = (0.5 * (logprobs - old_logprobs) ** 2).mean().item()
            ent = entropies.mean().item()

        return {
            "policy_loss": pg_loss.item(),
            "value_loss": vf_loss.item(),
            "total_loss": loss.item(),
            "clipfrac": clipfrac,
            "approx_kl": approx_kl,
            "entropy": ent,
            "kl_loss": kl_loss.item(),
            "entropy_bonus": entropy_bonus.item(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _aggregate_stats(self, stats_list: List[Dict]) -> Dict:
        """Average stats across all mini-batches."""
        if not stats_list:
            return {}
        keys = stats_list[0].keys()
        return {k: np.mean([s[k] for s in stats_list]) for k in keys}

    def save_checkpoint(self, output_dir: str, iteration):
        """Save model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"rl_iter_{iteration}")
        os.makedirs(save_path, exist_ok=True)

        # Save value head
        torch.save(
            self.model.v_head.state_dict(),
            os.path.join(save_path, "v_head.pt"),
        )

        # Save LoRA adapters (if using PEFT) or full model weights
        self.model.pretrained_model.save_pretrained(save_path)

        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(save_path, "optimizer.pt"),
        )

        # Save trainer state
        trainer_state = {"iteration": iteration, "kl_coef": self.kl_ctl.value}
        if self.lr_scheduler is not None:
            trainer_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(
            trainer_state,
            os.path.join(save_path, "trainer_state.pt"),
        )

        logger.info(f"Saved checkpoint at iteration {iteration} to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint (value head, LoRA adapters, optimizer, trainer state)."""
        # Load LoRA adapters if using PEFT
        if hasattr(self.model, "is_peft_model") and self.model.is_peft_model:
            adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
            if os.path.exists(adapter_config):
                from peft import set_peft_model_state_dict
                adapter_weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
                if not os.path.exists(adapter_weights_path):
                    adapter_weights_path = os.path.join(checkpoint_path, "adapter_model.bin")
                if os.path.exists(adapter_weights_path):
                    try:
                        if adapter_weights_path.endswith(".safetensors"):
                            from safetensors.torch import load_file
                            adapter_weights = load_file(adapter_weights_path, device=str(self.device))
                        else:
                            adapter_weights = torch.load(adapter_weights_path, map_location=self.device, weights_only=False)
                        set_peft_model_state_dict(self.model.pretrained_model, adapter_weights)
                        logger.info(f"Loaded LoRA adapters from {checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA adapters: {e}. Starting from current weights.")
                else:
                    logger.warning(f"adapter_config.json found but no weights file at {checkpoint_path}")
            else:
                logger.info("No adapter_config.json found; LoRA weights not loaded.")

        # Load value head
        v_head_path = os.path.join(checkpoint_path, "v_head.pt")
        if os.path.exists(v_head_path):
            self.model.v_head.load_state_dict(
                torch.load(v_head_path, map_location=self.device, weights_only=False)
            )
            logger.info(f"Loaded value head from {v_head_path}")

        # Load optimizer
        opt_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=False)
            )
            logger.info(f"Loaded optimizer from {opt_path}")

        # Load trainer state
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            self.current_step = state.get("iteration", 0)
            self.kl_ctl.value = state.get("kl_coef", self.config.init_kl_coef)
            if self.lr_scheduler is not None and "lr_scheduler" in state:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
                logger.info("Restored LR scheduler state")
            logger.info(f"Resumed from iteration {self.current_step}")

    def log_stats(self, stats: Dict, batch_info: Dict = None):
        """Log training statistics in a clear, grouped format."""
        it = self.current_step
        w = 62  # box width

        lines = []
        lines.append("")
        lines.append("\u2550" * w)
        lines.append(f"  ITERATION {it}  SUMMARY")
        lines.append("\u2550" * w)

        # ── Rollout Performance (most important) ──
        sr = stats.get("rollout/success_rate", None)
        mr = stats.get("rollout/mean_reward", None)
        ms = stats.get("rollout/mean_steps", None)
        dtg = stats.get("rollout/mean_dtg", None)
        lines.append("  Rollout Performance")
        lines.append("  " + "\u2500" * (w - 4))
        if sr is not None:
            lines.append(f"    Success Rate ........... {sr:.2%}")
        if mr is not None:
            lines.append(f"    Mean Reward ............ {mr:+.4f}")
        if ms is not None:
            lines.append(f"    Mean Steps / Episode ... {ms:.1f}")
        if dtg is not None:
            lines.append(f"    Distance to Goal ....... {dtg:.2f} m")

        # ── PPO Losses ──
        pl = stats.get("ppo/loss/policy", None)
        vl = stats.get("ppo/loss/value", None)
        tl = stats.get("ppo/loss/total", None)
        eb = stats.get("ppo/loss/entropy_bonus", None)
        kl_loss = stats.get("ppo/loss/kl", None)
        lines.append("")
        lines.append("  PPO Losses")
        lines.append("  " + "\u2500" * (w - 4))
        if pl is not None:
            lines.append(f"    Policy Loss ............ {pl:+.6f}")
        if vl is not None:
            lines.append(f"    Value  Loss ............ {vl:.6f}")
        if tl is not None:
            lines.append(f"    Total  Loss ............ {tl:.6f}")
        if kl_loss is not None and kl_loss != 0.0:
            lines.append(f"    KL     Loss ............ {kl_loss:.6f}")
        if eb is not None:
            lines.append(f"    Entropy (H) ............ {eb:.4f}")

        # ── Policy Diagnostics ──
        akl = stats.get("ppo/policy/approx_kl", None)
        rkl = stats.get("ppo/policy/ref_kl", None)
        cf = stats.get("ppo/policy/clipfrac", None)
        lines.append("")
        lines.append("  Policy Diagnostics")
        lines.append("  " + "\u2500" * (w - 4))
        if akl is not None:
            lines.append(f"    Approx KL .............. {akl:.6f}")
        if rkl is not None:
            lines.append(f"    Ref KL (old-ref) ....... {rkl:+.6f}")
        if cf is not None:
            lines.append(f"    Clip Fraction .......... {cf:.2%}")
        kl_coef = stats.get("ppo/kl_coef", None)
        if kl_coef is not None:
            lines.append(f"    KL Coefficient ......... {kl_coef:.6f}")

        # ── Value / Returns ──
        rm = stats.get("ppo/returns/mean", None)
        rv = stats.get("ppo/returns/var", None)
        vm = stats.get("ppo/val/mean", None)
        ve = stats.get("ppo/val/error", None)
        lines.append("")
        lines.append("  Value Head & Returns")
        lines.append("  " + "\u2500" * (w - 4))
        if rm is not None:
            lines.append(f"    Returns  (mean) ........ {rm:+.4f}")
        if rv is not None:
            lines.append(f"    Returns  (var) ......... {rv:.4f}")
        if vm is not None:
            lines.append(f"    V-pred   (mean) ........ {vm:+.4f}")
        if ve is not None:
            lines.append(f"    V-pred   error ......... {ve:.4f}")

        # ── Optimizer ──
        lr = stats.get("optim/lr", None)
        skipped = stats.get("ppo/skipped_oom_minibatches", 0)
        epochs_done = stats.get("ppo/optim/completed_epochs", None)
        early = stats.get("ppo/optim/early_stop", 0)
        lines.append("")
        lines.append("  Optimizer")
        lines.append("  " + "\u2500" * (w - 4))
        if lr is not None:
            lines.append(f"    Learning Rate .......... {lr:.2e}")
        if epochs_done is not None:
            lines.append(f"    PPO Epochs Done ........ {int(epochs_done)}")
        if skipped and skipped > 0:
            lines.append(f"    OOM Skipped Batches .... {int(skipped)}  \u26a0")
        if early and early > 0:
            lines.append(f"    Early Stopped .......... YES \u26a0")

        # ── Timing ──
        total = stats.get("ppo/timing/total", None)
        t_ref = stats.get("ppo/timing/forward_ref", None)
        t_opt = stats.get("ppo/timing/optimization", None)
        t_iter = stats.get("timing/iter_seconds", None)
        lines.append("")
        lines.append("  Timing")
        lines.append("  " + "\u2500" * (w - 4))
        if t_iter is not None:
            m, s = divmod(int(t_iter), 60)
            lines.append(f"    Iteration Wall Time .... {m}m {s}s")
        if total is not None:
            m, s = divmod(int(total), 60)
            lines.append(f"    PPO Step Time .......... {m}m {s}s")
        if t_ref is not None:
            lines.append(f"    Ref-Model Forward ...... {t_ref:.1f}s")
        if t_opt is not None:
            m, s = divmod(int(t_opt), 60)
            lines.append(f"    Optimization ........... {m}m {s}s")

        lines.append("\u2550" * w)
        lines.append("")

        logger.info("\n".join(lines))

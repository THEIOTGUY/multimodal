"""
VLN Reward Function for the MindDrive RL framework.

Implements a fully dense reward signal (Block 4):

    ┌──────────────────────────────────────────────────────────────────┐
    │  Reward Signal (Fully Dense)                                     │
    │                                                                  │
    │  Per-step:                                                       │
    │    Progress toward goal:  progress_scale * (d_{t-1} - d_t)      │
    │    Collision:             collision_penalty per collision         │
    │    Early-STOP penalty:    early_stop_penalty (within min_steps)  │
    │                                                                  │
    │  Terminal (continuous, not binary):                              │
    │    Success:  success_reward × SPL  (efficiency-weighted)         │
    │    Failure:  failure_reward + (success_reward - failure_reward)  │
    │              × norm_progress                                     │
    │              where norm_progress = 1 - final_dtg / initial_dtg  │
    └──────────────────────────────────────────────────────────────────┘

The terminal reward is continuous — the agent receives partial credit
proportional to how close it got to the goal and how efficient its path was.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RewardConfig:
    """Configuration for the VLN reward function."""

    # ── Sparse terminal rewards (MindDrive Block 4) ──
    success_reward: float = 1.0
    """Reward when the agent reaches the goal (within success_distance)."""

    failure_reward: float = -1.0
    """Reward on crash, collision limit, or exceeding max steps."""

    step_reward: float = 0.0
    """Per-step reward for safe navigation (default: 0 = no cost)."""

    # ── Dense shaping ──
    use_progress_reward: bool = True
    """If True, add shaped progress reward: d_{t-1} - d_t toward goal."""

    progress_scale: float = 0.1
    """Scaling factor for progress-based shaping reward."""

    # ── Collision-based penalties ──
    collision_penalty: float = -0.05
    """Per-collision penalty (applied when collision detected)."""

    use_collision_penalty: bool = True
    """Whether to penalize individual collisions."""

    # ── Episode limits ──
    max_episode_steps: int = 500
    """Maximum steps before the episode is considered failed."""

    success_distance: float = 3.0
    """Distance threshold (meters) for success (must match Habitat config)."""

    # ── STOP-bias penalty ──
    early_stop_penalty: float = -0.5
    """Penalty applied when STOP is taken within min_steps_before_stop of episode start."""

    min_steps_before_stop: int = 5
    """Minimum steps required before a STOP action is taken without penalty."""

    # ── Continuous terminal reward ──
    use_dtg_terminal: bool = True
    """If True, replace binary +1/-1 terminal with continuous DTG+SPL based reward."""


class VLNRewardFunction:
    """
    Computes per-step and terminal rewards for VLN episodes.

    The reward structure matches the MindDrive diagram:
    - Safe Step → 0
    - Success (Target Reached) → +1
    - Crash / Time Limit → -1

    Usage::

        reward_fn = VLNRewardFunction(RewardConfig())

        # At the beginning of each episode:
        reward_fn.reset()

        # At each step:
        reward = reward_fn.step_reward(metrics, step, done)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._prev_distance_to_goal: Optional[float] = None
        self._initial_distance_to_goal: Optional[float] = None

    def reset(self):
        """Call at the beginning of each episode."""
        self._prev_distance_to_goal = None
        self._initial_distance_to_goal = None

    def step_reward(
        self,
        metrics: Dict,
        step: int,
        done: bool,
        action_idx: int = -1,
    ) -> float:
        """
        Compute the reward for a single step.

        Parameters
        ----------
        metrics : dict
            Habitat environment metrics. Expected keys:
            - ``distance_to_goal`` : float
            - ``success`` : bool (only meaningful at episode end)
            - ``collisions.is_collision`` (optional)
        step : int
            Current step number (0-indexed).
        done : bool
            Whether the episode has ended.
        action_idx : int
            Action index taken (0 = STOP). Used for early-stop penalty.

        Returns
        -------
        float
            The reward signal for this step.
        """
        cfg = self.config
        reward = cfg.step_reward

        distance_to_goal = metrics.get("distance_to_goal", None)

        # Capture initial DTG on the very first step for terminal normalisation.
        if self._initial_distance_to_goal is None and distance_to_goal is not None:
            self._initial_distance_to_goal = distance_to_goal

        # ── Dense progress shaping ──
        if cfg.use_progress_reward and distance_to_goal is not None:
            if self._prev_distance_to_goal is not None:
                progress = self._prev_distance_to_goal - distance_to_goal
                reward += cfg.progress_scale * progress
            self._prev_distance_to_goal = distance_to_goal
        elif distance_to_goal is not None:
            self._prev_distance_to_goal = distance_to_goal

        # ── Collision penalty ──
        if cfg.use_collision_penalty:
            collisions = metrics.get("collisions", {})
            is_collision = collisions.get("is_collision", False)
            if is_collision:
                reward += cfg.collision_penalty

        # ── Early-STOP penalty (discourages STOP-bias from IL pre-training) ──
        if (
            action_idx == 0
            and step < cfg.min_steps_before_stop
            and not metrics.get("success", False)
        ):
            reward += cfg.early_stop_penalty

        # ── Terminal reward (continuous, DTG + path-efficiency based) ──
        if done:
            success = metrics.get("success", False)

            if cfg.use_dtg_terminal:
                # Normalised progress: fraction of initial distance covered.
                # 0.0 = no progress, 1.0 = exactly at goal.
                initial_dtg = self._initial_distance_to_goal
                final_dtg = distance_to_goal if distance_to_goal is not None else (
                    0.0 if success else (initial_dtg or 0.0)
                )
                if initial_dtg and initial_dtg > 0:
                    norm_progress = max(0.0, min(1.0, 1.0 - final_dtg / initial_dtg))
                else:
                    norm_progress = 1.0 if success else 0.0

                if success:
                    # Weight by SPL (path efficiency): shorter path = higher reward.
                    # SPL = shortest_path / max(path_taken, shortest_path)
                    spl = float(metrics.get("spl", norm_progress))
                    reward += cfg.success_reward * max(spl, norm_progress)
                else:
                    # Linear interpolation between failure_reward and success_reward.
                    # No progress  → failure_reward  (-1.0)
                    # Full progress → success_reward  (+1.0)
                    reward += cfg.failure_reward + (
                        cfg.success_reward - cfg.failure_reward
                    ) * norm_progress
            else:
                # Fallback: binary terminal signal
                reward += cfg.success_reward if success else cfg.failure_reward

        return reward

# StreamVLN RL — MindDrive Framework
# Reinforcement Learning for Vision-and-Language Navigation using PPO
#
# Architecture (from MindDrive diagram):
#   1. Actor-Critic Brain (StreamVLN + ValueHead)
#   2. Low-Rank Adaptation (LoRA) & KL-Penalty
#   3. Online Rollouts (Habitat Simulation Loop)
#   4. Dense Rewards & Advantage Estimation

from .streamvln_value_head import StreamVLNWithValueHead, create_reference_model
from .reward import VLNRewardFunction, RewardConfig
from .rollout_collector import (
    RolloutCollector,
    RolloutConfig,
    trajectories_to_transition_batch,
)
from .streamvln_ppo_trainer import StreamVLNPPOTrainer, StreamVLNPPOConfig

__all__ = [
    "StreamVLNWithValueHead",
    "create_reference_model",
    "VLNRewardFunction",
    "RewardConfig",
    "RolloutCollector",
    "RolloutConfig",
    "trajectories_to_transition_batch",
    "StreamVLNPPOTrainer",
    "StreamVLNPPOConfig",
]

"""
Block Blast AI - PPO + CNN Implementation
==========================================

Reinforcement learning implementation for Block Blast using Proximal Policy
Optimization (PPO) with Convolutional Neural Networks.

Main components:
- ppo_env.py: Gymnasium-compatible environment with CNN-friendly observations
- train_ppo_cnn.py: Training script using stable-baselines3
- eval_ppo_cnn.py: Evaluation script for trained models
- blockblast/: Core game logic and piece definitions

Author: Rishabh Goenka
"""

__version__ = "1.0.0"
__author__ = "Rishabh Goenka"

from ppo_env import BlockBlastEnv, make_env

__all__ = ["BlockBlastEnv", "make_env"]


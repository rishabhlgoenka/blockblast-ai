#!/usr/bin/env python3
"""
PPO + CNN Training Script for Block Blast
==========================================

Train a PPO agent with a CNN policy on Block Blast.
Uses stable-baselines3 for PPO implementation and a custom CNN feature extractor.

The CNN processes the 4-channel observation:
    - Channel 0: Board state (8x8)
    - Channels 1-3: Available pieces (each 8x8)

Architecture:
    Input (4, 8, 8) -> CNN -> Flatten -> MLP -> Policy + Value heads

Usage:
    python train_ppo_cnn.py --timesteps 1000000
    python train_ppo_cnn.py --timesteps 5000000 --learning-rate 0.0001
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO  # Using regular PPO, not MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from ppo_env import BlockBlastEnv


class BlockBlastCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Block Blast observations.
    
    Processes 4-channel observation (board + 3 pieces) with CNN layers,
    then flattens for policy and value heads.
    
    Architecture:
        Conv2D(4->32, 3x3) -> ReLU ->
        Conv2D(32->64, 3x3) -> ReLU ->
        Conv2D(64->64, 3x3) -> ReLU ->
        Flatten -> Linear(64*2*2 -> 256) -> ReLU
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Initialize the CNN feature extractor.
        
        Args:
            observation_space: Observation space (should be (4, 8, 8))
            features_dim: Number of output features (input to policy/value heads)
        """
        super().__init__(observation_space, features_dim)
        
        # Verify observation shape
        n_input_channels = observation_space.shape[0]  # Should be 4
        
        # CNN layers
        self.cnn = nn.Sequential(
            # First conv: 4 channels -> 32 channels
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Second conv: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Third conv: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Flatten
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]
        
        # Linear layer to produce feature vector
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            observations: Batch of observations (batch_size, 4, 8, 8)
            
        Returns:
            Feature vectors (batch_size, features_dim)
        """
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)


class TrainingCallback(BaseCallback):
    """
    Custom callback to log episode statistics during training.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
    
    def _on_step(self) -> bool:
        """
        Called after each step.
        Check if episode ended and log statistics.
        """
        # Access the first environment (we use DummyVecEnv with 1 env)
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            # Episode ended
            info = self.locals.get('infos', [{}])[0]
            
            if 'episode' in info:
                # SB3 automatically adds episode stats to info
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Get game score from info
                score = info.get('score', 0)
                self.episode_scores.append(score)
                
                # Print progress every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    mean_score = np.mean(self.episode_scores[-100:])
                    mean_length = np.mean(self.episode_lengths[-100:])
                    
                    print(f"\nEpisode {len(self.episode_rewards):5d} | "
                          f"Reward: {episode_reward:7.1f} | "
                          f"Score: {score:5d} | "
                          f"Length: {episode_length:4d}")
                    print(f"  Last 100 episodes -> "
                          f"Avg Reward: {mean_reward:7.1f} | "
                          f"Avg Score: {mean_score:6.1f} | "
                          f"Avg Length: {mean_length:5.1f}")
        
        return True


class CurriculumCallback(BaseCallback):
    """
    Callback to dynamically adjust curriculum difficulty during training.
    
    Curriculum phases:
    - Phase 1 (0-200k): 60% pre-filled -> Easy line clearing discovery
    - Phase 2 (200k-500k): 40% pre-filled -> Building towards clears
    - Phase 3 (500k+): 0% pre-filled -> Full game complexity
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.current_fill_ratio = 0.6
    
    def _on_step(self) -> bool:
        """Update curriculum based on timesteps."""
        timesteps = self.num_timesteps
        
        # Determine curriculum phase
        if timesteps < 200_000:
            # Phase 1: Easy (60% pre-filled)
            new_ratio = 0.6
        elif timesteps < 500_000:
            # Phase 2: Medium (40% pre-filled)
            new_ratio = 0.4
        else:
            # Phase 3: Hard (empty board)
            new_ratio = 0.0
        
        # Update environment if ratio changed
        if new_ratio != self.current_fill_ratio:
            self.current_fill_ratio = new_ratio
            # Access the actual environment inside DummyVecEnv
            env = self.training_env.envs[0]
            env.curriculum_fill_ratio = new_ratio
            print(f"\n{'='*70}")
            print(f"CURRICULUM UPDATE at {timesteps:,} steps")
            print(f"New fill ratio: {new_ratio:.1%} -> ", end="")
            if new_ratio == 0.6:
                print("PHASE 1: Easy (discovering line clears)")
            elif new_ratio == 0.4:
                print("PHASE 2: Medium (building towards clears)")
            else:
                print("PHASE 3: Hard (full game complexity)")
            print(f"{'='*70}\n")
        
        return True


def train_ppo(
    timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 512,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    save_path: str = "models/ppo_cnn_blockblast.zip",
    checkpoint_freq: int = 100_000,
    log_dir: str = "ppo_logs"
):
    """
    Train PPO agent with CNN policy on Block Blast.
    
    Args:
        timesteps: Total timesteps to train
        learning_rate: Learning rate for optimizer
        n_steps: Steps to collect before each update
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        seed: Random seed
        save_path: Path to save final model
        checkpoint_freq: Frequency to save checkpoints
        log_dir: Directory for tensorboard logs
    """
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models/ppo_checkpoints", exist_ok=True)
    
    print("="*70)
    print("Block Blast PPO + CNN Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {timesteps:,}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE Lambda: {gae_lambda}")
    print(f"  Clip range: {clip_range}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Value coef: {vf_coef}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Seed: {seed}")
    print(f"  Save path: {save_path}")
    print(f"  Log directory: {log_dir}")
    print(f"\nStarting training...\n")
    
    # Create environment with MIXED curriculum learning
    # Cycles through: 60%, empty, 40%, 10%, empty, 60%, empty each episode
    # This prevents overfitting to one difficulty level
    def make_env():
        return BlockBlastEnv(seed=seed, use_mixed_curriculum=True)
    
    # Wrap in DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([make_env])
    
    # Policy kwargs with custom CNN
    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128])  # Separate heads after CNN
    )
    
    # Create regular PPO model (no action masking - simpler and more robust)
    # Environment handles invalid actions by forcing valid ones
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed
    )
    
    # Setup callbacks
    training_callback = TrainingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="models/ppo_checkpoints",
        name_prefix="ppo_cnn",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # Train
    print("\n" + "="*70)
    print("MIXED CURRICULUM LEARNING ENABLED")
    print("  Each episode cycles through difficulty levels:")
    print("    Episode 1: 60% pre-filled")
    print("    Episode 2: Empty board")
    print("    Episode 3: 40% pre-filled")
    print("    Episode 4: 10% pre-filled")
    print("    Episode 5: Empty board")
    print("    Episode 6: 60% pre-filled")
    print("    Episode 7: Empty board")
    print("    ... pattern repeats ...")
    print("  This prevents overfitting to one difficulty level!")
    print("="*70)
    print("Training started...")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=timesteps,
        callback=[training_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(save_path)
    print(f"\n✓ Model saved to {save_path}")
    
    # Print training summary
    if len(training_callback.episode_rewards) > 0:
        print("\n" + "="*70)
        print("Training Summary")
        print("="*70)
        print(f"Episodes completed: {len(training_callback.episode_rewards)}")
        print(f"Mean reward (last 100): {np.mean(training_callback.episode_rewards[-100:]):.2f}")
        print(f"Mean score (last 100): {np.mean(training_callback.episode_scores[-100:]):.2f}")
        print(f"Best score: {max(training_callback.episode_scores)}")
        print("="*70)
    
    env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train PPO + CNN agent for Block Blast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1M timesteps, ~1-2 hours)
  python train_ppo_cnn.py --timesteps 1000000
  
  # Standard training (5M timesteps)
  python train_ppo_cnn.py --timesteps 5000000
  
  # Long training with custom learning rate
  python train_ppo_cnn.py --timesteps 10000000 --learning-rate 0.0001
        """
    )
    
    # Training parameters
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1_000_000,
        help='Total timesteps to train (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2048,
        help='Steps to collect before each update (default: 2048)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Minibatch size (default: 512)'
    )
    
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=10,
        help='Number of epochs per update (default: 10)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (default: 0.99)'
    )
    
    parser.add_argument(
        '--clip-range',
        type=float,
        default=0.2,
        help='PPO clip range (default: 0.2)'
    )
    
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.01,
        help='Entropy coefficient (default: 0.01)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--save-path',
        type=str,
        default='models/ppo_cnn_blockblast.zip',
        help='Path to save trained model (default: models/ppo_cnn_blockblast.zip)'
    )
    
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=100_000,
        help='Save checkpoint every N timesteps (default: 100,000)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='ppo_logs',
        help='Directory for tensorboard logs (default: ppo_logs)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {args.timesteps}")
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {args.n_steps}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {args.n_epochs}")
    if not 0 <= args.gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {args.gamma}")
    if args.clip_range <= 0:
        raise ValueError(f"clip_range must be positive, got {args.clip_range}")
    if args.ent_coef < 0:
        raise ValueError(f"ent_coef must be non-negative, got {args.ent_coef}")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train
    try:
        train_ppo(
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            seed=args.seed,
            save_path=args.save_path,
            checkpoint_freq=args.checkpoint_freq,
            log_dir=args.log_dir
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()


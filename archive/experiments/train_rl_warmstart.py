#!/usr/bin/env python3
"""
RL Training with Warmstart from RLHF v2

Goal: Push from 65 avg → 100 avg using RL exploration
Strategy: Start from trained RLHF v2 model, not random init
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from ppo_env import BlockBlastEnv
from train_ppo_cnn import BlockBlastCNN  # Custom CNN for our small board
import numpy as np


class ProgressCallback(BaseCallback):
    """Log progress during training."""
    
    def __init__(self, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Quick evaluation
            scores = []
            for _ in range(10):
                obs = self.training_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, done, info = self.training_env.step(action)
                    if done:
                        # Extract score from first env
                        if hasattr(self.training_env, 'get_attr'):
                            score = self.training_env.get_attr('state')[0].score
                        else:
                            score = self.training_env.envs[0].state.score
                        scores.append(score)
            
            mean_score = np.mean(scores)
            print(f"\n{'='*70}")
            print(f"Steps: {self.n_calls:,} | Avg Score: {mean_score:.1f} | Best: {max(scores)}")
            print(f"{'='*70}\n")
            
            if mean_score > self.best_mean_reward:
                self.best_mean_reward = mean_score
                # Save best model
                self.model.save("models/rl_warmstart_best.zip")
                print(f"🎉 New best avg score: {mean_score:.1f} - Model saved!")
        
        return True


def train_with_warmstart(
    warmstart_model: str = "models/human_imitation_rlhf_v2.zip",
    total_timesteps: int = 5_000_000,
    save_freq: int = 100_000,
    learning_rate: float = 0.00005,  # Lower LR for fine-tuning
):
    """
    Train PPO starting from RLHF v2 model.
    
    Args:
        warmstart_model: Path to starting model
        total_timesteps: Total training steps (5-10M recommended)
        save_freq: How often to save checkpoints
        learning_rate: Learning rate (lower for warmstart)
    """
    
    print("\n" + "="*70)
    print("RL WARMSTART TRAINING - Strategy 2")
    print("="*70)
    print(f"Starting model: {warmstart_model}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Target: 100+ average score")
    print("="*70 + "\n")
    
    # Create environment with mixed curriculum
    def make_env():
        return BlockBlastEnv(seed=42, use_mixed_curriculum=True)
    
    env = DummyVecEnv([make_env])
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/rl_warmstart", exist_ok=True)
    
    # Create new model with proper training parameters and custom CNN
    print("Creating new PPO model with warmstart from RLHF v2...")
    
    # Use custom CNN architecture (required for our small 8x8 board)
    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128])
    )
    
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log="logs/rl_warmstart",
        policy_kwargs=policy_kwargs,
        seed=42
    )
    
    # Load the trained policy from RLHF v2
    print(f"Loading trained policy from: {warmstart_model}")
    trained_model = PPO.load(warmstart_model)
    model.policy.load_state_dict(trained_model.policy.state_dict())
    
    print(f"✓ Policy loaded successfully from RLHF v2")
    print(f"Starting from: ~65 avg score")
    print(f"Learning rate: {learning_rate}")
    print(f"Exploration bonus: {model.ent_coef}")
    print("")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/checkpoints",
        name_prefix="rl_warmstart"
    )
    
    progress_callback = ProgressCallback(eval_freq=10000)
    
    # Train!
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print("This will take 12-24 hours for 5M steps.")
    print("The model will save checkpoints every 100k steps.")
    print("Monitor progress in logs/rl_warmstart/")
    print("")
    print("Expected progression:")
    print("  100k steps:  ~68 avg")
    print("  500k steps:  ~75 avg")
    print("  1M steps:    ~82 avg")
    print("  3M steps:    ~90 avg")
    print("  5M steps:    ~95-100 avg (target!)")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback],
        log_interval=10,
        tb_log_name="rl_warmstart",
        reset_num_timesteps=True  # Start fresh timestep counter
    )
    
    # Save final model
    final_path = "models/rl_warmstart_final.zip"
    model.save(final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final model saved: {final_path}")
    print(f"Best model saved: models/rl_warmstart_best.zip")
    print("")
    print("Evaluate with:")
    print(f"  python eval_ppo_cnn.py --model {final_path} --episodes 100")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Warmstart Training")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Total timesteps (default: 5M)")
    parser.add_argument("--model", type=str, default="models/human_imitation_rlhf_v2.zip",
                        help="Warmstart model path")
    parser.add_argument("--lr", type=float, default=0.00005,
                        help="Learning rate (default: 5e-5)")
    
    args = parser.parse_args()
    
    train_with_warmstart(
        warmstart_model=args.model,
        total_timesteps=args.timesteps,
        learning_rate=args.lr
    )


#!/usr/bin/env python3
"""
Curriculum Strategy Testing Script
===================================

Tests multiple curriculum strategies with short training runs (50-100k steps)
to find which approach works best before committing to long training.

Each strategy is trained, evaluated, and scored. Best strategy is identified.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from ppo_env import BlockBlastEnv
from train_ppo_cnn import BlockBlastCNN


# Define different curriculum strategies to test
CURRICULUM_STRATEGIES = {
    "no_curriculum": {
        "name": "No Curriculum (Empty Boards Only)",
        "use_mixed": False,
        "fixed_ratio": 0.0,
        "pattern": None,
        "description": "Always train on empty boards - baseline"
    },
    
    "heavy_start": {
        "name": "Heavy Pre-fill Start",
        "use_mixed": True,
        "pattern": [0.6, 0.6, 0.4, 0.4, 0.2, 0.0, 0.0],
        "description": "Start easy with lots of pre-filled, gradually transition"
    },
    
    "mixed_balanced": {
        "name": "Mixed Balanced (Current)",
        "use_mixed": True,
        "pattern": [0.6, 0.0, 0.4, 0.1, 0.0, 0.6, 0.0],
        "description": "Mix all difficulty levels - 43% empty boards"
    },
    
    "mixed_empty_heavy": {
        "name": "Mixed Empty-Heavy",
        "use_mixed": True,
        "pattern": [0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.1],
        "description": "More empty board practice - 57% empty boards"
    },
    
    "gradual_descent": {
        "name": "Gradual Difficulty Descent",
        "use_mixed": True,
        "pattern": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        "description": "Smooth difficulty curve from 60% to empty"
    },
    
    "alternating_simple": {
        "name": "Simple Alternating",
        "use_mixed": True,
        "pattern": [0.6, 0.0, 0.6, 0.0],
        "description": "Simple pattern: hard, easy, hard, easy"
    },
}


def create_env_for_strategy(strategy_config: Dict, seed: int = 42):
    """Create environment based on strategy configuration."""
    if strategy_config["use_mixed"]:
        # Create custom environment with specific pattern
        class CustomEnv(BlockBlastEnv):
            def __init__(self, seed=None):
                super().__init__(seed=seed, use_mixed_curriculum=False)
                self.mixed_curriculum_pattern = strategy_config["pattern"]
                self.use_mixed_curriculum = True
                self.episode_count = 0
        
        return CustomEnv(seed=seed)
    else:
        # Fixed ratio or no curriculum
        return BlockBlastEnv(
            seed=seed,
            curriculum_fill_ratio=strategy_config.get("fixed_ratio", 0.0)
        )


def train_strategy(
    strategy_name: str,
    strategy_config: Dict,
    timesteps: int = 50000,
    seed: int = 42
) -> str:
    """
    Train a model with the given curriculum strategy.
    
    Returns:
        Path to saved model
    """
    print(f"\n{'='*70}")
    print(f"Training: {strategy_config['name']}")
    print(f"Description: {strategy_config['description']}")
    print(f"Timesteps: {timesteps:,}")
    print(f"{'='*70}\n")
    
    # Create environment
    def make_env():
        return create_env_for_strategy(strategy_config, seed=seed)
    
    env = DummyVecEnv([make_env])
    
    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128])
    )
    
    # Create model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    train_time = time.time() - start_time
    
    # Save model
    os.makedirs("models/strategy_tests", exist_ok=True)
    save_path = f"models/strategy_tests/{strategy_name}.zip"
    model.save(save_path)
    
    print(f"\n✓ Training completed in {train_time:.1f} seconds")
    print(f"✓ Model saved to {save_path}\n")
    
    return save_path


def evaluate_strategy(model_path: str, num_episodes: int = 30) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating {model_path}...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment (empty boards for evaluation)
    env = BlockBlastEnv(seed=42)
    
    # Run episodes
    scores = []
    lines_cleared = []
    episode_lengths = []
    rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        scores.append(info.get('score', 0))
        lines_cleared.append(info.get('lines_cleared_total', 0))
        episode_lengths.append(episode_length)
        rewards.append(episode_reward)
    
    # Calculate statistics
    results = {
        'avg_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
        'avg_lines': float(np.mean(lines_cleared)),
        'total_lines': int(np.sum(lines_cleared)),
        'avg_length': float(np.mean(episode_lengths)),
        'avg_reward': float(np.mean(rewards)),
        'max_reward': float(np.max(rewards)),
    }
    
    print(f"  Avg Score: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Max Score: {results['max_score']}")
    print(f"  Avg Lines: {results['avg_lines']:.2f}")
    print(f"  Avg Reward: {results['avg_reward']:.2f}\n")
    
    return results


def rank_strategies(all_results: Dict[str, Dict]) -> List[tuple]:
    """
    Rank strategies based on composite score.
    
    Composite score = avg_score * 0.5 + avg_lines * 10 + avg_reward * 0.001
    """
    rankings = []
    
    for strategy_name, results in all_results.items():
        # Composite score emphasizing both game score and lines cleared
        composite = (
            results['avg_score'] * 0.5 +      # Game score
            results['avg_lines'] * 10.0 +     # Lines cleared (highly weighted)
            results['avg_reward'] * 0.001     # RL rewards
        )
        
        rankings.append((strategy_name, composite, results))
    
    # Sort by composite score (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings


def main():
    """Main testing workflow."""
    print("="*70)
    print("CURRICULUM STRATEGY TESTING")
    print("="*70)
    print(f"\nTesting {len(CURRICULUM_STRATEGIES)} different curriculum strategies")
    print("Each strategy will be trained for 50k timesteps and evaluated")
    print("\n" + "="*70 + "\n")
    
    timesteps_per_test = 50000
    eval_episodes = 30
    
    all_results = {}
    
    # Test each strategy
    for strategy_name, strategy_config in CURRICULUM_STRATEGIES.items():
        try:
            # Train
            model_path = train_strategy(
                strategy_name,
                strategy_config,
                timesteps=timesteps_per_test
            )
            
            # Evaluate
            results = evaluate_strategy(model_path, num_episodes=eval_episodes)
            
            # Store results with metadata
            all_results[strategy_name] = {
                'strategy_config': strategy_config,
                'model_path': model_path,
                'results': results
            }
            
        except Exception as e:
            print(f"ERROR testing {strategy_name}: {e}\n")
            continue
    
    # Rank strategies
    print("\n" + "="*70)
    print("FINAL RANKINGS")
    print("="*70 + "\n")
    
    rankings = rank_strategies({k: v['results'] for k, v in all_results.items()})
    
    for rank, (strategy_name, composite_score, results) in enumerate(rankings, 1):
        config = all_results[strategy_name]['strategy_config']
        
        print(f"{rank}. {config['name']}")
        print(f"   Composite Score: {composite_score:.2f}")
        print(f"   Avg Score: {results['avg_score']:.2f}")
        print(f"   Avg Lines: {results['avg_lines']:.2f}")
        print(f"   Max Score: {results['max_score']}")
        print(f"   Description: {config['description']}")
        print()
    
    # Save results to JSON
    output_file = "models/strategy_tests/test_results.json"
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {
            'rankings': [
                {
                    'rank': rank,
                    'strategy': strategy_name,
                    'composite_score': composite_score,
                    'config': all_results[strategy_name]['strategy_config'],
                    'results': results
                }
                for rank, (strategy_name, composite_score, results) in enumerate(rankings, 1)
            ],
            'test_parameters': {
                'timesteps_per_test': timesteps_per_test,
                'eval_episodes': eval_episodes
            }
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Full results saved to {output_file}")
    
    # Print recommendation
    best_strategy, best_score, best_results = rankings[0]
    best_config = all_results[best_strategy]['strategy_config']
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\n🏆 Best Strategy: {best_config['name']}")
    print(f"   {best_config['description']}")
    print(f"\n   Performance:")
    print(f"   - Average Score: {best_results['avg_score']:.2f}")
    print(f"   - Average Lines: {best_results['avg_lines']:.2f}")
    print(f"   - Max Score: {best_results['max_score']}")
    print(f"\n   Use this strategy for full 1M+ training!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
PPO + CNN Evaluation Script for Block Blast
============================================

Evaluate a trained PPO agent with CNN policy.
Loads a saved model and runs evaluation episodes to compute statistics.

Usage:
    python eval_ppo_cnn.py --episodes 200
    python eval_ppo_cnn.py --model-path models/ppo_cnn_blockblast.zip --episodes 100
    python eval_ppo_cnn.py --episodes 500 --render
"""

import argparse
import os
import json
from typing import List, Dict, Any

import numpy as np
from stable_baselines3 import PPO  # Using regular PPO

from ppo_env import BlockBlastEnv
from train_ppo_cnn import BlockBlastCNN  # Import custom CNN for model loading


def evaluate_model(
    model_path: str,
    num_episodes: int = 200,
    seed: int = 42,
    render: bool = False,
    save_results: bool = True,
    results_path: str = "results/ppo_eval.json"
) -> Dict[str, Any]:
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to saved model (.zip file)
        num_episodes: Number of episodes to evaluate
        seed: Random seed for evaluation
        render: Whether to render episodes (human mode)
        save_results: Whether to save results to file
        results_path: Path to save results JSON
        
    Returns:
        Dictionary with evaluation statistics
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print("="*70)
    print("Block Blast PPO + CNN Evaluation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model path: {model_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Seed: {seed}")
    print(f"  Render: {render}")
    print(f"\nLoading model...")
    
    # Load model
    model = PPO.load(model_path)
    
    print("✓ Model loaded successfully")
    print(f"\nRunning evaluation...\n")
    
    # Create environment
    env = BlockBlastEnv(seed=seed, render_mode='human' if render else None)
    
    # Statistics
    episode_scores = []
    episode_rewards = []
    episode_lengths = []
    episode_lines_cleared = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        lines_cleared_episode = 0
        
        while not done:
            # Use deterministic policy for evaluation
            # No action masking - environment handles invalid actions
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if info.get('valid', False):
                lines_cleared_episode += info.get('lines_cleared', 0)
            
            # Render if requested
            if render:
                env.render()
        
        # Get final score
        score = info.get('score', 0)
        
        # Store statistics
        episode_scores.append(score)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_lines_cleared.append(lines_cleared_episode)
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode + 1:4d}/{num_episodes} | "
                  f"Score: {score:5d} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Length: {episode_length:4d} | "
                  f"Lines: {lines_cleared_episode:3d}")
    
    env.close()
    
    # Compute statistics
    scores_array = np.array(episode_scores)
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    lines_array = np.array(episode_lines_cleared)
    
    results = {
        'model_path': model_path,
        'num_episodes': num_episodes,
        'seed': seed,
        'statistics': {
            'score': {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'min': int(np.min(scores_array)),
                'max': int(np.max(scores_array)),
                'median': float(np.median(scores_array)),
                'q25': float(np.percentile(scores_array, 25)),
                'q75': float(np.percentile(scores_array, 75))
            },
            'reward': {
                'mean': float(np.mean(rewards_array)),
                'std': float(np.std(rewards_array)),
                'min': float(np.min(rewards_array)),
                'max': float(np.max(rewards_array))
            },
            'length': {
                'mean': float(np.mean(lengths_array)),
                'std': float(np.std(lengths_array)),
                'min': int(np.min(lengths_array)),
                'max': int(np.max(lengths_array))
            },
            'lines_cleared': {
                'mean': float(np.mean(lines_array)),
                'std': float(np.std(lines_array)),
                'min': int(np.min(lines_array)),
                'max': int(np.max(lines_array)),
                'total': int(np.sum(lines_array))
            }
        },
        'episodes': {
            'scores': [int(s) for s in episode_scores],
            'rewards': [float(r) for r in episode_rewards],
            'lengths': [int(l) for l in episode_lengths],
            'lines_cleared': [int(l) for l in episode_lines_cleared]
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"\nScore Statistics:")
    print(f"  Mean:   {results['statistics']['score']['mean']:8.2f}")
    print(f"  Std:    {results['statistics']['score']['std']:8.2f}")
    print(f"  Min:    {results['statistics']['score']['min']:8d}")
    print(f"  Max:    {results['statistics']['score']['max']:8d}")
    print(f"  Median: {results['statistics']['score']['median']:8.2f}")
    print(f"  Q25:    {results['statistics']['score']['q25']:8.2f}")
    print(f"  Q75:    {results['statistics']['score']['q75']:8.2f}")
    
    print(f"\nReward Statistics:")
    print(f"  Mean:   {results['statistics']['reward']['mean']:8.2f}")
    print(f"  Std:    {results['statistics']['reward']['std']:8.2f}")
    print(f"  Min:    {results['statistics']['reward']['min']:8.2f}")
    print(f"  Max:    {results['statistics']['reward']['max']:8.2f}")
    
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean:   {results['statistics']['length']['mean']:8.2f}")
    print(f"  Std:    {results['statistics']['length']['std']:8.2f}")
    print(f"  Min:    {results['statistics']['length']['min']:8d}")
    print(f"  Max:    {results['statistics']['length']['max']:8d}")
    
    print(f"\nLines Cleared:")
    print(f"  Mean:   {results['statistics']['lines_cleared']['mean']:8.2f}")
    print(f"  Total:  {results['statistics']['lines_cleared']['total']:8d}")
    
    # Save results
    if save_results:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
    
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained PPO + CNN agent for Block Blast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation (200 episodes)
  python eval_ppo_cnn.py --episodes 200
  
  # Evaluate specific model
  python eval_ppo_cnn.py --model-path models/ppo_checkpoints/ppo_cnn_500000_steps.zip
  
  # Evaluation with rendering (watch agent play)
  python eval_ppo_cnn.py --episodes 10 --render
  
  # Long evaluation for better statistics
  python eval_ppo_cnn.py --episodes 1000
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/ppo_cnn_blockblast.zip',
        help='Path to trained model (default: models/ppo_cnn_blockblast.zip)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=200,
        help='Number of evaluation episodes (default: 200)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes (watch agent play)'
    )
    
    parser.add_argument(
        '--results-path',
        type=str,
        default='results/ppo_eval.json',
        help='Path to save results (default: results/ppo_eval.json)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.episodes <= 0:
        raise ValueError(f"episodes must be positive, got {args.episodes}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Run evaluation
    try:
        evaluate_model(
            model_path=args.model_path,
            num_episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            save_results=not args.no_save,
            results_path=args.results_path
        )
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()


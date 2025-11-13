#!/usr/bin/env python3
"""
Quick Model Evaluation Script
==============================

Evaluates a trained model on Block Blast.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from ppo_env import BlockBlastEnv


def evaluate_model(model_path: str, num_episodes: int = 100, render: bool = False):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to model
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = BlockBlastEnv(render_mode='human' if render else None)
    
    scores = []
    lengths = []
    
    print(f"Evaluating for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if render:
                env.render()
        
        score = info.get('score', 0)
        scores.append(score)
        lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Score: {score:5d} | "
                  f"Avg: {np.mean(scores):.1f}")
    
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Mean score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Median score: {np.median(scores):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Worst score: {min(scores)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print("="*70)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate Block Blast model')
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.episodes, args.render)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""Comprehensive test of all models on updated game."""

import numpy as np
from stable_baselines3 import PPO
from ppo_env import BlockBlastEnv

def test_model(model_path, name, episodes=100):
    """Test a model and return statistics."""
    print(f"\nTesting: {name}")
    print("=" * 70)
    
    try:
        model = PPO.load(model_path)
        env = BlockBlastEnv()
        
        scores = []
        lengths = []
        
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            score = info.get('score', 0)
            scores.append(score)
            lengths.append(steps)
            
            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep+1}/{episodes} | Score: {score:5d} | Avg: {np.mean(scores):.1f}")
        
        print(f"\n  {'='*66}")
        print(f"  Mean score:   {np.mean(scores):7.2f} ± {np.std(scores):.2f}")
        print(f"  Median score: {np.median(scores):7.2f}")
        print(f"  Best score:   {max(scores):7d}")
        print(f"  Worst score:  {min(scores):7d}")
        print(f"  Mean length:  {np.mean(lengths):7.2f}")
        print(f"  {'='*66}")
        
        return {
            'name': name,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'best': max(scores),
            'worst': min(scores),
            'length': np.mean(lengths)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("=" * 70)
    print("COMPREHENSIVE MODEL TESTING")
    print("Game Version: Weighted Pieces + Board Validation")
    print("=" * 70)
    
    models = [
        ('models/ppo_cnn_blockblast.zip', 'Baseline (3M PPO)'),
        ('models/rl_warmstart_best.zip', 'RL Warmstart Best'),
        ('models/ppo_expert_trained.zip', 'Expert-Trained'),
    ]
    
    results = []
    for path, name in models:
        result = test_model(path, name, episodes=100)
        if result:
            results.append(result)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'Mean':>10} {'Median':>10} {'Best':>10} {'Length':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['mean']:>10.2f} {r['median']:>10.2f} {r['best']:>10d} {r['length']:>10.2f}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['mean'])
        print("\n" + "=" * 70)
        print(f"🏆 WINNER: {best['name']} with {best['mean']:.2f} mean score")
        print("=" * 70)

if __name__ == '__main__':
    main()

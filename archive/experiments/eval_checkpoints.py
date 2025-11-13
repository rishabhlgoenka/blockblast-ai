#!/usr/bin/env python3
"""
Evaluate all training checkpoints to track progress.
"""

import os
import glob
from stable_baselines3 import PPO
from ppo_env import BlockBlastEnv
import numpy as np


def evaluate_checkpoint(model_path: str, episodes: int = 50):
    """Evaluate a single checkpoint."""
    env = BlockBlastEnv()
    model = PPO.load(model_path)
    
    scores = []
    lines = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        scores.append(env.state.score)
        lines.append(env.state.lines_cleared_total)
    
    return {
        'avg_score': np.mean(scores),
        'max_score': np.max(scores),
        'avg_lines': np.mean(lines),
        'std_score': np.std(scores)
    }


def main():
    """Evaluate all checkpoints."""
    
    print("\n" + "="*70)
    print("EVALUATING ALL CHECKPOINTS")
    print("="*70)
    
    # Find all checkpoints
    checkpoints = sorted(glob.glob("models/checkpoints/rl_warmstart_*.zip"))
    
    if not checkpoints:
        print("No checkpoints found in models/checkpoints/")
        return
    
    print(f"Found {len(checkpoints)} checkpoints\n")
    
    # Also evaluate baseline
    baseline_path = "models/human_imitation_rlhf_v2.zip"
    if os.path.exists(baseline_path):
        print("Evaluating baseline (RLHF v2)...")
        baseline_results = evaluate_checkpoint(baseline_path, episodes=50)
        print(f"  Baseline: {baseline_results['avg_score']:.1f} avg, {baseline_results['max_score']} max\n")
    
    # Evaluate each checkpoint
    results = []
    
    for i, checkpoint in enumerate(checkpoints):
        # Extract step number from filename
        filename = os.path.basename(checkpoint)
        step_str = filename.split('_')[-2]  # e.g., "100000" from "rl_warmstart_100000_steps.zip"
        steps = int(step_str)
        
        print(f"Evaluating checkpoint {i+1}/{len(checkpoints)}: {steps:,} steps...")
        result = evaluate_checkpoint(checkpoint, episodes=50)
        result['steps'] = steps
        result['path'] = checkpoint
        results.append(result)
        
        print(f"  {steps:7,} steps: {result['avg_score']:5.1f} avg, {result['max_score']:3} max, {result['avg_lines']:.2f} lines")
    
    # Summary
    print("\n" + "="*70)
    print("PROGRESS SUMMARY")
    print("="*70)
    print(f"{'Steps':>10} | {'Avg Score':>9} | {'Max Score':>9} | {'Lines/Game':>11} | {'Improvement':>12}")
    print("-"*70)
    
    if os.path.exists(baseline_path):
        baseline_avg = baseline_results['avg_score']
        print(f"{'Baseline':>10} | {baseline_avg:9.1f} | {baseline_results['max_score']:9} | {baseline_results['avg_lines']:11.2f} | {'---':>12}")
    else:
        baseline_avg = 65.1  # Known v2 avg
    
    for result in results:
        improvement = result['avg_score'] - baseline_avg
        improvement_pct = (improvement / baseline_avg) * 100
        print(f"{result['steps']:10,} | {result['avg_score']:9.1f} | {result['max_score']:9} | {result['avg_lines']:11.2f} | {improvement:+6.1f} ({improvement_pct:+.1f}%)")
    
    # Best checkpoint
    best = max(results, key=lambda x: x['avg_score'])
    print("\n" + "="*70)
    print("🏆 BEST CHECKPOINT")
    print("="*70)
    print(f"Steps: {best['steps']:,}")
    print(f"Avg Score: {best['avg_score']:.1f} (+{best['avg_score'] - baseline_avg:.1f} from baseline)")
    print(f"Max Score: {best['max_score']}")
    print(f"Path: {best['path']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


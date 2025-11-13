#!/usr/bin/env python3
"""Verify no impossible situations and comprehensive testing."""

import numpy as np
from stable_baselines3 import PPO
from ppo_env import BlockBlastEnv
from blockblast.core import BlockBlastGame

def test_model_detailed(model_path, name, episodes=100):
    """Test a model with detailed tracking."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    model = PPO.load(model_path)
    env = BlockBlastEnv()
    
    scores = []
    lengths = []
    impossible_count = 0
    valid_moves_per_turn = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        episode_valid_moves = []
        
        while not done:
            # Check for impossible situations
            state = env.state
            valid_actions = env.game.get_valid_actions(state)
            
            if len(valid_actions) == 0:
                impossible_count += 1
                break
            
            episode_valid_moves.append(len(valid_actions))
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        score = info.get('score', 0)
        scores.append(score)
        lengths.append(steps)
        if len(episode_valid_moves) > 0:
            valid_moves_per_turn.extend(episode_valid_moves)
        
        if (ep + 1) % 25 == 0:
            print(f"Episode {ep+1:3d}/{episodes} | Score: {score:5d} | Avg: {np.mean(scores):6.2f} | Impossible: {impossible_count}")
    
    print(f"\n{'─'*70}")
    print(f"RESULTS:")
    print(f"  Mean Score:        {np.mean(scores):7.2f} ± {np.std(scores):.2f}")
    print(f"  Median Score:      {np.median(scores):7.2f}")
    print(f"  Best Score:        {max(scores):7d}")
    print(f"  Worst Score:       {min(scores):7d}")
    print(f"  Mean Length:       {np.mean(lengths):7.2f}")
    print(f"  Impossible Games:  {impossible_count} / {episodes} ({100*impossible_count/episodes:.1f}%)")
    if valid_moves_per_turn:
        print(f"  Avg Valid Moves:   {np.mean(valid_moves_per_turn):7.2f}")
    print(f"{'─'*70}")
    
    return {
        'name': name,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'median': np.median(scores),
        'best': max(scores),
        'worst': min(scores),
        'length': np.mean(lengths),
        'impossible': impossible_count,
        'scores': scores
    }

def test_random_play(episodes=100):
    """Test random placement."""
    print(f"\n{'='*70}")
    print(f"Testing: Random Play (Baseline)")
    print(f"{'='*70}")
    
    game = BlockBlastGame()
    scores = []
    lengths = []
    impossible_count = 0
    
    for ep in range(episodes):
        state = game.new_game()
        steps = 0
        
        while not state.done:
            valid_actions = game.get_valid_actions(state)
            
            if len(valid_actions) == 0:
                impossible_count += 1
                break
            
            action = valid_actions[np.random.randint(len(valid_actions))]
            state, reward, done, info = game.apply_action(state, action)
            steps += 1
        
        scores.append(state.score)
        lengths.append(steps)
        
        if (ep + 1) % 25 == 0:
            print(f"Episode {ep+1:3d}/{episodes} | Score: {state.score:5d} | Avg: {np.mean(scores):6.2f} | Impossible: {impossible_count}")
    
    print(f"\n{'─'*70}")
    print(f"RESULTS:")
    print(f"  Mean Score:        {np.mean(scores):7.2f} ± {np.std(scores):.2f}")
    print(f"  Median Score:      {np.median(scores):7.2f}")
    print(f"  Best Score:        {max(scores):7d}")
    print(f"  Worst Score:       {min(scores):7d}")
    print(f"  Mean Length:       {np.mean(lengths):7.2f}")
    print(f"  Impossible Games:  {impossible_count} / {episodes} ({100*impossible_count/episodes:.1f}%)")
    print(f"{'─'*70}")
    
    return {
        'name': 'Random Play',
        'mean': np.mean(scores),
        'std': np.std(scores),
        'median': np.median(scores),
        'best': max(scores),
        'worst': min(scores),
        'length': np.mean(lengths),
        'impossible': impossible_count,
        'scores': scores
    }

def main():
    print("="*70)
    print("COMPREHENSIVE VERIFICATION TEST")
    print("Game: Weighted Pieces + Board Validation")
    print("Episodes: 100 per model")
    print("="*70)
    
    models = [
        ('models/ppo_cnn_blockblast.zip', 'Baseline (3M PPO)'),
        ('models/rl_warmstart_best.zip', 'RL Warmstart Best'),
        ('models/ppo_expert_trained.zip', 'Expert-Trained'),
    ]
    
    results = []
    
    # Test all models
    for path, name in models:
        result = test_model_detailed(path, name, episodes=100)
        results.append(result)
    
    # Test random play
    random_result = test_random_play(episodes=100)
    results.append(random_result)
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Mean':>8} {'Median':>8} {'Best':>6} {'Worst':>6} {'Impossible':>11}")
    print(f"{'─'*70}")
    for r in results:
        print(f"{r['name']:<25} {r['mean']:>8.2f} {r['median']:>8.2f} {r['best']:>6d} {r['worst']:>6d} {r['impossible']:>6d} ({100*r['impossible']/100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("VERIFICATION:")
    total_impossible = sum(r['impossible'] for r in results)
    total_games = len(results) * 100
    print(f"  Total impossible situations: {total_impossible} / {total_games} games ({100*total_impossible/total_games:.2f}%)")
    
    if total_impossible == 0:
        print("  ✅ NO IMPOSSIBLE SITUATIONS DETECTED!")
        print("  Board validation is working perfectly!")
    else:
        print(f"  ⚠️  Found {total_impossible} impossible situations")
        print("  Board validation may need adjustment")
    
    # Best model
    best = max(results[:-1], key=lambda x: x['mean'])  # Exclude random
    print(f"\n🏆 BEST MODEL: {best['name']}")
    print(f"   Mean: {best['mean']:.2f} (vs Random: {random_result['mean']:.2f})")
    print(f"   Improvement: {100*(best['mean']/random_result['mean']-1):.1f}% over random")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

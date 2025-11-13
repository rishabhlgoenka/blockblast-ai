#!/usr/bin/env python3
"""
Manual RLHF (Reinforcement Learning from Human Feedback)

Watch the AI play and rate its moves from -10 to +10.
Use your feedback to fine-tune the model to play better!

Similar to how ChatGPT was trained with human preferences.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from ppo_env import BlockBlastEnv
from train_ppo_cnn import BlockBlastCNN
import time


class HumanFeedbackDataset(Dataset):
    """Dataset of (state, action, human_reward) tuples."""
    
    def __init__(self, feedback_data):
        """
        Args:
            feedback_data: List of (obs, action, human_score) tuples
        """
        self.observations = []
        self.actions = []
        self.scores = []
        
        for obs, action, score in feedback_data:
            self.observations.append(obs)
            self.actions.append(action)
            self.scores.append(score / 10.0)  # Normalize to [-1, 1]
        
        self.observations = np.array(self.observations, dtype=np.float32) / 255.0
        self.actions = np.array(self.actions, dtype=np.int64)
        self.scores = np.array(self.scores, dtype=np.float32)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.observations[idx]),
            torch.LongTensor([self.actions[idx]]),
            torch.FloatTensor([self.scores[idx]])
        )


def collect_human_feedback(model_path: str, num_games: int = 5):
    """
    Have the AI play games while human rates each move.
    
    Returns:
        List of (observation, action, human_score) tuples
    """
    print("\n" + "="*70)
    print("MANUAL RLHF - COLLECT HUMAN FEEDBACK")
    print("="*70)
    print("\nYou will watch the AI play and rate each move:")
    print("  -10 = Terrible move (very bad)")
    print("   -5 = Bad move")
    print("    0 = Neutral/OK move")
    print("   +5 = Good move")
    print("  +10 = Excellent move (very good)")
    print("")
    print("The AI will learn from your ratings!")
    print("="*70)
    
    input("\nPress ENTER to start...")
    
    # Load model
    env = BlockBlastEnv(render_mode='ansi')
    model = PPO.load(model_path)
    
    feedback_data = []
    
    for game_num in range(num_games):
        print("\n" + "="*70)
        print(f"GAME {game_num + 1}/{num_games}")
        print("="*70)
        
        obs, _ = env.reset()
        done = False
        move_num = 0
        
        while not done:
            move_num += 1
            
            # Show current state
            print(f"\n--- Move {move_num} ---")
            print(env.render())
            print(f"Score: {env.state.score}, Combo: {env.state.combo}x")
            
            # AI makes a move
            action, _ = model.predict(obs, deterministic=False)
            
            # Show what AI is about to do
            from blockblast.env import decode_action
            piece_idx, x, y = decode_action(action)
            print(f"\nAI chose: Piece {piece_idx} at position ({y}, {x})")
            
            # Apply action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Show result
            print("\nAfter move:")
            print(env.render())
            print(f"Score: {env.state.score}, Combo: {env.state.combo}x")
            if info.get('lines_cleared', 0) > 0:
                print(f"✓ Cleared {info['lines_cleared']} lines!")
            
            # Get human feedback
            while True:
                try:
                    rating_input = input("\nRate this move (-10 to +10, or 's' to skip game): ").strip()
                    
                    if rating_input.lower() == 's':
                        print("Skipping rest of this game...")
                        done = True
                        break
                    
                    rating = int(rating_input)
                    if -10 <= rating <= 10:
                        feedback_data.append((obs, action, rating))
                        print(f"✓ Recorded: {rating}/10")
                        break
                    else:
                        print("Please enter a number between -10 and +10")
                except ValueError:
                    print("Invalid input. Enter a number between -10 and +10, or 's' to skip")
            
            obs = next_obs
            
            if done:
                print(f"\nGame Over! Final Score: {env.state.score}")
                print(f"Lines Cleared: {env.state.lines_cleared_total}")
                break
    
    print("\n" + "="*70)
    print(f"✓ Collected {len(feedback_data)} rated moves from {num_games} games")
    print("="*70)
    
    return feedback_data


def fine_tune_with_feedback(model_path: str, feedback_data, epochs: int = 30):
    """
    Fine-tune the model using human feedback.
    
    The model learns to:
    - Maximize actions with positive scores
    - Minimize actions with negative scores
    """
    print("\n" + "="*70)
    print("FINE-TUNING WITH HUMAN FEEDBACK")
    print("="*70)
    print(f"Training data: {len(feedback_data)} rated moves")
    print(f"Epochs: {epochs}")
    print("")
    
    # Load the model's policy network
    model = PPO.load(model_path)
    policy = model.policy
    
    # Create dataset
    dataset = HumanFeedbackDataset(feedback_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=0.0001)
    
    # Training loop
    print("Training...")
    for epoch in range(epochs):
        total_loss = 0.0
        
        for obs_batch, action_batch, score_batch in dataloader:
            optimizer.zero_grad()
            
            # Get policy predictions
            features = policy.extract_features(obs_batch)
            action_logits = policy.action_net(features)
            
            # Calculate loss: we want to increase probability of high-scored actions
            # and decrease probability of low-scored actions
            log_probs = torch.log_softmax(action_logits, dim=1)
            selected_log_probs = log_probs.gather(1, action_batch)
            
            # Loss: negative log probability weighted by human score
            # High scores → want to maximize probability (minimize negative log prob)
            # Low scores → want to minimize probability (maximize negative log prob)
            loss = -(selected_log_probs * score_batch).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save fine-tuned model
    output_path = model_path.replace('.zip', '_rlhf.zip')
    model.save(output_path)
    
    print(f"\n✓ Fine-tuned model saved to: {output_path}")
    
    return output_path


def evaluate_rlhf_model(model_path: str, episodes: int = 20):
    """Evaluate the RLHF fine-tuned model."""
    print("\n" + "="*70)
    print("EVALUATING RLHF MODEL")
    print("="*70)
    
    env = BlockBlastEnv()
    model = PPO.load(model_path)
    
    scores = []
    lines_cleared = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        scores.append(env.state.score)
        lines_cleared.append(env.state.lines_cleared_total)
        
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep + 1}/{episodes}: Score={env.state.score}, Lines={env.state.lines_cleared_total}")
    
    print("\n" + "="*70)
    print("RLHF RESULTS")
    print("="*70)
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Average Lines: {np.mean(lines_cleared):.2f}")
    print("="*70)


def main():
    """Main RLHF pipeline."""
    print("\n" + "="*70)
    print("MANUAL RLHF - Train AI with YOUR Feedback!")
    print("="*70)
    print("\nThis is how ChatGPT was trained!")
    print("You rate the AI's moves, and it learns from your preferences.")
    print("")
    
    model_path = "models/human_imitation.zip"
    
    print(f"Starting model: {model_path}")
    print("")
    
    # Step 1: Collect feedback
    num_games = int(input("How many games to rate? (recommend 5-10): ").strip() or "5")
    feedback_data = collect_human_feedback(model_path, num_games)
    
    if len(feedback_data) == 0:
        print("\n❌ No feedback collected. Exiting.")
        return
    
    # Save feedback
    with open('human_feedback.pkl', 'wb') as f:
        pickle.dump(feedback_data, f)
    print(f"\n✓ Feedback saved to: human_feedback.pkl")
    
    # Step 2: Fine-tune
    print("\nStarting fine-tuning...")
    rlhf_model_path = fine_tune_with_feedback(model_path, feedback_data, epochs=30)
    
    # Step 3: Evaluate
    print("\nEvaluating improved model...")
    evaluate_rlhf_model(rlhf_model_path, episodes=20)
    
    print("\n" + "="*70)
    print("✅ RLHF COMPLETE!")
    print("="*70)
    print(f"\nOriginal model: {model_path}")
    print(f"RLHF model: {rlhf_model_path}")
    print("\nThe AI learned from your feedback!")
    print("="*70)


if __name__ == "__main__":
    main()


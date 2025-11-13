#!/usr/bin/env python3
"""
Behavioral Cloning from Human Demonstrations
=============================================

Train an agent to imitate human gameplay using supervised learning.
Then fine-tune with RL to surpass human performance.

Steps:
1. Load human demonstrations
2. Train agent with behavioral cloning (supervised learning)
3. Evaluate imitation performance
4. Fine-tune with RL
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ppo_env import BlockBlastEnv
from train_ppo_cnn import BlockBlastCNN
import os


class HumanDemonstrationDataset(Dataset):
    """PyTorch dataset for human demonstrations."""
    
    def __init__(self, demonstrations):
        """
        Args:
            demonstrations: List of (observation, action_id) tuples
        """
        self.observations = []
        self.actions = []
        
        for obs, action_id in demonstrations:
            self.observations.append(obs)
            self.actions.append(action_id)
        
        self.observations = np.array(self.observations, dtype=np.float32) / 255.0  # Normalize
        self.actions = np.array(self.actions, dtype=np.int64)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.observations[idx]), torch.LongTensor([self.actions[idx]])


def load_demonstrations(filepath: str = "human_demonstrations.pkl"):
    """Load human demonstrations from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No demonstrations found at {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both old format (list) and new format (dict with 'demonstrations' key)
    if isinstance(data, dict):
        demonstrations = data['demonstrations']
        metadata = data.get('metadata', {})
        print(f"✓ Loaded {len(demonstrations)} demonstrations from {filepath}")
        if metadata:
            print(f"  Episodes: {metadata.get('num_episodes', 'unknown')}")
    else:
        demonstrations = data
        print(f"✓ Loaded {len(demonstrations)} demonstrations from {filepath}")
    
    return demonstrations


def behavioral_cloning(demonstrations, epochs: int = 50, batch_size: int = 64):
    """
    Train agent using behavioral cloning (supervised learning).
    
    Args:
        demonstrations: List of (obs, action) pairs from human
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained PPO model (with learned policy)
    """
    print("\n" + "="*70)
    print("BEHAVIORAL CLONING - Learning from Human")
    print("="*70)
    
    # Create dataset
    dataset = HumanDemonstrationDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training set: {len(dataset)} examples")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    
    # Create PPO model (we'll train its policy network with supervised learning)
    env = DummyVecEnv([lambda: BlockBlastEnv()])
    
    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128])
    )
    
    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    # Extract policy network
    policy = model.policy
    policy.train()
    
    # Optimizer for supervised learning
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining with supervised learning...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for obs_batch, action_batch in dataloader:
            obs_batch = obs_batch
            action_batch = action_batch.squeeze()
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get policy logits
            features = policy.extract_features(obs_batch)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            action_logits = policy.action_net(latent_pi)
            
            # Compute loss
            loss = criterion(action_logits, action_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            
            # Accuracy
            predicted = torch.argmax(action_logits, dim=1)
            accuracy = (predicted == action_batch).float().mean().item()
            epoch_accuracy += accuracy
            
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    print(f"\n✓ Behavioral cloning complete!")
    print(f"  Final accuracy: {avg_accuracy:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/human_imitation.zip")
    print(f"✓ Model saved to models/human_imitation.zip")
    
    return model


def evaluate_imitation(model_path: str = "models/human_imitation.zip", num_episodes: int = 20):
    """Evaluate the imitation learning model."""
    print("\n" + "="*70)
    print("EVALUATING IMITATION MODEL")
    print("="*70)
    
    model = PPO.load(model_path)
    env = BlockBlastEnv()
    
    scores = []
    lines_cleared = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1
        
        scores.append(info.get('score', 0))
        lines_cleared.append(env.state.lines_cleared_total)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1}: Score={scores[-1]}, Lines={lines_cleared[-1]}")
    
    print("\n" + "="*70)
    print("IMITATION LEARNING RESULTS")
    print("="*70)
    print(f"Average Score:       {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score:           {np.max(scores)}")
    print(f"Average Lines:       {np.mean(lines_cleared):.2f}")
    print(f"Total Lines:         {np.sum(lines_cleared)}")
    print("="*70)
    
    return np.mean(scores), np.mean(lines_cleared)


def rl_fine_tuning(model_path: str = "models/human_imitation.zip", timesteps: int = 500000):
    """
    Fine-tune the imitation model with RL to surpass human performance.
    """
    print("\n" + "="*70)
    print("RL FINE-TUNING - Improving Beyond Human")
    print("="*70)
    print(f"Timesteps: {timesteps:,}")
    
    # Create environment
    def make_env():
        return BlockBlastEnv(seed=42)
    
    env = DummyVecEnv([make_env])
    
    # Load imitation model and set environment
    model = PPO.load(model_path, env=env)
    
    # Continue training with RL
    print("\nContinuing training with RL...")
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True
    )
    
    # Save fine-tuned model
    model.save("models/human_imitation_finetuned.zip")
    print(f"\n✓ Fine-tuned model saved to models/human_imitation_finetuned.zip")
    
    return model


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("TRAIN FROM HUMAN DEMONSTRATIONS")
    print("="*70)
    
    # Step 1: Load demonstrations
    try:
        demonstrations = load_demonstrations()
    except FileNotFoundError:
        print("\n❌ No human demonstrations found!")
        print("   Run 'python human_play.py' first to record demonstrations.")
        return
    
    if len(demonstrations) < 100:
        print(f"\n⚠️  Only {len(demonstrations)} demonstrations - recommend at least 500")
        print("   Continuing with available data...")
    else:
        print(f"\n✓ {len(demonstrations)} demonstrations - good amount of training data!")
    
    # Step 2: Behavioral cloning
    print("\n" + "="*70)
    print("PHASE 1: BEHAVIORAL CLONING")
    print("="*70)
    model = behavioral_cloning(demonstrations, epochs=50)
    
    # Step 3: Evaluate imitation
    print("\n" + "="*70)
    print("PHASE 2: EVALUATION")
    print("="*70)
    avg_score, avg_lines = evaluate_imitation()
    
    # Step 4: RL fine-tuning
    if avg_score > 30:  # If imitation is decent
        print("\n" + "="*70)
        print("PHASE 3: RL FINE-TUNING")
        print("="*70)
        
        print(f"\nImitation achieved {avg_score:.1f} avg score.")
        print("Proceeding with RL fine-tuning...")
        timesteps = 500000  # Auto-default to 500k timesteps
        
        rl_fine_tuning(timesteps=timesteps)
        
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        evaluate_imitation("models/human_imitation_finetuned.zip")
    else:
        print(f"\n⚠️  Imitation performance too low ({avg_score:.1f})")
        print("   Consider recording more demonstrations or playing better games")


if __name__ == "__main__":
    main()


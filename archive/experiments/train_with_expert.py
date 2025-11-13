#!/usr/bin/env python3
"""
Train PPO Model with Expert Demonstrations
==========================================

Uses the expert solver to generate high-quality demonstrations,
then trains the existing RL model using behavior cloning + PPO.

Process:
1. Load best existing model
2. Generate expert demonstrations using solver
3. Train using imitation learning (behavior cloning)
4. Continue with PPO for fine-tuning
"""

import argparse
import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle

from ppo_env import BlockBlastEnv
from expert_solver import ExpertSolver, GreedySolver
from blockblast.core import GameState, Action


class ExpertDataGenerator:
    """Generate expert demonstration data using the solver."""
    
    def __init__(self, solver_type: str = 'greedy', look_ahead: int = 2):
        """
        Initialize data generator.
        
        Args:
            solver_type: 'greedy' or 'full' (full is slower but better)
            look_ahead: How many moves to look ahead (only for full solver)
        """
        if solver_type == 'greedy':
            self.solver = GreedySolver()
        else:
            self.solver = ExpertSolver(max_depth=look_ahead)
        
        self.env = BlockBlastEnv()
        self.solver_type = solver_type
        self.look_ahead = look_ahead
    
    def generate_episode(self, max_steps: int = 200) -> Tuple[List[np.ndarray], List[int], int]:
        """
        Generate one expert episode.
        
        Returns:
            (observations, actions, total_score)
        """
        observations = []
        actions = []
        
        obs, info = self.env.reset()
        done = False
        steps = 0
        total_score = 0
        
        while not done and steps < max_steps:
            # Get expert action using solver
            state = self.env.state
            
            if state is None:
                break
            
            if self.solver_type == 'greedy':
                best_action = self.solver.get_best_move(state)
            else:
                best_action = self.solver.get_best_move(state, look_ahead=self.look_ahead)
            
            if best_action is None:
                # No valid moves
                break
            
            # Convert Action to environment action format
            # Env expects: piece_idx * 192 + row * 12 + col
            # We need to convert from game coordinates to env coordinates
            env_action = self._convert_action(best_action)
            
            # Store observation and action
            observations.append(obs.copy())
            actions.append(env_action)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            total_score = info.get('score', 0)
            
            steps += 1
        
        return observations, actions, total_score
    
    def _convert_action(self, action: Action) -> int:
        """
        Convert core Action to environment action index.
        
        Action in core: piece_index (0-2), row (-4 to 11), col (-4 to 11)
        Action in env: flat index from 0 to 575 (3 pieces × 12 rows × 16 cols)
        
        But env actually uses 12x12 grid for simplicity.
        """
        # Adjust coordinates from game space (-4 to 11) to env space (0 to 11)
        env_row = action.row + 4
        env_col = action.col + 4
        
        # Clamp to valid range
        env_row = max(0, min(11, env_row))
        env_col = max(0, min(11, env_col))
        
        # Calculate flat index: piece_idx * 144 + row * 12 + col
        env_action = action.piece_index * 144 + env_row * 12 + env_col
        
        return env_action
    
    def generate_dataset(
        self, 
        num_episodes: int = 1000,
        min_score_threshold: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate a dataset of expert demonstrations.
        
        Args:
            num_episodes: Number of episodes to generate
            min_score_threshold: Only keep episodes with score >= this
            
        Returns:
            (observations, actions, scores)
        """
        all_observations = []
        all_actions = []
        all_scores = []
        
        print(f"Generating {num_episodes} expert episodes...")
        print(f"Solver: {self.solver_type}, Look-ahead: {self.look_ahead}")
        print(f"Minimum score threshold: {min_score_threshold}")
        print()
        
        good_episodes = 0
        
        for episode in tqdm(range(num_episodes), desc="Generating episodes"):
            observations, actions, score = self.generate_episode()
            
            # Only keep good episodes
            if score >= min_score_threshold and len(observations) > 0:
                all_observations.extend(observations)
                all_actions.extend(actions)
                all_scores.append(score)
                good_episodes += 1
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_score = np.mean(all_scores) if all_scores else 0
                print(f"\n  Episodes {episode+1}/{num_episodes}")
                print(f"  Good episodes: {good_episodes} ({100*good_episodes/(episode+1):.1f}%)")
                print(f"  Average score: {avg_score:.1f}")
                if all_scores:
                    print(f"  Best score: {max(all_scores)}")
                    print(f"  Total demonstrations: {len(all_observations)}")
        
        print(f"\n✓ Generated {len(all_observations)} demonstrations from {good_episodes} episodes")
        print(f"  Average score: {np.mean(all_scores):.1f}")
        print(f"  Best score: {max(all_scores) if all_scores else 0}")
        
        return (
            np.array(all_observations, dtype=np.float32),
            np.array(all_actions, dtype=np.int64),
            all_scores
        )


class BehaviorCloningTrainer:
    """Train model using behavior cloning on expert demonstrations."""
    
    def __init__(self, model: PPO, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: PPO model to train
            device: Device to train on
        """
        self.model = model
        self.device = device
        self.model.policy.to(device)
    
    def train(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        batch_size: int = 256,
        epochs: int = 10,
        learning_rate: float = 1e-4
    ):
        """
        Train using behavior cloning.
        
        Args:
            observations: Expert observations
            actions: Expert actions
            batch_size: Batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        print(f"\nTraining with Behavior Cloning...")
        print(f"  Dataset size: {len(observations)} samples")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")
        print()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        
        dataset_size = len(observations)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            
            # Mini-batch training
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                
                # Forward pass
                with torch.no_grad():
                    features = self.model.policy.extract_features(batch_obs)
                
                action_logits = self.model.policy.action_net(
                    self.model.policy.mlp_extractor.forward_actor(features)
                )
                
                # Compute loss
                loss = criterion(action_logits, batch_actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(action_logits, dim=1)
                accuracy = (predictions == batch_actions).float().mean()
                epoch_accuracy += accuracy.item()
                
                num_batches += 1
            
            # Print epoch stats
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}")


def train_with_expert(
    model_path: str,
    num_episodes: int = 1000,
    solver_type: str = 'greedy',
    look_ahead: int = 2,
    min_score: int = 100,
    bc_epochs: int = 10,
    bc_lr: float = 1e-4,
    ppo_timesteps: int = 500_000,
    save_path: str = "models/ppo_expert_trained.zip",
    save_demonstrations: bool = True
):
    """
    Main training function.
    
    Args:
        model_path: Path to existing model to load
        num_episodes: Number of expert episodes to generate
        solver_type: 'greedy' or 'full'
        look_ahead: Look-ahead depth (for full solver)
        min_score: Minimum score to keep episode
        bc_epochs: Behavior cloning epochs
        bc_lr: Behavior cloning learning rate
        ppo_timesteps: Additional PPO training timesteps
        save_path: Path to save final model
        save_demonstrations: Whether to save the dataset
    """
    print("="*70)
    print("Training PPO Model with Expert Demonstrations")
    print("="*70)
    print()
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("expert_data", exist_ok=True)
    
    # Load existing model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    print()
    
    # Generate expert demonstrations
    print("Step 1: Generating Expert Demonstrations")
    print("-" * 70)
    generator = ExpertDataGenerator(solver_type=solver_type, look_ahead=look_ahead)
    observations, actions, scores = generator.generate_dataset(
        num_episodes=num_episodes,
        min_score_threshold=min_score
    )
    
    # Save demonstrations
    if save_demonstrations:
        demo_path = "expert_data/expert_demonstrations.pkl"
        print(f"\nSaving demonstrations to {demo_path}...")
        with open(demo_path, 'wb') as f:
            pickle.dump({
                'observations': observations,
                'actions': actions,
                'scores': scores,
                'solver_type': solver_type,
                'look_ahead': look_ahead
            }, f)
        print("✓ Demonstrations saved")
    
    # Behavior cloning training
    print("\n" + "="*70)
    print("Step 2: Behavior Cloning Training")
    print("-" * 70)
    trainer = BehaviorCloningTrainer(model)
    trainer.train(
        observations=observations,
        actions=actions,
        epochs=bc_epochs,
        learning_rate=bc_lr
    )
    print("✓ Behavior cloning complete")
    
    # Additional PPO training
    if ppo_timesteps > 0:
        print("\n" + "="*70)
        print("Step 3: Additional PPO Fine-tuning")
        print("-" * 70)
        print(f"Training for {ppo_timesteps:,} additional timesteps...")
        print()
        
        # Create environment
        env = DummyVecEnv([lambda: BlockBlastEnv(use_mixed_curriculum=True)])
        model.set_env(env)
        
        # Continue training
        model.learn(total_timesteps=ppo_timesteps, progress_bar=True)
        print("✓ PPO fine-tuning complete")
    
    # Save final model
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)
    print("✓ Model saved")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Final model saved to: {save_path}")
    print(f"Demonstrations used: {len(observations)}")
    print(f"Average expert score: {np.mean(scores):.1f}")
    print(f"Best expert score: {max(scores)}")


def main():
    parser = argparse.ArgumentParser(
        description='Train PPO model with expert demonstrations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/ppo_cnn_blockblast.zip',
        help='Path to existing model to load'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of expert episodes to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--solver',
        type=str,
        choices=['greedy', 'full'],
        default='greedy',
        help='Solver type: greedy (fast) or full (slow but better)'
    )
    
    parser.add_argument(
        '--look-ahead',
        type=int,
        default=2,
        help='Look-ahead depth for full solver (default: 2)'
    )
    
    parser.add_argument(
        '--min-score',
        type=int,
        default=100,
        help='Minimum score to keep episode (default: 100)'
    )
    
    parser.add_argument(
        '--bc-epochs',
        type=int,
        default=10,
        help='Behavior cloning epochs (default: 10)'
    )
    
    parser.add_argument(
        '--bc-lr',
        type=float,
        default=1e-4,
        help='Behavior cloning learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--ppo-timesteps',
        type=int,
        default=500_000,
        help='Additional PPO timesteps (default: 500,000)'
    )
    
    parser.add_argument(
        '--save-path',
        type=str,
        default='models/ppo_expert_trained.zip',
        help='Path to save final model'
    )
    
    parser.add_argument(
        '--no-save-demos',
        action='store_true',
        help='Do not save demonstration dataset'
    )
    
    args = parser.parse_args()
    
    train_with_expert(
        model_path=args.model,
        num_episodes=args.episodes,
        solver_type=args.solver,
        look_ahead=args.look_ahead,
        min_score=args.min_score,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        ppo_timesteps=args.ppo_timesteps,
        save_path=args.save_path,
        save_demonstrations=not args.no_save_demos
    )


if __name__ == '__main__':
    main()


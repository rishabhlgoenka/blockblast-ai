"""
Training script for Snake RL Agent
Trains a DQN agent to play the Snake game.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from snake_env import SnakeEnv
from dqn_agent import DQNAgent


def train(
    episodes=10000,
    max_steps=1000,
    save_freq=100,
    render=False,
    model_dir='models',
    log_freq=10
):
    """
    Train the DQN agent.
    
    Args:
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        save_freq: Frequency of saving model (in episodes)
        render: Whether to render the game during training
        model_dir: Directory to save models
        log_freq: Frequency of logging progress (in episodes)
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = SnakeEnv(render_mode=render)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        hidden_size=256,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    # Training metrics
    scores = []
    avg_scores = []
    epsilons = []
    losses = []
    
    print("=" * 60)
    print("Starting training...")
    print(f"Episodes: {episodes}")
    print(f"Model directory: {model_dir}")
    print("=" * 60)
    
    best_avg_score = -float('inf')
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        score = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Render if enabled
            if render:
                env.render(fps=50)
            
            score = info['score']
            state = next_state
            
            if done:
                break
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        
        # Record metrics
        scores.append(score)
        epsilons.append(agent.epsilon)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        
        # Calculate moving average
        window = min(100, len(scores))
        avg_score = np.mean(scores[-window:])
        avg_scores.append(avg_score)
        
        # Log progress
        if episode % log_freq == 0:
            print(f"Episode {episode}/{episodes} | "
                  f"Score: {score} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Save model periodically and when achieving best average score
        if episode % save_freq == 0:
            model_path = os.path.join(model_dir, f'dqn_snake_ep{episode}.pth')
            agent.save(model_path)
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_model_path = os.path.join(model_dir, 'dqn_snake_best.pth')
            agent.save(best_model_path)
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'dqn_snake_final.pth')
    agent.save(final_model_path)
    
    # Close environment
    env.close()
    
    # Plot training results
    plot_results(scores, avg_scores, epsilons, losses, model_dir)
    
    print("=" * 60)
    print("Training completed!")
    print(f"Final average score: {avg_scores[-1]:.2f}")
    print(f"Best average score: {best_avg_score:.2f}")
    print(f"Models saved in: {model_dir}")
    print("=" * 60)


def plot_results(scores, avg_scores, epsilons, losses, save_dir):
    """
    Plot training results.
    
    Args:
        scores: List of scores per episode
        avg_scores: List of average scores
        epsilons: List of epsilon values
        losses: List of loss values
        save_dir: Directory to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot scores
    ax1.plot(scores, alpha=0.3, label='Score')
    ax1.plot(avg_scores, label='Avg Score (100 episodes)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Score over Episodes')
    ax1.legend()
    ax1.grid(True)
    
    # Plot epsilon
    ax2.plot(epsilons, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate (Epsilon)')
    ax2.grid(True)
    
    # Plot loss
    ax3.plot(losses, color='red', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.grid(True)
    
    # Plot score distribution (histogram)
    ax4.hist(scores, bins=30, color='green', alpha=0.7)
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(plot_path)
    print(f"Training plots saved to {plot_path}")
    plt.close()


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train DQN agent for Snake game')
    
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes to train (default: 10000)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Frequency of saving model in episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render the game during training')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='Frequency of logging in episodes (default: 10)')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run a quick sanity check with 10 episodes')
    
    args = parser.parse_args()
    
    # If sanity check mode, override some parameters
    if args.sanity_check:
        print("\n" + "=" * 60)
        print("SANITY CHECK MODE")
        print("Running with 10 episodes for quick testing...")
        print("=" * 60 + "\n")
        args.episodes = 10
        args.save_freq = 5
        args.log_freq = 1
        args.model_dir = 'models_sanity_check'
    
    # Run training
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        render=args.render,
        model_dir=args.model_dir,
        log_freq=args.log_freq
    )


if __name__ == '__main__':
    main()


"""
Watch/Evaluation script for Snake RL Agent
Visualizes a trained DQN agent playing the Snake game.
"""

import argparse
import time
from snake_env import SnakeEnv
from dqn_agent import DQNAgent


def watch(
    model_path,
    episodes=10,
    fps=10,
    epsilon=0.0
):
    """
    Watch the trained agent play the game.
    
    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to watch
        fps: Frames per second (game speed)
        epsilon: Exploration rate (0.0 for pure exploitation)
    """
    # Initialize environment with rendering enabled
    env = SnakeEnv(render_mode=True)
    
    # Initialize agent
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        hidden_size=256,
        epsilon_start=epsilon  # Use specified epsilon (typically 0 for evaluation)
    )
    
    # Load the trained model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    
    print("=" * 60)
    print("Watching trained agent play Snake...")
    print(f"Episodes: {episodes}")
    print(f"FPS: {fps}")
    print(f"Epsilon: {epsilon}")
    print("Press Ctrl+C to stop early")
    print("=" * 60 + "\n")
    
    # Statistics
    scores = []
    steps_list = []
    
    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            score = 0
            steps = 0
            done = False
            
            print(f"\nEpisode {episode}/{episodes}")
            
            while not done:
                # Select action (no exploration in watch mode unless epsilon specified)
                action = agent.select_action(state, training=False)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Render the game
                env.render(fps=fps)
                
                score = info['score']
                steps = info['steps']
                state = next_state
            
            scores.append(score)
            steps_list.append(steps)
            
            print(f"Episode {episode} finished - Score: {score}, Steps: {steps}")
            
            # Small delay between episodes
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Close environment
        env.close()
        
        # Print statistics
        if scores:
            print("\n" + "=" * 60)
            print("Statistics:")
            print(f"Average Score: {sum(scores) / len(scores):.2f}")
            print(f"Max Score: {max(scores)}")
            print(f"Min Score: {min(scores)}")
            print(f"Average Steps: {sum(steps_list) / len(steps_list):.2f}")
            print("=" * 60)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Watch trained DQN agent play Snake')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model (.pth file)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to watch (default: 10)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second / game speed (default: 10)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                       help='Exploration rate (default: 0.0 for pure exploitation)')
    
    args = parser.parse_args()
    
    # Run watch mode
    watch(
        model_path=args.model_path,
        episodes=args.episodes,
        fps=args.fps,
        epsilon=args.epsilon
    )


if __name__ == '__main__':
    main()


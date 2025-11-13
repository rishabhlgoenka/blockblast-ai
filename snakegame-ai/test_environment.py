"""
Quick test script to verify the Snake environment is working correctly.
This runs a few episodes with random actions to ensure everything is set up properly.
"""

from snake_env import SnakeEnv
import numpy as np


def test_environment():
    """Test the Snake environment with random actions"""
    print("=" * 60)
    print("Testing Snake Environment")
    print("=" * 60)
    
    # Create environment without rendering for speed
    env = SnakeEnv(render_mode=False)
    print("✓ Environment created successfully")
    
    # Test reset
    state = env.reset()
    print(f"✓ Environment reset - State shape: {state.shape}")
    print(f"  Initial state: {state}")
    
    # Run a few episodes with random actions
    num_test_episodes = 5
    print(f"\nRunning {num_test_episodes} test episodes with random actions...")
    
    for episode in range(1, num_test_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:  # Limit steps for testing
            # Random action
            action = np.random.randint(0, 3)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        print(f"  Episode {episode}: Score={info['score']}, Steps={steps}, Total Reward={total_reward:.2f}")
    
    print("✓ All episodes completed successfully")
    
    # Clean up
    env.close()
    print("✓ Environment closed")
    
    print("\n" + "=" * 60)
    print("Environment test PASSED!")
    print("You can now run training with: python train_agent.py --sanity_check")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_environment()
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        print("Please check your installation and try again.")
        raise


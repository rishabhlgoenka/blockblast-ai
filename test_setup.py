#!/usr/bin/env python3
"""
Quick test script to verify BlockBlastAICNN setup.

Run this after installation to ensure everything is working:
    python test_setup.py
"""

import sys
import numpy as np


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        print(f"  ✓ Gymnasium {gym.__version__}")
    except ImportError as e:
        print(f"  ✗ Gymnasium import failed: {e}")
        return False
    
    try:
        import stable_baselines3 as sb3
        print(f"  ✓ Stable-Baselines3 {sb3.__version__}")
    except ImportError as e:
        print(f"  ✗ Stable-Baselines3 import failed: {e}")
        print("    Install with: pip install stable-baselines3")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    return True


def test_environment():
    """Test that the environment works correctly."""
    print("\nTesting BlockBlastEnv...")
    
    try:
        from ppo_env import BlockBlastEnv
        
        # Create environment
        env = BlockBlastEnv(seed=42)
        print("  ✓ Environment created")
        
        # Check observation space
        obs_space = env.observation_space
        assert obs_space.shape == (4, 8, 8), f"Wrong obs shape: {obs_space.shape}"
        print(f"  ✓ Observation space: {obs_space.shape}")
        
        # Check action space
        action_space = env.action_space
        assert action_space.n == 507, f"Wrong action space size: {action_space.n}"
        print(f"  ✓ Action space: Discrete({action_space.n})")
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (4, 8, 8), f"Wrong obs shape after reset: {obs.shape}"
        assert obs.dtype == np.uint8, f"Wrong dtype: {obs.dtype}"
        assert obs.min() >= 0 and obs.max() <= 255, f"Values out of range: [{obs.min()}, {obs.max()}]"
        print("  ✓ Reset works")
        
        # Test a few steps
        for i in range(5):
            # Try random valid action (action 52 is often valid at start)
            action = 52 + i * 10  # Try different actions
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                obs, info = env.reset()
        
        print("  ✓ Step works")
        
        env.close()
        print("  ✓ Environment tests passed!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_architecture():
    """Test that the CNN architecture can be created."""
    print("\nTesting CNN architecture...")
    
    try:
        import torch
        import gymnasium as gym
        from gymnasium import spaces
        from train_ppo_cnn import BlockBlastCNN
        
        # Create dummy observation space (uint8 for image-based CNN)
        obs_space = spaces.Box(low=0, high=255, shape=(4, 8, 8), dtype=np.uint8)
        
        # Create CNN
        cnn = BlockBlastCNN(obs_space, features_dim=256)
        print("  ✓ CNN created")
        
        # Test forward pass
        batch_size = 4
        dummy_obs = torch.zeros(batch_size, 4, 8, 8)
        features = cnn(dummy_obs)
        
        assert features.shape == (batch_size, 256), f"Wrong output shape: {features.shape}"
        print(f"  ✓ Forward pass works: {dummy_obs.shape} -> {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("BlockBlastAICNN Setup Test")
    print("="*70)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment tests failed.")
        sys.exit(1)
    
    # Test CNN
    if not test_cnn_architecture():
        print("\n❌ CNN tests failed.")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYou're ready to train! Try:")
    print("  python train_ppo_cnn.py --timesteps 100000")
    print("\nOr evaluate a trained model:")
    print("  python eval_ppo_cnn.py --episodes 10")
    print("="*70)


if __name__ == '__main__':
    main()


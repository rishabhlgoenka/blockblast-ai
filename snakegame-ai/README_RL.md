# Snake Game Reinforcement Learning Agent

This project implements a Deep Q-Network (DQN) agent that learns to play the Snake game using reinforcement learning.

## Overview

The project consists of several components:

1. **Original Game**: `snake-pygame-master/Snake Game.py` - The original playable Snake game (unchanged)
2. **Environment Wrapper**: `snake_env.py` - Gym-style environment interface
3. **DQN Agent**: `dqn_agent.py` - Deep Q-Network implementation with replay buffer
4. **Training Script**: `train_agent.py` - Script to train the agent
5. **Watch Script**: `watch_agent.py` - Script to visualize trained agent playing

## Installation

1. Make sure you have Python 3.7+ installed

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pygame torch numpy matplotlib
```

## Original Game

You can still play the original human-controlled Snake game:

```bash
cd snake-pygame-master
python "Snake Game.py"
```

Controls: Arrow keys or WASD to move, ESC to quit

## RL Agent Details

### State Representation (11 features)
The agent observes:
- **Danger detection** (3 features): Danger straight ahead, to the right, to the left
- **Current direction** (4 features): One-hot encoding of UP, DOWN, LEFT, RIGHT
- **Food location** (4 features): Food is above, below, left, or right of snake head

### Action Space (3 actions)
Actions are **relative to current direction**:
- **0**: Turn left
- **1**: Go straight (maintain direction)
- **2**: Turn right

### Reward Structure
- **+10**: Eating an apple (food)
- **-10**: Dying (collision with wall or self)
- **-0.01**: Per step (encourages efficiency and shorter paths)

### DQN Architecture
- Input layer: 11 neurons (state features)
- Hidden layer 1: 256 neurons (ReLU activation)
- Hidden layer 2: 256 neurons (ReLU activation)
- Output layer: 3 neurons (Q-values for each action)

### Training Features
- **Experience Replay**: Stores and samples past experiences to break correlation
- **Target Network**: Separate target network updated periodically for stability
- **Epsilon-Greedy Exploration**: Starts at 1.0 (pure exploration) and decays to 0.01
- **Gradient Clipping**: Prevents exploding gradients

## Usage

### 1. Sanity Check (Quick Test)

Before running long training, verify everything works:

```bash
python train_agent.py --sanity_check
```

This runs 10 episodes quickly to ensure there are no crashes. Models will be saved to `models_sanity_check/`.

### 2. Training the Agent

Basic training with default parameters (10,000 episodes):

```bash
python train_agent.py
```

Custom training with options:

```bash
python train_agent.py --episodes 5000 --save_freq 50 --log_freq 10
```

**Available options:**
- `--episodes`: Number of episodes to train (default: 10000)
- `--max_steps`: Maximum steps per episode (default: 1000)
- `--save_freq`: Save model every N episodes (default: 100)
- `--render`: Enable visual rendering during training (slower)
- `--model_dir`: Directory to save models (default: models)
- `--log_freq`: Log progress every N episodes (default: 10)

**Training Output:**
- Models saved periodically: `models/dqn_snake_ep{N}.pth`
- Best model (highest avg score): `models/dqn_snake_best.pth`
- Final model: `models/dqn_snake_final.pth`
- Training plots: `models/training_results.png`

**Example Training Session:**
```bash
# Train for 20,000 episodes, save every 200 episodes
python train_agent.py --episodes 20000 --save_freq 200 --log_freq 20
```

### 3. Watching the Trained Agent

After training, watch your agent play:

```bash
python watch_agent.py --model_path models/dqn_snake_best.pth
```

**Available options:**
- `--model_path`: Path to saved model (required)
- `--episodes`: Number of episodes to watch (default: 10)
- `--fps`: Game speed in frames per second (default: 10)
- `--epsilon`: Exploration rate for testing (default: 0.0)

**Examples:**

Watch the best model for 5 episodes at slow speed:
```bash
python watch_agent.py --model_path models/dqn_snake_best.pth --episodes 5 --fps 5
```

Watch final model at fast speed:
```bash
python watch_agent.py --model_path models/dqn_snake_final.pth --fps 30
```

Watch a specific checkpoint:
```bash
python watch_agent.py --model_path models/dqn_snake_ep5000.pth --episodes 20 --fps 15
```

## Training Tips

1. **Start Small**: Use `--sanity_check` first to verify everything works

2. **Monitor Progress**: Watch the logged output for:
   - Score increasing over time
   - Epsilon decreasing (less random exploration)
   - Average score improving

3. **Patience**: Early episodes will have low scores (0-2) as the agent explores randomly. Improvement typically starts after 500-1000 episodes.

4. **Expected Performance**:
   - Episodes 1-500: Random behavior, scores 0-2
   - Episodes 500-2000: Learning basic survival, scores 2-5
   - Episodes 2000-5000: Improving strategy, scores 5-15
   - Episodes 5000+: Good performance, scores 10-30+

5. **Adjust Hyperparameters**: Edit `dqn_agent.py` to tune:
   - Learning rate
   - Hidden layer size
   - Epsilon decay rate
   - Batch size

## File Structure

```
snakegame-ai/
├── snake-pygame-master/
│   ├── Snake Game.py          # Original playable game (unchanged)
│   └── README.md              # Original game README
├── snake_env.py               # Environment wrapper (NEW)
├── dqn_agent.py              # DQN agent implementation (NEW)
├── train_agent.py            # Training script (NEW)
├── watch_agent.py            # Visualization script (NEW)
├── requirements.txt          # Python dependencies (NEW)
├── README_RL.md              # This file (NEW)
└── models/                   # Saved models (created during training)
    ├── dqn_snake_best.pth
    ├── dqn_snake_final.pth
    ├── dqn_snake_ep100.pth
    └── training_results.png
```

## Implementation Details

### Environment (`snake_env.py`)
- Wraps the original game logic into a Gym-style interface
- Provides `reset()`, `step()`, and `render()` methods
- Handles collision detection and reward calculation
- Can run with or without rendering for faster training

### Agent (`dqn_agent.py`)
- Implements Deep Q-Network with experience replay
- Uses separate target network for stability
- Epsilon-greedy exploration with decay
- Supports GPU acceleration if CUDA is available

### Training (`train_agent.py`)
- Main training loop with episode management
- Periodic model checkpoints
- Tracks and plots training metrics
- Command-line interface for easy configuration

### Watching (`watch_agent.py`)
- Loads trained models
- Runs in evaluation mode (no training)
- Displays game visually with Pygame
- Shows performance statistics

## Troubleshooting

**Issue**: "No module named 'pygame'"
- **Solution**: Run `pip install pygame`

**Issue**: Training is very slow
- **Solution**: Don't use `--render` during training. Rendering slows down training significantly.

**Issue**: Agent not improving
- **Solution**: 
  - Train for more episodes (try 10,000+)
  - Check if epsilon is decaying (should decrease over time)
  - Try adjusting learning rate in `dqn_agent.py`

**Issue**: "CUDA out of memory"
- **Solution**: Reduce batch size in `dqn_agent.py` (line with `batch_size=64`)

**Issue**: Can't find saved model
- **Solution**: Check the `models/` directory. Use the full path if needed.

## Future Improvements

Possible enhancements:
- Implement Double DQN or Dueling DQN
- Add more sophisticated state representations (e.g., CNN on game grid)
- Implement prioritized experience replay
- Add curriculum learning (start with easier settings)
- Multi-step returns for better credit assignment
- Implement A3C or PPO for comparison

## Credits

- Original Snake Game: From the `snake-pygame-master` repository
- DQN Algorithm: Based on "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Implementation: Custom wrapper and agent for this specific game

## License

The RL implementation is provided as-is for educational purposes.


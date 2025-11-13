# Snake RL Agent - Implementation Summary

## What Was Built

A complete reinforcement learning system for training an AI agent to play Snake using Deep Q-Network (DQN).

## Files Created

### Core Implementation (4 files)

1. **`snake_env.py`** (373 lines)
   - Gym-style environment wrapper around the original Snake game
   - **State space**: 11 features (danger detection + direction + food location)
   - **Action space**: 3 actions (turn left, straight, right - relative to current direction)
   - **Rewards**: +10 (eat apple), -10 (die), -0.01 (per step)
   - Can run with or without rendering (rendering disabled during training for speed)

2. **`dqn_agent.py`** (296 lines)
   - Deep Q-Network implementation
   - Experience replay buffer (stores 100k transitions)
   - Epsilon-greedy exploration (starts at 1.0, decays to 0.01)
   - Target network (updated every 1000 steps)
   - 3-layer neural network (11 → 256 → 256 → 3)
   - GPU support (automatically uses CUDA if available)

3. **`train_agent.py`** (216 lines)
   - Training script with command-line interface
   - Saves models periodically and tracks best model
   - Generates training plots (score, epsilon, loss)
   - Includes sanity check mode for quick testing
   - Logs progress during training

4. **`watch_agent.py`** (133 lines)
   - Evaluation/visualization script
   - Loads trained models and displays agent playing
   - Adjustable game speed (FPS)
   - Shows statistics after episodes
   - Pure exploitation mode (no exploration)

### Documentation & Testing (4 files)

5. **`requirements.txt`**
   - Package dependencies (pygame, torch, numpy, matplotlib)

6. **`test_environment.py`** (62 lines)
   - Quick test to verify environment works
   - Runs 5 episodes with random actions

7. **`README_RL.md`** (Comprehensive documentation)
   - Detailed explanation of all components
   - State/action/reward specifications
   - Usage instructions
   - Troubleshooting guide
   - Implementation details

8. **`QUICKSTART.md`** (Quick reference guide)
   - Step-by-step usage instructions
   - Command reference table
   - Expected training timeline
   - Common issues and solutions

### Original Game (Preserved)

9. **`snake-pygame-master/Snake Game.py`** (UNCHANGED)
   - Original playable game still works
   - No modifications made

## Key Design Decisions

### 1. State Representation
Chose a compact 11-feature representation instead of raw pixels:
- **Pros**: Faster training, lower memory, simpler network
- **Cons**: Hand-engineered features instead of learned representations
- Features capture essential information: danger, direction, food location

### 2. Relative Actions
Actions are relative to current direction (left/straight/right) instead of absolute (up/down/left/right):
- **Pros**: More intuitive for agent, easier to learn, no invalid moves
- **Cons**: Requires direction tracking
- Prevents impossible moves (e.g., can't go directly opposite to current direction)

### 3. Reward Structure
Simple but effective rewards:
- **+10** for food: Strong positive reinforcement
- **-10** for death: Strong negative reinforcement  
- **-0.01** per step: Encourages efficiency, prevents infinite loops

### 4. DQN Architecture
Standard DQN with proven techniques:
- Experience replay: Breaks correlation between consecutive samples
- Target network: Stabilizes training
- Epsilon-greedy: Balances exploration and exploitation
- Gradient clipping: Prevents exploding gradients

## Current Game Structure Analysis

### Main Game Loop (Original Snake Game.py)
```
Lines 91-170: Main infinite while loop
  ├── Event handling (keyboard input)
  ├── Direction validation (prevent 180° turns)
  ├── Snake movement (update position based on direction)
  ├── Food collision detection
  ├── Snake body growth mechanics
  ├── Rendering (draw snake, food, score)
  └── Game over conditions
      ├── Wall collision (lines 157-160)
      └── Self collision (lines 162-164)
```

### Key Variables
- `snake_pos`: [x, y] - Head position
- `snake_body`: List of [x, y] - All body segments
- `food_pos`: [x, y] - Apple position
- `direction`: String ('UP', 'DOWN', 'LEFT', 'RIGHT')
- `score`: Integer - Number of apples eaten

### Game Grid
- Window: 720×480 pixels
- Block size: 10×10 pixels
- Grid: 72×48 cells
- Coordinates are multiples of 10

## How It Works

### Training Flow
```
1. Initialize environment and agent
2. For each episode:
   a. Reset environment → get initial state
   b. While not done:
      - Agent selects action (epsilon-greedy)
      - Environment executes action → returns (next_state, reward, done)
      - Store transition in replay buffer
      - Sample batch from buffer → train network
      - Update target network periodically
   c. Decay epsilon (reduce exploration)
   d. Save model checkpoints
3. Generate training plots
```

### Environment Step
```
1. Convert relative action to absolute direction
2. Move snake head in that direction
3. Check if food is eaten:
   - Yes: Increase score, spawn new food, reward +10
   - No: Remove tail segment, reward -0.01
4. Check for collisions (wall or self):
   - Yes: reward -10, done=True
5. Return (new_state, reward, done, info)
```

### Agent Action Selection
```
If training and random() < epsilon:
    return random action (exploration)
Else:
    return argmax(Q(state)) (exploitation)
```

## Usage Commands

### Quick Test
```bash
# Test environment (30 seconds)
python test_environment.py

# Sanity check training (1-2 minutes)
python train_agent.py --sanity_check
```

### Full Training
```bash
# Default: 10,000 episodes (~4-6 hours on CPU)
python train_agent.py

# Custom training
python train_agent.py --episodes 20000 --save_freq 500
```

### Watch Agent
```bash
# Watch best model
python watch_agent.py --model_path models/dqn_snake_best.pth

# Slow motion
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 5

# Multiple episodes
python watch_agent.py --model_path models/dqn_snake_best.pth --episodes 20
```

### Play Original Game
```bash
cd snake-pygame-master
python "Snake Game.py"
```

## Expected Performance

### Training Progress
| Episodes | Expected Behavior | Avg Score |
|----------|------------------|-----------|
| 0-500 | Random exploration | 0-2 |
| 500-2000 | Learning basics | 2-5 |
| 2000-5000 | Improving strategy | 5-15 |
| 5000-10000 | Good performance | 15-30+ |
| 10000+ | Optimal play | 30-50+ |

### Training Metrics
Saved in `models/training_results.png`:
1. **Score over episodes**: Shows learning progress
2. **Epsilon decay**: Shows exploration reduction
3. **Loss**: Shows network training stability
4. **Score distribution**: Shows performance consistency

## Model Files

Training creates these files in `models/`:
- `dqn_snake_best.pth` - **Best model** (highest avg score) - USE THIS
- `dqn_snake_final.pth` - Final model after all episodes
- `dqn_snake_ep{N}.pth` - Checkpoints every N episodes
- `training_results.png` - Training plots

Each `.pth` file contains:
- Policy network weights
- Target network weights
- Optimizer state
- Epsilon value
- Training step count

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Snake Game                         │
│              (Original Pygame Code)                  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ Wrapped by
                 ▼
┌─────────────────────────────────────────────────────┐
│              SnakeEnv (snake_env.py)                 │
│  ┌──────────────────────────────────────────────┐  │
│  │  reset() → state                              │  │
│  │  step(action) → (next_state, reward, done)   │  │
│  │  render() → display game                     │  │
│  └──────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ Used by
                 ▼
┌─────────────────────────────────────────────────────┐
│             DQNAgent (dqn_agent.py)                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  DQN Network (11→256→256→3)                 │  │
│  │  Replay Buffer (100k transitions)            │  │
│  │  select_action(state) → action               │  │
│  │  train_step() → update weights               │  │
│  └──────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ train_agent  │   │ watch_agent  │
│    .py       │   │    .py       │
│              │   │              │
│ Training     │   │ Evaluation   │
│ Loop         │   │ Mode         │
└──────────────┘   └──────────────┘
```

## What Was NOT Changed

✓ Original `Snake Game.py` - Completely untouched, still playable
✓ Original `README.md` - Preserved as-is
✓ Game graphics, colors, and mechanics - All preserved

## Next Steps (For User)

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test environment**: `python test_environment.py`
3. **Quick sanity check**: `python train_agent.py --sanity_check`
4. **Full training**: `python train_agent.py` (leave running for hours)
5. **Watch trained agent**: `python watch_agent.py --model_path models/dqn_snake_best.pth`

For detailed instructions, see:
- **QUICKSTART.md** - Quick start guide
- **README_RL.md** - Complete documentation

## Potential Improvements

Future enhancements could include:
- [ ] Double DQN (reduce overestimation bias)
- [ ] Dueling DQN (separate value and advantage streams)
- [ ] Prioritized experience replay (sample important transitions more often)
- [ ] CNN-based state representation (learn features from pixels)
- [ ] A3C or PPO (actor-critic methods)
- [ ] Curriculum learning (start with easier game settings)
- [ ] Multi-step returns (better credit assignment)
- [ ] Hindsight experience replay (learn from failures)

## Technical Notes

### GPU Acceleration
- Agent automatically uses CUDA if available
- Training is ~3-5x faster on GPU vs CPU
- Models can be trained on GPU and loaded on CPU (and vice versa)

### Memory Usage
- Replay buffer: ~150MB (100k transitions × 11 features × 2 states)
- Neural network: ~1MB (256×256 weights)
- Total: ~200-300MB during training

### Training Time Estimates
- **CPU (modern)**: 10k episodes in 4-6 hours
- **GPU (CUDA)**: 10k episodes in 1-2 hours
- Depends on CPU/GPU speed and epsilon decay

### Reproducibility
- Uses random initialization (not seeded by default)
- Each training run will be different
- To make reproducible, add random seeds in training script

## Credits & References

**Original Snake Game**: From `snake-pygame-master` repository

**DQN Algorithm**: 
- Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- DeepMind Nature paper

**Implementation**: Custom wrapper and agent designed specifically for this Snake game

---

**Implementation completed successfully!** All files are modular, well-commented, and ready to use.


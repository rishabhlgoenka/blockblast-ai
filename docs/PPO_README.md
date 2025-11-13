# Block Blast AI - PPO + CNN Implementation

This is a complete PPO (Proximal Policy Optimization) implementation for Block Blast using a CNN policy. This folder is a standalone copy of the original Block Blast project, adapted specifically for PPO training with convolutional neural networks.

## 🎯 Overview

**Key Differences from Original DQN Implementation:**

| Feature | Original DQN | This PPO+CNN |
|---------|--------------|--------------|
| **Algorithm** | Deep Q-Network | Proximal Policy Optimization |
| **Network** | Linear (MLP) | Convolutional Neural Network |
| **State Representation** | Hand-crafted features (139D vector) | Raw board + pieces (4×8×8 channels) |
| **Library** | Custom PyTorch | Stable-Baselines3 |
| **Learning** | Off-policy (replay buffer) | On-policy (direct policy gradient) |

## 📁 Project Structure

```
BlockBlastAICNN/
├── blockblast/           # Core game logic (copied from original)
│   ├── __init__.py
│   ├── core.py          # Game state, actions, scoring
│   ├── env.py           # Original DQN environment
│   ├── pieces.py        # 30 piece definitions
│   └── pygame_app.py    # GUI (not modified)
│
├── ppo_env.py           # 🆕 Gym-compatible environment with CNN observations
├── train_ppo_cnn.py     # 🆕 PPO training script
├── eval_ppo_cnn.py      # 🆕 Evaluation script
├── requirements.txt     # Updated with SB3 dependencies
├── PPO_README.md        # This file
│
├── models/              # Saved models
│   └── ppo_checkpoints/ # Training checkpoints
├── results/             # Evaluation results
└── ppo_logs/            # TensorBoard logs
```

## 🔧 Installation

### 1. Create Virtual Environment (Recommended)

```bash
cd BlockBlastAICNN
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `stable-baselines3>=2.0.0` - PPO implementation
- `gymnasium>=0.28.0` - Environment API
- `torch>=2.0.0` - Neural networks
- `numpy>=1.20.0` - Numerical operations

## 🎮 Observation Space

### CNN-Friendly Format: `(4, 8, 8)`

The environment provides channel-first observations perfect for CNNs:

```
Channel 0: Board State (8×8)
  - 0 = empty cell
  - 1 = filled cell

Channels 1-3: Available Pieces (each 8×8)
  - Each piece (5×5) is centered on an 8×8 grid
  - 0 = empty, 1 = piece block
  - Helps CNN learn spatial relationships
```

**Visual Example:**

```
Channel 0 (Board):      Channel 1 (Piece 1):    Channel 2 (Piece 2):
┌─────────┐             ┌─────────┐             ┌─────────┐
│ █ █ █ █ │             │ · · · · │             │ · · · · │
│ · · · · │             │ · █ █ · │             │ · · █ · │
│ █ · · · │             │ · █ · · │             │ · · █ · │
│ · · · █ │             │ · · · · │             │ · · █ · │
└─────────┘             └─────────┘             └─────────┘
```

**Why this format?**
- ✅ Standard CNN input format (channels-first)
- ✅ Spatial structure preserved (not flattened)
- ✅ Compatible with SB3's `CnnPolicy`
- ✅ Easier for CNN to learn spatial patterns

## 🎯 Action Space

**Discrete(507)** - Same as original DQN

- **3 pieces** × **13 positions** × **13 positions** = **507 actions**
- Position range: `-4` to `8` (allows edge placements)
- Invalid actions receive `-1` penalty (episode continues)

## 🎁 Reward Structure

```python
reward = score_delta + line_bonus + game_over_penalty

# Components:
score_delta        = score_after - score_before  # Base game scoring
line_bonus         = +10 per line cleared        # Encourage clearing
game_over_penalty  = -20 if game ends            # Discourage early death
invalid_action     = -1 (state unchanged)        # Learn valid moves
```

## 🚀 Training

### Basic Training

```bash
python train_ppo_cnn.py --timesteps 1000000
```

This will:
- Train for 1 million timesteps (~2-4 hours depending on hardware)
- Save checkpoints every 100k timesteps to `models/ppo_checkpoints/`
- Save final model to `models/ppo_cnn_blockblast.zip`
- Log training to `ppo_logs/` (view with TensorBoard)

### Advanced Training Options

```bash
# Long training with custom hyperparameters
python train_ppo_cnn.py \
    --timesteps 5000000 \
    --learning-rate 0.0001 \
    --n-steps 4096 \
    --batch-size 512 \
    --n-epochs 10 \
    --gamma 0.99 \
    --clip-range 0.2 \
    --ent-coef 0.01 \
    --checkpoint-freq 200000

# Quick test run
python train_ppo_cnn.py --timesteps 100000 --checkpoint-freq 25000
```

### Monitor Training Progress

```bash
# In another terminal, start TensorBoard
tensorboard --logdir=ppo_logs

# Open browser to http://localhost:6006
```

**Key metrics to watch:**
- `rollout/ep_rew_mean` - Average episode reward
- `train/entropy_loss` - Exploration (should decrease gradually)
- `train/policy_loss` - Policy improvement
- `train/value_loss` - Value function accuracy

## 📊 Evaluation

### Standard Evaluation

```bash
python eval_ppo_cnn.py --episodes 200
```

**Output:**
```
Evaluation Summary
======================================================================

Score Statistics:
  Mean:     245.50
  Std:       87.32
  Min:        42
  Max:       523
  Median:    231.00
  Q25:       178.25
  Q75:       298.50

✓ Results saved to results/ppo_eval.json
```

### Advanced Evaluation

```bash
# Evaluate specific checkpoint
python eval_ppo_cnn.py \
    --model-path models/ppo_checkpoints/ppo_cnn_500000_steps.zip \
    --episodes 500

# Watch agent play (renders episodes)
python eval_ppo_cnn.py --episodes 10 --render

# Long evaluation for publication-quality stats
python eval_ppo_cnn.py --episodes 1000 --results-path results/final_eval.json
```

## 🧠 CNN Architecture

The custom CNN feature extractor processes the 4-channel observation:

```
Input: (4, 8, 8)
    ↓
Conv2D(4 → 32, kernel=3x3, padding=1)
    ↓ ReLU
Conv2D(32 → 64, kernel=3x3, padding=1)
    ↓ ReLU
Conv2D(64 → 64, kernel=3x3, padding=0)
    ↓ ReLU
Flatten
    ↓
Linear(256 → 256)
    ↓ ReLU
    ├─→ Policy Head (128 neurons → 507 actions)
    └─→ Value Head (128 neurons → 1 value)
```

**Design Rationale:**
- Small input (8×8) → Small network (3 conv layers sufficient)
- Preserves spatial structure for piece-board reasoning
- Separate policy/value heads (standard actor-critic)

## ⚙️ Hyperparameters

### Default Values (Tuned for Block Blast)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `learning_rate` | `3e-4` | Standard for PPO |
| `n_steps` | `2048` | Rollout length before update |
| `batch_size` | `512` | Minibatch size for SGD |
| `n_epochs` | `10` | Optimization epochs per rollout |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE parameter (advantage estimation) |
| `clip_range` | `0.2` | PPO clipping (prevents large updates) |
| `ent_coef` | `0.01` | Entropy bonus (exploration) |
| `vf_coef` | `0.5` | Value function loss weight |

### Tuning Tips

**Agent too cautious / low scores?**
- ↑ Increase `ent_coef` to `0.02` (more exploration)
- ↑ Increase `clip_range` to `0.3` (larger policy updates)

**Training unstable / value loss spikes?**
- ↓ Decrease `learning_rate` to `1e-4`
- ↓ Decrease `clip_range` to `0.15`
- ↑ Increase `n_steps` to `4096` (more data per update)

**Converging too slowly?**
- ↑ Increase `learning_rate` to `5e-4`
- ↑ Increase `batch_size` to `1024`
- ↓ Decrease `n_epochs` to `5` (faster updates)

## 📈 Expected Performance

Based on architecture and problem complexity:

| Training Timesteps | Expected Avg Score | Training Time* |
|--------------------|-------------------|----------------|
| 100k | 50-100 | 15-30 min |
| 500k | 150-250 | 1-2 hours |
| 1M | 250-400 | 2-4 hours |
| 5M | 400-600+ | 10-20 hours |

*On modern GPU (RTX 3080 or similar)

## 🔬 Comparison with Original DQN

### Advantages of PPO + CNN:

✅ **No feature engineering** - CNN learns directly from raw board  
✅ **On-policy learning** - More stable, less prone to catastrophic forgetting  
✅ **Better exploration** - Entropy bonus encourages trying new strategies  
✅ **Standard library** - Less custom code, easier to debug/extend  

### Potential Drawbacks:

❌ **Sample efficiency** - PPO typically needs more episodes than DQN  
❌ **Memory** - CNN uses more GPU memory than MLP  
❌ **Training time** - Each update is slower due to CNN forward passes  

## 🛠️ Troubleshooting

### "No module named 'stable_baselines3'"

```bash
pip install stable-baselines3 gymnasium
```

### "CUDA out of memory"

Reduce batch size:
```bash
python train_ppo_cnn.py --batch-size 256 --n-steps 1024
```

### Agent keeps making invalid moves

This is normal early in training. The agent learns valid moves through negative rewards. After ~100k timesteps, invalid move rate should drop below 10%.

### Training very slow on CPU

PPO + CNN benefits greatly from GPU. Expected speeds:
- **CPU**: ~200-500 steps/sec
- **GPU**: ~2000-5000 steps/sec

Consider reducing `n_steps` or `batch_size` if training on CPU.

### Want to use original DQN instead?

The original DQN code is untouched in the parent directory. This folder is completely independent.

## 📚 Further Reading

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [Gymnasium API](https://gymnasium.farama.org/)

## 🤝 Contributing

This is a standalone fork of the original Block Blast DQN project. To modify:

1. All changes should stay within `BlockBlastAICNN/`
2. Do not modify files in the parent directory
3. Feel free to experiment with:
   - Different CNN architectures
   - Reward shaping
   - Hyperparameters
   - Alternative algorithms (A2C, SAC, etc.)

## 📄 License

Same as the original Block Blast project.

---

**Happy Training! 🎮🤖**


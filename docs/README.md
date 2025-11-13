# Block Blast AI - PPO + CNN

<div align="center">

**Train a Deep Reinforcement Learning agent to master Block Blast using PPO and Convolutional Neural Networks**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)

</div>

---

## 🎮 What is This?

This is a **standalone PPO (Proximal Policy Optimization) implementation** for the Block Blast puzzle game, using a **Convolutional Neural Network** to learn directly from the board state.

**Key Features:**
- 🧠 **CNN-based policy** - Learns spatial patterns from raw 8×8 board
- 🎯 **Production-ready** - Built on Stable-Baselines3
- 📊 **Full observability** - 4-channel observation (board + 3 pieces)
- ⚙️ **Highly configurable** - CLI args for all hyperparameters
- 📈 **Training monitoring** - TensorBoard integration
- 🔬 **Evaluation tools** - Comprehensive statistics and analysis

## 🚀 Quick Start

### 1. Install (2 minutes)

```bash
cd BlockBlastAICNN
pip install -r requirements.txt
python test_setup.py
```

### 2. Train (5-10 minutes for quick test)

```bash
python train_ppo_cnn.py --timesteps 100000
```

### 3. Evaluate

```bash
python eval_ppo_cnn.py --episodes 100
```

**That's it!** You've trained and evaluated your first PPO agent. 🎉

For more details, see **[QUICKSTART.md](QUICKSTART.md)**.

## 📊 What to Expect

| Training Time | Expected Avg Score | Timesteps |
|--------------|-------------------|-----------|
| 15-30 min | 50-100 | 100k |
| 1-2 hours | 150-250 | 500k |
| 2-4 hours | 250-400 | 1M |
| 10-20 hours | 400-600+ | 5M |

*Times for modern GPU (RTX 3080 or similar)*

## 🧠 How It Works

### Observation Space: `(4, 8, 8)`

```
Channel 0: Board State        Channels 1-3: Available Pieces
┌─────────┐                   ┌─────────┐ ┌─────────┐ ┌─────────┐
│ █ █ █ █ │                   │ · · · · │ │ · · · · │ │ · · · · │
│ · · · · │                   │ · █ █ · │ │ · · █ · │ │ · █ · · │
│ █ · · · │    →  CNN  →      │ · █ · · │ │ · · █ · │ │ · █ █ · │
│ · · · █ │                   │ · · · · │ │ · · █ · │ │ · · · · │
└─────────┘                   └─────────┘ └─────────┘ └─────────┘
```

### CNN Architecture

```
Input (4, 8, 8)
    ↓
Conv2D(4 → 32, 3×3) + ReLU
    ↓
Conv2D(32 → 64, 3×3) + ReLU
    ↓
Conv2D(64 → 64, 3×3) + ReLU
    ↓
Flatten + FC(256)
    ↓
    ├─→ Policy Head (128 → 507 actions)
    └─→ Value Head (128 → 1 value)
```

### Reward Function

```python
reward = score_increase + (10 × lines_cleared) - (20 if game_over)
```

- **Score increase**: Encourages placing pieces
- **Line bonus**: +10 per line (prioritize clearing)
- **Game over penalty**: -20 (avoid early termination)

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[PPO_README.md](PPO_README.md)** - Complete technical documentation
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - What was built and why

## 🎯 Common Use Cases

### Standard Training

```bash
# Train for 1M timesteps (~2-4 hours)
python train_ppo_cnn.py --timesteps 1000000

# Evaluate performance
python eval_ppo_cnn.py --episodes 200
```

### Hyperparameter Tuning

```bash
# More exploration
python train_ppo_cnn.py --timesteps 1000000 --ent-coef 0.02

# Faster learning
python train_ppo_cnn.py --timesteps 1000000 --learning-rate 0.0005

# More stable training
python train_ppo_cnn.py --timesteps 1000000 --learning-rate 0.0001 --clip-range 0.15
```

### Monitoring Training

```bash
# Terminal 1: Training
python train_ppo_cnn.py --timesteps 5000000

# Terminal 2: TensorBoard
tensorboard --logdir=ppo_logs
# Open browser to http://localhost:6006
```

### Watch Agent Play

```bash
python eval_ppo_cnn.py --episodes 10 --render
```

### Evaluate Checkpoints

```bash
# Compare different checkpoints
python eval_ppo_cnn.py --model-path models/ppo_checkpoints/ppo_cnn_100000_steps.zip --episodes 100
python eval_ppo_cnn.py --model-path models/ppo_checkpoints/ppo_cnn_500000_steps.zip --episodes 100
python eval_ppo_cnn.py --model-path models/ppo_cnn_blockblast.zip --episodes 100
```

## 🔬 Technical Details

### Algorithm: PPO (Proximal Policy Optimization)

**Why PPO?**
- ✅ More stable than vanilla policy gradients
- ✅ Better sample efficiency than A2C
- ✅ On-policy (no replay buffer issues)
- ✅ Industry standard (OpenAI, DeepMind use it)

### Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `learning_rate` | 3e-4 | Optimizer step size |
| `n_steps` | 2048 | Rollout length |
| `batch_size` | 512 | Minibatch size |
| `n_epochs` | 10 | Updates per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | Advantage estimation |
| `clip_range` | 0.2 | PPO clipping |
| `ent_coef` | 0.01 | Entropy bonus |

### Action Space: Discrete(507)

- **3 pieces** available at any time
- **13 × 13 = 169** possible positions per piece
- **3 × 169 = 507** total actions
- Invalid actions get -1 penalty (no episode termination)

## 📈 Training Tips

### Agent explores too much / random play
- Decrease `--ent-coef` (try 0.005)
- Training needs more time

### Agent too cautious / conservative
- Increase `--ent-coef` (try 0.02)
- Increase `--clip-range` (try 0.3)

### Training unstable / loss spikes
- Decrease `--learning-rate` (try 1e-4)
- Decrease `--clip-range` (try 0.15)

### Converging too slowly
- Increase `--learning-rate` (try 5e-4)
- Increase `--batch-size` (try 1024)

## 🛠️ Project Structure

```
BlockBlastAICNN/
├── ppo_env.py              # Gym environment (CNN observations)
├── train_ppo_cnn.py        # Training script
├── eval_ppo_cnn.py         # Evaluation script
├── test_setup.py           # Setup verification
│
├── blockblast/             # Core game logic (from original)
│   ├── core.py            # Game rules, scoring
│   ├── pieces.py          # 30 piece definitions
│   └── env.py             # Original DQN environment
│
├── models/                 # Saved models
│   └── ppo_checkpoints/   # Training checkpoints
├── results/                # Evaluation results (JSON)
├── ppo_logs/              # TensorBoard logs
│
└── docs/
    ├── README.md           # This file
    ├── QUICKSTART.md       # 5-minute guide
    ├── PPO_README.md       # Full technical docs
    └── PROJECT_SUMMARY.md  # Implementation details
```

## 🔄 Comparison with Original DQN

This folder is a **standalone copy** - the original DQN implementation in the parent directory is **completely untouched**.

| Feature | Original DQN | This PPO+CNN |
|---------|--------------|--------------|
| Algorithm | Deep Q-Network | Proximal Policy Optimization |
| Network | 3-layer MLP | CNN (3 conv + 2 FC) |
| Input | Hand-crafted 139D vector | Raw 4×8×8 channels |
| Library | Custom PyTorch | Stable-Baselines3 |
| Learning | Off-policy | On-policy |

**Both implementations are valid!** This PPO version explores a different approach using CNNs and modern RL libraries.

## ❓ FAQ

**Q: Do I need a GPU?**  
A: Not required, but strongly recommended. CPU training is 5-10× slower.

**Q: How much training time for good results?**  
A: 1M timesteps (2-4 hours on GPU) gives decent performance. 5M+ for best results.

**Q: Can I modify the CNN architecture?**  
A: Yes! Edit the `BlockBlastCNN` class in `train_ppo_cnn.py`.

**Q: What if training is unstable?**  
A: Lower the learning rate and clip range. See [PPO_README.md](PPO_README.md#troubleshooting).

**Q: How do I compare with the original DQN?**  
A: The original DQN code is in the parent directory, unchanged. Train both and compare!

**Q: Can I use this for other games?**  
A: Yes! The CNN architecture and PPO setup are general. Just adapt the observation space.

## 🤝 Contributing

This is a standalone project. Feel free to:
- Experiment with architectures
- Try different algorithms (A2C, SAC, etc.)
- Tune hyperparameters
- Compare with the original DQN

**Important:** All changes should stay within `BlockBlastAICNN/`. Don't modify files in the parent directory.

## 📄 Requirements

- Python 3.8+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- Gymnasium 0.28+
- NumPy 1.20+

See `requirements.txt` for full list.

## 🎓 Learning Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{blockblast_ppo_cnn,
  title = {Block Blast AI - PPO + CNN Implementation},
  author = {Block Blast AI Team},
  year = {2025},
  url = {https://github.com/yourusername/blockblast-ai}
}
```

## 📧 Support

- **Quick questions:** See [QUICKSTART.md](QUICKSTART.md)
- **Technical details:** See [PPO_README.md](PPO_README.md)
- **Implementation info:** See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## 📜 License

Same as the original Block Blast project.

---

<div align="center">

**Ready to train?** Start with [QUICKSTART.md](QUICKSTART.md)!

**Want details?** Read [PPO_README.md](PPO_README.md)!

**Happy Training! 🎮🤖**

</div>


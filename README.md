# Block Blast AI - PPO + CNN

A reinforcement learning agent that plays Block Blast using Proximal Policy Optimization (PPO) and Convolutional Neural Networks.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py

# Train a model (1M steps, ~2-4 hours)
python train_ppo_cnn.py --timesteps 1000000

# Evaluate the model
python eval_ppo_cnn.py --episodes 200
```

## 📚 Documentation

- **[Getting Started](docs/QUICKSTART.md)** - 5-minute quick start guide with common commands
- **[Full Documentation](docs/README.md)** - Complete project overview and features
- **[Technical Details](docs/PPO_README.md)** - Architecture, hyperparameters, and troubleshooting
- **[Project Summary](docs/PROJECT_SUMMARY.md)** - Implementation details and design decisions

## 🎮 What This Does

Trains a deep RL agent to play Block Blast using:
- **PPO Algorithm** - Stable, on-policy reinforcement learning
- **CNN Policy** - Learns spatial patterns from raw board state
- **Custom Environment** - Gym-compatible wrapper for Block Blast

## 📊 Core Components

| File | Purpose |
|------|---------|
| `ppo_env.py` | Gymnasium environment wrapper with CNN observations |
| `train_ppo_cnn.py` | PPO training script with custom CNN architecture |
| `eval_ppo_cnn.py` | Model evaluation with comprehensive statistics |
| `test_setup.py` | Setup verification and dependency checking |
| `blockblast/` | Core game logic (rules, pieces, scoring) |

## 🔬 Features

- ✅ CNN-based policy (learns spatial patterns)
- ✅ Stable-Baselines3 integration
- ✅ TensorBoard logging
- ✅ Automatic checkpointing
- ✅ Comprehensive evaluation tools
- ✅ Configurable hyperparameters via CLI

## 📈 Expected Performance

| Training Steps | Avg Score | Time (GPU) |
|---------------|-----------|------------|
| 100k | 50-100 | 15-30 min |
| 1M | 250-400 | 2-4 hours |
| 5M | 400-600+ | 10-20 hours |

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- Gymnasium 0.28+

See `requirements.txt` for full dependencies.

## 📁 Project Structure

```
BlockBlastAICNN/
├── README.md                    # This file
├── requirements.txt             # Dependencies
│
├── ppo_env.py                   # Gym environment
├── train_ppo_cnn.py             # Training script
├── eval_ppo_cnn.py              # Evaluation script
├── test_setup.py                # Setup verification
│
├── blockblast/                  # Game logic
│   ├── core.py                  # Game rules
│   ├── pieces.py                # Piece definitions
│   └── env.py                   # Original environment
│
├── docs/                        # Documentation
├── models/                      # Saved models
├── results/                     # Evaluation results
├── ppo_logs/                    # TensorBoard logs
└── archive/                     # Experimental code & analysis
```

## 🎓 Learn More

For detailed documentation, hyperparameter tuning guides, and troubleshooting:

👉 **See [docs/README.md](docs/README.md)** for the complete documentation

## 👥 Credits

This project combines original RL implementations by Rishabh Goenka with external code:

- **Block Blast AI & RL Training**: Developed by Rishabh Goenka
- **Snake Game (Original)**: By Rajat Dipta Biswas ([snake-pygame](https://github.com/rajatdiptabiswas/snake-pygame))
- **Snake Game RL Wrapper**: Developed by Rishabh Goenka

For complete attribution and licensing information, see **[CREDITS.md](CREDITS.md)**.

## 📜 License

See [CREDITS.md](CREDITS.md) for licensing information for this project and external dependencies.

---

**Ready to train?** Run `python test_setup.py` to get started!


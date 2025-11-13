# Models Directory

This directory contains trained PPO models and checkpoints.

## Directory Structure

```
models/
├── checkpoints/          # Training checkpoints (saved periodically)
├── ppo_checkpoints/      # PPO training checkpoints
├── strategy_tests/       # Models from different training strategies
└── *.zip                 # Trained model files
```

## Model Files

### Main Models

- `ppo_cnn_blockblast.zip` - Primary trained PPO + CNN model
- `rl_warmstart_final.zip` - Final model from RL warmstart training
- `rl_warmstart_best.zip` - Best performing model during warmstart training
- `human_imitation.zip` - Model trained on human demonstrations
- `human_imitation_rlhf.zip` - Model trained with reinforcement learning from human feedback

### Checkpoints

Checkpoint files are saved during training at regular intervals (e.g., every 100k steps):
- `rl_warmstart_100000_steps.zip`
- `rl_warmstart_200000_steps.zip`
- etc.

## Loading Models

To load and evaluate a model:

```bash
python eval_ppo_cnn.py --model-path models/ppo_cnn_blockblast.zip --episodes 200
```

To continue training from a checkpoint:

```bash
python train_ppo_cnn.py --load-model models/checkpoints/rl_warmstart_500000_steps.zip
```

## Model Format

Models are saved in stable-baselines3's `.zip` format, which includes:
- Policy network weights (CNN + MLP)
- Value network weights
- Optimizer state
- Training hyperparameters
- Normalization statistics

## Storage

**Note:** Model files (*.zip, *.pth) are excluded from git via .gitignore due to their size.
Trained models should be shared separately or via cloud storage if needed.


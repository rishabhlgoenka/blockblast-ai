# BlockBlastAICNN - Quick Start Guide

Get up and running with PPO + CNN in 5 minutes!

## 🚀 Setup (2 minutes)

```bash
cd BlockBlastAICNN

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_setup.py
```

You should see:
```
✅ ALL TESTS PASSED!
```

## 🎮 Train Your First Agent (3 minutes)

```bash
# Quick training run (100k timesteps, ~5-10 minutes)
python train_ppo_cnn.py --timesteps 100000
```

**What happens:**
- Agent learns to play Block Blast
- Progress printed every 10 episodes
- Checkpoints saved to `models/ppo_checkpoints/`
- Final model: `models/ppo_cnn_blockblast.zip`

**Expected output:**
```
Episode    10 | Reward:   -15.0 | Score:     8 | Length:   12
Episode    20 | Reward:    25.3 | Score:    42 | Length:   28
Episode    30 | Reward:    67.1 | Score:   128 | Length:   45
...
✓ Model saved to models/ppo_cnn_blockblast.zip
```

## 📊 Evaluate Your Agent

```bash
# Run 200 test episodes
python eval_ppo_cnn.py --episodes 200
```

**Output:**
```
Score Statistics:
  Mean:     156.42
  Std:       78.23
  Min:        15
  Max:       387
  
✓ Results saved to results/ppo_eval.json
```

## 🎯 Next Steps

### 1. Longer Training (Better Performance)

```bash
# Standard training (1M timesteps, ~2-4 hours)
python train_ppo_cnn.py --timesteps 1000000

# Long training (5M timesteps, ~10-20 hours)
python train_ppo_cnn.py --timesteps 5000000
```

### 2. Monitor Training with TensorBoard

```bash
# In another terminal
tensorboard --logdir=ppo_logs

# Open browser to http://localhost:6006
```

### 3. Watch Your Agent Play

```bash
python eval_ppo_cnn.py --episodes 5 --render
```

### 4. Evaluate Checkpoints

```bash
# Evaluate a specific checkpoint
python eval_ppo_cnn.py \
    --model-path models/ppo_checkpoints/ppo_cnn_500000_steps.zip \
    --episodes 100
```

## 🔧 Common Commands Reference

```bash
# Training
python train_ppo_cnn.py --timesteps 1000000
python train_ppo_cnn.py --timesteps 5000000 --learning-rate 0.0001
python train_ppo_cnn.py --timesteps 10000000 --checkpoint-freq 200000

# Evaluation
python eval_ppo_cnn.py --episodes 200
python eval_ppo_cnn.py --model-path models/ppo_cnn_blockblast.zip --episodes 500
python eval_ppo_cnn.py --episodes 10 --render

# Testing
python test_setup.py
```

## ⚙️ Quick Hyperparameter Tuning

### Agent Explores Too Much / Random Play

```bash
python train_ppo_cnn.py --timesteps 1000000 --ent-coef 0.005
```

### Agent Too Cautious / Conservative

```bash
python train_ppo_cnn.py --timesteps 1000000 --ent-coef 0.02
```

### Training Unstable / Loss Spikes

```bash
python train_ppo_cnn.py --timesteps 1000000 --learning-rate 0.0001 --clip-range 0.15
```

### Faster Convergence

```bash
python train_ppo_cnn.py --timesteps 1000000 --learning-rate 0.0005 --batch-size 1024
```

## 📚 Full Documentation

See `PPO_README.md` for:
- Complete architecture details
- Observation/action space explanation
- Reward structure
- Hyperparameter tuning guide
- Troubleshooting

## ❓ FAQ

**Q: How long should I train?**  
A: Start with 1M timesteps (2-4 hours). For best performance, try 5M+ timesteps.

**Q: My agent keeps making invalid moves**  
A: Normal in early training. Should improve after 100k timesteps.

**Q: Training is slow on my CPU**  
A: Expected. PPO+CNN benefits from GPU. Consider reducing `--n-steps 1024 --batch-size 256`.

**Q: How do I compare with the original DQN?**  
A: The original DQN code is in the parent directory, untouched. Train both and compare!

**Q: Can I modify the CNN architecture?**  
A: Yes! Edit `BlockBlastCNN` class in `train_ppo_cnn.py`.

## 🎓 What's Next?

1. ✅ Complete quickstart
2. 📖 Read `PPO_README.md` for deep dive
3. 🔬 Experiment with hyperparameters
4. 📊 Compare with original DQN
5. 🚀 Try alternative algorithms (A2C, SAC)
6. 🎮 Build your own improvements!

---

**Need Help?** Check `PPO_README.md` for detailed documentation and troubleshooting.


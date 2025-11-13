# 🐍 Snake Game RL Agent - START HERE

Welcome! This document will guide you through the complete Snake Reinforcement Learning project.

## 📋 Quick Overview

**What you have:**
- ✅ Original playable Snake game (preserved and working)
- ✅ Complete RL environment wrapper
- ✅ DQN agent with experience replay
- ✅ Training script with metrics tracking
- ✅ Visualization script to watch the AI play
- ✅ Comprehensive documentation

**What it does:**
- Trains an AI agent to play Snake using Deep Q-Learning
- Agent learns from rewards (+10 for food, -10 for death)
- Can achieve scores of 30+ after sufficient training
- Fully visual - you can watch the AI play in real-time

## 🚀 Quick Start (5 Minutes)

### Step 1: Install (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Test (30 seconds)
```bash
python test_environment.py
```

### Step 3: Sanity Check (2 minutes)
```bash
python train_agent.py --sanity_check
```

### Step 4: Watch (if you have a pre-trained model)
```bash
python watch_agent.py --model_path models/dqn_snake_best.pth
```

**All working?** ✅ You're ready to train!

## 📚 Documentation Guide

Choose what you need:

| Document | What's Inside | When to Read |
|----------|--------------|--------------|
| **QUICKSTART.md** | Step-by-step commands | First time using the project |
| **README_RL.md** | Complete documentation | Understanding how everything works |
| **IMPLEMENTATION_SUMMARY.md** | Technical details | Deep dive into the code |
| **COMPARISON.md** | Original vs RL comparison | Understanding the differences |
| **START_HERE.md** (this file) | Navigation guide | Right now! |

## 🎯 Your Training Journey

### Phase 1: Quick Test (2 minutes)
```bash
python train_agent.py --sanity_check
```
**Result:** Confirms everything works, creates test models

### Phase 2: Real Training (4-6 hours)
```bash
python train_agent.py --episodes 10000
```
**Result:** Trained model saved to `models/dqn_snake_best.pth`

### Phase 3: Watch Your Agent (Anytime)
```bash
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 10
```
**Result:** Visual demonstration of learned behavior

### Phase 4 (Optional): Extended Training
```bash
python train_agent.py --episodes 50000
```
**Result:** Even better performance!

## 🎮 Commands Reference

### Playing the Original Game
```bash
cd snake-pygame-master
python "Snake Game.py"
```

### Training Commands
```bash
# Sanity check (10 episodes, ~2 min)
python train_agent.py --sanity_check

# Standard training (10k episodes, ~4-6 hours)
python train_agent.py

# Extended training (20k episodes, ~8-12 hours)
python train_agent.py --episodes 20000

# Training with visual rendering (SLOW, for debugging)
python train_agent.py --episodes 100 --render
```

### Watching Commands
```bash
# Basic watch (best model, 10 episodes)
python watch_agent.py --model_path models/dqn_snake_best.pth

# Slow motion (5 FPS)
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 5

# Fast forward (30 FPS)
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 30

# Watch many episodes
python watch_agent.py --model_path models/dqn_snake_best.pth --episodes 50

# Watch a specific checkpoint
python watch_agent.py --model_path models/dqn_snake_ep5000.pth
```

## 🏗️ Project Structure

```
snakegame-ai/
│
├── 📄 START_HERE.md              ← You are here!
├── 📄 QUICKSTART.md              ← Quick commands reference
├── 📄 README_RL.md               ← Full documentation
├── 📄 IMPLEMENTATION_SUMMARY.md  ← Technical deep dive
├── 📄 COMPARISON.md              ← Original vs RL comparison
│
├── 🐍 snake_env.py               ← Environment wrapper (373 lines)
├── 🧠 dqn_agent.py               ← DQN agent (296 lines)
├── 🎓 train_agent.py             ← Training script (216 lines)
├── 👀 watch_agent.py             ← Visualization script (133 lines)
├── ✅ test_environment.py        ← Quick test script (62 lines)
│
├── 📦 requirements.txt           ← Dependencies
│
├── 📁 snake-pygame-master/
│   ├── 🎮 Snake Game.py         ← Original game (UNCHANGED)
│   └── 📄 README.md             ← Original README
│
└── 📁 models/                    ← Created during training
    ├── dqn_snake_best.pth       ← Best model (use this!)
    ├── dqn_snake_final.pth      ← Final model
    ├── dqn_snake_ep*.pth        ← Periodic checkpoints
    └── training_results.png      ← Training plots
```

## 🎓 What You'll Learn

By exploring this project, you'll understand:

1. **Reinforcement Learning Basics**
   - State representation
   - Action selection
   - Reward design

2. **Deep Q-Networks (DQN)**
   - Neural network architecture
   - Experience replay
   - Target networks
   - Epsilon-greedy exploration

3. **Environment Design**
   - Gym-style interfaces
   - Step/reset methods
   - Observation spaces

4. **Training Strategies**
   - Hyperparameter tuning
   - Checkpoint management
   - Metrics tracking

## 🔧 Technical Summary

### State Space (11 features)
```
[danger_straight, danger_right, danger_left,
 dir_up, dir_down, dir_left, dir_right,
 food_up, food_down, food_left, food_right]
```

### Action Space (3 actions)
```
0 = Turn left (relative to current direction)
1 = Go straight (maintain direction)
2 = Turn right (relative to current direction)
```

### Reward Structure
```
+10  = Eat apple (food)
-10  = Die (collision)
-0.01 = Each step (encourages efficiency)
```

### Neural Network
```
Input (11) → Dense(256) → ReLU → Dense(256) → ReLU → Output (3)
```

### Training Features
- ✅ Experience replay buffer (100k transitions)
- ✅ Target network (updated every 1000 steps)
- ✅ Epsilon decay (1.0 → 0.01)
- ✅ Gradient clipping
- ✅ GPU support (automatic)

## 📊 Expected Performance

| Training Stage | Episodes | Behavior | Avg Score |
|----------------|----------|----------|-----------|
| 🔴 Random | 0-500 | Random exploration | 0-2 |
| 🟡 Learning | 500-2000 | Basic survival | 2-5 |
| 🟢 Improving | 2000-5000 | Strategic play | 5-15 |
| 🔵 Good | 5000-10000 | Strong performance | 15-30 |
| 🟣 Excellent | 10000+ | Near-optimal play | 30-50+ |

## ⚡ Quick Tips

### Before Training
1. ✅ Run `python test_environment.py` first
2. ✅ Use `--sanity_check` to verify setup
3. ✅ Don't use `--render` for long training (too slow!)

### During Training
1. 📊 Monitor the console output
2. 🎯 Score should increase over time
3. 📉 Epsilon should decrease (less exploration)
4. ⏱️ Be patient - training takes hours

### After Training
1. 🏆 Use `dqn_snake_best.pth` (best average score)
2. 👀 Watch with `--fps 10` (good speed)
3. 📊 Check `training_results.png` for plots
4. 🔄 Try longer training for better results

## 🆘 Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame
```

### "No module named 'torch'"
```bash
pip install torch
```

### Training too slow?
- ❌ Don't use `--render` during training
- ✅ Let it run overnight
- ✅ Use GPU if available (automatic)

### Can't find model file?
```bash
# List available models
ls -la models/

# Use full path
python watch_agent.py --model_path /full/path/to/models/dqn_snake_best.pth
```

### Agent not improving?
- Train for more episodes (try 20k)
- Check if epsilon is decaying
- Make sure training isn't interrupted
- Review `training_results.png` for issues

## 🎯 Recommended Workflow

### First Time User (Total: ~10 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test environment
python test_environment.py

# 3. Quick sanity check
python train_agent.py --sanity_check

# 4. Watch the untrained agent (will be bad!)
python watch_agent.py --model_path models_sanity_check/dqn_snake_final.pth --fps 10
```

### Serious Training (Total: overnight)
```bash
# Start training before bed
python train_agent.py --episodes 20000

# Next morning: watch your trained agent!
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 10
```

### Researcher/Developer
```bash
# 1. Read the implementation
cat IMPLEMENTATION_SUMMARY.md

# 2. Review code
cat snake_env.py
cat dqn_agent.py

# 3. Modify hyperparameters in dqn_agent.py
# 4. Train with custom settings
python train_agent.py --episodes 50000 --save_freq 1000

# 5. Compare different checkpoints
python watch_agent.py --model_path models/dqn_snake_ep10000.pth
python watch_agent.py --model_path models/dqn_snake_ep30000.pth
python watch_agent.py --model_path models/dqn_snake_best.pth
```

## 🏆 Success Criteria

You'll know the agent is learning when:

- ✅ Average score increases over episodes
- ✅ Agent survives longer (more steps)
- ✅ Less random behavior (epsilon decreasing)
- ✅ Consistently scores 15+ after 5000 episodes
- ✅ Occasionally scores 30+ after 10000 episodes

## 📞 Need More Help?

Refer to these documents:

1. **Command not working?** → `QUICKSTART.md`
2. **How does it work?** → `README_RL.md`
3. **Want implementation details?** → `IMPLEMENTATION_SUMMARY.md`
4. **Original vs RL differences?** → `COMPARISON.md`

## 🎉 You're Ready!

Everything is set up and documented. Here's your action plan:

```bash
# Step 1: Verify installation
pip install -r requirements.txt
python test_environment.py

# Step 2: Quick test
python train_agent.py --sanity_check

# Step 3: Start real training
python train_agent.py --episodes 10000

# Step 4: Watch your trained agent
python watch_agent.py --model_path models/dqn_snake_best.pth
```

**Happy training! 🚀**

---

*For detailed explanations, see `README_RL.md` or `IMPLEMENTATION_SUMMARY.md`*


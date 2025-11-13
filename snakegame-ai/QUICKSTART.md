# Quick Start Guide - Snake RL Agent

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Environment

Verify everything is working:

```bash
python test_environment.py
```

You should see output showing 5 test episodes with random actions.

### 3. Sanity Check Training (1 minute)

Run a quick 10-episode training to make sure nothing crashes:

```bash
python train_agent.py --sanity_check
```

This will:
- Train for just 10 episodes (very fast)
- Save models to `models_sanity_check/`
- Show you what training output looks like

### 4. Full Training (Recommended: Run Overnight)

Start real training:

```bash
python train_agent.py --episodes 10000
```

Or with custom settings:

```bash
python train_agent.py --episodes 20000 --save_freq 500 --log_freq 50
```

**Training will take several hours depending on your hardware.**

Progress will be logged every 10 episodes by default. You'll see:
```
Episode 10/10000 | Score: 1 | Avg Score: 0.50 | Epsilon: 0.904 | Loss: 0.0234
Episode 20/10000 | Score: 0 | Avg Score: 0.45 | Epsilon: 0.818 | Loss: 0.0456
...
```

Models are saved to the `models/` directory:
- `dqn_snake_best.pth` - Best performing model (use this one!)
- `dqn_snake_final.pth` - Final model after all episodes
- `dqn_snake_ep{N}.pth` - Checkpoints every N episodes

### 5. Watch Your Trained Agent Play

After training completes, watch it play:

```bash
python watch_agent.py --model_path models/dqn_snake_best.pth
```

Adjust game speed:

```bash
# Slow motion (5 FPS)
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 5

# Fast forward (30 FPS)
python watch_agent.py --model_path models/dqn_snake_best.pth --fps 30
```

Watch more episodes:

```bash
python watch_agent.py --model_path models/dqn_snake_best.pth --episodes 20
```

### 6. Play the Original Game (Human)

The original game is still playable:

```bash
cd snake-pygame-master
python "Snake Game.py"
```

Use arrow keys or WASD to control the snake.

## Quick Command Reference

| Command | Purpose |
|---------|---------|
| `python test_environment.py` | Test environment setup |
| `python train_agent.py --sanity_check` | Quick 10-episode test |
| `python train_agent.py` | Full training (10k episodes) |
| `python train_agent.py --episodes 5000` | Custom episode count |
| `python watch_agent.py --model_path models/dqn_snake_best.pth` | Watch best model |
| `python watch_agent.py --model_path models/dqn_snake_best.pth --fps 5` | Watch in slow motion |

## Expected Training Timeline

- **Minutes 1-10**: Random behavior, scores mostly 0-1
- **Minutes 10-30**: Starting to learn, occasional score of 2-3
- **Hour 1-2**: Better survival, scores 3-8
- **Hour 2-4**: Decent strategy, scores 8-15
- **Hour 4+**: Good performance, scores 15-30+

## Troubleshooting

**"No module named 'pygame'"**
```bash
pip install pygame
```

**Training too slow?**
- Don't use `--render` flag during training
- Make sure you're not running other heavy programs

**Want to stop training early?**
- Press Ctrl+C
- Your progress is saved! Use the latest checkpoint in `models/`

**Want to resume training?**
- The current implementation trains from scratch
- To continue, you'd need to load a model in `train_agent.py` (see `agent.load()` method)

## What's Next?

After successful training:

1. Compare different checkpoints:
   ```bash
   python watch_agent.py --model_path models/dqn_snake_ep1000.pth
   python watch_agent.py --model_path models/dqn_snake_ep5000.pth
   python watch_agent.py --model_path models/dqn_snake_final.pth
   ```

2. Check training plots: Open `models/training_results.png`

3. Try longer training for better performance:
   ```bash
   python train_agent.py --episodes 50000
   ```

4. Experiment with hyperparameters in `dqn_agent.py`:
   - Learning rate
   - Hidden layer size
   - Epsilon decay rate
   - Batch size

## Need Help?

See `README_RL.md` for detailed documentation on:
- State representation
- Action space
- Reward structure
- DQN architecture
- Advanced usage


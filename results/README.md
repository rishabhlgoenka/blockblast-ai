# Results Directory

This directory stores evaluation results and performance metrics.

## Files

- `ppo_eval.json` - Evaluation statistics from trained PPO models

## Evaluation Results Format

Evaluation results are saved as JSON files with the following structure:

```json
{
  "model_path": "path/to/model.zip",
  "num_episodes": 200,
  "seed": 42,
  "mean_score": 450.5,
  "std_score": 120.3,
  "min_score": 150,
  "max_score": 750,
  "median_score": 425,
  "percentile_25": 350,
  "percentile_75": 550,
  "mean_reward": 445.2,
  "mean_episode_length": 85.5,
  "mean_lines_cleared": 12.3,
  "timestamp": "2024-11-13T10:30:00"
}
```

## Generating Results

Run the evaluation script to generate new results:

```bash
python eval_ppo_cnn.py --episodes 200 --save-results results/my_eval.json
```

## Analysis

Results can be loaded and analyzed using Python:

```python
import json

with open('results/ppo_eval.json', 'r') as f:
    results = json.load(f)

print(f"Mean score: {results['mean_score']:.1f}")
print(f"Std dev: {results['std_score']:.1f}")
```

## Comparing Models

To compare different models, evaluate each and save to separate files:

```bash
python eval_ppo_cnn.py --model-path models/model_a.zip --episodes 200
python eval_ppo_cnn.py --model-path models/model_b.zip --episodes 200
```

Then compare the JSON results to determine which model performs better.


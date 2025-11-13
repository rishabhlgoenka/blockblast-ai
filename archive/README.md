# Archive Directory

This directory contains experimental code, old implementations, and analysis documents from the development process.

## Structure

### experiments/
Experimental scripts used during development and research:

- `expert_agent_v2.py` / `expert_agent.py` / `expert_solver.py` - Various expert agent implementations
- `analyze_pieces.py` - Analysis of piece distribution and difficulty
- `comprehensive_test.py` - Comprehensive testing suite
- `eval_checkpoints.py` - Checkpoint evaluation utilities
- `evaluate_model.py` - Model evaluation scripts
- `fix_piece_generation.py` / `fix_smart_piece_generation.py` - Piece generation improvements
- `human_play_gui.py` / `human_play.py` - GUI for human gameplay
- `manual_rlhf.py` / `rlhf_gui.py` - Reinforcement Learning from Human Feedback experiments
- `test_curriculum_strategies.py` - Curriculum learning strategy tests
- `test_smart_generation.py` / `test_weighted_pieces.py` - Piece generation testing
- `train_from_human.py` - Training from human demonstrations
- `train_rl_warmstart.py` - RL warmstart training experiments
- `train_with_expert.py` - Training with expert guidance
- `verify_test.py` - Verification utilities

### docs/
Analysis and investigation documents:

- `FINAL_RESULTS_ANALYSIS.md` - Final results from experiments
- `INVESTIGATION_SUMMARY.md` - Summary of investigations
- `PIECE_ANALYSIS_FINDINGS.md` - Findings from piece analysis
- `TRAINING_SUMMARY.md` - Training process summary

### data/
Training data and demonstrations:

- `human_demonstrations.pkl` - Human gameplay demonstrations
- `human_feedback.pkl` - Human feedback data
- `expert_data/` - Expert agent gameplay data

### logs/
Training and evaluation logs:

- `baseline_evaluation.log`
- `comprehensive_test_results.log`
- `expert_evaluation.log`
- `expert_training_full.log`
- `human_training_CORRECT.log`
- `rl_warmstart_evaluation.log`

### models/
Old or experimental model checkpoints

## Status

The code in this directory is **archived** and may not be maintained. It is kept for:

1. **Historical reference** - Understanding the development process
2. **Experimental results** - Documenting what was tried
3. **Analysis** - Research findings and insights

## Active Development

For current, maintained code, see the root directory:

- `train_ppo_cnn.py` - Current training implementation
- `eval_ppo_cnn.py` - Current evaluation implementation
- `ppo_env.py` - Current environment implementation
- `blockblast/` - Core game logic

## Note

If you're looking for working, tested code, **do not use the files in this archive**.
Use the files in the root directory instead.


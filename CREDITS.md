# Credits and Attribution

This project combines original work with external code and game implementations. This document provides proper attribution for all external sources.

## Block Blast AI Implementation

**Primary Developer:** Rishabh Goenka

The reinforcement learning implementation, training infrastructure, and environment wrappers for Block Blast were developed by Rishabh Goenka. This includes:

- PPO + CNN training implementation (`train_ppo_cnn.py`, `ppo_env.py`, `eval_ppo_cnn.py`)
- Custom CNN feature extractor for Block Blast observations
- Block Blast game logic implementation (`blockblast/` package)
- Comprehensive documentation and training guides
- Evaluation tools and analysis scripts

## Snake Game - External Code

### Original Snake Game

**Project:** Snake Eater  
**Author:** Rajat Dipta Biswas  
**Source:** [https://github.com/rajatdiptabiswas/snake-pygame](https://github.com/rajatdiptabiswas/snake-pygame)  
**Location in this repo:** `snakegame-ai/snake-pygame-master/`

The original Snake game implementation (in `snake-pygame-master/Snake Game.py`) is preserved unchanged from the original repository. This provides a human-playable version of the classic Snake game built with Pygame.

**Acknowledgements from original author:**
- [Pygame Documentation](https://www.pygame.org/docs/)
- Udemy: Python Game Development course

### Snake Game RL Wrapper

**Developer:** Rishabh Goenka

The reinforcement learning wrapper and DQN agent for the Snake game were developed specifically for this project:

- `snakegame-ai/snake_env.py` - Gym-style environment wrapper
- `snakegame-ai/dqn_agent.py` - Deep Q-Network implementation
- `snakegame-ai/train_agent.py` - Training script
- `snakegame-ai/watch_agent.py` - Visualization script
- `snakegame-ai/test_environment.py` - Testing utilities
- All documentation files (README_RL.md, START_HERE.md, QUICKSTART.md, etc.)

## Third-Party Libraries and Frameworks

This project builds upon several open-source libraries:

- **Stable-Baselines3** - Reinforcement learning algorithms (PPO implementation)
- **PyTorch** - Deep learning framework
- **Gymnasium (OpenAI Gym)** - RL environment interface standard
- **Pygame** - Game development library
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization and plotting

## Algorithms and Techniques

### Proximal Policy Optimization (PPO)

The PPO algorithm implementation is provided by Stable-Baselines3, based on:

**Paper:** "Proximal Policy Optimization Algorithms"  
**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  
**Year:** 2017  
**Link:** [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

### Deep Q-Network (DQN)

The DQN algorithm used in the Snake game agent is based on:

**Paper:** "Human-level control through deep reinforcement learning"  
**Authors:** Volodymyr Mnih et al.  
**Year:** 2015  
**Published in:** Nature  
**Link:** [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)

## Block Blast Game

The Block Blast game logic implemented in `blockblast/` is inspired by the popular mobile game "Block Blast." The implementation includes:

- 8×8 game board
- 30 piece types (single blocks, lines, L-shapes, T-shapes, squares, plus signs)
- Line clearing mechanics (rows and columns)
- Combo system
- Original scoring algorithm

This is an independent implementation created for educational and research purposes in reinforcement learning.

## License

### This Project

The reinforcement learning implementations, training scripts, and documentation created by Rishabh Goenka are provided for educational and research purposes.

### External Code

- **Snake Game (snake-pygame-master/):** Refer to the original repository for licensing information
- **Stable-Baselines3:** MIT License
- **PyTorch:** BSD-style License
- **Gymnasium:** MIT License
- **Pygame:** LGPL License

## Contact

For questions about the RL implementations or this project:
- Primary Developer: Rishabh Goenka

For questions about the original Snake game:
- Original Author: Rajat Dipta Biswas
- Repository: [https://github.com/rajatdiptabiswas/snake-pygame](https://github.com/rajatdiptabiswas/snake-pygame)

---

*Last Updated: November 2024*


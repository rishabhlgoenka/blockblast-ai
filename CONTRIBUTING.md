# Contributing to BlockBlastAICNN

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BlockBlastAICNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python test_setup.py
```

## Project Structure

```
BlockBlastAICNN/
├── blockblast/           # Core game logic
│   ├── core.py          # Game rules and state management
│   ├── pieces.py        # Piece definitions (30 piece types)
│   ├── env.py           # Original environment wrapper
│   └── pygame_app.py    # GUI for human play
│
├── ppo_env.py           # Gymnasium-compatible environment
├── train_ppo_cnn.py     # PPO training script
├── eval_ppo_cnn.py      # Model evaluation script
├── test_setup.py        # Setup verification
│
├── snakegame-ai/        # Snake game RL implementation
│
├── docs/                # Documentation
├── models/              # Saved models and checkpoints
├── results/             # Evaluation results
├── archive/             # Experimental and archived code
└── ppo_logs/            # TensorBoard logs
```

## Code Style

### Python

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible
- Use meaningful variable and function names

### Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters (flexible for readability)
- Use double quotes for strings
- Add blank lines between functions and classes

### Example

```python
def calculate_score(blocks_placed: int, lines_cleared: int) -> int:
    """
    Calculate score for a move.
    
    Args:
        blocks_placed: Number of blocks in the placed piece
        lines_cleared: Number of lines cleared
    
    Returns:
        Score earned for this move
    """
    score = blocks_placed
    if lines_cleared > 0:
        bonus = LINE_CLEAR_BONUS[min(lines_cleared - 1, 5)]
        score += bonus
    return score
```

## Making Changes

### Branch Naming

- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Refactor: `refactor/description`

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Longer explanation if needed...

- Detail 1
- Detail 2
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(training): add curriculum learning strategy
fix(env): correct reward calculation for combos
docs(readme): update installation instructions
```

## Testing

### Running Tests

```bash
# Verify setup
python test_setup.py

# Test environment
python -c "from ppo_env import BlockBlastEnv; env = BlockBlastEnv(); print('OK')"
```

### Before Submitting

1. Run the setup test: `python test_setup.py`
2. Check for obvious errors
3. Test your changes with a short training run if applicable
4. Update documentation if you changed functionality

## Adding Features

### New Reward Function

1. Modify reward calculation in `ppo_env.py`
2. Document the change in code comments
3. Test with a short training run
4. Update `docs/` if it's a major change

### New Training Strategy

1. Add as a new script or modify `train_ppo_cnn.py`
2. Document hyperparameters and expected behavior
3. Run experiments and document results
4. Consider adding to `archive/experiments/` if experimental

### New Environment Feature

1. Update `blockblast/core.py` for game logic changes
2. Update `ppo_env.py` for observation/action changes
3. Verify with `test_setup.py`
4. Update CNN architecture if observation shape changes

## Documentation

### Required Documentation

- **Docstrings**: All public functions, classes, and methods
- **README updates**: For new features or changed workflows
- **Code comments**: For complex logic or non-obvious decisions
- **Type hints**: For function arguments and returns

### Documentation Style

```python
def function_name(arg1: int, arg2: str) -> bool:
    """
    One-line summary of what the function does.
    
    More detailed explanation if needed. Explain the purpose,
    not just what the code does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When this happens
    
    Example:
        >>> function_name(42, "test")
        True
    """
```

## Pull Request Process

1. **Create a branch** from `main` for your changes
2. **Make your changes** following the code style guidelines
3. **Test your changes** thoroughly
4. **Update documentation** as needed
5. **Submit PR** with clear description of changes
6. **Address review feedback** promptly

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Changes Made
- Change 1
- Change 2

## Testing
How were the changes tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No new warnings or errors
```

## External Code

### Attribution

When incorporating external code:

1. Add to `CREDITS.md` with source and license
2. Keep original copyright notices
3. Document modifications made
4. Ensure license compatibility

### Snake Game

The snake game in `snakegame-ai/snake-pygame-master/` is external code.
See `CREDITS.md` for attribution.

## Questions or Issues?

- Check existing documentation in `docs/`
- Look for similar issues in the issue tracker
- Review `CREDITS.md` for external code questions

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project
- Show empathy towards other contributors
- Accept constructive criticism gracefully

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to BlockBlastAICNN! 🎉


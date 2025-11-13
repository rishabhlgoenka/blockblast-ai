"""
Block Blast Game - Python implementation with pygame and RL environment.

This package provides:
- Core game logic (rules, pieces, scoring)
- RL environment wrapper
- Pygame-based GUI for human play
"""

__version__ = "1.0.0"
__author__ = "Rishabh Goenka"

from blockblast.core import BlockBlastGame, GameState, Action
from blockblast.pieces import Piece, get_piece, ALL_PIECES

__all__ = [
    "BlockBlastGame",
    "GameState",
    "Action",
    "Piece",
    "get_piece",
    "ALL_PIECES",
]


"""
Piece definitions and helpers for Block Blast game.

Each piece is represented as a 5x5 binary matrix where 1 indicates a filled cell.
"""
from typing import List, Tuple
import numpy as np

# Type alias for piece shape
PieceShape = List[List[int]]


class Piece:
    """Represents a game piece with its shape and metadata."""
    
    def __init__(self, piece_id: int, shape: PieceShape, priority: int = 1):
        """
        Initialize a piece.
        
        Args:
            piece_id: Unique identifier for this piece type
            shape: 5x5 binary matrix representing the piece
            priority: Priority/weight for random selection (unused in simplified version)
        """
        self.id = piece_id
        self.shape = np.array(shape, dtype=np.int8)
        self.priority = priority
        
        # Calculate bounding box (for efficient collision detection)
        rows, cols = np.where(self.shape == 1)
        if len(rows) > 0:
            self.min_row = rows.min()
            self.max_row = rows.max()
            self.min_col = cols.min()
            self.max_col = cols.max()
            self.width = self.max_col - self.min_col + 1
            self.height = self.max_row - self.min_row + 1
        else:
            self.min_row = self.max_row = self.min_col = self.max_col = 0
            self.width = self.height = 0
    
    def get_cells(self) -> List[Tuple[int, int]]:
        """
        Get list of relative (row, col) coordinates of filled cells.
        Coordinates are relative to the top-left of the 5x5 matrix.
        
        Returns:
            List of (row, col) tuples for filled cells
        """
        rows, cols = np.where(self.shape == 1)
        return list(zip(rows.tolist(), cols.tolist()))
    
    def get_block_count(self) -> int:
        """Get the number of blocks in this piece."""
        return int(np.sum(self.shape))
    
    def __repr__(self) -> str:
        return f"Piece(id={self.id}, blocks={self.get_block_count()})"


# All 30 piece definitions from the original game
# Each is a 5x5 matrix. Priority values are from the original game.
PIECE_DEFINITIONS = [
    # 0: Single block
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 1
    },
    # 1: Vertical line (4 blocks)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 16
    },
    # 2: Horizontal line (4 blocks)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0]],
        "priority": 16
    },
    # 3: L-shape (bottom-left)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 1, 0, 0]],
        "priority": 4
    },
    # 4: L-shape (bottom-right)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 1, 0]],
        "priority": 4
    },
    # 5: T-shape (pointing down)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 4
    },
    # 6: T-shape variant
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 0]],
        "priority": 4
    },
    # 7: T-shape (pointing up)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [1, 1, 1, 0, 0]],
        "priority": 4
    },
    # 8: T-shape variant
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 1, 1]],
        "priority": 4
    },
    # 9: L-shape (top-left)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 4
    },
    # 10: L-shape (top-right)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 4
    },
    # 11: Cross/Plus shape
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 8
    },
    # 12: T-shape (pointing right)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0]],
        "priority": 8
    },
    # 13: Cross variant
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 8
    },
    # 14: T-shape (pointing left)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 8
    },
    # 15: 2x2 square
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0]],
        "priority": 16
    },
    # 16: Z-shape (left)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 8
    },
    # 17: Z-shape variant
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 8
    },
    # 18: S-shape (horizontal)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]],
        "priority": 8
    },
    # 19: Z-shape (horizontal)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0]],
        "priority": 8
    },
    # 20: Long L (bottom-left)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [1, 1, 1, 0, 0]],
        "priority": 5
    },
    # 21: Long L (bottom-right)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 1, 1]],
        "priority": 5
    },
    # 22: Long L (top-left)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 5
    },
    # 23: Long L (top-right)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 5
    },
    # 24: Vertical line (5 blocks)
    {
        "shape": [[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]],
        "priority": 6
    },
    # 25: Horizontal line (5 blocks)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1]],
        "priority": 6
    },
    # 26: Vertical 2x3 rectangle
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0]],
        "priority": 6
    },
    # 27: 3x2 rectangle (horizontal)
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0]],
        "priority": 6
    },
    # 28: 3x3 square
    {
        "shape": [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0]],
        "priority": 6
    },
]


def create_all_pieces() -> List[Piece]:
    """Create all 30 game pieces."""
    return [
        Piece(piece_id=i, shape=piece_def["shape"], priority=piece_def["priority"])
        for i, piece_def in enumerate(PIECE_DEFINITIONS)
    ]


# Create global piece library
ALL_PIECES = create_all_pieces()


def get_piece(piece_id: int) -> Piece:
    """Get a piece by its ID."""
    if 0 <= piece_id < len(ALL_PIECES):
        return ALL_PIECES[piece_id]
    raise ValueError(f"Invalid piece ID: {piece_id}")


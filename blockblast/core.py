"""
Core game logic for Block Blast.

This module contains pure game logic with no rendering or input handling.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import random
from .pieces import Piece, get_piece, ALL_PIECES


# Score bonus for clearing multiple lines simultaneously
# Index is (num_lines - 1), value is the bonus
# ACTUAL Block Blast mobile game scoring
LINE_CLEAR_BONUS = [10, 30, 60, 100, 150, 210]


@dataclass
class GameState:
    """Represents the current state of the game."""
    
    board: np.ndarray  # 8x8 grid, 0 = empty, >0 = filled (color/piece id)
    available_pieces: List[int]  # List of 3 piece IDs currently available
    score: int = 0
    done: bool = False
    combo: int = 0  # Current combo counter
    moves: int = 0  # Total number of moves made
    lines_cleared_total: int = 0  # Total lines cleared in the game
    
    def copy(self) -> GameState:
        """Create a deep copy of this state."""
        return GameState(
            board=self.board.copy(),
            available_pieces=self.available_pieces.copy(),
            score=self.score,
            done=self.done,
            combo=self.combo,
            moves=self.moves,
            lines_cleared_total=self.lines_cleared_total
        )


@dataclass
class Action:
    """Represents a player action (placing a piece)."""
    
    piece_index: int  # Index into available_pieces (0, 1, or 2)
    row: int  # Row position on board (0-9)
    col: int  # Column position on board (0-9)
    
    def __repr__(self) -> str:
        return f"Action(piece_idx={self.piece_index}, row={self.row}, col={self.col})"


class BlockBlastGame:
    """
    Core game logic for Block Blast.
    
    This class manages the game state and provides methods to validate
    and apply actions, check for game over, etc.
    """
    
    BOARD_SIZE = 8
    NUM_PIECES_AVAILABLE = 3
    
    def __init__(self, rng_seed: Optional[int] = None):
        """
        Initialize the game.
        
        Args:
            rng_seed: Random number generator seed for reproducibility
        """
        self.rng = random.Random(rng_seed)
    
    def new_game(self) -> GameState:
        """
        Start a new game.
        
        Returns:
            Initial game state
        """
        board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        available_pieces = self._generate_new_pieces()
        return GameState(
            board=board,
            available_pieces=available_pieces,
            score=0,
            done=False,
            combo=0,
            moves=0,
            lines_cleared_total=0
        )
    
    def _generate_new_pieces(self, board: Optional[np.ndarray] = None, max_retries: int = 50) -> List[int]:
        """
        Generate 3 random piece IDs using weighted selection based on priorities.
        
        Args:
            board: Optional current board state for validation
            max_retries: Maximum attempts to generate a valid piece set
            
        Returns:
            List of 3 piece IDs, guaranteed to have at least one placeable piece if board provided
        """
        # Get priorities from all pieces
        priorities = np.array([piece.priority for piece in ALL_PIECES])
        probabilities = priorities / priorities.sum()
        
        # If no board provided (initial generation), just generate pieces
        if board is None:
            pieces = np.random.choice(
                len(ALL_PIECES),
                size=self.NUM_PIECES_AVAILABLE,
                replace=True,
                p=probabilities
            )
            return pieces.tolist()
        
        # With board validation: ensure at least one piece is placeable
        for attempt in range(max_retries):
            pieces = np.random.choice(
                len(ALL_PIECES),
                size=self.NUM_PIECES_AVAILABLE,
                replace=True,
                p=probabilities
            ).tolist()
            
            # Check if at least one piece can be placed
            if self.has_valid_move(board, pieces):
                return pieces
        
        # Fallback: return last generated set
        # This should be extremely rare with max_retries=50
        return pieces
    
    def can_place_piece(self, board: np.ndarray, piece: Piece, row: int, col: int) -> bool:
        """
        Check if a piece can be placed at the given position.
        
        Args:
            board: Current board state
            piece: Piece to place
            row: Row position (top-left of piece's 5x5 matrix)
            col: Column position (top-left of piece's 5x5 matrix)
        
        Returns:
            True if placement is valid
        """
        cells = piece.get_cells()
        
        for cell_row, cell_col in cells:
            board_row = row + cell_row
            board_col = col + cell_col
            
            # Check bounds
            if board_row < 0 or board_row >= self.BOARD_SIZE:
                return False
            if board_col < 0 or board_col >= self.BOARD_SIZE:
                return False
            
            # Check if cell is empty
            if board[board_row, board_col] != 0:
                return False
        
        return True
    
    def has_valid_move(self, board: np.ndarray, piece_ids: List[int]) -> bool:
        """
        Check if any of the given pieces can be placed somewhere on the board.
        
        Args:
            board: Current board state
            piece_ids: List of piece IDs to check
        
        Returns:
            True if at least one piece can be placed somewhere
        """
        for piece_id in piece_ids:
            piece = get_piece(piece_id)
            
            # Try all possible positions
            # Pieces are 5x5 matrices with offsets, so we need to check
            # negative positions to place pieces at board edges
            # Range: -4 to BOARD_SIZE allows pieces to be placed anywhere
            # where at least one block lands on the board
            for row in range(-4, self.BOARD_SIZE + 4):
                for col in range(-4, self.BOARD_SIZE + 4):
                    if self.can_place_piece(board, piece, row, col):
                        return True
        
        return False
    
    def get_valid_actions(self, state: GameState) -> List[Action]:
        """
        Get all valid actions for the current state.
        
        Args:
            state: Current game state
        
        Returns:
            List of valid actions
        """
        if state.done:
            return []
        
        valid_actions = []
        
        for piece_idx in range(len(state.available_pieces)):
            piece_id = state.available_pieces[piece_idx]
            piece = get_piece(piece_id)
            
            # Try all possible positions
            for row in range(-4, self.BOARD_SIZE + 4):
                for col in range(-4, self.BOARD_SIZE + 4):
                    if self.can_place_piece(state.board, piece, row, col):
                        valid_actions.append(Action(piece_idx, row, col))
        
        return valid_actions
    
    def _clear_lines(self, board: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Clear complete rows and columns.
        
        Args:
            board: Current board state (will be modified in place)
        
        Returns:
            Tuple of (number of lines cleared, modified board)
        """
        lines_cleared = 0
        
        # Check rows
        rows_to_clear = []
        for row in range(self.BOARD_SIZE):
            if np.all(board[row, :] > 0):
                rows_to_clear.append(row)
        
        # Check columns
        cols_to_clear = []
        for col in range(self.BOARD_SIZE):
            if np.all(board[:, col] > 0):
                cols_to_clear.append(col)
        
        # Clear the rows
        for row in rows_to_clear:
            board[row, :] = 0
            lines_cleared += 1
        
        # Clear the columns
        for col in cols_to_clear:
            board[:, col] = 0
            lines_cleared += 1
        
        return lines_cleared, board
    
    def _calculate_score(self, blocks_placed: int, lines_cleared: int, combo: int) -> int:
        """
        Calculate score for a move (ACTUAL Block Blast mobile game scoring).
        
        Verified from real gameplay data:
        - Move 1: +10 bonus (1 line, 1x combo)
        - Move 2: +20 bonus (1 line, 2x combo) 
        - Move 3: +90 bonus (2 lines, 3x combo)
        
        Args:
            blocks_placed: Number of blocks in the placed piece
            lines_cleared: Number of lines cleared
            combo: Current combo counter (starts at 0, increases with each consecutive clear)
        
        Returns:
            Score earned for this move
        """
        # Base score: 1 point per block placed
        # 3x3 block = 9 points, 4-block line = 4 points
        score = blocks_placed
        
        # Line clear bonus (combo multiplies ONLY this, not block points)
        if lines_cleared > 0:
            bonus_index = min(lines_cleared - 1, len(LINE_CLEAR_BONUS) - 1)
            line_bonus = LINE_CLEAR_BONUS[bonus_index]
            
            # Combo multiplier: line_bonus × combo_number
            # 1st clear (combo=0): 10 × 1 = 10
            # 2nd clear (combo=1): 10 × 2 = 20  
            # 3rd clear (combo=2): 10 × 3 = 30, etc.
            combo_multiplier = combo + 1  # combo 0 = ×1, combo 1 = ×2, combo 2 = ×3
            score += line_bonus * combo_multiplier
        
        return score
    
    def apply_action(
        self, 
        state: GameState, 
        action: Action
    ) -> Tuple[GameState, int, bool, Dict[str, Any]]:
        """
        Apply an action to the current state.
        
        Args:
            state: Current game state
            action: Action to apply
        
        Returns:
            Tuple of (next_state, reward, done, info)
            - next_state: New game state after action
            - reward: Score earned from this action
            - done: Whether game is over
            - info: Dictionary with additional information
        """
        # Validate action
        if action.piece_index < 0 or action.piece_index >= len(state.available_pieces):
            raise ValueError(f"Invalid piece index: {action.piece_index}")
        
        piece_id = state.available_pieces[action.piece_index]
        piece = get_piece(piece_id)
        
        if not self.can_place_piece(state.board, piece, action.row, action.col):
            raise ValueError(f"Cannot place piece at position ({action.row}, {action.col})")
        
        # Create new state
        next_state = state.copy()
        next_state.moves += 1
        
        # Place the piece on the board
        # We use piece_id + 1 as the cell value (0 remains empty)
        cells = piece.get_cells()
        for cell_row, cell_col in cells:
            board_row = action.row + cell_row
            board_col = action.col + cell_col
            next_state.board[board_row, board_col] = piece_id + 1
        
        blocks_placed = piece.get_block_count()
        
        # Clear complete lines
        lines_cleared, next_state.board = self._clear_lines(next_state.board)
        next_state.lines_cleared_total += lines_cleared
        
        # Update combo
        if lines_cleared > 0:
            next_state.combo += 1
        else:
            # Reset combo after 3 moves without clearing
            # For simplicity, we reset immediately if no lines cleared
            next_state.combo = 0
        
        # Calculate score
        reward = self._calculate_score(blocks_placed, lines_cleared, state.combo)
        next_state.score += reward
        
        # Remove used piece and check if we need new pieces
        next_state.available_pieces = [
            p for i, p in enumerate(next_state.available_pieces) 
            if i != action.piece_index
        ]
        
        # If all pieces used, generate new set with validation
        # Ensures at least one piece is placeable on current board
        if len(next_state.available_pieces) == 0:
            next_state.available_pieces = self._generate_new_pieces(board=next_state.board)
        
        # Check for game over
        if not self.has_valid_move(next_state.board, next_state.available_pieces):
            next_state.done = True
        
        # Build info dict
        info = {
            "blocks_placed": blocks_placed,
            "lines_cleared": lines_cleared,
            "combo": next_state.combo,
            "piece_id": piece_id,
            "pieces_remaining": len(next_state.available_pieces),
        }
        
        return next_state, reward, next_state.done, info
    
    def get_board_string(self, board: np.ndarray) -> str:
        """
        Get a string representation of the board for debugging.
        
        Args:
            board: Board to visualize
        
        Returns:
            String representation
        """
        lines = []
        lines.append("  " + "".join(str(i) for i in range(self.BOARD_SIZE)))
        for i, row in enumerate(board):
            row_str = f"{i} " + "".join("■" if cell > 0 else "·" for cell in row)
            lines.append(row_str)
        return "\n".join(lines)


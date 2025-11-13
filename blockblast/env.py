"""
RL-style environment wrapper for Block Blast.

Observation Format:
    {
        "board": np.ndarray of shape (8, 8), dtype=np.float32
                 0 for empty cells, 1 for filled cells
        "pieces": np.ndarray of shape (3, 5, 5), dtype=np.float32
                  Binary masks for the 3 available pieces
                  1 for occupied cells, 0 for empty padding
    }

Action Encoding:
    Discrete action space with 507 actions (3 pieces × 13 × 13 positions)
    action_id = piece_idx * (GRID_SIZE * GRID_SIZE) + (y+4) * GRID_SIZE + (x+4)
    where:
        piece_idx ∈ {0, 1, 2}
        x, y ∈ {-4..8} (top-left position of piece's 5x5 matrix relative to board)
    
    The extended range allows pieces to be placed at board edges where blocks
    may extend outside the position coordinate due to piece shape offsets.

Reward Definition (Improved for Better Learning):
    - Base reward: score_after - score_before (game's internal scoring)
    - Line clear bonus: +LINE_CLEAR_BONUS per line cleared (encourages clearing)
    - Survival bonus: +SURVIVAL_BONUS per valid move (encourages longer games)
    - Game over penalty: GAME_OVER_PENALTY if game ends (discourages premature death)
    - Invalid action: -1.0 penalty, state unchanged

This environment wraps the existing core game logic and provides a clean
RL interface separate from the pygame UI.
"""

# Reward shaping constants (easily tunable)
LINE_CLEAR_BONUS = 10.0     # Bonus per line cleared (INCREASED - prioritize line clearing!)
SURVIVAL_BONUS = 0.0        # Removed - don't reward just surviving
GAME_OVER_PENALTY = -20.0   # Penalty when game ends (negative to discourage)
INVALID_ACTION_PENALTY = -1.0  # Penalty for invalid moves
from typing import Dict, Optional, Tuple, Any
import numpy as np
from .core import BlockBlastGame, GameState, Action
from .pieces import get_piece


# Action space constants
NUM_PIECES = 3
BOARD_W = 8
BOARD_H = 8
GRID_SIZE = 13  # Range from -4 to 8 (inclusive) = 13 positions
POS_OFFSET = 4  # Offset to convert -4..8 to 0..12
NUM_ACTIONS = NUM_PIECES * GRID_SIZE * GRID_SIZE  # 507


def encode_action(piece_idx: int, x: int, y: int) -> int:
    """
    Encode a (piece_idx, x, y) tuple into a discrete action ID.
    
    Args:
        piece_idx: Index of piece to place (0, 1, or 2)
        x: Column position on board (-4 to 8)
        y: Row position on board (-4 to 8)
    
    Returns:
        action_id in range [0, 507)
    
    Example:
        >>> encode_action(0, 0, 0)  # piece 0 at position (0, 0)
        52
    """
    x_offset = x + POS_OFFSET  # Convert -4..8 to 0..12
    y_offset = y + POS_OFFSET  # Convert -4..8 to 0..12
    return piece_idx * (GRID_SIZE * GRID_SIZE) + y_offset * GRID_SIZE + x_offset


def decode_action(action_id: int) -> Tuple[int, int, int]:
    """
    Decode a discrete action ID into (piece_idx, x, y).
    
    Args:
        action_id: Integer action ID in range [0, 507)
    
    Returns:
        Tuple of (piece_idx, x, y) where x, y are in range [-4, 8]
    
    Example:
        >>> decode_action(52)
        (0, 0, 0)
    """
    piece_idx = action_id // (GRID_SIZE * GRID_SIZE)
    remainder = action_id % (GRID_SIZE * GRID_SIZE)
    y_offset = remainder // GRID_SIZE
    x_offset = remainder % GRID_SIZE
    x = x_offset - POS_OFFSET  # Convert 0..12 to -4..8
    y = y_offset - POS_OFFSET  # Convert 0..12 to -4..8
    return piece_idx, x, y


def state_to_obs(state: GameState) -> Dict[str, np.ndarray]:
    """
    Convert internal GameState to RL-friendly observation dict.
    
    Args:
        state: Current game state
    
    Returns:
        Dictionary with:
            - "board": (8, 8) float32 array, 0=empty, 1=filled
            - "pieces": (3, 5, 5) float32 array, binary piece masks
    """
    # Convert board to float32, binarize (any non-zero value becomes 1)
    board = (state.board > 0).astype(np.float32)
    
    # Get piece shapes as float32 arrays
    pieces = np.zeros((3, 5, 5), dtype=np.float32)
    for i, piece_id in enumerate(state.available_pieces):
        if i < 3:  # Only take first 3 pieces
            piece = get_piece(piece_id)
            pieces[i] = piece.shape.astype(np.float32)
    
    return {
        "board": board,
        "pieces": pieces,
    }


class BlockBlastEnv:
    """
    Gym-like RL environment for Block Blast game.
    
    This environment wraps the existing core game logic (in blockblast.core)
    and provides a clean interface for RL training. The pygame UI remains
    separate and unchanged.
    
    Action Space:
        Discrete(192): 3 pieces × 8×8 board positions
    
    Observation Space:
        Dict with "board" (8,8) and "pieces" (3,5,5), both float32
    
    Reward:
        - Score increase from action
        - +0.01 survival bonus if valid action and not done
        - -10.0 penalty if game ends
        - -1.0 penalty for invalid action
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.game = BlockBlastGame(rng_seed=seed)
        self.state: Optional[GameState] = None
        self._seed = seed
        
        # Action space size
        self.num_actions = NUM_ACTIONS
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Start a new episode and return the initial observation.
        
        Returns:
            obs: Dictionary with "board" (8,8) and "pieces" (3,5,5)
        """
        self.state = self.game.new_game()
        return state_to_obs(self.state)
    
    def step(self, action_id: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Apply one action and return (obs, reward, done, info).
        
        Reward calculation (improved for better learning):
            1. Base: score_after - score_before (internal game scoring)
            2. Line clear bonus: +LINE_CLEAR_BONUS per line cleared
            3. Survival bonus: +SURVIVAL_BONUS if valid action and not done
            4. Game over penalty: GAME_OVER_PENALTY if game ends
            5. Invalid action penalty: INVALID_ACTION_PENALTY, state unchanged
        
        Args:
            action_id: Integer in [0, 192) representing the action
        
        Returns:
            obs: New observation dict
            reward: Float reward for this step
            done: Whether episode is finished
            info: Dict with diagnostic information
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # CHECK FOR GAME OVER FIRST - if no valid moves exist, game is done
        # This prevents infinite loops where agent keeps trying invalid actions
        if not self.state.done:
            if not self.game.has_valid_move(self.state.board, self.state.available_pieces):
                self.state.done = True
        
        # Decode action
        piece_idx, x, y = decode_action(action_id)
        
        # Check if piece index is valid
        if piece_idx >= len(self.state.available_pieces):
            # Invalid piece index - treat as invalid action
            obs = state_to_obs(self.state)
            return obs, INVALID_ACTION_PENALTY, self.state.done, {
                "valid": False,
                "reason": "invalid_piece_index",
                "piece_idx": piece_idx,
            }
        
        # Get the piece
        piece_id = self.state.available_pieces[piece_idx]
        piece = get_piece(piece_id)
        
        # Convert (x, y) board position to piece placement position
        # The piece is defined as a 5×5 matrix, but we need to place it
        # such that its blocks land at the intended board position
        # For simplicity, we interpret (x, y) as the position where the
        # piece's top-left occupied cell should go
        
        # Find the offset of the first occupied cell in the piece
        cells = piece.get_cells()
        if not cells:
            # Empty piece (shouldn't happen, but handle it)
            obs = state_to_obs(self.state)
            return obs, INVALID_ACTION_PENALTY, self.state.done, {
                "valid": False,
                "reason": "empty_piece",
            }
        
        # Use the piece's natural 5×5 position directly
        # The core game logic expects (row, col) relative to the 5×5 piece matrix
        # We need to find where to place the piece's origin so its blocks land correctly
        
        # For simplicity: treat (x, y) as the offset from top-left of piece matrix
        # This means action (x, y) places piece origin at board position (x, y)
        row_offset = y
        col_offset = x
        
        # Check if action is valid using core game logic
        score_before = self.state.score
        
        if not self.game.can_place_piece(self.state.board, piece, row_offset, col_offset):
            # Invalid placement - return penalty, state unchanged
            obs = state_to_obs(self.state)
            return obs, INVALID_ACTION_PENALTY, self.state.done, {
                "valid": False,
                "reason": "invalid_placement",
                "piece_idx": piece_idx,
                "x": x,
                "y": y,
            }
        
        # Valid action - apply it using core game logic
        action = Action(piece_idx, row_offset, col_offset)
        
        try:
            self.state, base_reward, done, info = self.game.apply_action(self.state, action)
        except Exception as e:
            # Should not happen if can_place_piece returned True, but handle it
            obs = state_to_obs(self.state)
            return obs, INVALID_ACTION_PENALTY, self.state.done, {
                "valid": False,
                "reason": f"apply_action_failed: {e}",
            }
        
        # Calculate improved reward with better shaping
        score_after = self.state.score
        
        # 1. Base reward: score delta from game's internal scoring
        reward = float(score_after - score_before)
        
        # 2. Line clear bonus: explicit reward for clearing lines
        #    This encourages the agent to prioritize line clearing
        lines_cleared = info.get("lines_cleared", 0)
        if lines_cleared > 0:
            reward += LINE_CLEAR_BONUS * lines_cleared
        
        # 3. Survival bonus: small reward for each valid move
        #    Encourages the agent to keep playing longer
        if not done:
            reward += SURVIVAL_BONUS
        
        # 4. Game over penalty: discourage early termination
        #    This is negative, so we ADD it (it's already negative)
        if done:
            reward += GAME_OVER_PENALTY
        
        # Get new observation
        obs = state_to_obs(self.state)
        
        # Add info with reward breakdown for debugging
        info.update({
            "valid": True,
            "piece_idx": piece_idx,
            "x": x,
            "y": y,
            "score_before": score_before,
            "score_after": score_after,
            "total_reward": reward,
            "lines_cleared": lines_cleared,
        })
        
        return obs, reward, done, info
    
    def get_valid_action_mask(self) -> np.ndarray:
        """
        Return a boolean array indicating which actions are valid.
        
        Returns:
            mask: Boolean array of shape (507,) where mask[a] is True
                  if action_id a is legal in the current state
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        
        # For each available piece
        for piece_idx in range(len(self.state.available_pieces)):
            piece_id = self.state.available_pieces[piece_idx]
            piece = get_piece(piece_id)
            
            # Try all board positions in the extended range
            # Pieces are 5x5 matrices, so we need range -4 to +8 to cover all valid placements
            for y in range(-4, BOARD_H + 4):
                for x in range(-4, BOARD_W + 4):
                    # Check if this placement is valid
                    if self.game.can_place_piece(self.state.board, piece, y, x):
                        action_id = encode_action(piece_idx, x, y)
                        mask[action_id] = True
        
        return mask
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment for debugging (text mode, no pygame).
        
        Args:
            mode: "human" prints to console, "ansi" returns string
        
        Returns:
            String representation if mode="ansi", else None
        """
        if self.state is None:
            return None
        
        lines = []
        lines.append(f"\n{'='*40}")
        lines.append(f"Score: {self.state.score} | Moves: {self.state.moves} | "
                    f"Combo: {self.state.combo}")
        lines.append(f"{'='*40}")
        
        # Board
        lines.append("\nBoard:")
        lines.append(self.game.get_board_string(self.state.board))
        
        # Available pieces
        lines.append(f"\nAvailable Pieces ({len(self.state.available_pieces)}):")
        for i, piece_id in enumerate(self.state.available_pieces):
            piece = get_piece(piece_id)
            lines.append(f"\nPiece {i} (ID: {piece_id}, blocks: {piece.get_block_count()}):")
            for row in piece.shape:
                lines.append("  " + "".join("█" if cell else "·" for cell in row))
        
        if self.state.done:
            lines.append("\n*** GAME OVER ***")
        
        output = "\n".join(lines)
        
        if mode == "human":
            print(output)
            return None
        elif mode == "ansi":
            return output
        else:
            raise ValueError(f"Invalid render mode: {mode}")
    
    @property
    def current_state(self) -> Optional[GameState]:
        """Get the current game state (for debugging/inspection)."""
        return self.state


def make_env(seed: Optional[int] = None) -> BlockBlastEnv:
    """
    Factory function to create a new environment.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        New BlockBlastEnv instance
    """
    return BlockBlastEnv(seed=seed)

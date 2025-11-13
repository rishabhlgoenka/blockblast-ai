"""
PPO-Compatible Gym Environment for Block Blast with CNN-Friendly Observations
==============================================================================

This environment wraps the existing Block Blast game logic and provides:
1. Standard Gym API (reset, step, observation_space, action_space)
2. CNN-friendly observations in channel-first format: (C, H, W)
3. Compatible with stable-baselines3 PPO + CnnPolicy

Observation Format (Channel-First for CNNs):
-------------------------------------------
Shape: (C, 8, 8) where C = 4 channels
    - Channel 0: Board occupancy (0=empty, 1=filled)
    - Channel 1: Piece 1 encoding (piece blocks placed on virtual 8x8 grid)
    - Channel 2: Piece 2 encoding
    - Channel 3: Piece 3 encoding

Each piece is encoded by placing its 5x5 shape in the center of an 8x8 grid,
making it easy for the CNN to reason about piece-board spatial relationships.

Action Space:
------------
Discrete(507): 3 pieces × 13 × 13 positions
    - 507 discrete actions representing all possible (piece, position) pairs
    - Position range extends from -4 to 8 to handle edge placements

Rewards:
-------
- Base reward: score increase from placing piece
- Line clear bonus: +10 per line cleared (encourages clearing)
- Game over penalty: -20 if game ends (discourages early termination)
- Invalid action penalty: -1 (state unchanged, episode not terminated)

Invalid Actions:
---------------
Invalid actions receive a -1 penalty but do NOT end the episode.
The agent must learn to avoid invalid actions through negative rewards.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional

from blockblast.core import BlockBlastGame, GameState, Action
from blockblast.pieces import get_piece
from blockblast.env import (
    encode_action, decode_action, 
    LINE_CLEAR_BONUS, GAME_OVER_PENALTY, INVALID_ACTION_PENALTY,
    NUM_ACTIONS, BOARD_W, BOARD_H
)


class BlockBlastEnv(gym.Env):
    """
    Gymnasium-compatible environment for Block Blast with CNN-friendly observations.
    
    Designed for use with stable-baselines3 PPO and CnnPolicy.
    """
    
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}
    
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None, max_steps: int = 1000, 
                 curriculum_fill_ratio: float = 0.0, use_mixed_curriculum: bool = False):
        """
        Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility
            render_mode: Rendering mode ('human' or 'ansi')
            max_steps: Maximum steps per episode (prevents infinite loops)
            curriculum_fill_ratio: Ratio of board to pre-fill (0.0 = empty, 0.5 = half full)
                                   Used for fixed curriculum learning
            use_mixed_curriculum: If True, randomly mix difficulty levels each episode
                                  Pattern: 60% (10% of time), empty, 40%, 10%, empty, 60%, empty...
        """
        super().__init__()
        
        self.game = BlockBlastGame(rng_seed=seed)
        self.state: Optional[GameState] = None
        self._seed = seed
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.curriculum_fill_ratio = curriculum_fill_ratio
        self.use_mixed_curriculum = use_mixed_curriculum
        
        # Mixed curriculum pattern: [60%, empty, 40%, 10%, empty, 60%, empty]
        self.mixed_curriculum_pattern = [0.6, 0.0, 0.4, 0.1, 0.0, 0.6, 0.0]
        self.episode_count = 0
        
        # Define observation space: (C, H, W) = (4, 8, 8)
        # Channel-first format for CNNs
        # - Channel 0: Board (0=empty, 1=filled)
        # - Channels 1-3: Three available pieces encoded on 8x8 grid
        # Using uint8 with [0, 255] range for image-based CNN policies
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 8, 8),
            dtype=np.uint8
        )
        
        # Define action space: 507 discrete actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # Episode statistics
        self.episode_reward = 0.0
        self.episode_steps = 0
    
    def _get_valid_actions(self, state: GameState) -> np.ndarray:
        """
        Get mask of valid actions for current state.
        
        Returns:
            Binary mask array of shape (NUM_ACTIONS,) where 1 = valid, 0 = invalid
        """
        valid_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        
        for piece_idx, piece_id in enumerate(state.available_pieces):
            piece = get_piece(piece_id)
            
            # Try all possible positions for this piece
            for y in range(-4, 9):  # -4 to 8
                for x in range(-4, 9):
                    if self.game.can_place_piece(state.board, piece, y, x):
                        action_id = encode_action(piece_idx, x, y)
                        valid_mask[action_id] = True
        
        return valid_mask
    
    def action_masks(self) -> np.ndarray:
        """
        Get valid action mask for current state (for action masking support).
        
        Returns:
            Binary mask where 1 = valid action, 0 = invalid
            
        Note: If game is done or no valid moves, returns all True to avoid
        masking errors (the episode will terminate anyway).
        """
        if self.state is None:
            return np.ones(NUM_ACTIONS, dtype=bool)
        
        # If game is already done, return all-true mask to avoid masking errors
        if self.state.done:
            return np.ones(NUM_ACTIONS, dtype=bool)
        
        # Get valid actions
        valid_mask = self._get_valid_actions(self.state)
        
        # If no valid moves exist, return all-true mask (episode will end on next step anyway)
        if not valid_mask.any():
            return np.ones(NUM_ACTIONS, dtype=bool)
        
        return valid_mask
    
    def _encode_observation(self, state: GameState) -> np.ndarray:
        """
        Convert GameState to CNN-friendly observation.
        
        Args:
            state: Current game state
            
        Returns:
            Observation array of shape (4, 8, 8) with uint8 values in [0, 255]
        """
        obs = np.zeros((4, 8, 8), dtype=np.uint8)
        
        # Channel 0: Board occupancy (0 or 255)
        obs[0] = (state.board > 0).astype(np.uint8) * 255
        
        # Channels 1-3: Encode each piece on an 8x8 grid
        # We place each 5x5 piece centered on the 8x8 grid to help CNN
        # understand spatial relationships
        for i, piece_id in enumerate(state.available_pieces[:3]):  # Take first 3 pieces
            if i < 3:  # Ensure we have 3 piece channels
                piece = get_piece(piece_id)
                piece_shape = piece.shape.astype(np.uint8) * 255  # 5x5, scale to [0, 255]
                
                # Place piece in center of 8x8 grid (offset by 1-2 to center)
                # This helps the CNN see piece-board spatial relationships
                offset_row = (8 - 5) // 2  # = 1
                offset_col = (8 - 5) // 2  # = 1
                
                obs[i + 1, offset_row:offset_row+5, offset_col:offset_col+5] = piece_shape
        
        return obs
    
    def _prefill_board(self, state: GameState, fill_ratio: float) -> GameState:
        """
        Pre-fill the board with blocks for curriculum learning.
        Creates partial rows/columns to make line clearing easier to discover.
        
        Args:
            state: Current game state
            fill_ratio: Ratio of cells to fill (0.0-1.0)
        
        Returns:
            Modified game state with pre-filled board
        """
        if fill_ratio <= 0.0:
            return state
        
        total_cells = BOARD_W * BOARD_H
        cells_to_fill = int(total_cells * fill_ratio)
        
        # Strategy: Fill rows/columns partially (not randomly) to create clear opportunities
        # This helps agent discover that completing rows = reward
        rng = np.random.RandomState(self._seed)
        
        # Fill some rows partially (leave 1-2 gaps for easy completion)
        rows_to_fill = min(int(fill_ratio * 8), 7)  # Don't fill all rows
        for row_idx in range(rows_to_fill):
            # Fill 6-7 cells in this row (leaving 1-2 gaps)
            num_to_fill = rng.randint(6, 8)
            cols = rng.choice(BOARD_W, size=num_to_fill, replace=False)
            for col in cols:
                state.board[row_idx, col] = 1  # Mark as filled
        
        return state
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return initial observation.
        
        Args:
            seed: Random seed for this episode
            options: Additional options (unused)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Handle seeding
        if seed is not None:
            self._seed = seed
            self.game = BlockBlastGame(rng_seed=seed)
        
        # Reset game state
        self.state = self.game.new_game()
        
        # Apply curriculum learning: pre-fill board if needed
        if self.use_mixed_curriculum:
            # Mixed curriculum: cycle through pattern each episode
            fill_ratio = self.mixed_curriculum_pattern[self.episode_count % len(self.mixed_curriculum_pattern)]
            self.episode_count += 1
            if fill_ratio > 0.0:
                self.state = self._prefill_board(self.state, fill_ratio)
        elif self.curriculum_fill_ratio > 0.0:
            # Fixed curriculum (legacy)
            self.state = self._prefill_board(self.state, self.curriculum_fill_ratio)
        
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Return initial observation and empty info dict
        obs = self._encode_observation(self.state)
        info = {
            'score': self.state.score,
            'moves': self.state.moves,
            'combo': self.state.combo
        }
        
        return obs, info
    
    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action and return the result.
        
        IMPORTANT: In Block Blast, you MUST make a valid move or the game ends.
        Invalid moves shouldn't exist - if no valid moves available, game is over.
        
        Args:
            action_id: Integer action ID in [0, 507)
            
        Returns:
            observation: New observation after action
            reward: Reward received
            terminated: Whether episode ended naturally (game over)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.episode_steps += 1
        
        # Check if max steps reached (truncate episode)
        truncated = self.episode_steps >= self.max_steps
        
        # Get valid actions FIRST
        valid_mask = self._get_valid_actions(self.state)
        
        # If NO valid moves exist, GAME OVER!
        if not valid_mask.any():
            self.state.done = True
            obs = self._encode_observation(self.state)
            reward = GAME_OVER_PENALTY
            self.episode_reward += reward
            
            info = {
                'valid': False,
                'reason': 'no_valid_moves',
                'score': self.state.score,
                'moves': self.state.moves,
                'episode_reward': self.episode_reward,
                'game_over': True
            }
            
            return obs, reward, True, truncated, info
        
        # If agent chose invalid action, force a random valid action
        # (This shouldn't happen with proper action masking in training)
        if not valid_mask[action_id]:
            valid_actions = np.where(valid_mask)[0]
            action_id = np.random.choice(valid_actions)
        
        # Decode action
        piece_idx, x, y = decode_action(action_id)
        
        # Get the piece
        piece_id = self.state.available_pieces[piece_idx]
        piece = get_piece(piece_id)
        
        # Store score before move
        score_before = self.state.score
        
        # Apply the action (must be valid at this point)
        action = Action(piece_idx, y, x)
        self.state, base_reward, done, game_info = self.game.apply_action(self.state, action)
        
        # Calculate reward with DENSE SHAPING - reward progress towards line clears!
        score_after = self.state.score
        reward = float(score_after - score_before)
        
        # Track occupied cells
        occupied_cells_after = (self.state.board > 0).sum()
        
        #  ==== DENSE REWARDS: Reward intermediate progress! ====
        
        # 1. Row/Column completion progress (DENSE FEEDBACK!)
        row_column_bonus = 0.0
        for row in self.state.board:
            filled = (row > 0).sum()
            if filled == 7:  # 7/8 filled - almost there!
                row_column_bonus += 25.0
            elif filled == 6:  # 6/8 filled
                row_column_bonus += 10.0
            elif filled == 5:  # 5/8 filled
                row_column_bonus += 3.0
        
        for col_idx in range(BOARD_W):
            col = self.state.board[:, col_idx]
            filled = (col > 0).sum()
            if filled == 7:  # Almost complete column!
                row_column_bonus += 25.0
            elif filled == 6:
                row_column_bonus += 10.0
            elif filled == 5:
                row_column_bonus += 3.0
        
        reward += row_column_bonus
        
        # 2. EXPONENTIAL line clear rewards (keep existing - these are huge!)
        lines_cleared = game_info.get('lines_cleared', 0)
        if lines_cleared == 1:
            reward += 100.0  # Increased from 50
        elif lines_cleared == 2:
            reward += 300.0  # Increased from 150
        elif lines_cleared == 3:
            reward += 700.0  # Increased from 350
        elif lines_cleared == 4:
            reward += 1500.0  # Increased from 700
        elif lines_cleared >= 5:
            reward += 3000.0  # Increased from 1500
        
        # 3. EXPONENTIAL combo bonuses
        if self.state.combo >= 2:
            reward += (2 ** self.state.combo) * 15.0  # Increased multiplier
        
        # 4. Strategic placement bonus (NEW!)
        # Reward if action created completion opportunity without immediately clearing
        if lines_cleared == 0 and row_column_bonus > 20:
            reward += 15.0  # "Good setup move"
        
        # 5. Small survival bonus - reward staying alive
        reward += 0.5
        
        # 6. Moderate density penalty (not too harsh to allow building)
        occupied_cells = occupied_cells_after
        board_density = occupied_cells / (BOARD_W * BOARD_H)
        
        if score_after < 100:
            penalty_multiplier = 30.0  # Reduced from 100
        elif score_after < 150:
            penalty_multiplier = 15.0  # Reduced from 50
        else:
            penalty_multiplier = 5.0   # Reduced from 20/5
        
        reward -= (board_density ** 2) * penalty_multiplier
        
        # 7. Game over penalty
        if done:
            reward -= 50.0  # Reduced from 100
        
        self.episode_reward += reward
        
        # Get new observation
        obs = self._encode_observation(self.state)
        
        # Build info dict
        info = {
            'valid': True,
            'score': self.state.score,
            'moves': self.state.moves,
            'combo': self.state.combo,
            'lines_cleared': lines_cleared,
            'episode_reward': self.episode_reward,
            'blocks_placed': game_info.get('blocks_placed', 0)
        }
        
        return obs, reward, done, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            String representation if render_mode='ansi', else None
        """
        if self.state is None:
            return None
        
        if self.render_mode == 'human' or self.render_mode == 'ansi':
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
            for i, piece_id in enumerate(self.state.available_pieces[:3]):
                piece = get_piece(piece_id)
                lines.append(f"\nPiece {i} (ID: {piece_id}, blocks: {piece.get_block_count()}):")
                for row in piece.shape:
                    lines.append("  " + "".join("█" if cell else "·" for cell in row))
            
            if self.state.done:
                lines.append("\n*** GAME OVER ***")
            
            output = "\n".join(lines)
            
            if self.render_mode == 'human':
                print(output)
                return None
            else:
                return output
        
        return None
    
    def close(self):
        """Clean up resources."""
        pass


def make_env(seed: Optional[int] = None, render_mode: Optional[str] = None) -> BlockBlastEnv:
    """
    Factory function to create a new environment.
    
    Args:
        seed: Random seed for reproducibility
        render_mode: Rendering mode
        
    Returns:
        New BlockBlastEnv instance
    """
    return BlockBlastEnv(seed=seed, render_mode=render_mode)


#!/usr/bin/env python3
"""
Improved Expert Agent v2 for Block Blast
=========================================

MUCH better heuristics that should achieve 80-120+ scores.

Key improvements:
1. Explicit line-clearing detection and prioritization
2. Strategic placement to CREATE line-clearing opportunities
3. Look-ahead evaluation
4. Better board state assessment
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from ppo_env import BlockBlastEnv, encode_action, decode_action
from blockblast.core import BlockBlastGame, GameState, Action
from blockblast.pieces import get_piece


class ImprovedExpertAgent:
    """
    Significantly improved heuristic expert agent.
    Target: 80-120+ average score with 3-5+ lines cleared per episode.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize improved expert agent."""
        self.game = BlockBlastGame(rng_seed=seed)
        self.env = BlockBlastEnv(seed=seed)
    
    def get_valid_actions(self, state: GameState) -> List[int]:
        """Get all valid action IDs for current state."""
        valid_actions = []
        
        for piece_idx in range(len(state.available_pieces)):
            piece_id = state.available_pieces[piece_idx]
            piece = get_piece(piece_id)
            
            for row in range(8):
                for col in range(8):
                    if self.game.can_place_piece(state.board, piece, row, col):
                        action_id = encode_action(piece_idx, col, row)
                        valid_actions.append(action_id)
        
        return valid_actions
    
    def simulate_action(self, state: GameState, action_id: int) -> Tuple[GameState, dict]:
        """
        Simulate an action and return resulting state + info.
        """
        piece_idx, x, y = decode_action(action_id)
        action = Action(piece_idx, y, x)
        
        simulated_state = state.copy()
        try:
            next_state, reward, done, info = self.game.apply_action(simulated_state, action)
            return next_state, info
        except:
            return state.copy(), {'lines_cleared': 0, 'valid': False}
    
    def evaluate_action(self, state: GameState, action_id: int) -> float:
        """
        Comprehensive action evaluation.
        Returns a score for how good this action is.
        """
        next_state, info = self.simulate_action(state, action_id)
        
        if not info.get('valid', True):
            return -1000.0  # Invalid action
        
        score = 0.0
        
        # ===== PRIORITY 1: LINE CLEARING (MASSIVE BONUS) =====
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            # Exponential reward for line clearing
            score += 1000.0 * (lines_cleared ** 2)  # 1k, 4k, 9k, 16k...
            return score  # If we clear lines, this is automatically best!
        
        # ===== PRIORITY 2: CREATING LINE-CLEARING OPPORTUNITIES =====
        # Count rows/columns that are 1 cell away from completion
        almost_complete_score = 0.0
        
        for row in next_state.board:
            empty_count = (row == 0).sum()
            if empty_count == 1:  # 7/8 filled!
                almost_complete_score += 200.0
            elif empty_count == 2:  # 6/8 filled
                almost_complete_score += 50.0
        
        for col_idx in range(8):
            col = next_state.board[:, col_idx]
            empty_count = (col == 0).sum()
            if empty_count == 1:  # 7/8 filled!
                almost_complete_score += 200.0
            elif empty_count == 2:  # 6/8 filled
                almost_complete_score += 50.0
        
        score += almost_complete_score
        
        # ===== PRIORITY 3: BOARD CLEANLINESS =====
        # Heavily penalize filling up the board
        density = (next_state.board > 0).sum() / 64.0
        score -= (density ** 3) * 500.0  # Cubic penalty!
        
        # ===== PRIORITY 4: AVOID ISOLATED PIECES =====
        # Penalize creating hard-to-fill gaps
        empty_cells = (next_state.board == 0)
        
        # Count isolated empty cells (surrounded by filled cells)
        isolation_penalty = 0.0
        for row in range(8):
            for col in range(8):
                if empty_cells[row, col]:
                    # Check neighbors
                    neighbors_filled = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < 8 and 0 <= nc < 8:
                            if next_state.board[nr, nc] > 0:
                                neighbors_filled += 1
                    
                    # If cell is surrounded, it's bad
                    if neighbors_filled >= 3:
                        isolation_penalty += 50.0
        
        score -= isolation_penalty
        
        # ===== PRIORITY 5: PREFER EDGES AND BUILDING FROM EXISTING =====
        # Reward placing near existing pieces (easier to create lines)
        piece_idx, x, y = decode_action(action_id)
        piece = get_piece(state.available_pieces[piece_idx])
        
        adjacency_bonus = 0.0
        for i in range(piece.shape.shape[0]):
            for j in range(piece.shape.shape[1]):
                if piece.shape[i, j]:
                    piece_row, piece_col = y + i, x + j
                    if 0 <= piece_row < 8 and 0 <= piece_col < 8:
                        # Check if next to existing pieces
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = piece_row + dr, piece_col + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                if state.board[nr, nc] > 0:  # Original board
                                    adjacency_bonus += 5.0
        
        score += adjacency_bonus
        
        return score
    
    def select_action(self, state: GameState) -> Optional[int]:
        """
        Select best action using improved heuristics.
        """
        valid_actions = self.get_valid_actions(state)
        
        if not valid_actions:
            return None
        
        # Evaluate all actions
        action_scores = [(action_id, self.evaluate_action(state, action_id)) 
                        for action_id in valid_actions]
        
        # Sort by score (descending)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get best action
        best_action, best_score = action_scores[0]
        
        return best_action
    
    def play_episode(self, verbose: bool = False) -> Tuple[int, int, List[Tuple[np.ndarray, int]]]:
        """
        Play one episode using improved expert policy.
        
        Returns:
            score: Final game score
            moves: Number of moves made
            trajectory: List of (observation, action) pairs
        """
        obs, info = self.env.reset()
        trajectory = []
        done = False
        moves = 0
        
        while not done:
            action = self.select_action(self.env.state)
            
            if action is None:
                break
            
            trajectory.append((obs.copy(), action))
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            moves += 1
            
            if verbose and moves % 10 == 0:
                print(f"Move {moves}: Score={info.get('score', 0)}, Lines={self.env.state.lines_cleared_total}")
        
        score = info.get('score', 0)
        lines = self.env.state.lines_cleared_total
        
        return score, moves, trajectory, lines


def test_improved_expert(num_episodes: int = 30):
    """Test the improved expert agent."""
    print("="*70)
    print("Testing Improved Expert Agent v2")
    print("="*70)
    
    expert = ImprovedExpertAgent(seed=42)
    
    scores = []
    lines_cleared_list = []
    moves_list = []
    
    print(f"\nPlaying {num_episodes} test episodes...\n")
    
    for episode in range(num_episodes):
        score, moves, trajectory, lines = expert.play_episode()
        
        scores.append(score)
        lines_cleared_list.append(lines)
        moves_list.append(moves)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1:3d}: Score={score:4d}, Lines={lines:2d}, Moves={moves:3d}")
    
    print("\n" + "="*70)
    print("Improved Expert Agent Performance")
    print("="*70)
    print(f"Average Score:       {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score:           {np.max(scores)}")
    print(f"Min Score:           {np.min(scores)}")
    print(f"Average Lines:       {np.mean(lines_cleared_list):.2f}")
    print(f"Total Lines:         {np.sum(lines_cleared_list)}")
    print(f"Average Moves:       {np.mean(moves_list):.2f}")
    print("="*70)
    
    avg_score = np.mean(scores)
    avg_lines = np.mean(lines_cleared_list)
    
    if avg_score >= 100 and avg_lines >= 4:
        print("\n🏆 EXCELLENT! Expert is strong enough for imitation learning!")
    elif avg_score >= 70 and avg_lines >= 2:
        print("\n✓ GOOD! Expert is decent - should work for imitation learning.")
    elif avg_score >= 50:
        print("\n⚠️  MODERATE. Expert is okay but could be better.")
    else:
        print("\n❌ LOW. Expert needs more improvement.")
    
    return avg_score, avg_lines


if __name__ == "__main__":
    avg_score, avg_lines = test_improved_expert(30)


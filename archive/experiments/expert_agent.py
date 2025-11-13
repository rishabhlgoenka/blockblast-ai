#!/usr/bin/env python3
"""
Expert Heuristic Agent for Block Blast
=======================================

Rule-based "teacher" agent that uses smart heuristics to play Block Blast well.
This agent will generate demonstrations for imitation learning.

Heuristics:
1. Always prioritize moves that clear lines
2. Prefer moves that complete rows/columns (75%+ full)
3. Prefer moves that don't increase board density
4. Avoid filling corners when possible
"""

import numpy as np
from typing import List, Tuple, Optional
from ppo_env import BlockBlastEnv, encode_action, decode_action
from blockblast.core import BlockBlastGame, GameState, Action
from blockblast.pieces import get_piece


class ExpertAgent:
    """
    Heuristic expert agent that plays Block Blast using smart rules.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize expert agent."""
        self.game = BlockBlastGame(rng_seed=seed)
        self.env = BlockBlastEnv(seed=seed)
    
    def get_valid_actions(self, state: GameState) -> List[int]:
        """Get all valid action IDs for current state."""
        valid_actions = []
        
        for piece_idx in range(len(state.available_pieces)):
            piece_id = state.available_pieces[piece_idx]
            piece = get_piece(piece_id)
            
            # Try all positions on board
            for row in range(8):
                for col in range(8):
                    if self.game.can_place_piece(state.board, piece, row, col):
                        action_id = encode_action(piece_idx, col, row)
                        valid_actions.append(action_id)
        
        return valid_actions
    
    def count_lines_cleared(self, state: GameState, action_id: int) -> int:
        """
        Simulate action and count how many lines would be cleared.
        """
        # Decode action
        piece_idx, x, y = decode_action(action_id)
        
        # Create action
        action = Action(piece_idx, y, x)
        
        # Simulate (don't modify original state)
        simulated_state = state.copy()
        try:
            next_state, reward, done, info = self.game.apply_action(simulated_state, action)
            return info.get('lines_cleared', 0)
        except:
            return 0
    
    def calculate_board_density_after(self, state: GameState, action_id: int) -> float:
        """Calculate board density if we take this action."""
        # Decode action
        piece_idx, x, y = decode_action(action_id)
        action = Action(piece_idx, y, x)
        
        # Simulate
        simulated_state = state.copy()
        try:
            next_state, _, _, _ = self.game.apply_action(simulated_state, action)
            occupied = (next_state.board > 0).sum()
            return occupied / (8 * 8)
        except:
            return 1.0  # Worst case
    
    def get_row_column_completeness(self, state: GameState, action_id: int) -> float:
        """
        Score based on how many rows/columns are close to completion after action.
        Returns: score (higher = more rows/columns nearly complete)
        """
        # Decode action
        piece_idx, x, y = decode_action(action_id)
        action = Action(piece_idx, y, x)
        
        # Simulate
        simulated_state = state.copy()
        try:
            next_state, _, _, _ = self.game.apply_action(simulated_state, action)
            board = next_state.board
            
            score = 0.0
            
            # Check rows
            for row in board:
                filled = (row > 0).sum()
                if filled == 7:  # Almost complete
                    score += 10.0
                elif filled == 6:
                    score += 5.0
                elif filled >= 4:
                    score += 1.0
            
            # Check columns
            for col_idx in range(8):
                col = board[:, col_idx]
                filled = (col > 0).sum()
                if filled == 7:  # Almost complete
                    score += 10.0
                elif filled == 6:
                    score += 5.0
                elif filled >= 4:
                    score += 1.0
            
            return score
        except:
            return 0.0
    
    def select_action(self, state: GameState) -> Optional[int]:
        """
        Select best action using expert heuristics.
        
        IMPROVED Strategy:
        1. ALWAYS clear lines if possible (top priority!)
        2. Place pieces that build towards completing rows
        3. Avoid dense placements
        4. Keep board relatively empty
        """
        valid_actions = self.get_valid_actions(state)
        
        if not valid_actions:
            return None  # No valid moves - game over
        
        # Phase 1: Check if any action clears lines
        # If so, ONLY consider line-clearing moves
        line_clearing_actions = []
        for action_id in valid_actions:
            lines = self.count_lines_cleared(state, action_id)
            if lines > 0:
                line_clearing_actions.append((action_id, lines))
        
        if line_clearing_actions:
            # Pick the action that clears MOST lines
            line_clearing_actions.sort(key=lambda x: x[1], reverse=True)
            best_actions = [a for a, l in line_clearing_actions if l == line_clearing_actions[0][1]]
            return np.random.choice(best_actions)
        
        # Phase 2: No line clears available, use heuristics
        action_scores = []
        
        for action_id in valid_actions:
            score = 0.0
            
            # 1. Row/column completion potential (strongly reward)
            completeness = self.get_row_column_completeness(state, action_id)
            score += completeness * 5.0  # Amplify importance
            
            # 2. HEAVILY penalize increasing board density
            density = self.calculate_board_density_after(state, action_id)
            score -= density * 50.0  # Strong penalty
            
            # 3. Bonus for keeping board empty
            current_density = (state.board > 0).sum() / 64.0
            if density < current_density + 0.1:  # Doesn't increase much
                score += 10.0
            
            action_scores.append((action_id, score))
        
        # Select action with highest score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        best_action, best_score = action_scores[0]
        
        # If multiple actions have same top score, pick randomly among them
        top_actions = [a for a, s in action_scores if abs(s - best_score) < 0.1]
        if len(top_actions) > 1:
            best_action = np.random.choice(top_actions)
        
        return best_action
    
    def play_episode(self) -> Tuple[int, int, List[Tuple[np.ndarray, int]]]:
        """
        Play one episode using expert policy.
        
        Returns:
            score: Final game score
            moves: Number of moves made
            trajectory: List of (observation, action) pairs for imitation learning
        """
        obs, info = self.env.reset()
        trajectory = []
        done = False
        moves = 0
        
        while not done:
            # Get expert action
            action = self.select_action(self.env.state)
            
            if action is None:
                break  # No valid moves
            
            # Store for imitation learning
            trajectory.append((obs.copy(), action))
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            moves += 1
        
        score = info.get('score', 0)
        
        return score, moves, trajectory


def test_expert():
    """Test the expert agent and show its performance."""
    print("="*70)
    print("Testing Expert Agent")
    print("="*70)
    
    expert = ExpertAgent(seed=42)
    
    scores = []
    lines_cleared_list = []
    moves_list = []
    
    print("\nPlaying 20 test episodes...\n")
    
    for episode in range(20):
        score, moves, trajectory = expert.play_episode()
        lines_cleared = expert.env.state.lines_cleared_total
        
        scores.append(score)
        lines_cleared_list.append(lines_cleared)
        moves_list.append(moves)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1:2d}: Score={score:4d}, Lines={lines_cleared:2d}, Moves={moves:3d}")
    
    print("\n" + "="*70)
    print("Expert Agent Performance")
    print("="*70)
    print(f"Average Score:       {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score:           {np.max(scores)}")
    print(f"Min Score:           {np.min(scores)}")
    print(f"Average Lines:       {np.mean(lines_cleared_list):.2f}")
    print(f"Total Lines:         {np.sum(lines_cleared_list)}")
    print(f"Average Moves:       {np.mean(moves_list):.2f}")
    print("="*70)
    
    return np.mean(scores)


if __name__ == "__main__":
    avg_score = test_expert()
    
    if avg_score < 60:
        print("\n⚠️  Expert performance is low. May need better heuristics.")
    elif avg_score < 100:
        print("\n✓  Expert is decent. Good enough for imitation learning.")
    else:
        print("\n🏆 Expert is strong! Excellent for imitation learning.")


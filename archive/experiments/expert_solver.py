#!/usr/bin/env python3
"""
Expert Solver for Block Blast
==============================

Implements the exhaustive search algorithm from the JavaScript solver.
Finds optimal move sequences that maximize score through combos and line clears.

The solver uses:
1. Exhaustive search through all possible placements
2. Combo-weighted scoring (higher combos = exponentially more valuable)
3. Line clear multipliers
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from blockblast.core import GameState, Action, BlockBlastGame
from blockblast.pieces import get_piece


@dataclass
class SolverMove:
    """Represents a move in the solver's search."""
    action: Action
    score_earned: int
    lines_cleared: int
    combo: int
    board_after: np.ndarray


@dataclass
class Solution:
    """Complete solution path."""
    moves: List[SolverMove]
    total_score: int
    final_combo: int
    
    def __repr__(self):
        return f"Solution(moves={len(self.moves)}, score={self.total_score}, combo={self.final_combo})"


class ExpertSolver:
    """
    Expert solver that finds optimal move sequences.
    
    Uses exhaustive search with pruning to find high-scoring solutions.
    """
    
    def __init__(self, max_depth: int = 3):
        """
        Initialize solver.
        
        Args:
            max_depth: Maximum search depth (number of moves to look ahead)
        """
        self.game = BlockBlastGame()
        self.max_depth = max_depth
        self.solutions_found = 0
        self.nodes_explored = 0
    
    def solve(self, state: GameState, depth: int = 1) -> List[Solution]:
        """
        Find all possible solutions from current state up to given depth.
        
        Args:
            state: Current game state
            depth: How many moves to look ahead
            
        Returns:
            List of solutions sorted by score (best first)
        """
        self.solutions_found = 0
        self.nodes_explored = 0
        
        solutions = []
        self._recursive_search(state, [], depth, solutions)
        
        # Sort by total score (descending)
        solutions.sort(key=lambda s: s.total_score, reverse=True)
        
        return solutions
    
    def _recursive_search(
        self, 
        state: GameState, 
        current_moves: List[SolverMove],
        depth_remaining: int,
        solutions: List[Solution]
    ):
        """
        Recursive exhaustive search.
        
        Args:
            state: Current state
            current_moves: Moves taken so far
            depth_remaining: How many more moves to explore
            solutions: List to accumulate solutions
        """
        self.nodes_explored += 1
        
        # Get all valid actions
        valid_actions = self.game.get_valid_actions(state)
        
        if not valid_actions:
            # Dead end - no valid moves
            if len(current_moves) > 0:
                solutions.append(Solution(
                    moves=current_moves.copy(),
                    total_score=sum(m.score_earned for m in current_moves),
                    final_combo=current_moves[-1].combo if current_moves else 0
                ))
                self.solutions_found += 1
            return
        
        # Try each valid action
        for action in valid_actions:
            self.nodes_explored += 1
            
            # Apply action
            try:
                next_state, reward, done, info = self.game.apply_action(state, action)
                
                # Create move record
                move = SolverMove(
                    action=action,
                    score_earned=reward,
                    lines_cleared=info['lines_cleared'],
                    combo=next_state.combo,
                    board_after=next_state.board.copy()
                )
                
                new_moves = current_moves + [move]
                
                # If we've exhausted all pieces or reached depth limit, save solution
                if len(next_state.available_pieces) == 0 or depth_remaining <= 1 or done:
                    solutions.append(Solution(
                        moves=new_moves,
                        total_score=sum(m.score_earned for m in new_moves),
                        final_combo=next_state.combo
                    ))
                    self.solutions_found += 1
                else:
                    # Continue searching
                    self._recursive_search(next_state, new_moves, depth_remaining - 1, solutions)
                    
            except ValueError:
                # Invalid action (shouldn't happen if get_valid_actions is correct)
                continue
    
    def get_best_move(self, state: GameState, look_ahead: int = 1) -> Optional[Action]:
        """
        Get the best single move for the current state.
        
        Args:
            state: Current game state
            look_ahead: How many moves to look ahead (1 = greedy, higher = smarter)
            
        Returns:
            Best action, or None if no valid moves
        """
        solutions = self.solve(state, depth=look_ahead)
        
        if not solutions:
            return None
        
        # Return first move of best solution
        best_solution = solutions[0]
        if len(best_solution.moves) > 0:
            return best_solution.moves[0].action
        
        return None
    
    def evaluate_action(self, state: GameState, action: Action) -> float:
        """
        Evaluate how good an action is.
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            Score (higher is better)
        """
        try:
            next_state, reward, done, info = self.game.apply_action(state, action)
            
            # Base score is the reward
            score = reward
            
            # Bonus for combos (exponential)
            if next_state.combo > 0:
                score += 50 * (2 ** next_state.combo)
            
            # Bonus for clearing multiple lines
            if info['lines_cleared'] > 1:
                score += 100 * info['lines_cleared']
            
            # Penalty for filling board (less empty space = worse)
            filled_cells = np.sum(next_state.board > 0)
            density = filled_cells / (8 * 8)
            score -= density * 20
            
            # Bonus for having valid moves remaining
            if not done:
                score += 10
            
            return score
            
        except ValueError:
            return -1000.0  # Invalid action


class GreedySolver:
    """
    Faster greedy solver that just picks the best immediate move.
    """
    
    def __init__(self):
        self.game = BlockBlastGame()
        self.expert = ExpertSolver(max_depth=1)
    
    def get_best_move(self, state: GameState) -> Optional[Action]:
        """Get best immediate move (greedy)."""
        valid_actions = self.game.get_valid_actions(state)
        
        if not valid_actions:
            return None
        
        # Evaluate each action
        best_action = None
        best_score = -float('inf')
        
        for action in valid_actions:
            score = self.expert.evaluate_action(state, action)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action


def test_solver():
    """Test the solver on a simple game."""
    print("Testing Expert Solver...")
    print("=" * 70)
    
    game = BlockBlastGame(rng_seed=42)
    state = game.new_game()
    
    print("Initial state:")
    print(game.get_board_string(state.board))
    print(f"Available pieces: {state.available_pieces}")
    print()
    
    # Test greedy solver
    greedy = GreedySolver()
    action = greedy.get_best_move(state)
    
    if action:
        print(f"Greedy best move: {action}")
        next_state, reward, done, info = game.apply_action(state, action)
        print(f"Reward: {reward}, Lines cleared: {info['lines_cleared']}, Combo: {next_state.combo}")
        print(game.get_board_string(next_state.board))
    
    print("\n" + "=" * 70)
    
    # Test full solver with look-ahead
    print("\nTesting full solver with look-ahead...")
    solver = ExpertSolver(max_depth=3)
    solutions = solver.solve(state, depth=2)
    
    print(f"Found {len(solutions)} solutions")
    print(f"Explored {solver.nodes_explored} nodes")
    
    if solutions:
        print(f"\nTop 3 solutions:")
        for i, sol in enumerate(solutions[:3]):
            print(f"{i+1}. {sol}")
            if len(sol.moves) > 0:
                print(f"   First move: {sol.moves[0].action}")


if __name__ == '__main__':
    test_solver()


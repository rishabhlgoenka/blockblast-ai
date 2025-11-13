#!/usr/bin/env python3
"""
Human-Playable Block Blast with Demo Recording
===============================================

Play Block Blast in the terminal and record your gameplay for imitation learning!

Controls:
- Use numbers 0-2 to select piece
- Use arrow keys or WASD to move cursor
- Press SPACE or ENTER to place piece
- Press Q to quit
"""

import numpy as np
import sys
import os
import json
import pickle
from typing import List, Tuple, Optional
from ppo_env import BlockBlastEnv
from blockblast.core import GameState, Action
from blockblast.pieces import get_piece


class HumanPlayer:
    """
    Interactive Block Blast game for human players.
    Records gameplay for imitation learning.
    """
    
    def __init__(self, record_demos: bool = True):
        self.env = BlockBlastEnv()
        self.record_demos = record_demos
        self.demonstrations = []  # Will store (obs, action) pairs
        
    def render_board(self, state: GameState):
        """Render the game board to terminal."""
        print("\n" + "="*50)
        print(f"SCORE: {state.score} | MOVES: {state.moves} | COMBO: {state.combo}")
        print("="*50)
        
        # Print column numbers
        print("   ", end="")
        for col in range(8):
            print(f" {col} ", end="")
        print()
        
        # Print board
        for row in range(8):
            print(f" {row} ", end="")
            for col in range(8):
                if state.board[row, col] > 0:
                    print(" ■ ", end="")
                else:
                    print(" · ", end="")
            print()
        
        print()
    
    def render_pieces(self, state: GameState):
        """Show available pieces."""
        print("AVAILABLE PIECES:")
        print("-" * 50)
        
        for idx in range(len(state.available_pieces)):
            piece_id = state.available_pieces[idx]
            piece = get_piece(piece_id)
            
            print(f"\nPiece {idx}:")
            for row in range(piece.shape.shape[0]):
                print("  ", end="")
                for col in range(piece.shape.shape[1]):
                    if piece.shape[row, col]:
                        print("■ ", end="")
                    else:
                        print("  ", end="")
                print()
        print()
    
    def get_valid_actions_display(self, state: GameState) -> List[Tuple[int, int, int]]:
        """Get valid actions as (piece_idx, row, col) tuples."""
        valid_actions = []
        
        for piece_idx in range(len(state.available_pieces)):
            piece_id = state.available_pieces[piece_idx]
            piece = get_piece(piece_id)
            
            for row in range(8):
                for col in range(8):
                    if self.env.game.can_place_piece(state.board, piece, row, col):
                        valid_actions.append((piece_idx, row, col))
        
        return valid_actions
    
    def show_valid_moves(self, state: GameState, selected_piece: int):
        """Show where selected piece can be placed."""
        piece_id = state.available_pieces[selected_piece]
        piece = get_piece(piece_id)
        
        print(f"\nValid placements for Piece {selected_piece} (marked with *):")
        print("   ", end="")
        for col in range(8):
            print(f" {col} ", end="")
        print()
        
        for row in range(8):
            print(f" {row} ", end="")
            for col in range(8):
                if state.board[row, col] > 0:
                    print(" ■ ", end="")
                elif self.env.game.can_place_piece(state.board, piece, row, col):
                    print(" * ", end="")
                else:
                    print(" · ", end="")
            print()
        print()
    
    def get_human_action(self, state: GameState) -> Optional[Action]:
        """Get action from human player via terminal input."""
        valid_actions = self.get_valid_actions_display(state)
        
        if not valid_actions:
            print("\n❌ NO VALID MOVES! GAME OVER!")
            return None
        
        print(f"\n{len(valid_actions)} valid moves available.")
        
        # Step 1: Select piece
        while True:
            piece_input = input(f"\nSelect piece (0-{len(state.available_pieces)-1}) or 'q' to quit: ").strip().lower()
            
            if piece_input == 'q':
                return None
            
            try:
                piece_idx = int(piece_input)
                if 0 <= piece_idx < len(state.available_pieces):
                    break
                else:
                    print(f"Invalid piece! Choose 0-{len(state.available_pieces)-1}")
            except ValueError:
                print("Please enter a number or 'q'")
        
        # Show valid placements
        self.show_valid_moves(state, piece_idx)
        
        # Step 2: Select position
        while True:
            pos_input = input("Enter position as 'row,col' (e.g., '3,5'): ").strip()
            
            if pos_input.lower() == 'q':
                return None
            
            try:
                parts = pos_input.split(',')
                if len(parts) != 2:
                    print("Format: row,col (e.g., '3,5')")
                    continue
                
                row = int(parts[0].strip())
                col = int(parts[1].strip())
                
                if (piece_idx, row, col) in valid_actions:
                    return Action(piece_idx, row, col)
                else:
                    print("Invalid placement! Choose a position marked with *")
            except ValueError:
                print("Please enter numbers in format: row,col")
    
    def play_episode(self, episode_num: int = 1) -> Tuple[int, int]:
        """
        Play one episode with human control.
        Records demonstrations if enabled.
        """
        print("\n" + "="*70)
        print(f"EPISODE {episode_num} - STARTING NEW GAME")
        print("="*70)
        
        obs, info = self.env.reset()
        episode_trajectory = []
        done = False
        moves = 0
        
        while not done:
            self.render_board(self.env.state)
            self.render_pieces(self.env.state)
            
            # Get human action
            action = self.get_human_action(self.env.state)
            
            if action is None:
                print("\nGame ended (quit or no valid moves)")
                break
            
            # Convert to action_id for recording
            from ppo_env import encode_action
            action_id = encode_action(action.piece_index, action.col, action.row)
            
            # Record demonstration
            if self.record_demos:
                episode_trajectory.append((obs.copy(), action_id))
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action_id)
            done = terminated or truncated
            moves += 1
            
            print(f"\n✓ Move {moves}: Placed piece {action.piece_index} at ({action.row}, {action.col})")
            print(f"  Score: {info.get('score', 0)}, Lines cleared this move: {info.get('lines_cleared_total', 0)}")
        
        # Final score
        final_score = info.get('score', 0)
        total_lines = self.env.state.lines_cleared_total
        
        print("\n" + "="*70)
        print(f"EPISODE {episode_num} COMPLETE!")
        print(f"Final Score: {final_score}")
        print(f"Lines Cleared: {total_lines}")
        print(f"Moves Made: {moves}")
        print("="*70)
        
        # Save trajectory
        if self.record_demos and episode_trajectory:
            self.demonstrations.extend(episode_trajectory)
            print(f"✓ Recorded {len(episode_trajectory)} demonstrations")
        
        return final_score, total_lines
    
    def save_demonstrations(self, filename: str = "human_demos.pkl"):
        """Save recorded demonstrations to file."""
        if not self.demonstrations:
            print("No demonstrations to save!")
            return
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.demonstrations, f)
        
        print(f"\n✓ Saved {len(self.demonstrations)} demonstrations to {filepath}")
        
        # Also save metadata
        metadata = {
            'num_demonstrations': len(self.demonstrations),
            'observation_shape': self.demonstrations[0][0].shape if self.demonstrations else None
        }
        
        with open(f"data/{filename.replace('.pkl', '_metadata.json')}", 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main function for human gameplay."""
    print("\n" + "="*70)
    print("BLOCK BLAST - HUMAN PLAYER")
    print("Record demonstrations for AI training!")
    print("="*70)
    
    num_episodes = input("\nHow many games do you want to play? (recommended: 20-50): ").strip()
    try:
        num_episodes = int(num_episodes)
    except:
        num_episodes = 5
        print(f"Using default: {num_episodes} episodes")
    
    player = HumanPlayer(record_demos=True)
    
    scores = []
    lines_cleared = []
    
    for episode in range(1, num_episodes + 1):
        score, lines = player.play_episode(episode)
        scores.append(score)
        lines_cleared.append(lines)
        
        if episode < num_episodes:
            cont = input("\nPress ENTER for next game, or 'q' to quit: ").strip().lower()
            if cont == 'q':
                break
    
    # Summary
    print("\n" + "="*70)
    print("GAMEPLAY SUMMARY")
    print("="*70)
    print(f"Episodes Played:     {len(scores)}")
    print(f"Average Score:       {np.mean(scores):.2f}")
    print(f"Best Score:          {np.max(scores)}")
    print(f"Average Lines:       {np.mean(lines_cleared):.2f}")
    print(f"Total Lines:         {np.sum(lines_cleared)}")
    print(f"Demonstrations:      {len(player.demonstrations)}")
    print("="*70)
    
    # Save demonstrations
    if player.demonstrations:
        player.save_demonstrations()
        print("\n✓ Ready for AI training!")
    else:
        print("\n⚠️  No demonstrations recorded")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Exiting...")
        sys.exit(0)


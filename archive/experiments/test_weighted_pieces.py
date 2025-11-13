"""
Test the impact of implementing weighted piece generation based on priorities.
Compare uniform vs weighted piece selection.
"""

import numpy as np
import sys
from collections import Counter

sys.path.insert(0, '/Users/rishabh/blockblast-ai/BlockBlastAICNN')

from blockblast.core import BlockBlastGame, GameState
from blockblast.pieces import ALL_PIECES, get_piece

class WeightedBlockBlastGame(BlockBlastGame):
    """Block Blast game with weighted piece generation."""
    
    def _generate_new_pieces(self):
        """Generate 3 random piece IDs using weighted selection based on priority."""
        # Get all priorities
        priorities = np.array([piece.priority for piece in ALL_PIECES])
        probabilities = priorities / priorities.sum()
        
        # Sample with replacement using priorities as weights
        pieces = np.random.choice(
            len(ALL_PIECES),
            size=self.NUM_PIECES_AVAILABLE,
            replace=True,
            p=probabilities
        )
        
        return pieces.tolist()

def simulate_games(game_class, num_games=1000, max_moves=1000):
    """Simulate games and return statistics."""
    game = game_class()
    
    scores = []
    moves_list = []
    board_fills = []
    piece_usage = Counter()
    game_over_pieces = Counter()
    
    for i in range(num_games):
        state = game.new_game()
        moves = 0
        
        while not state.done and moves < max_moves:
            valid_actions = game.get_valid_actions(state)
            if not valid_actions:
                break
            
            # Pick random valid action
            action = valid_actions[np.random.randint(len(valid_actions))]
            
            # Track piece usage
            piece_id = state.available_pieces[action.piece_index]
            piece_usage[piece_id] += 1
            
            state, reward, done, info = game.apply_action(state, action)
            moves += 1
        
        scores.append(state.score)
        moves_list.append(moves)
        board_fills.append(np.sum(state.board > 0) / 64)
        
        # Track pieces at game over
        for piece_id in state.available_pieces:
            game_over_pieces[piece_id] += 1
    
    return {
        'scores': scores,
        'moves': moves_list,
        'board_fills': board_fills,
        'piece_usage': piece_usage,
        'game_over_pieces': game_over_pieces
    }

def print_statistics(name, results):
    """Print statistics for a set of results."""
    scores = results['scores']
    moves = results['moves']
    fills = results['board_fills']
    
    print(f"\n{name}:")
    print(f"  Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Min/Max: {np.min(scores)}/{np.max(scores)}")
    print(f"  25th/75th percentile: {np.percentile(scores, 25):.1f}/{np.percentile(scores, 75):.1f}")
    print(f"  Moves: {np.mean(moves):.1f} ± {np.std(moves):.1f}")
    print(f"  Board fill: {np.mean(fills)*100:.1f}%")
    
    return np.mean(scores)

def main():
    print("\n" + "=" * 80)
    print("WEIGHTED vs UNIFORM PIECE GENERATION COMPARISON")
    print("=" * 80)
    
    num_games = 2000
    print(f"\nSimulating {num_games} games with each approach...")
    
    print("\n1. Running UNIFORM (current) piece generation...")
    uniform_results = simulate_games(BlockBlastGame, num_games=num_games)
    
    print("2. Running WEIGHTED (priority-based) piece generation...")
    weighted_results = simulate_games(WeightedBlockBlastGame, num_games=num_games)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    uniform_avg = print_statistics("UNIFORM (current)", uniform_results)
    weighted_avg = print_statistics("WEIGHTED (priority-based)", weighted_results)
    
    improvement = ((weighted_avg - uniform_avg) / uniform_avg) * 100
    
    print("\n" + "=" * 80)
    print("PIECE USAGE DISTRIBUTION")
    print("=" * 80)
    
    print("\nTop 15 pieces generated (UNIFORM):")
    total_uniform = sum(uniform_results['piece_usage'].values())
    for piece_id, count in uniform_results['piece_usage'].most_common(15):
        size = get_piece(piece_id).get_block_count()
        priority = get_piece(piece_id).priority
        pct = count / total_uniform * 100
        print(f"  Piece {piece_id:2d} (size={size}, priority={priority:2d}): {count:5d} times ({pct:.1f}%)")
    
    print("\nTop 15 pieces generated (WEIGHTED):")
    total_weighted = sum(weighted_results['piece_usage'].values())
    for piece_id, count in weighted_results['piece_usage'].most_common(15):
        size = get_piece(piece_id).get_block_count()
        priority = get_piece(piece_id).priority
        pct = count / total_weighted * 100
        print(f"  Piece {piece_id:2d} (size={size}, priority={priority:2d}): {count:5d} times ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("PIECES AT GAME OVER")
    print("=" * 80)
    
    print("\nTop 10 pieces at game over (UNIFORM):")
    for piece_id, count in uniform_results['game_over_pieces'].most_common(10):
        size = get_piece(piece_id).get_block_count()
        print(f"  Piece {piece_id:2d} (size={size}): {count:4d} times ({count/num_games*100:.1f}%)")
    
    print("\nTop 10 pieces at game over (WEIGHTED):")
    for piece_id, count in weighted_results['game_over_pieces'].most_common(10):
        size = get_piece(piece_id).get_block_count()
        print(f"  Piece {piece_id:2d} (size={size}): {count:4d} times ({count/num_games*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    print(f"\nWeighted generation score improvement: {improvement:+.1f}%")
    
    if improvement > 10:
        print("\n✓ SIGNIFICANT IMPROVEMENT with weighted generation!")
        print("  Recommendation: Implement priority-based piece selection")
    elif improvement > 0:
        print("\n→ Modest improvement with weighted generation")
        print("  The priority system helps slightly but isn't the main issue")
    else:
        print("\n✗ No improvement with weighted generation")
        print("  The priority system doesn't address the score plateau")
    
    print(f"\nKey finding: Random play averages {uniform_avg:.1f} points")
    print(f"Your agents plateau at ~60 points")
    print(f"\nThis suggests:")
    print(f"  1. The game is inherently difficult with current piece set")
    print(f"  2. Agents aren't learning much beyond random play")
    print(f"  3. May need different piece set or game modifications to achieve higher scores")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()


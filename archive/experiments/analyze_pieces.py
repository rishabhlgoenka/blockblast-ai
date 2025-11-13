"""
Analyze piece generation and placement difficulty in Block Blast.
This script investigates:
1. How pieces are generated (uniform vs weighted)
2. What % of piece sets are impossible to place on empty board
3. What % of piece sets cause game over at different board fill levels
4. Statistics on problematic piece combinations
"""

import numpy as np
import sys
from collections import Counter, defaultdict
from typing import List, Tuple

# Add the blockblast module to path (not 'blockblast copy')
sys.path.insert(0, '/Users/rishabh/blockblast-ai/BlockBlastAICNN')

from blockblast.core import BlockBlastGame, GameState
from blockblast.pieces import ALL_PIECES, get_piece

def analyze_piece_generation():
    """Analyze how pieces are generated."""
    print("=" * 80)
    print("PIECE GENERATION ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal number of piece types: {len(ALL_PIECES)}")
    print(f"\nPiece priorities (from original game):")
    
    priorities = {}
    for i, piece in enumerate(ALL_PIECES):
        size = piece.get_block_count()
        priority = piece.priority
        if priority not in priorities:
            priorities[priority] = []
        priorities[priority].append((i, size))
    
    for priority in sorted(priorities.keys()):
        pieces = priorities[priority]
        print(f"  Priority {priority:2d}: {len(pieces):2d} pieces - IDs {[p[0] for p in pieces]}")
        print(f"              Sizes: {[p[1] for p in pieces]}")
    
    # Calculate theoretical weighted distribution
    total_priority = sum(p.priority for p in ALL_PIECES)
    print(f"\nTotal priority sum: {total_priority}")
    print(f"\nIf priorities were used (weighted distribution):")
    for priority in sorted(priorities.keys()):
        pieces = priorities[priority]
        prob = (priority * len(pieces)) / total_priority * 100
        print(f"  Priority {priority:2d} pieces: {prob:.1f}% chance")
    
    print(f"\n⚠️  CURRENT IMPLEMENTATION: Uniform random (each piece has {100/len(ALL_PIECES):.2f}% chance)")
    print(f"⚠️  Priority values are IGNORED in current implementation!")
    
    return priorities

def test_empty_board_placements(num_trials=10000):
    """Test what % of random piece sets can't be placed on empty board."""
    print("\n" + "=" * 80)
    print("EMPTY BOARD PLACEMENT TEST")
    print("=" * 80)
    
    game = BlockBlastGame(rng_seed=None)
    impossible_count = 0
    piece_combo_failures = Counter()
    
    for trial in range(num_trials):
        # Generate random pieces
        pieces = [np.random.randint(0, len(ALL_PIECES)) for _ in range(3)]
        
        # Create empty board
        empty_board = np.zeros((8, 8), dtype=np.int8)
        
        # Check if at least one piece can be placed
        can_place = game.has_valid_move(empty_board, pieces)
        
        if not can_place:
            impossible_count += 1
            piece_combo = tuple(sorted(pieces))
            piece_combo_failures[piece_combo] += 1
    
    print(f"\nTrials: {num_trials}")
    print(f"Impossible to place ANY piece on EMPTY board: {impossible_count} ({impossible_count/num_trials*100:.2f}%)")
    
    if piece_combo_failures:
        print(f"\nTop 10 piece combinations that fail on empty board:")
        for combo, count in piece_combo_failures.most_common(10):
            sizes = [get_piece(p).get_block_count() for p in combo]
            print(f"  Pieces {combo} (sizes {sizes}): {count} times")
    else:
        print(f"\n✓ All random piece combinations CAN be placed on an empty board!")
    
    return impossible_count / num_trials

def simulate_game_until_failure(game, max_moves=1000):
    """Play randomly until game over, return stats."""
    state = game.new_game()
    moves = 0
    
    while not state.done and moves < max_moves:
        valid_actions = game.get_valid_actions(state)
        if not valid_actions:
            break
        
        # Pick random valid action
        action = valid_actions[np.random.randint(len(valid_actions))]
        state, reward, done, info = game.apply_action(state, action)
        moves += 1
    
    board_fill = np.sum(state.board > 0) / (8 * 8)
    
    return {
        'score': state.score,
        'moves': moves,
        'board_fill': board_fill,
        'final_pieces': state.available_pieces.copy(),
        'done': state.done
    }

def test_game_over_scenarios(num_games=1000):
    """Simulate games to see when/how they end."""
    print("\n" + "=" * 80)
    print("GAME OVER SCENARIO ANALYSIS")
    print("=" * 80)
    
    game = BlockBlastGame()
    
    scores = []
    moves_list = []
    board_fills = []
    final_piece_sets = []
    
    for i in range(num_games):
        if (i + 1) % 100 == 0:
            print(f"  Simulating game {i+1}/{num_games}...")
        
        result = simulate_game_until_failure(game)
        scores.append(result['score'])
        moves_list.append(result['moves'])
        board_fills.append(result['board_fill'])
        if result['done']:
            final_piece_sets.append(tuple(sorted(result['final_pieces'])))
    
    print(f"\n{num_games} random games simulated:")
    print(f"\nScore statistics:")
    print(f"  Mean: {np.mean(scores):.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Std: {np.std(scores):.1f}")
    print(f"  Min: {np.min(scores)}")
    print(f"  Max: {np.max(scores)}")
    print(f"  25th percentile: {np.percentile(scores, 25):.1f}")
    print(f"  75th percentile: {np.percentile(scores, 75):.1f}")
    
    print(f"\nMoves per game:")
    print(f"  Mean: {np.mean(moves_list):.1f}")
    print(f"  Median: {np.median(moves_list):.1f}")
    
    print(f"\nBoard fill at game over:")
    print(f"  Mean: {np.mean(board_fills)*100:.1f}%")
    print(f"  Median: {np.median(board_fills)*100:.1f}%")
    print(f"  Min: {np.min(board_fills)*100:.1f}%")
    print(f"  Max: {np.max(board_fills)*100:.1f}%")
    
    # Analyze piece sizes at game over
    if final_piece_sets:
        piece_combo_counter = Counter(final_piece_sets)
        print(f"\nTop 15 piece combinations at game over:")
        for combo, count in piece_combo_counter.most_common(15):
            sizes = [get_piece(p).get_block_count() for p in combo]
            total_blocks = sum(sizes)
            print(f"  Pieces {combo} (sizes {sizes}, total {total_blocks} blocks): {count} times ({count/num_games*100:.1f}%)")
    
    return scores, board_fills, final_piece_sets

def analyze_piece_size_distribution():
    """Analyze the distribution of piece sizes."""
    print("\n" + "=" * 80)
    print("PIECE SIZE DISTRIBUTION")
    print("=" * 80)
    
    size_counts = Counter()
    size_priorities = defaultdict(list)
    
    for i, piece in enumerate(ALL_PIECES):
        size = piece.get_block_count()
        priority = piece.priority
        size_counts[size] += 1
        size_priorities[size].append((i, priority))
    
    print(f"\nDistribution of piece sizes:")
    for size in sorted(size_counts.keys()):
        count = size_counts[size]
        pieces = size_priorities[size]
        avg_priority = np.mean([p[1] for p in pieces])
        print(f"  {size} blocks: {count:2d} pieces (avg priority: {avg_priority:.1f})")
        # Show piece IDs
        piece_ids = [p[0] for p in pieces]
        print(f"             IDs: {piece_ids}")
    
    # Calculate expected size in uniform vs weighted
    uniform_expected = sum(piece.get_block_count() for piece in ALL_PIECES) / len(ALL_PIECES)
    
    total_priority = sum(p.priority for p in ALL_PIECES)
    weighted_expected = sum(piece.get_block_count() * piece.priority for piece in ALL_PIECES) / total_priority
    
    print(f"\nExpected blocks per piece:")
    print(f"  Uniform random (current): {uniform_expected:.2f} blocks")
    print(f"  Weighted by priority (original game): {weighted_expected:.2f} blocks")
    print(f"  Difference: {uniform_expected - weighted_expected:.2f} blocks per piece")
    print(f"  Per 3-piece set: {3 * (uniform_expected - weighted_expected):.2f} blocks")

def test_board_fill_scenarios(num_trials=5000):
    """Test placement success rate at different board fill levels."""
    print("\n" + "=" * 80)
    print("BOARD FILL LEVEL ANALYSIS")
    print("=" * 80)
    
    game = BlockBlastGame()
    
    fill_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for fill_level in fill_levels:
        print(f"\nTesting at {fill_level*100:.0f}% board fill...")
        impossible_count = 0
        
        for trial in range(num_trials):
            # Create board with given fill level
            board = np.zeros((8, 8), dtype=np.int8)
            num_cells_to_fill = int(64 * fill_level)
            
            # Randomly fill cells
            if num_cells_to_fill > 0:
                flat_indices = np.random.choice(64, num_cells_to_fill, replace=False)
                for idx in flat_indices:
                    row, col = idx // 8, idx % 8
                    board[row, col] = 1
            
            # Generate random pieces
            pieces = [np.random.randint(0, len(ALL_PIECES)) for _ in range(3)]
            
            # Check if at least one piece can be placed
            can_place = game.has_valid_move(board, pieces)
            
            if not can_place:
                impossible_count += 1
        
        print(f"  Impossible rate: {impossible_count/num_trials*100:.2f}% ({impossible_count}/{num_trials})")

def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "BLOCK BLAST PIECE ANALYSIS" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 1. Analyze piece generation
    analyze_piece_generation()
    
    # 2. Analyze piece sizes
    analyze_piece_size_distribution()
    
    # 3. Test empty board
    test_empty_board_placements(num_trials=50000)
    
    # 4. Test different fill levels
    test_board_fill_scenarios(num_trials=5000)
    
    # 5. Simulate games to see typical scores/endings
    scores, board_fills, final_pieces = test_game_over_scenarios(num_games=1000)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 80)
    
    print("\n1. PIECE GENERATION:")
    print("   - Current implementation uses UNIFORM random selection")
    print("   - Original game priorities are IGNORED")
    print("   - This means larger pieces (5-9 blocks) appear more often than intended")
    
    print("\n2. IMPOSSIBLE PLACEMENTS:")
    print("   - On empty board: Very rare (likely 0%)")
    print("   - But as board fills, certain piece combinations become impossible")
    
    print("\n3. SCORE PLATEAU:")
    avg_score = np.mean(scores)
    print(f"   - Random play achieves mean score of {avg_score:.1f}")
    print(f"   - Your agents plateau around 60")
    if avg_score < 100:
        print(f"   - Random play also scores low, suggesting piece generation IS an issue")
    else:
        print(f"   - Random play scores higher, suggesting learning/strategy issues")
    
    print("\n4. RECOMMENDATIONS:")
    print("   a) Consider implementing weighted piece generation using priorities")
    print("   b) Smaller pieces (1-4 blocks) should appear more frequently")
    print("   c) The score plateau might be due to:")
    print("      - Agent not learning long-term board management")
    print("      - Reward function not incentivizing board cleanliness")
    print("      - Limited exploration of strategic placements")
    print("      - Piece generation being too difficult (if random play also scores low)")
    
    print("\n" + "=" * 80)
    print()

if __name__ == "__main__":
    main()

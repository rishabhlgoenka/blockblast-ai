#!/usr/bin/env python3
"""
Pygame RLHF - Watch AI play and rate its moves visually!

Watch the AI play Block Blast in a beautiful GUI and rate each move with number keys.
"""

import pygame
import sys
import pickle
import numpy as np
from stable_baselines3 import PPO
from ppo_env import BlockBlastEnv  # Use PPO-compatible environment
from blockblast.pieces import get_piece
from blockblast.env import decode_action  # For displaying move info


# Colors (matching original game)
COLOR_BG = (20, 20, 40)
COLOR_GRID = (40, 40, 80)
COLOR_CELL_EMPTY = (30, 30, 60)
COLOR_CELL_FILLED = (100, 200, 255)
COLOR_PIECE = (255, 200, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_POSITIVE = (100, 255, 100)
COLOR_NEGATIVE = (255, 100, 100)
COLOR_NEUTRAL = (200, 200, 200)

# Layout
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BOARD_CELL_SIZE = 55
BOARD_LEFT = 50
BOARD_TOP = 100

# Panel
PANEL_LEFT = 600
PANEL_WIDTH = 550


class RLHFApp:
    """Pygame app for visual RLHF - watch AI play and rate moves."""
    
    def __init__(self, model_path: str):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RLHF - Rate AI Moves")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Load model
        self.model = PPO.load(model_path)
        self.env = BlockBlastEnv()
        
        # State
        self.obs = None
        self.waiting_for_rating = False
        self.current_rating = 0
        self.last_action = None
        self.last_obs = None
        self.game_active = False
        self.done = False
        
        # Feedback collection
        self.feedback_data = []
        self.games_completed = 0
        self.moves_rated = 0
        self.target_games = 0
        
        # Animation
        self.show_rating_display = False
        self.rating_display_timer = 0
        
        # Instructions state
        self.showing_instructions = True
    
    def start_new_game(self):
        """Start a new game."""
        self.obs, _ = self.env.reset()
        self.game_active = True
        self.done = False
        self.waiting_for_rating = False
        print(f"\n=== Game {self.games_completed + 1}/{self.target_games} Started ===")
    
    def ai_make_move(self):
        """AI makes one move."""
        if self.done:
            return
        
        # Save state before move
        self.last_obs = self.obs.copy()
        
        # AI predicts action
        self.last_action, _ = self.model.predict(self.obs, deterministic=False)
        
        # Apply action
        self.obs, reward, terminated, truncated, info = self.env.step(self.last_action)
        self.done = terminated or truncated
        
        # Wait for human rating
        self.waiting_for_rating = True
        
        # Print move info
        piece_idx, x, y = decode_action(self.last_action)
        print(f"Move: Piece {piece_idx} at ({y}, {x}) | Score: {self.env.state.score}")
        if info.get('lines_cleared', 0) > 0:
            print(f"  ✓ Cleared {info['lines_cleared']} lines!")
    
    def record_rating(self, rating: int):
        """Record human rating for last move."""
        if not self.waiting_for_rating:
            return
        
        # Save feedback
        self.feedback_data.append((self.last_obs, self.last_action, rating))
        self.moves_rated += 1
        self.waiting_for_rating = False
        
        # Show rating animation
        self.show_rating_display = True
        self.rating_display_timer = 60  # Show for 1 second (60 frames)
        self.current_rating = rating
        
        print(f"  → Rated: {rating:+d}/10")
        
        # Check if game over
        if self.done:
            self.games_completed += 1
            print(f"Game Over! Score: {self.env.state.score}")
            print(f"Progress: {self.games_completed}/{self.target_games} games, {self.moves_rated} moves rated")
            
            if self.games_completed < self.target_games:
                # Auto-start next game after brief pause
                pygame.time.wait(1000)
                self.start_new_game()
            else:
                self.game_active = False
                print("\n=== All games completed! ===")
                print(f"Total feedback: {len(self.feedback_data)} moves")
    
    def draw_instructions(self):
        """Draw instruction screen."""
        self.screen.fill(COLOR_BG)
        
        y = 100
        title = self.font_large.render("RLHF - Train AI with YOUR Feedback", True, COLOR_TEXT)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, y))
        
        y += 80
        instructions = [
            "How it works:",
            "",
            "1. Watch the AI play Block Blast",
            "2. Rate each move using NUMBER KEYS:",
            "",
            "   1-9 = Rate from +1 to +9 (good moves)",
            "   0   = Rate 0 (neutral/ok move)",
            "   -   = Rate -5 (bad move)",
            "   =   = Rate +10 (excellent move!)",
            "",
            "3. AI learns from YOUR ratings to play better!",
            "",
            "Rating Guide:",
            "  +10: Excellent! (cleared lines, smart setup)",
            "  +5:  Good move (solid play)",
            "   0:  Neutral (neither good nor bad)",
            "  -5:  Bad move (missed opportunity)",
            "",
            f"Games to rate: {self.target_games}",
            "",
            "Press SPACE to start!"
        ]
        
        for line in instructions:
            text = self.font_small.render(line, True, COLOR_TEXT)
            self.screen.blit(text, (100, y))
            y += 35
    
    def draw_board(self):
        """Draw the game board."""
        board = self.env.state.board
        
        for row in range(8):
            for col in range(8):
                x = BOARD_LEFT + col * BOARD_CELL_SIZE
                y = BOARD_TOP + row * BOARD_CELL_SIZE
                
                # Cell background
                if board[row, col] > 0:
                    color = COLOR_CELL_FILLED
                else:
                    color = COLOR_CELL_EMPTY
                
                pygame.draw.rect(self.screen, color, (x, y, BOARD_CELL_SIZE - 2, BOARD_CELL_SIZE - 2))
                pygame.draw.rect(self.screen, COLOR_GRID, (x, y, BOARD_CELL_SIZE - 2, BOARD_CELL_SIZE - 2), 1)
    
    def draw_pieces(self):
        """Draw current pieces."""
        pieces_y = BOARD_TOP + 8 * BOARD_CELL_SIZE + 30
        pieces_x = BOARD_LEFT
        
        label = self.font.render("Current Pieces:", True, COLOR_TEXT)
        self.screen.blit(label, (pieces_x, pieces_y - 30))
        
        for i, piece_id in enumerate(self.env.state.available_pieces):
            if piece_id is None:
                continue
            
            piece = get_piece(piece_id)
            piece_offset_x = pieces_x + i * 120
            
            for dy, dx in piece.get_cells():
                x = piece_offset_x + dx * 30
                y = pieces_y + dy * 30
                pygame.draw.rect(self.screen, COLOR_PIECE, (x, y, 28, 28))
    
    def draw_info_panel(self):
        """Draw info panel."""
        x = PANEL_LEFT
        y = 100
        
        # Title
        title = self.font_large.render("AI Playing...", True, COLOR_TEXT)
        self.screen.blit(title, (x, y))
        y += 60
        
        # Game stats
        stats = [
            ("Score:", str(self.env.state.score)),
            ("Moves:", str(self.env.state.moves)),
            ("Combo:", f"{self.env.state.combo}x"),
            ("Lines:", str(self.env.state.lines_cleared_total)),
            ("", ""),
            ("Games:", f"{self.games_completed + 1}/{self.target_games}"),
            ("Rated:", str(self.moves_rated)),
        ]
        
        for label, value in stats:
            if label:
                text_label = self.font.render(label, True, COLOR_NEUTRAL)
                text_value = self.font.render(value, True, COLOR_TEXT)
                self.screen.blit(text_label, (x, y))
                self.screen.blit(text_value, (x + 150, y))
            y += 40
        
        # Rating prompt
        y += 40
        if self.waiting_for_rating:
            prompt = self.font_large.render("RATE THIS MOVE!", True, COLOR_POSITIVE)
            self.screen.blit(prompt, (x, y))
            y += 50
            
            hint1 = self.font_small.render("1-9: +1 to +9", True, COLOR_TEXT)
            hint2 = self.font_small.render("0: Neutral", True, COLOR_TEXT)
            hint3 = self.font_small.render("-: Bad (-5)", True, COLOR_TEXT)
            hint4 = self.font_small.render("=: Excellent (+10)", True, COLOR_TEXT)
            
            self.screen.blit(hint1, (x, y))
            self.screen.blit(hint2, (x, y + 30))
            self.screen.blit(hint3, (x, y + 60))
            self.screen.blit(hint4, (x, y + 90))
        else:
            waiting = self.font.render("Waiting for AI...", True, COLOR_NEUTRAL)
            self.screen.blit(waiting, (x, y))
        
        # Show last rating
        if self.show_rating_display:
            y = 400
            rating_color = COLOR_POSITIVE if self.current_rating > 0 else (COLOR_NEGATIVE if self.current_rating < 0 else COLOR_NEUTRAL)
            rating_text = self.font_large.render(f"Rated: {self.current_rating:+d}/10", True, rating_color)
            self.screen.blit(rating_text, (x, y))
    
    def draw_completion_screen(self):
        """Draw completion screen."""
        self.screen.fill(COLOR_BG)
        
        y = 200
        title = self.font_large.render("Feedback Collection Complete!", True, COLOR_POSITIVE)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, y))
        
        y += 80
        stats = [
            f"Games Played: {self.games_completed}",
            f"Moves Rated: {self.moves_rated}",
            "",
            "Data saved to: human_feedback.pkl",
            "",
            "Press SPACE to start fine-tuning...",
            "Press ESC to exit"
        ]
        
        for line in stats:
            text = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
            y += 50
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Instructions screen
                if self.showing_instructions:
                    if event.key == pygame.K_SPACE:
                        self.showing_instructions = False
                        self.start_new_game()
                    return True
                
                # Completion screen
                if not self.game_active and self.games_completed >= self.target_games:
                    if event.key == pygame.K_SPACE:
                        return False  # Start fine-tuning
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    return True
                
                # Rating keys
                if self.waiting_for_rating:
                    rating = None
                    
                    # Number keys 0-9
                    if pygame.K_0 <= event.key <= pygame.K_9:
                        rating = event.key - pygame.K_0
                    # Minus key for -5
                    elif event.key == pygame.K_MINUS:
                        rating = -5
                    # Equals key for +10
                    elif event.key == pygame.K_EQUALS:
                        rating = 10
                    
                    if rating is not None:
                        self.record_rating(rating)
        
        return True
    
    def update(self):
        """Update game state."""
        if not self.game_active:
            return
        
        # Update rating display timer
        if self.show_rating_display:
            self.rating_display_timer -= 1
            if self.rating_display_timer <= 0:
                self.show_rating_display = False
        
        # AI makes move if not waiting for rating
        if not self.waiting_for_rating and not self.done:
            pygame.time.wait(500)  # Brief pause between moves
            self.ai_make_move()
    
    def draw(self):
        """Draw everything."""
        if self.showing_instructions:
            self.draw_instructions()
        elif not self.game_active and self.games_completed >= self.target_games:
            self.draw_completion_screen()
        else:
            self.screen.fill(COLOR_BG)
            
            # Title
            title = self.font_large.render("RLHF Training", True, COLOR_TEXT)
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
            
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()
    
    def run(self, num_games: int):
        """Run the RLHF collection loop."""
        self.target_games = num_games
        
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        return self.feedback_data


def fine_tune_with_feedback(model_path: str, feedback_data, epochs: int = 30):
    """Fine-tune model with collected feedback (same as manual_rlhf.py)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from train_ppo_cnn import BlockBlastCNN
    
    class HumanFeedbackDataset(Dataset):
        def __init__(self, feedback_data):
            self.observations = []
            self.actions = []
            self.scores = []
            
            for obs, action, score in feedback_data:
                self.observations.append(obs)
                self.actions.append(action)
                self.scores.append(score / 10.0)  # Normalize to [-1, 1]
            
            self.observations = np.array(self.observations, dtype=np.float32) / 255.0
            self.actions = np.array(self.actions, dtype=np.int64)
            self.scores = np.array(self.scores, dtype=np.float32)
        
        def __len__(self):
            return len(self.actions)
        
        def __getitem__(self, idx):
            return (
                torch.FloatTensor(self.observations[idx]),
                torch.LongTensor([self.actions[idx]]),
                torch.FloatTensor([self.scores[idx]])
            )
    
    print("\n" + "="*70)
    print("FINE-TUNING WITH HUMAN FEEDBACK")
    print("="*70)
    print(f"Training data: {len(feedback_data)} rated moves")
    print(f"Epochs: {epochs}")
    print("")
    
    # Load model
    model = PPO.load(model_path)
    policy = model.policy
    
    # Create dataset
    dataset = HumanFeedbackDataset(feedback_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=0.0001)
    
    # Training loop
    print("Training...")
    for epoch in range(epochs):
        total_loss = 0.0
        
        for obs_batch, action_batch, score_batch in dataloader:
            optimizer.zero_grad()
            
            # Get policy predictions
            features = policy.extract_features(obs_batch)
            action_logits = policy.action_net(features)
            
            # Calculate loss
            log_probs = torch.log_softmax(action_logits, dim=1)
            selected_log_probs = log_probs.gather(1, action_batch)
            loss = -(selected_log_probs * score_batch).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save fine-tuned model with versioning
    import re
    if '_v2' in model_path:
        output_path = model_path.replace('_v2.zip', '_v3.zip')
    elif '_rlhf' in model_path and '_v' not in model_path:
        output_path = model_path.replace('_rlhf.zip', '_rlhf_v2.zip')
    else:
        output_path = model_path.replace('.zip', '_rlhf.zip')
    model.save(output_path)
    
    print(f"\n✓ Fine-tuned model saved to: {output_path}")
    return output_path


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("PYGAME RLHF - Visual AI Training!")
    print("="*70)
    print("\nWatch the AI play in a beautiful GUI and rate its moves.")
    print("This is how ChatGPT was trained - from human feedback!")
    print("")
    
    # Use the latest RLHF model (iterative improvement)
    # Check for v2, otherwise use v1
    import os
    if os.path.exists("models/human_imitation_rlhf_v2.zip"):
        model_path = "models/human_imitation_rlhf_v2.zip"
    else:
        model_path = "models/human_imitation_rlhf.zip"
    
    # Get number of games
    try:
        num_games = int(input("How many games to rate? (recommend 5-10): ").strip() or "5")
    except ValueError:
        num_games = 5
    
    print(f"\nStarting RLHF with {num_games} games...")
    print("The Pygame window will open shortly.")
    print("")
    
    # Run RLHF collection
    app = RLHFApp(model_path)
    feedback_data = app.run(num_games)
    
    if len(feedback_data) == 0:
        print("\n❌ No feedback collected. Exiting.")
        return
    
    # Save feedback
    with open('human_feedback.pkl', 'wb') as f:
        pickle.dump(feedback_data, f)
    print(f"✓ Feedback saved to: human_feedback.pkl")
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    rlhf_model_path = fine_tune_with_feedback(model_path, feedback_data, epochs=30)
    
    # Evaluate
    print("\nEvaluating improved model...")
    from eval_ppo_cnn import evaluate_model
    evaluate_model(rlhf_model_path, episodes=20, render=False)
    
    print("\n" + "="*70)
    print("✅ RLHF COMPLETE!")
    print("="*70)
    print(f"Original model: {model_path}")
    print(f"RLHF model: {rlhf_model_path}")
    print("\nThe AI learned from your feedback!")
    print("="*70)


if __name__ == "__main__":
    main()


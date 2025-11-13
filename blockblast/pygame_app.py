"""
Pygame frontend for Block Blast.

Human-playable version of the game with mouse controls.
"""
import sys
from typing import Optional, Tuple, List
import numpy as np

try:
    import pygame
except ImportError:
    print("Error: pygame is not installed. Install it with: pip install pygame")
    sys.exit(1)

from .env import BlockBlastEnv
from .core import Action
from .pieces import get_piece


# Colors (RGB)
COLORS = {
    'background': (20, 24, 29),
    'grid_line': (40, 48, 58),
    'empty_cell': (30, 36, 44),
    'text': (220, 220, 220),
    'piece_preview_bg': (35, 42, 51),
    'selected': (255, 255, 100),
    'valid_preview': (100, 255, 100, 128),
    'invalid_preview': (255, 100, 100, 128),
    'game_over': (200, 50, 50),
}

# Piece colors (matching the 7 colors from original game)
PIECE_COLORS = [
    (139, 195, 74),   # Green
    (255, 193, 7),    # Amber
    (3, 169, 244),    # Light Blue
    (233, 30, 99),    # Pink
    (156, 39, 176),   # Purple
    (255, 87, 34),    # Deep Orange
    (0, 188, 212),    # Cyan
]


class PygameApp:
    """Pygame application for playing Block Blast."""
    
    def __init__(self, window_width: int = 1000, window_height: int = 700):
        """
        Initialize the pygame app.
        
        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        pygame.init()
        
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Block Blast")
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Game state
        self.env = BlockBlastEnv()
        self.env.reset()
        
        # UI layout parameters
        self.board_size = 500  # Size of the board display area
        self.cell_size = self.board_size // 8
        self.board_x = 50
        self.board_y = 100
        
        # Piece preview area
        self.preview_area_x = self.board_x + self.board_size + 50
        self.preview_area_y = self.board_y
        self.preview_cell_size = 25
        self.preview_spacing = 150
        
        # Selection state
        self.selected_piece_idx: Optional[int] = None
        self.dragging = False
        self.drag_offset = (0, 0)
        self.preview_position: Optional[Tuple[int, int]] = None
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.running = True
    
    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(self.fps)
        
        pygame.quit()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_mouse_down(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    self.handle_mouse_up(event.pos)
            
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset game
                    self.env.reset()
                    self.selected_piece_idx = None
                    self.dragging = False
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def handle_mouse_down(self, pos: Tuple[int, int]):
        """Handle mouse button down."""
        if self.env.current_state and self.env.current_state.done:
            return
        
        # Check if clicked on a piece preview
        piece_idx = self.get_piece_at_position(pos)
        if piece_idx is not None:
            self.selected_piece_idx = piece_idx
            self.dragging = True
            # Calculate drag offset
            piece_rect = self.get_piece_preview_rect(piece_idx)
            self.drag_offset = (pos[0] - piece_rect[0], pos[1] - piece_rect[1])
    
    def handle_mouse_up(self, pos: Tuple[int, int]):
        """Handle mouse button up."""
        if not self.dragging or self.selected_piece_idx is None:
            return
        
        if self.env.current_state and self.env.current_state.done:
            self.dragging = False
            return
        
        # Try to place the piece
        if self.preview_position is not None:
            row, col = self.preview_position

            # Encode the action: piece_idx, x (col), y (row)
            from .env import encode_action
            action_id = encode_action(self.selected_piece_idx, col, row)
            
            # Apply the action using action ID
            try:
                obs, reward, done, info = self.env.step(action_id)
                if not info.get('valid', False):
                    # Invalid action, give feedback (optional)
                    print(f"Invalid placement: {info.get('reason', 'unknown')}")
            except Exception as e:
                print(f"Error applying action: {e}")
                import traceback
                traceback.print_exc()
        
        self.selected_piece_idx = None
        self.dragging = False
        self.preview_position = None
    
    def handle_mouse_motion(self, pos: Tuple[int, int]):
        """Handle mouse motion."""
        if self.dragging and self.selected_piece_idx is not None:
            # Update preview position
            self.preview_position = self.get_board_position_from_mouse(pos)
    
    def get_piece_at_position(self, pos: Tuple[int, int]) -> Optional[int]:
        """Check if mouse is over a piece preview."""
        if not self.env.current_state:
            return None
        
        for i in range(len(self.env.current_state.available_pieces)):
            rect = self.get_piece_preview_rect(i)
            if (rect[0] <= pos[0] <= rect[0] + rect[2] and
                rect[1] <= pos[1] <= rect[1] + rect[3]):
                return i
        
        return None
    
    def get_piece_preview_rect(self, piece_idx: int) -> Tuple[int, int, int, int]:
        """Get the rectangle for a piece preview."""
        x = self.preview_area_x
        y = self.preview_area_y + piece_idx * self.preview_spacing
        size = 5 * self.preview_cell_size
        return (x, y, size, size)
    
    def get_board_position_from_mouse(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert mouse position to board cell coordinates."""
        if self.selected_piece_idx is None:
            return None
        
        # Get piece
        piece_id = self.env.current_state.available_pieces[self.selected_piece_idx]
        piece = get_piece(piece_id)
        
        # Calculate where the piece origin would be
        # Account for drag offset and piece centering
        piece_x = pos[0] - self.drag_offset[0]
        piece_y = pos[1] - self.drag_offset[1]
        
        # Convert to board coordinates (top-left of piece's 5x5 matrix)
        row = (piece_y - self.board_y) // self.cell_size
        col = (piece_x - self.board_x) // self.cell_size
        
        return (row, col)
    
    def update(self):
        """Update game state."""
        pass  # Game state is updated by event handlers
    
    def render(self):
        """Render the game."""
        self.screen.fill(COLORS['background'])
        
        # Render UI elements
        self.render_header()
        self.render_board()
        self.render_piece_previews()
        
        # Render dragged piece if dragging
        if self.dragging and self.selected_piece_idx is not None:
            mouse_pos = pygame.mouse.get_pos()
            self.render_dragged_piece(mouse_pos)
        
        # Render game over screen
        if self.env.current_state and self.env.current_state.done:
            self.render_game_over()
        
        pygame.display.flip()
    
    def render_header(self):
        """Render score and stats."""
        if not self.env.current_state:
            return
        
        state = self.env.current_state
        
        # Title
        title = self.font_large.render("Block Blast", True, COLORS['text'])
        self.screen.blit(title, (self.board_x, 20))
        
        # Stats
        stats_text = f"Score: {state.score}  |  Moves: {state.moves}  |  Combo: {state.combo}x"
        stats = self.font_small.render(stats_text, True, COLORS['text'])
        self.screen.blit(stats, (self.board_x, 70))
    
    def render_board(self):
        """Render the game board."""
        if not self.env.current_state:
            return
        
        board = self.env.current_state.board
        
        # Draw cells
        for row in range(8):
            for col in range(8):
                x = self.board_x + col * self.cell_size
                y = self.board_y + row * self.cell_size
                
                # Determine cell color
                cell_value = board[row, col]
                if cell_value > 0:
                    # Filled cell - use piece color
                    color = PIECE_COLORS[(cell_value - 1) % len(PIECE_COLORS)]
                else:
                    color = COLORS['empty_cell']
                
                # Draw cell
                pygame.draw.rect(self.screen, color, 
                               (x, y, self.cell_size, self.cell_size))
                
                # Draw grid line
                pygame.draw.rect(self.screen, COLORS['grid_line'], 
                               (x, y, self.cell_size, self.cell_size), 1)
        
        # Draw preview of where piece would be placed
        if self.dragging and self.preview_position is not None:
            self.render_placement_preview()
    
    def render_placement_preview(self):
        """Render preview of piece placement on board."""
        if not self.dragging or self.selected_piece_idx is None or self.preview_position is None:
            return
        
        piece_id = self.env.current_state.available_pieces[self.selected_piece_idx]
        piece = get_piece(piece_id)
        row, col = self.preview_position
        
        # Check if placement is valid
        valid = self.env.game.can_place_piece(
            self.env.current_state.board, piece, row, col
        )
        
        color = COLORS['valid_preview'] if valid else COLORS['invalid_preview']
        
        # Create semi-transparent surface
        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        surface.fill(color)
        
        # Draw piece cells
        cells = piece.get_cells()
        for cell_row, cell_col in cells:
            board_row = row + cell_row
            board_col = col + cell_col
            
            if 0 <= board_row < 8 and 0 <= board_col < 8:
                x = self.board_x + board_col * self.cell_size
                y = self.board_y + board_row * self.cell_size
                self.screen.blit(surface, (x, y))
    
    def render_piece_previews(self):
        """Render available pieces."""
        if not self.env.current_state:
            return
        
        for i, piece_id in enumerate(self.env.current_state.available_pieces):
            if i == self.selected_piece_idx and self.dragging:
                continue  # Don't draw the piece being dragged
            
            piece = get_piece(piece_id)
            rect = self.get_piece_preview_rect(i)
            
            # Draw background
            pygame.draw.rect(self.screen, COLORS['piece_preview_bg'], rect)
            pygame.draw.rect(self.screen, COLORS['grid_line'], rect, 2)
            
            # Draw piece
            color = PIECE_COLORS[piece_id % len(PIECE_COLORS)]
            self.render_piece_shape(piece, rect[0], rect[1], 
                                   self.preview_cell_size, color)
            
            # Draw piece info
            info_text = f"Piece {i+1}"
            info = self.font_small.render(info_text, True, COLORS['text'])
            self.screen.blit(info, (rect[0], rect[1] - 25))
    
    def render_dragged_piece(self, pos: Tuple[int, int]):
        """Render piece being dragged."""
        if self.selected_piece_idx is None:
            return
        
        piece_id = self.env.current_state.available_pieces[self.selected_piece_idx]
        piece = get_piece(piece_id)
        color = PIECE_COLORS[piece_id % len(PIECE_COLORS)]
        
        # Draw at mouse position with offset
        x = pos[0] - self.drag_offset[0]
        y = pos[1] - self.drag_offset[1]
        
        self.render_piece_shape(piece, x, y, self.preview_cell_size, color, alpha=200)
    
    def render_piece_shape(self, piece, x: int, y: int, cell_size: int, 
                          color: Tuple[int, int, int], alpha: int = 255):
        """Render a piece shape."""
        cells = piece.get_cells()
        
        for cell_row, cell_col in cells:
            cx = x + cell_col * cell_size
            cy = y + cell_row * cell_size
            
            if alpha < 255:
                # Create semi-transparent surface
                surface = pygame.Surface((cell_size - 2, cell_size - 2), pygame.SRCALPHA)
                surface.fill((*color, alpha))
                self.screen.blit(surface, (cx + 1, cy + 1))
            else:
                pygame.draw.rect(self.screen, color, 
                               (cx + 1, cy + 1, cell_size - 2, cell_size - 2))
            
            # Draw border
            pygame.draw.rect(self.screen, COLORS['grid_line'], 
                           (cx, cy, cell_size, cell_size), 1)
    
    def render_game_over(self):
        """Render game over screen."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        text = self.font_large.render("GAME OVER", True, COLORS['game_over'])
        text_rect = text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50))
        self.screen.blit(text, text_rect)
        
        # Final score
        score_text = self.font_medium.render(
            f"Final Score: {self.env.current_state.score}", 
            True, COLORS['text']
        )
        score_rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 20))
        self.screen.blit(score_text, score_rect)
        
        # Instructions
        restart_text = self.font_small.render(
            "Press R to restart or ESC to quit", 
            True, COLORS['text']
        )
        restart_rect = restart_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 80))
        self.screen.blit(restart_text, restart_rect)


def main():
    """Main entry point."""
    app = PygameApp()
    app.run()


if __name__ == "__main__":
    main()


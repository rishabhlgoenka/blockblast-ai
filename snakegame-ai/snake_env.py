"""
Snake Environment Wrapper
Provides a Gym-style interface for the Snake game for RL training.
"""

import pygame
import random
import numpy as np
from enum import Enum


class Direction(Enum):
    """Enum for snake directions"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeEnv:
    """
    Snake game environment for reinforcement learning.
    
    Action space: 3 actions
        0 = Turn left (relative to current direction)
        1 = Go straight (maintain direction)
        2 = Turn right (relative to current direction)
    
    State space: 11 features
        - Danger straight ahead (1 or 0)
        - Danger to the right (1 or 0)
        - Danger to the left (1 or 0)
        - Current direction: UP (1 or 0)
        - Current direction: DOWN (1 or 0)
        - Current direction: LEFT (1 or 0)
        - Current direction: RIGHT (1 or 0)
        - Food location: above (1 or 0)
        - Food location: below (1 or 0)
        - Food location: left (1 or 0)
        - Food location: right (1 or 0)
    
    Reward structure:
        +10 for eating an apple
        -10 for dying (collision with wall or self)
        -0.01 per step to encourage efficiency
    """
    
    def __init__(self, frame_size_x=720, frame_size_y=480, block_size=10, render_mode=False):
        """
        Initialize the Snake environment.
        
        Args:
            frame_size_x: Width of the game window
            frame_size_y: Height of the game window
            block_size: Size of each grid block
            render_mode: Whether to render the game visually
        """
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.block_size = block_size
        self.render_mode = render_mode
        
        # Initialize Pygame only if rendering
        if self.render_mode:
            pygame.init()
            pygame.display.set_caption('Snake RL Agent')
            self.game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
            self.fps_controller = pygame.time.Clock()
            
            # Colors
            self.black = pygame.Color(0, 0, 0)
            self.white = pygame.Color(255, 255, 255)
            self.red = pygame.Color(255, 0, 0)
            self.green = pygame.Color(0, 255, 0)
        
        # Game state variables
        self.snake_pos = None
        self.snake_body = None
        self.food_pos = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.max_steps_without_food = 100 * ((frame_size_x * frame_size_y) // (block_size * block_size))
        
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation (state)
        """
        # Initialize snake position (centered)
        start_x = self.frame_size_x // 2
        start_y = self.frame_size_y // 2
        start_x = (start_x // self.block_size) * self.block_size
        start_y = (start_y // self.block_size) * self.block_size
        
        self.snake_pos = [start_x, start_y]
        self.snake_body = [
            [start_x, start_y],
            [start_x - self.block_size, start_y],
            [start_x - (2 * self.block_size), start_y]
        ]
        
        # Random food position
        self.food_pos = self._spawn_food()
        
        # Start moving right
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        
        return self._get_state()
    
    def _spawn_food(self):
        """Spawn food at a random position not occupied by snake"""
        while True:
            food_pos = [
                random.randrange(1, (self.frame_size_x // self.block_size)) * self.block_size,
                random.randrange(1, (self.frame_size_y // self.block_size)) * self.block_size
            ]
            # Make sure food doesn't spawn on snake body
            if food_pos not in self.snake_body:
                return food_pos
    
    def _get_state(self):
        """
        Get current state representation.
        
        Returns:
            numpy array of 11 features
        """
        head = self.snake_pos
        
        # Points around the head based on current direction
        point_l = None
        point_r = None
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        point_left = [head[0] - self.block_size, head[1]]
        point_right = [head[0] + self.block_size, head[1]]
        
        # Determine left, right, and straight based on current direction
        if self.direction == Direction.UP:
            point_straight = point_u
            point_l = point_left
            point_r = point_right
        elif self.direction == Direction.DOWN:
            point_straight = point_d
            point_l = point_right
            point_r = point_left
        elif self.direction == Direction.LEFT:
            point_straight = point_left
            point_l = point_d
            point_r = point_u
        elif self.direction == Direction.RIGHT:
            point_straight = point_right
            point_l = point_u
            point_r = point_d
        
        # Check for danger in each direction (collision with wall or self)
        danger_straight = self._is_collision(point_straight)
        danger_right = self._is_collision(point_r)
        danger_left = self._is_collision(point_l)
        
        # Current direction one-hot encoding
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        
        # Food location relative to head
        food_up = self.food_pos[1] < head[1]
        food_down = self.food_pos[1] > head[1]
        food_left = self.food_pos[0] < head[0]
        food_right = self.food_pos[0] > head[0]
        
        # Create state array
        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_u,
            dir_d,
            dir_l,
            dir_r,
            food_up,
            food_down,
            food_left,
            food_right
        ], dtype=int)
        
        return state
    
    def _is_collision(self, point):
        """
        Check if a point results in collision.
        
        Args:
            point: [x, y] coordinates to check
            
        Returns:
            True if collision, False otherwise
        """
        # Wall collision
        if (point[0] < 0 or point[0] >= self.frame_size_x or
            point[1] < 0 or point[1] >= self.frame_size_y):
            return True
        
        # Self collision
        if point in self.snake_body[1:]:
            return True
        
        return False
    
    def step(self, action):
        """
        Take an action and advance the environment by one step.
        
        Args:
            action: 0 (turn left), 1 (straight), 2 (turn right)
            
        Returns:
            observation: current state
            reward: reward for this step
            done: whether episode is finished
            info: additional information (score, steps)
        """
        self.steps += 1
        self.steps_without_food += 1
        
        # Convert relative action to absolute direction
        self._update_direction(action)
        
        # Move the snake
        if self.direction == Direction.UP:
            self.snake_pos[1] -= self.block_size
        elif self.direction == Direction.DOWN:
            self.snake_pos[1] += self.block_size
        elif self.direction == Direction.LEFT:
            self.snake_pos[0] -= self.block_size
        elif self.direction == Direction.RIGHT:
            self.snake_pos[0] += self.block_size
        
        # Insert new head position
        self.snake_body.insert(0, list(self.snake_pos))
        
        # Initialize reward
        reward = -0.01  # Small negative reward per step
        done = False
        
        # Check if food is eaten
        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 10  # Big positive reward for eating food
            self.food_pos = self._spawn_food()
            self.steps_without_food = 0
        else:
            # Remove tail if no food eaten
            self.snake_body.pop()
        
        # Check for game over conditions
        if self._is_collision(self.snake_pos):
            reward = -10  # Big negative reward for dying
            done = True
        
        # Check if snake is stuck (too many steps without eating)
        if self.steps_without_food > self.max_steps_without_food:
            reward = -10
            done = True
        
        # Get new state
        observation = self._get_state()
        
        # Info dictionary
        info = {
            'score': self.score,
            'steps': self.steps
        }
        
        return observation, reward, done, info
    
    def _update_direction(self, action):
        """
        Update direction based on relative action.
        
        Args:
            action: 0 (left), 1 (straight), 2 (right)
        """
        if action == 0:  # Turn left
            if self.direction == Direction.UP:
                self.direction = Direction.LEFT
            elif self.direction == Direction.DOWN:
                self.direction = Direction.RIGHT
            elif self.direction == Direction.LEFT:
                self.direction = Direction.DOWN
            elif self.direction == Direction.RIGHT:
                self.direction = Direction.UP
        elif action == 1:  # Go straight
            pass  # Direction remains the same
        elif action == 2:  # Turn right
            if self.direction == Direction.UP:
                self.direction = Direction.RIGHT
            elif self.direction == Direction.DOWN:
                self.direction = Direction.LEFT
            elif self.direction == Direction.LEFT:
                self.direction = Direction.UP
            elif self.direction == Direction.RIGHT:
                self.direction = Direction.DOWN
    
    def render(self, fps=25):
        """
        Render the game state.
        
        Args:
            fps: Frames per second for game speed
        """
        if not self.render_mode:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Fill background
        self.game_window.fill(self.black)
        
        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green,
                           pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        
        # Draw food
        pygame.draw.rect(self.game_window, self.white,
                        pygame.Rect(self.food_pos[0], self.food_pos[1], 
                                  self.block_size, self.block_size))
        
        # Draw score
        font = pygame.font.SysFont('consolas', 20)
        score_surface = font.render(f'Score: {self.score}', True, self.white)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)
        
        # Update display
        pygame.display.update()
        self.fps_controller.tick(fps)
    
    def close(self):
        """Clean up resources"""
        if self.render_mode:
            pygame.quit()


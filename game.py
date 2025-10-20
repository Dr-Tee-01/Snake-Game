
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame modules
pygame.init()
font = pygame.font.SysFont('arial', 25)


# --- Game Directions ---
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Represent a point (x, y) in the grid
Point = namedtuple('Point', 'x, y')

# --- Colors (RGB format) ---
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# --- Game Config ---
BLOCK_SIZE = 20   # size of one block (snake segment or food)
SPEED = 40        # game speed (frames per second)


class SnakeGameAI:
    """
    Snake game environment for training an AI agent.
    Provides state transitions, rewards, and rendering.
    """

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # Setup game display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()

        # Initialize the game state
        self.reset()

    def reset(self):
        """
        Reset the game state after game over or at start.
        """
        self.direction = Direction.RIGHT   # initial direction

        # Snake starts in the middle of the screen
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0   # counts frames since last reset

    def _place_food(self):
        """
        Place food at a random location not occupied by the snake.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        # Ensure food is not on the snake body
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Execute one step in the game:
        - Process user events
        - Move snake according to chosen action
        - Check for collisions (game over)
        - Update reward
        - Update UI and return state
        """
        self.frame_iteration += 1

        # 1. Handle quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move the snake
        self._move(action)
        self.snake.insert(0, self.head)  # update new head position

        # 3. Check for collisions
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # remove tail (normal movement)

        # 5. Update display and control game speed
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return reward, game status, and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Check if the snake collides with walls or itself.
        """
        if pt is None:
            pt = self.head

        # Boundary check
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True

        # Self collision
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Render snake, food, and score on the screen.
        """
        self.display.fill(BLACK)

        # Draw snake
        for block in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(block.x + 4, block.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Update snake's direction based on the action.
        Action is encoded as [straight, right turn, left turn].
        """
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        # Decide new direction
        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[idx]  # keep moving straight
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = directions[(idx + 1) % 4]  # turn right
        else:  # [0, 0, 1]
            new_dir = directions[(idx - 1) % 4]  # turn left

        self.direction = new_dir

        # Move head in the new direction
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

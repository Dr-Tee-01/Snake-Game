import pygame
import random
from enum import Enum
from collections import namedtuple

# Initialize pygame modules (graphics, sound, etc.)
pygame.init()

# Load font for rendering text on screen
font = pygame.font.Font('arial.ttf', 25)
# Alternatively, you can use a system font:
# font = pygame.font.SysFont('arial', 25)

# Possible movement directions for the snake
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Represents a point in the grid (x, y)
Point = namedtuple('Point', 'x, y')

# Define some RGB color constants
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Size of each block in the grid (snake segment, food, etc.)
BLOCK_SIZE = 20
# Game speed (frames per second)
SPEED = 20


class SnakeGame:
    """
    A simple Snake game implemented using pygame.
    Handles game logic such as movement, collision detection,
    food placement, and UI rendering.
    """

    def __init__(self, w=640, h=480):
        """Initialize game window, snake state, and food placement."""
        self.w = w
        self.h = h

        # Set up game display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()

        # Initial snake state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)

        # Snake starts with 3 blocks (head + 2 body parts)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        # Score tracking
        self.score = 0

        # Place the first piece of food
        self.food = None
        self._place_food()

    def _place_food(self):
        """Place food randomly inside the grid (not overlapping the snake)."""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        # If food spawns on the snake, try again
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        """
        Execute one frame of the game:
        - Process user input
        - Move snake
        - Check for collisions
        - Place food / update score
        - Update UI
        Returns:
            (game_over: bool, score: int)
        """

        # 1. Handle user input events (quit / key presses)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # 2. Move snake (update head position)
        self._move(self.direction)
        self.snake.insert(0, self.head)  # insert new head at the front

        # 3. Check if game over (collision detection)
        if self._is_collision():
            return True, self.score

        # 4. Check if snake ate food
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            # If no food eaten, remove the tail (snake stays same length)
            self.snake.pop()

        # 5. Update UI and regulate speed
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return current state
        return False, self.score

    def _is_collision(self):
        """Check if the snake hit the wall or itself."""
        # Collision with wall
        if (self.head.x >= self.w or self.head.x < 0 or
            self.head.y >= self.h or self.head.y < 0):
            return True

        # Collision with itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """Render the game state on the screen."""
        # Fill background
        self.display.fill(BLACK)

        # Draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score text
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        # Refresh the display
        pygame.display.flip()

    def _move(self, direction):
        """Update snake's head position based on chosen direction."""
        x, y = self.head.x, self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


if __name__ == '__main__':
    game = SnakeGame()

    # Main game loop
    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print('Final Score:', score)
    pygame.quit()

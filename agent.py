import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# ---- Hyperparameters ----
MAX_MEMORY = 100_000     # maximum number of past experiences stored
BATCH_SIZE = 1000        # number of samples used per training step
LR = 0.001               # learning rate for optimizer


class Agent:
    """
    Reinforcement Learning Agent for Snake using Deep Q-Learning.
    Handles memory storage, state extraction, and decision-making.
    """

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0     # exploration rate (decreases over time)
        self.gamma = 0.9     # discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)   # experience replay buffer
        self.model = Linear_QNet(11, 256, 3)     # input=11 state features, hidden=256, output=3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        """
        Convert the current game environment into a numerical state vector.
        State encodes:
          - dangers ahead/left/right
          - current movement direction
          - relative position of the food
        """
        head = game.snake[0]

        # Points in each direction relative to the snake's head
        left_block = Point(head.x - 20, head.y)
        right_block = Point(head.x + 20, head.y)
        up_block = Point(head.x, head.y - 20)
        down_block = Point(head.x, head.y + 20)

        # Current movement direction
        moving_left = game.direction == Direction.LEFT
        moving_right = game.direction == Direction.RIGHT
        moving_up = game.direction == Direction.UP
        moving_down = game.direction == Direction.DOWN

        state = [
            # --- Danger detection ---
            (moving_right and game.is_collision(right_block)) or
            (moving_left  and game.is_collision(left_block)) or
            (moving_up    and game.is_collision(up_block)) or
            (moving_down  and game.is_collision(down_block)),

            # Danger to the right of current movement
            (moving_up    and game.is_collision(right_block)) or
            (moving_down  and game.is_collision(left_block)) or
            (moving_left  and game.is_collision(up_block)) or
            (moving_right and game.is_collision(down_block)),

            # Danger to the left of current movement
            (moving_down  and game.is_collision(right_block)) or
            (moving_up    and game.is_collision(left_block)) or
            (moving_right and game.is_collision(up_block)) or
            (moving_left  and game.is_collision(down_block)),

            # --- Movement direction ---
            moving_left,
            moving_right,
            moving_up,
            moving_down,

            # --- Food location relative to snake ---
            game.food.x < game.head.x,  # food is left
            game.food.x > game.head.x,  # food is right
            game.food.y < game.head.y,  # food is above
            game.food.y > game.head.y   # food is below
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in memory for replay training.
        Each experience = (state, action, reward, next_state, done).
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train on a batch of past experiences (experience replay).
        """
        if len(self.memory) > BATCH_SIZE:
            # Randomly sample a subset if memory is large
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            # Otherwise use all available memory
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train immediately on the most recent experience (online update).
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Choose an action using epsilon-greedy strategy:
          - With probability epsilon: random action (exploration).
          - Otherwise: best action from the Q-network (exploitation).
        """
        self.epsilon = max(0, 80 - self.n_games)  # epsilon decreases as games increase
        move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Random move
            action_idx = random.randint(0, 2)
        else:
            # Use model prediction
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()

        move[action_idx] = 1
        return move


def train():
    """
    Main training loop:
      - Plays the game
      - Collects experiences
      - Updates the Q-network
      - Tracks performance (scores & averages)
    """
    scores = []
    avg_scores = []
    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # --- Step 1: Get current state ---
        old_state = agent.get_state(game)

        # --- Step 2: Decide action ---
        action = agent.get_action(old_state)

        # --- Step 3: Perform action & observe reward ---
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)

        # --- Step 4: Train on this single step ---
        agent.train_short_memory(old_state, action, reward, new_state, done)

        # --- Step 5: Store experience ---
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            # Reset game & train on a batch of past experiences
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save model if a new high score is reached
            if score > best_score:
                best_score = score
                agent.model.save()

            # Logging
            print(f"Game {agent.n_games} | Score {score} | Record {best_score}")

            # Plot scores
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            avg_scores.append(mean_score)
            plot(scores, avg_scores)


if __name__ == "__main__":
    train()

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
import cv2
import pygame
from helper import plot, close_plot_video
import os


# ---- Hyperparameters ----
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def save_checkpoint(self, filename="model_checkpoint.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
        }, filename)
        print(f"✅ Saved checkpoint at game {self.n_games}")

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3)   # input=11, hidden=256, output=3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        head = game.snake[0]

        # Directions relative to head
        left_block = Point(head.x - 20, head.y)
        right_block = Point(head.x + 20, head.y)
        up_block = Point(head.x, head.y - 20)
        down_block = Point(head.x, head.y + 20)

        moving_left = game.direction == Direction.LEFT
        moving_right = game.direction == Direction.RIGHT
        moving_up = game.direction == Direction.UP
        moving_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (moving_right and game.is_collision(right_block)) or
            (moving_left  and game.is_collision(left_block)) or
            (moving_up    and game.is_collision(up_block)) or
            (moving_down  and game.is_collision(down_block)),

            # Danger right
            (moving_up    and game.is_collision(right_block)) or
            (moving_down  and game.is_collision(left_block)) or
            (moving_left  and game.is_collision(up_block)) or
            (moving_right and game.is_collision(down_block)),

            # Danger left
            (moving_down  and game.is_collision(right_block)) or
            (moving_up    and game.is_collision(left_block)) or
            (moving_right and game.is_collision(up_block)) or
            (moving_left  and game.is_collision(down_block)),

            # Movement direction
            moving_left,
            moving_right,
            moving_up,
            moving_down,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)
        move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            idx = torch.argmax(prediction).item()

        move[idx] = 1
        return move


def train():
    scores = []
    avg_scores = []
    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGameAI()

    if agent.n_games % 10000 == 0:
        agent.save_checkpoint("checkpoint_latest.pth")

    # Before loop
    pygame.display.init()
    game = SnakeGameAI()

    # Create a video writer for Snake gameplay
    os.makedirs("videos", exist_ok=True)
    snake_video_path = "videos/snake_training.mp4"
    snake_writer = cv2.VideoWriter(snake_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (game.w, game.h))


    while True:

        # Capture the current PyGame frame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.rot90(frame)  # rotate surface
        frame = np.flipud(frame)  # fix orientation
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        snake_writer.write(frame_bgr)

        old_state = agent.get_state(game)
        action = agent.get_action(old_state)
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save()

            print(f"Game {agent.n_games} | Score {score} | Record {best_score}")

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            avg_scores.append(mean_score)
            plot(scores, avg_scores)
    snake_writer.release()
    close_plot_video()
    pygame.quit()
    agent.model.save("final_model.pth")



if __name__ == "__main__":
    train()

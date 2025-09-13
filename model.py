import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    """
    A simple 2-layer fully connected neural network
    used to approximate the Q-function in reinforcement learning.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # input → hidden
        self.fc2 = nn.Linear(hidden_size, output_size)  # hidden → output

    def forward(self, x):
        """
        Forward pass through the network.
        Uses ReLU activation for hidden layer.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # raw output (Q-values for each action)
        return x

    def save(self, filename="model.pth"):
        """
        Save the trained model parameters to disk.
        """
        model_dir = "./model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        filepath = os.path.join(model_dir, filename)
        torch.save(self.state_dict(), filepath)


class QTrainer:
    """
    Handles training of the Q-learning agent using the given model.
    Performs updates to minimize the difference between predicted
    Q-values and target Q-values.
    """
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma  # discount factor
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # loss between predicted and target Q-values

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs one training step (backpropagation).

        Args:
            state: Current state (tensor or array).
            action: Action taken (one-hot encoded).
            reward: Reward received.
            next_state: State after taking the action.
            done: Whether the episode ended after this step.
        """
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If we have a single sample, expand dims → shape (1, x)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # 1. Predicted Q-values for the current state
        pred = self.model(state)

        # 2. Create target tensor as a copy of predictions
        target = pred.clone()

        # Update only the Q-value corresponding to the action taken
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:  # Bellman equation update
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Find which action was taken (argmax because one-hot encoding)
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Compute loss (difference between target and prediction)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)

        # Backpropagate and update weights
        loss.backward()
        self.optimizer.step()

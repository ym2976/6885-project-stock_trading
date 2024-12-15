import random
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import Portfolio

import torch
import torch.nn as nn


class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.value_fc = nn.Linear(32, 16)
        self.value_out = nn.Linear(16, 1)
        self.adv_fc = nn.Linear(32, 16)
        self.adv_out = nn.Linear(16, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.relu(self.value_fc(x))
        value = self.value_out(value)
        adv = self.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - adv_mean
        return Q


class SimpleDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimpleDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        Q = self.fc4(x)
        return Q


class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name="",
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999,
                 buffer_size=1000, max_memory=1000, lr=0.0001,
                 double_dqn=False, dueling_dqn=True, target_update_freq=100):
        super().__init__(balance=balance)
        self.model_type = 'DuelingDQN'
        self.state_dim = state_dim
        self.action_dim = 3
        self.is_eval = is_eval
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=max_memory)
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.dueling_dqn:
            self.model = DuelingDQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_model = DuelingDQNNetwork(self.state_dim, self.action_dim).to(self.device)
        else:
            self.model = SimpleDQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_model = SimpleDQNNetwork(self.state_dim, self.action_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        if self.is_eval and model_name:
            self.load(model_name)

        self.update_target_network()
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.current_price = 0.0

    def adjust_epsilon(self, episode, total_episodes):
        if episode < total_episodes * 0.5:
            self.epsilon = max(self.epsilon_min, 1.0 - episode / (total_episodes * 0.3))
        elif episode < total_episodes * 0.9:
            self.epsilon = max(self.epsilon_min, 0.5 - (episode - total_episodes * 0.3) / (total_episodes * 0.4))
        else:
            self.epsilon = self.epsilon_min

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.target_model.to(device)

    def reset(self):
        self.reset_portfolio()

    def remember(self, state, action, reward, next_state, done):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=self.device)
            current_Q = self.predict(state_tensor)
            current_Q_value = current_Q[0, action]
            if self.double_dqn:
                next_actions = self.model(next_state_tensor).argmax(dim=1, keepdim=True)
                next_Q = self.target_model(next_state_tensor).gather(1, next_actions).squeeze(1)
            else:
                next_Q = self.target_model(next_state_tensor).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_Q.item()
            td_error = abs(target - current_Q_value.item())
        self.memory.append((state, action, reward, next_state, done, td_error))

    def sample_memory(self):
        td_errors = np.array([exp[5] for exp in self.memory])
        probabilities = td_errors / (td_errors.sum() + 1e-6)
        indices = np.random.choice(len(self.memory), self.buffer_size, p=probabilities)
        return [self.memory[i] for i in indices]

    def predict(self, state):
        with torch.no_grad():
            Q_values = self.model(state)
        return Q_values

    def act(self, state):
        if self.is_eval:
            Q_values = self.predict(state)
            Q_values = self.apply_action_mask(Q_values)
            return torch.argmax(Q_values, dim=1).item()
        if np.random.rand() <= self.epsilon:
            valid_actions = self.get_valid_actions()
            return random.choice(valid_actions)
        else:
            Q_values = self.predict(state)
            Q_values = self.apply_action_mask(Q_values)
            return torch.argmax(Q_values, dim=1).item()

    def set_current_price(self, price):
        self.current_price = price

    def get_valid_actions(self):
        valid_actions = [0, 1, 2]
        if len(self.inventory) == 0:
            valid_actions.remove(2)
        if self.balance < self.current_price:
            valid_actions.remove(1)
        return valid_actions

    def apply_action_mask(self, Q_values):
        valid_actions = self.get_valid_actions()
        mask = torch.zeros_like(Q_values)
        mask[:, valid_actions] = 1
        Q_values[mask == 0] = -float('inf')
        return Q_values

    def experience_replay(self):
        if len(self.memory) < self.buffer_size:
            return 0.0
        mini_batch = self.sample_memory()
        states = np.array([x[0] for x in mini_batch])
        state_batch = torch.from_numpy(states).float().to(self.device).squeeze(1)
        next_states = np.array([x[3] for x in mini_batch])
        next_state_batch = torch.from_numpy(next_states).float().to(self.device).squeeze(1)
        actions_batch = torch.tensor([x[1] for x in mini_batch], dtype=torch.long, device=self.device).view(-1, 1)
        rewards_batch = torch.tensor([x[2] for x in mini_batch], dtype=torch.float, device=self.device)
        done_batch = torch.tensor([x[4] for x in mini_batch], dtype=torch.bool, device=self.device)
        q_values = self.model(state_batch).squeeze(1)
        current_Q = q_values.gather(1, actions_batch).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.model(next_state_batch).argmax(dim=1, keepdim=True)
                next_Q = self.target_model(next_state_batch).gather(1, next_actions).squeeze(1)
            else:
                next_Q = self.target_model(next_state_batch).max(dim=1)[0]
            target = rewards_batch + (1 - done_batch.float()) * self.gamma * next_Q
        loss = self.criterion(current_Q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}. Parameters norm:")

    def load(self, model_name):
        filepath = f"saved_models/{model_name}.pth"
        if os.path.isfile(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.eval()
            print(f"Evaluating using model: {model_name}.pth, Model {model_name} successfully loaded for evaluation.")
        else:
            print(f"Model file {filepath} not found. Starting from scratch.")

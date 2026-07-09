# DQN agent
# src/agent.py

# ε-greedy action selection
# Replay buffer interaction
# Bellman target computation
# Loss computation
# Optimizer step
# Target network update

# State
#   │
#   ▼
# agent.py
#   │
#   ├── Calls DQN
#   ├── Chooses action
#   ├── Stores experience
#   ├── Samples replay buffer
#   ├── Computes Bellman target
#   ├── Updates DQN weights
#   └── Updates target network

import random
import torch
import numpy as np

from dqn import DQN


class DQNAgent:
    def __init__(self, state_dim, action_dim, device, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.epsilon = epsilon

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        0 = Allow
        1 = Inspect
        """

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        return action

    def update_target_network(self):
        """
        Copy policy network weights into target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, min_epsilon=0.05, decay_rate=0.995):
        """
        Reduce exploration over time.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
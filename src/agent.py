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

# It should not know anything about the LLM. The LLM belongs in environment.py / llm_critic.py.

# src/agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=100000
    ):

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        # Main DQN
        self.policy_network = DQN(
            state_dim,
            action_dim
        )

        # Target DQN
        self.target_network = DQN(
            state_dim,
            action_dim
        )

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

    #######################################################
    # ε-greedy action selection
    #######################################################

    def select_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, 1)

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(state)

        return torch.argmax(q_values).item()

    #######################################################
    # Store transition
    #######################################################

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    #######################################################
    # Learn
    #######################################################

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(
                self.batch_size
            )

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)

        actions = torch.LongTensor(actions)

        rewards = torch.FloatTensor(rewards)

        dones = torch.FloatTensor(dones)

    ###################################################
    # Current Q(s,a)
    ###################################################

        current_q = self.policy_network(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze()

    ###################################################
    # max Q(s',a')
    ###################################################

        with torch.no_grad():

            next_q = self.target_network(next_states)

            max_next_q = next_q.max(1)[0]

    ###################################################
    # Bellman Target
    ###################################################

        target = rewards + \
                 self.gamma * \
                 max_next_q * \
                 (1 - dones)

    ###################################################
    # Loss
    ###################################################

        loss = self.loss_fn(
            current_q,
            target
        )

    ###################################################
    # Backpropagation
    ###################################################

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    ###################################################
    # Decay epsilon
    ###################################################

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    #######################################################
    # Copy policy network → target network
    #######################################################

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
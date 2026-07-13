# src/agent.py

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Standard DQN agent.

    Responsibilities
    ----------------
    - Select actions using epsilon-greedy exploration
    - Store transitions in the replay buffer
    - Sample mini-batches
    - Compute Bellman targets
    - Update policy-network weights
    - Periodically synchronize the target network

    The agent does not create or manage the Gymnasium environment.
    That responsibility belongs to train.py and environment.py.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        device: Optional[str] = None,
    ) -> None:

        if state_dim <= 0:
            raise ValueError("state_dim must be greater than zero.")

        if action_dim <= 1:
            raise ValueError("action_dim must be at least 2.")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        if not 0.0 <= epsilon_min <= epsilon <= 1.0:
            raise ValueError(
                "Expected 0 <= epsilon_min <= epsilon <= 1."
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        # Online/policy network:
        # updated after every learning step.
        self.policy_network = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        # Target network:
        # used to compute stable Bellman targets.
        self.target_network = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.update_target_network()
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size
        )

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
    ) -> int:
        """
        Select an action using epsilon-greedy exploration.

        Parameters
        ----------
        state:
            One normalized network-flow state vector.

        explore:
            If False, always select the greedy action.
            Useful during evaluation.
        """

        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.policy_network.eval()

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one transition in replay memory.

        During the baseline experiment, reward is R_env.
        After PBRS integration, reward will be R_total.
        """

        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def learn(self) -> Optional[float]:
        """
        Perform one DQN optimization step.

        Returns
        -------
        float or None
            Training loss, or None if there are not enough
            replay-buffer samples yet.
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        self.policy_network.train()

        # Q_theta(s, a) for the actions actually taken.
        all_current_q = self.policy_network(states)

        current_q = all_current_q.gather(
            dim=1,
            index=actions.unsqueeze(1),
        ).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            all_next_q = self.target_network(next_states)

            max_next_q = all_next_q.max(
                dim=1
            ).values

            # Bellman target:
            # y = r + gamma * max Q_target(s', a')
            # The future term is removed for terminal states.
            target_q = rewards + (
                self.gamma
                * max_next_q
                * (1.0 - dones)
            )

        loss = self.loss_fn(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Optional protection against unstable gradients.
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        return float(loss.item())

    def update_target_network(self) -> None:
        """
        Copy policy-network weights into the target network.
        """

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

    def save(self, path: str) -> None:
        """Save the policy-network parameters."""

        torch.save(
            self.policy_network.state_dict(),
            path,
        )

    def load(self, path: str) -> None:
        """Load policy-network parameters."""

        state_dict = torch.load(
            path,
            map_location=self.device,
        )

        self.policy_network.load_state_dict(state_dict)
        self.update_target_network()


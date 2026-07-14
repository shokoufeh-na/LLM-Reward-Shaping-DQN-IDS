# src/agent.py

import math
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
    Standard DQN agent for network anomaly detection.

    Responsibilities
    ----------------
    - Select actions using epsilon-greedy exploration
    - Store transitions in replay memory
    - Sample random mini-batches
    - Compute Bellman targets
    - Update policy-network weights
    - Synchronize the target network

    The agent does not create or manage the Gymnasium environment.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        eps_start: float = 0.90,
        eps_end: float = 0.05,
        eps_decay: float = 1000.0,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        hidden_dim: int = 128,
        device: Optional[str] = None,
    ) -> None:

        if state_dim <= 0:
            raise ValueError("state_dim must be greater than zero.")

        if action_dim <= 1:
            raise ValueError("action_dim must be at least 2.")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        if not 0.0 <= eps_end <= eps_start <= 1.0:
            raise ValueError(
                "Expected 0 <= eps_end <= eps_start <= 1."
            )

        if eps_decay <= 0:
            raise ValueError("eps_decay must be greater than zero.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if buffer_size < batch_size:
            raise ValueError(
                "buffer_size must be greater than or equal to batch_size."
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        # Epsilon-greedy parameters
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.current_epsilon = eps_start

        # Select device automatically unless explicitly supplied.
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print(f"DQN agent using device: {self.device}")

        # Policy network:
        # updated during every optimization step.
        self.policy_network = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Target network:
        # used to calculate more stable Bellman targets.
        self.target_network = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Initially, both networks have identical weights.
        self.update_target_network()
        self.target_network.eval()

        # AdamW follows the official PyTorch DQN tutorial more closely.
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=learning_rate,
            amsgrad=True,
        )

        # Huber loss is less sensitive to large TD errors than MSE.
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size
        )

    def _calculate_epsilon(self) -> float:
        """
        Calculate the exponentially decayed epsilon value.

        Formula
        -------
        epsilon = eps_end
                  + (eps_start - eps_end)
                  * exp(-steps_done / eps_decay)
        """

        return self.eps_end + (
            self.eps_start - self.eps_end
        ) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
    ) -> int:
        """
        Select an action.

        During training:
            Uses epsilon-greedy exploration.

        During evaluation:
            Set explore=False to always select the greedy action.
        """

        if explore:
            self.current_epsilon = self._calculate_epsilon()
            self.steps_done += 1

            if random.random() < self.current_epsilon:
                return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.policy_network.eval()

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        action = q_values.argmax(dim=1).item()

        return int(action)

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

        Baseline:
            reward = R_env

        LLM-guided PBRS:
            reward = R_total
                   = R_env + F(s, s')
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
            Loss value, or None if the replay buffer does not yet
            contain enough samples.
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

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        self.policy_network.train()

        # Policy network predicts Q-values for every action.
        all_current_q = self.policy_network(states)

        # Select Q(s, a) only for actions actually taken.
        current_q = all_current_q.gather(
            dim=1,
            index=actions.unsqueeze(1),
        ).squeeze(1)

        # Compute max_a' Q_target(s', a') using the frozen target network.
        with torch.no_grad():
            all_next_q = self.target_network(next_states)

            max_next_q = all_next_q.max(
                dim=1
            ).values

            # Bellman target:
            #
            # y = reward + gamma * max_a' Q_target(s', a')
            #
            # For terminal states, the future term is removed.
            target_q = rewards + (
                self.gamma
                * max_next_q
                * (1.0 - dones)
            )

        # Temporal-difference loss
        loss = self.loss_fn(
            current_q,
            target_q,
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients for additional stability.
        torch.nn.utils.clip_grad_value_(
            self.policy_network.parameters(),
            clip_value=100.0,
        )

        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """
        Hard target-network update.

        Copies all policy-network weights into the target network.
        Call this periodically from train.py, for example every
        1,000 environment steps.
        """

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

    def save(self, path: str) -> None:
        """
        Save the learned policy-network weights.
        """

        torch.save(
            {
                "policy_network_state_dict":
                    self.policy_network.state_dict(),
                "target_network_state_dict":
                    self.target_network.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "steps_done":
                    self.steps_done,
                "current_epsilon":
                    self.current_epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load a saved DQN checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.policy_network.load_state_dict(
            checkpoint["policy_network_state_dict"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.steps_done = checkpoint.get(
            "steps_done",
            0,
        )

        self.current_epsilon = checkpoint.get(
            "current_epsilon",
            self.eps_start,
        )

        self.policy_network.eval()
        self.target_network.eval()

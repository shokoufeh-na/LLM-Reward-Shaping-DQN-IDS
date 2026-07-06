# DQN Network
# src/dqn.py

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for network anomaly detection.

    Input:
        state vector s with shape: (batch_size, state_dim)

    Output:
        Q-values for each action:
        Q(s, Allow), Q(s, Inspect)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns Q-values for all actions.
        """
        return self.network(state)


if __name__ == "__main__":
    # Example test
    state_dim = 78      # CICIDS2017 has about 78 input features
    action_dim = 2      # 0 = Allow, 1 = Inspect

    model = DQN(state_dim, action_dim)

    sample_state = torch.randn(1, state_dim)

    q_values = model(sample_state)

    print("Q-values:", q_values)
    print("Q(s, Allow):", q_values[0, 0].item())
    print("Q(s, Inspect):", q_values[0, 1].item())

# RL Environment (Renv)
# don't include the LLM yet
# connects the dataset, the DQN, the hidden labels, the LLM, and the reward shaping.

# src/environment.py

import numpy as np


class NetworkEnvironment:
    """
    Reinforcement Learning Environment for CICIDS2017.

    Action Space
    ------------
    0 : Allow
    1 : Inspect

    State
    -----
    One normalized network flow.

    Reward
    ------
    Renv
    """

    def __init__(self, states, labels):

        self.states = states
        self.labels = labels

        self.current_index = 0

        self.num_samples = len(states)

    def reset(self):
        """
        Reset the environment.
        """

        self.current_index = 0

        return self.states[self.current_index]
# Current State

# ↓

# Hidden Label

# ↓

# Compute Renv

# ↓

# Move to Next State

# ↓

# Return

    def step(self, action):
        """
        Execute one action.

        Returns
        -------
        next_state
        reward
        done
        info
        """

        true_label = self.labels[self.current_index]

        reward = self.compute_reward(
            action,
            true_label
        )

        self.current_index += 1

        done = self.current_index >= self.num_samples

        if done:

            next_state = np.zeros_like(
                self.states[0]
            )

        else:

            next_state = self.states[
                self.current_index
            ]

        info = {
            "label": true_label
        }

        return (
            next_state,
            reward,
            done,
            info
        )

    def compute_reward(
            self,
            action,
            label):
        """
        Environmental reward.

        label
        -----
        0 = BENIGN
        1 = ATTACK
        """

        if label == 0:

            if action == 0:
                return 1.0

            else:
                return -1.0

        else:

            if action == 1:
                return 1.0

            else:
                return -1.0
# The replay buffer is a memory of past experiences. Its purpose is to:

# Store transitions (s,a,R total,s′).
# Randomly sample mini-batches for training.
# Reduce correlation between consecutive samples.
# Reuse past experiences, making learning more stable and data-efficient.

# src/replay_buffer.py

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores transitions:
        (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=100000):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of experiences stored.
        """

        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Store one transition in the replay buffer.

        Parameters
        ----------
        state : ndarray
            Current state s

        action : int
            Action selected by the DQN

        reward : float
            Total reward (R_total)

        next_state : ndarray
            Next state s'

        done : bool
            Whether the episode has ended
        """

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch.

        Returns
        -------
        states
        actions
        rewards
        next_states
        dones
        """

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        """
        Number of stored experiences.
        """

        return len(self.buffer)


if __name__ == "__main__":

    buffer = ReplayBuffer(capacity=10)

    for i in range(5):

        state = np.random.rand(78)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = np.random.rand(78)
        done = False

        buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    print("Replay Buffer Size:", len(buffer))

    states, actions, rewards, next_states, dones = buffer.sample(2)

    print("States shape:", states.shape)
    print("Actions:", actions)
    print("Rewards:", rewards)
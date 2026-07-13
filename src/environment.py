# src/environment.py

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Action definitions
ALLOW = 0
INSPECT = 1

# Label definitions
BENIGN = 0
ATTACK = 1


class NetworkEnvironment(gym.Env):
    """
    Gymnasium environment for CICIDS2017 network anomaly detection.

    State
    -----
    One normalized CICIDS2017 network-flow feature vector.

    Actions
    -------
    0: Allow
    1: Inspect

    Environmental Reward
    --------------------
    Correct decision: +1.0
    Incorrect decision: -1.0

    Notes
    -----
    The DQN receives only the state vector. The environment keeps the
    corresponding binary label internally and uses it to calculate R_env.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        states: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        super().__init__()

        self.states = np.asarray(states, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

        self._validate_inputs()

        self.num_samples = len(self.states)
        self.state_dim = self.states.shape[1]
        self.current_index = 0

        # Two discrete actions:
        # 0 = Allow
        # 1 = Inspect
        self.action_space = spaces.Discrete(2)

        # State vectors are Z-score standardized, so values are not
        # necessarily restricted to [0, 1].
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

    def _validate_inputs(self) -> None:
        """Validate states and labels before training begins."""

        if self.states.ndim != 2:
            raise ValueError(
                "states must be a two-dimensional array with shape "
                "(number_of_samples, number_of_features)."
            )

        if self.labels.ndim != 1:
            raise ValueError(
                "labels must be a one-dimensional array."
            )

        if len(self.states) == 0:
            raise ValueError("states cannot be empty.")

        if len(self.states) != len(self.labels):
            raise ValueError(
                "states and labels must contain the same number of samples."
            )

        valid_labels = {BENIGN, ATTACK}
        observed_labels = set(np.unique(self.labels).tolist())

        if not observed_labels.issubset(valid_labels):
            raise ValueError(
                "labels must be binary: 0 for BENIGN and 1 for ATTACK. "
                f"Observed labels: {observed_labels}"
            )

        if not np.isfinite(self.states).all():
            raise ValueError(
                "states contain NaN or infinite values. "
                "Clean the features before creating the environment."
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to the first network flow.

        Returns
        -------
        observation:
            First state vector.

        info:
            Additional environment information.
        """

        super().reset(seed=seed)

        self.current_index = 0

        observation = self.states[self.current_index].copy()

        info = {
            "sample_index": self.current_index,
        }

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """
        Execute the selected action for the current network flow.

        Parameters
        ----------
        action:
            0 for Allow or 1 for Inspect.

        Returns
        -------
        next_state:
            The next normalized network-flow state.

        reward:
            Environmental reward R_env.

        terminated:
            True when all samples have been processed.

        truncated:
            Always False because this environment currently has no
            external time limit.

        info:
            Diagnostic information. The agent should not use
            true_label for action selection during training.
        """

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected {ALLOW} (Allow) or {INSPECT} (Inspect)."
            )

        if self.current_index >= self.num_samples:
            raise RuntimeError(
                "The episode has already terminated. Call reset() "
                "before calling step() again."
            )

        true_label = int(self.labels[self.current_index])

        reward = self.compute_environment_reward(
            action=action,
            true_label=true_label,
        )

        current_sample_index = self.current_index
        self.current_index += 1

        terminated = self.current_index >= self.num_samples
        truncated = False

        if terminated:
            # Gymnasium still requires an observation when the episode ends.
            next_state = np.zeros(
                self.state_dim,
                dtype=np.float32,
            )
        else:
            next_state = self.states[self.current_index].copy()

        info = {
            "sample_index": current_sample_index,
            "true_label": true_label,
            "action_correct": reward > 0,
            "environment_reward": reward,
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    @staticmethod
    def compute_environment_reward(
        action: int,
        true_label: int,
    ) -> float:
        """
        Calculate the original environmental reward R_env.

        Reward table
        ------------
        BENIGN + Allow   = +1
        BENIGN + Inspect = -1
        ATTACK + Inspect = +1
        ATTACK + Allow   = -1
        """

        correct_action = (
            (true_label == BENIGN and action == ALLOW)
            or
            (true_label == ATTACK and action == INSPECT)
        )

        return 1.0 if correct_action else -1.0


if __name__ == "__main__":
    # Small test using fake normalized state vectors.
    sample_states = np.array(
        [
            [0.20, -0.50, 1.10],
            [-0.80, 1.30, 0.40],
            [1.20, 0.10, -0.60],
        ],
        dtype=np.float32,
    )

    # 0 = BENIGN, 1 = ATTACK
    sample_labels = np.array([0, 1, 0], dtype=np.int64)

    env = NetworkEnvironment(
        states=sample_states,
        labels=sample_labels,
    )

    state, info = env.reset()

    print("Initial state:", state)
    print("Reset info:", info)

    terminated = False

    while not terminated:
        action = env.action_space.sample()

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        action_name = "Allow" if action == ALLOW else "Inspect"

        print(
            f"Action={action_name}, "
            f"Reward={reward}, "
            f"Terminated={terminated}, "
            f"Info={info}"
        )

        state = next_state

# src/reward_shaping.py

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    """
    Stores the components of the reward calculation.
    """

    environment_reward: float
    shaping_reward: float
    total_reward: float
    phi_current: float
    phi_next: float


class PotentialBasedRewardShaper:
    """
    Potential-Based Reward Shaping (PBRS).

    Formula
    -------
    F(s, s') = gamma * Phi(s') - Phi(s)

    Total reward
    ------------
    R_total = R_env + scale * F(s, s')
    """

    def __init__(
        self,
        gamma: float = 0.99,
        scale: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        gamma:
            Discount factor used by PBRS.

        scale:
            Optional coefficient controlling the strength of the
            shaping reward. Keep this at 1.0 for the standard form.
        """

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        if scale < 0.0:
            raise ValueError("scale must be non-negative.")

        self.gamma = gamma
        self.scale = scale

    @staticmethod
    def _validate_potential(phi: float, name: str) -> float:
        """
        Validate one LLM-generated potential score.
        """

        phi = float(phi)

        if not 0.0 <= phi <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0. "
                f"Received {phi}."
            )

        return phi

    def compute_shaping_reward(
        self,
        phi_current: float,
        phi_next: float,
        done: bool = False,
    ) -> float:
        """
        Compute the PBRS shaping reward.

        Parameters
        ----------
        phi_current:
            Semantic potential Phi(s) for the current state.

        phi_next:
            Semantic potential Phi(s') for the next state.

        done:
            Whether the transition ends the episode.

        Returns
        -------
        float
            Shaping reward F(s, s').
        """

        phi_current = self._validate_potential(
            phi_current,
            "phi_current",
        )

        phi_next = self._validate_potential(
            phi_next,
            "phi_next",
        )

        # For a terminal transition, there is no future state value.
        effective_phi_next = 0.0 if done else phi_next

        shaping_reward = (
            self.gamma * effective_phi_next
            - phi_current
        )

        return self.scale * shaping_reward

    def combine_rewards(
        self,
        environment_reward: float,
        shaping_reward: float,
    ) -> float:
        """
        Combine R_env and the PBRS shaping reward.
        """

        return float(environment_reward) + float(shaping_reward)

    def calculate(
        self,
        environment_reward: float,
        phi_current: float,
        phi_next: float,
        done: bool = False,
    ) -> RewardBreakdown:
        """
        Compute shaping reward and total reward together.

        Returns
        -------
        RewardBreakdown
            All reward components for logging and debugging.
        """

        shaping_reward = self.compute_shaping_reward(
            phi_current=phi_current,
            phi_next=phi_next,
            done=done,
        )

        total_reward = self.combine_rewards(
            environment_reward=environment_reward,
            shaping_reward=shaping_reward,
        )

        return RewardBreakdown(
            environment_reward=float(environment_reward),
            shaping_reward=shaping_reward,
            total_reward=total_reward,
            phi_current=float(phi_current),
            phi_next=float(phi_next),
        )


if __name__ == "__main__":
    shaper = PotentialBasedRewardShaper(
        gamma=0.99,
        scale=1.0,
    )

    result = shaper.calculate(
        environment_reward=1.0,
        phi_current=0.40,
        phi_next=0.80,
        done=False,
    )

    print("Environment reward:", result.environment_reward)
    print("Phi(s):", result.phi_current)
    print("Phi(s'):", result.phi_next)
    print("Shaping reward:", result.shaping_reward)
    print("Total reward:", result.total_reward)

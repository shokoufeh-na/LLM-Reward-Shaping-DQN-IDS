# src/train.py

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import joblib
from pathlib import Path

from agent import DQNAgent
from data_loader import load_all_csvs
from environment import NetworkEnvironment
from feature_extractor import (
    build_training_states,
    build_evaluation_states,
)
from llm_critic import LLMCritic
from reward_shaping import PotentialBasedRewardShaper


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    This affects:
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA, when available
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Improve determinism for NVIDIA CUDA operations.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}.")


def state_to_feature_dictionary(
    normalized_state: np.ndarray,
    scaler,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Convert one normalized DQN state back to named,
    original-scale network-flow features for the LLM.

    The DQN receives normalized values, while the LLM receives
    semantically meaningful values such as:

        Flow Duration: 125000
        Destination Port: 443
    """

    state_2d = np.asarray(
        normalized_state,
        dtype=np.float64,
    ).reshape(1, -1)

    original_values = scaler.inverse_transform(state_2d)[0]

    return {
        feature_name: float(feature_value)
        for feature_name, feature_value in zip(
            feature_names,
            original_values,
        )
    }


def evaluate_potential(
    critic: LLMCritic,
    normalized_state: np.ndarray,
    scaler,
) -> float:
    """
    Convert a normalized state to named features and ask
    the LLM Security Critic for Phi(s).
    """

    named_state = state_to_feature_dictionary(
        normalized_state=normalized_state,
        scaler=scaler,
        feature_names = scaler.feature_names_in_,
    )

    return critic.evaluate(named_state)


def train(
    epochs: int = 1,
    use_llm: bool = False,
    max_steps_per_epoch: Optional[int] = None,
    target_update_interval: int = 1000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_size: int = 100_000,
    seed: int = 42,
) -> None:
    """
    Train one of two models.

    Baseline DQN
    ------------
    reward = R_env

    Proposed LLM-guided DQN
    -----------------------
    reward = R_total
           = R_env + F(s, s')
    """

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if target_update_interval <= 0:
        raise ValueError(
            "target_update_interval must be greater than zero."
        )

    if max_steps_per_epoch is not None and max_steps_per_epoch <= 0:
        raise ValueError(
            "max_steps_per_epoch must be greater than zero."
        )

    # Seed Python, NumPy, and PyTorch before creating
    # the environment, agent, and neural networks.
    set_random_seed(seed)
    
    print("Loading training data...")

    train_dataframe = load_all_csvs(
        "data/train"
    )

    print("Preparing standardized training states...")

    train_states, train_labels, scaler = (
        build_training_states(
            train_dataframe
        )
    )
   
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        scaler,
        MODELS_DIR / "standard_scaler.joblib"
    )

    train_states = np.asarray(
        train_states,
        dtype=np.float32,
    )

    train_labels = np.asarray(
        train_labels,
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)

    benign_indices = np.where(train_labels == 0)[0]
    attack_indices = np.where(train_labels == 1)[0]

    samples_per_class = min(
        len(benign_indices),
        len(attack_indices),
    )

    selected_benign = rng.choice(
        benign_indices,
        size=samples_per_class,
        replace=False,
    )

    selected_attack = rng.choice(
        attack_indices,
        size=samples_per_class,
        replace=False,
    )

    selected_indices = np.concatenate(
        [selected_benign, selected_attack]
    )
    # shuffle before creating the environment
    rng.shuffle(selected_indices)

    train_states = train_states[selected_indices]
    train_labels = train_labels[selected_indices]

    print("\nBalanced training subset")
    print(f"Benign: {np.sum(train_labels == 0):,}")
    print(f"Attack: {np.sum(train_labels == 1):,}")

    print(f"Training states: {train_states.shape}")
    print(f"Training labels: {train_labels.shape}")


    environment = NetworkEnvironment(
        states=train_states,
        labels=train_labels,
    )

    # Seed the Gymnasium action space.
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)

    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    llm_critic: Optional[LLMCritic] = None
    reward_shaper: Optional[PotentialBasedRewardShaper] = None

    if use_llm:
        print("LLM-guided PBRS enabled.")

        llm_critic = LLMCritic(
            model="llama3"
        )

        reward_shaper = PotentialBasedRewardShaper(
            gamma=gamma,
            scale=1.0,
        )
    else:
        print("Training baseline DQN using R_env only.")

    global_step = 0
    best_epoch_reward = float("-inf")

    reward_history: list[float] = []
    environment_reward_history: list[float] = []
    shaping_reward_history: list[float] = []
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        # A deterministic seed is supplied to reset().
        # Adding the epoch index allows reproducible but distinct
        # environment randomness if randomization is added later.
        epoch_seed = seed + epoch - 1

        state, reset_info = environment.reset(
            seed=epoch_seed
        )

        terminated = False
        truncated = False

        epoch_total_reward = 0.0
        epoch_environment_reward = 0.0
        epoch_shaping_reward = 0.0
        epoch_losses: list[float] = []

        step_in_epoch = 0

        phi_current: Optional[float] = None

        if use_llm:
            assert llm_critic is not None

            phi_current = evaluate_potential(
                critic=llm_critic,
                normalized_state=state,
                scaler=scaler,
            )

        while not terminated and not truncated:
            action = agent.select_action(
                state=state,
                explore=True,
            )

            (
                next_state,
                environment_reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            done = terminated or truncated

            total_reward = float(environment_reward)
            shaping_reward = 0.0

            if use_llm:
                assert llm_critic is not None
                assert reward_shaper is not None
                assert phi_current is not None

                if done:
                    phi_next = 0.0
                else:
                    phi_next = evaluate_potential(
                    critic=llm_critic,
                    normalized_state=state,
                    scaler=scaler,
)

                reward_result = reward_shaper.calculate(
                    environment_reward=environment_reward,
                    phi_current=phi_current,
                    phi_next=phi_next,
                    done=done,
                )

                shaping_reward = (
                    reward_result.shaping_reward
                )

                total_reward = (
                    reward_result.total_reward
                )

                # Reuse Phi(s') as Phi(s) in the next step.
                phi_current = phi_next

            agent.remember(
                state=state,
                action=action,
                reward=total_reward,
                next_state=next_state,
                done=done,
            )

            loss = agent.learn()

            if loss is not None:
                epoch_losses.append(loss)
                loss_history.append(loss)

            global_step += 1
            step_in_epoch += 1

            if global_step % target_update_interval == 0:
                agent.update_target_network()

                print(
                    "Target network updated at "
                    f"global step {global_step:,}."
                )

            epoch_total_reward += total_reward
            epoch_environment_reward += environment_reward
            epoch_shaping_reward += shaping_reward

            state = next_state

            if (
                max_steps_per_epoch is not None
                and step_in_epoch >= max_steps_per_epoch
            ):
                truncated = True

        average_loss = (
            float(np.mean(epoch_losses))
            if epoch_losses
            else float("nan")
        )

        reward_history.append(epoch_total_reward)
        environment_reward_history.append(
            epoch_environment_reward
        )
        shaping_reward_history.append(
            epoch_shaping_reward
        )

        print(
            f"\nEpoch {epoch}/{epochs}"
            f"\n  Epoch seed: {epoch_seed}"
            f"\n  Steps: {step_in_epoch:,}"
            f"\n  R_env sum: "
            f"{epoch_environment_reward:.4f}"
            f"\n  Shaping reward sum: "
            f"{epoch_shaping_reward:.4f}"
            f"\n  Total reward sum: "
            f"{epoch_total_reward:.4f}"
            f"\n  Average loss: {average_loss:.6f}"
            f"\n  Epsilon: {agent.current_epsilon:.6f}"
            f"\n  Replay-buffer size: "
            f"{len(agent.replay_buffer):,}\n"
        )

        if epoch_total_reward > best_epoch_reward:
            best_epoch_reward = epoch_total_reward

            model_name = (
                "best_llm_pbrs_dqn.pt"
                if use_llm
                else "best_baseline_dqn.pt"
            )

            model_path = MODELS_DIR / model_name

            agent.save(str(model_path))

            print(f"Saved best model to: {model_path}")

    result_prefix = (
        "llm_pbrs"
        if use_llm
        else "baseline"
    )

    np.savetxt(
        RESULTS_DIR / f"{result_prefix}_reward_history.csv",
        np.asarray(
            reward_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    np.savetxt(
        RESULTS_DIR
        / f"{result_prefix}_environment_reward_history.csv",
        np.asarray(
            environment_reward_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    np.savetxt(
        RESULTS_DIR
        / f"{result_prefix}_shaping_reward_history.csv",
        np.asarray(
            shaping_reward_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    np.savetxt(
        RESULTS_DIR / f"{result_prefix}_loss_history.csv",
        np.asarray(
            loss_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    print("Training completed.")
    print(f"Experiment seed: {seed}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a baseline or LLM-guided DQN for "
            "CICIDS2017 network anomaly detection."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes through the training environment.",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable the LLM Security Critic and PBRS.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional maximum number of environment steps "
            "per epoch. Useful for initial testing."
        ),
    )

    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
        help=(
            "Number of environment steps between "
            "target-network synchronizations."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward discount factor.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Replay-buffer mini-batch size.",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="Maximum replay-buffer capacity.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible experiments.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    train(
        epochs=arguments.epochs,
        use_llm=arguments.use_llm,
        max_steps_per_epoch=arguments.max_steps,
        target_update_interval=(
            arguments.target_update_interval
        ),
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        batch_size=arguments.batch_size,
        buffer_size=arguments.buffer_size,
        seed=arguments.seed,
    )

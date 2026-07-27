# src/train.py

import argparse
import random
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agent import DQNAgent
from data_loader import load_all_csvs
from environment import NetworkEnvironment
from feature_extractor import build_training_states
from llm_critic import LLMCritic
from reward_shaping import PotentialBasedRewardShaper


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_SPLIT_DIR = PROJECT_ROOT / "data_split"

TRAIN_SPLIT_DIR = DATA_SPLIT_DIR / "train"
VALIDATION_SPLIT_DIR = DATA_SPLIT_DIR / "validation"
TEST_SPLIT_DIR = DATA_SPLIT_DIR / "test"


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(
        f"Random seed set to {seed}."
    )


# ============================================================
# RANDOM TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_random_data_split(
    dataframe: pd.DataFrame,
    seed: int = 42,
):
    """
    Randomly divide the complete CICIDS2017 dataset into:

        Training:   60%
        Validation: 20%
        Test:       20%

    Stratification uses the ORIGINAL CICIDS2017 Label
    column so attack types are distributed across the
    three subsets.

    Example labels:

        BENIGN
        DoS Hulk
        DDoS
        PortScan
        FTP-Patator
        SSH-Patator
        Web Attack
        etc.
    """

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns.str.strip()
    )

    if "Label" not in dataframe.columns:
        raise KeyError(
            "'Label' column was not found."
        )

    # Clean spaces in label strings.
    dataframe["Label"] = (
        dataframe["Label"]
        .astype(str)
        .str.strip()
    )

    print(
        "\nCreating random stratified "
        "train/validation/test split..."
    )

    # --------------------------------------------------------
    # First split:
    #
    # 60% TRAIN
    # 40% TEMPORARY
    # --------------------------------------------------------

    train_dataframe, temporary_dataframe = (
        train_test_split(
            dataframe,
            test_size=0.40,
            random_state=seed,
            stratify=dataframe["Label"],
        )
    )

    # --------------------------------------------------------
    # Second split:
    #
    # Temporary 40% becomes:
    #
    # 20% VALIDATION
    # 20% TEST
    # --------------------------------------------------------

    (
        validation_dataframe,
        test_dataframe,
    ) = train_test_split(
        temporary_dataframe,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_dataframe["Label"],
    )

    train_dataframe = (
        train_dataframe
        .reset_index(drop=True)
    )

    validation_dataframe = (
        validation_dataframe
        .reset_index(drop=True)
    )

    test_dataframe = (
        test_dataframe
        .reset_index(drop=True)
    )

    print(
        "\nRandom Dataset Split"
    )

    print(
        "=============================="
    )

    print(
        f"Training:   "
        f"{len(train_dataframe):,} "
        f"({len(train_dataframe) / len(dataframe):.1%})"
    )

    print(
        f"Validation: "
        f"{len(validation_dataframe):,} "
        f"({len(validation_dataframe) / len(dataframe):.1%})"
    )

    print(
        f"Test:       "
        f"{len(test_dataframe):,} "
        f"({len(test_dataframe) / len(dataframe):.1%})"
    )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


# ============================================================
# SAVE RANDOM SPLIT
# ============================================================

def save_data_split(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """
    Save the random split so evaluate.py can later load
    exactly the same validation and test datasets.
    """

    TRAIN_SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEST_SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        TRAIN_SPLIT_DIR
        / "train.csv"
    )

    validation_path = (
        VALIDATION_SPLIT_DIR
        / "validation.csv"
    )

    test_path = (
        TEST_SPLIT_DIR
        / "test.csv"
    )

    train_dataframe.to_csv(
        train_path,
        index=False,
    )

    validation_dataframe.to_csv(
        validation_path,
        index=False,
    )

    test_dataframe.to_csv(
        test_path,
        index=False,
    )

    print(
        "\nSaved random dataset split:"
    )

    print(
        f"  Train:      {train_path}"
    )

    print(
        f"  Validation: {validation_path}"
    )

    print(
        f"  Test:       {test_path}"
    )


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

def balance_training_data(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
):
    """
    Create a 50/50 BENIGN/ATTACK training subset.

    IMPORTANT:
    Only training data are balanced.

    Validation and test data remain untouched.
    """

    rng = np.random.default_rng(
        seed
    )

    benign_indices = np.where(
        train_labels == 0
    )[0]

    attack_indices = np.where(
        train_labels == 1
    )[0]

    if (
        len(benign_indices) == 0
        or len(attack_indices) == 0
    ):
        raise ValueError(
            "Training data must contain both "
            "BENIGN and ATTACK samples."
        )

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
        [
            selected_benign,
            selected_attack,
        ]
    )

    # Randomize the order so the environment
    # does not receive all benign traffic first.
    rng.shuffle(
        selected_indices
    )

    balanced_states = train_states[
        selected_indices
    ]

    balanced_labels = train_labels[
        selected_indices
    ]

    print(
        "\nBalanced Training Subset"
    )

    print(
        "=============================="
    )

    print(
        f"BENIGN: "
        f"{np.sum(balanced_labels == 0):,}"
    )

    print(
        f"ATTACK: "
        f"{np.sum(balanced_labels == 1):,}"
    )

    print(
        f"Total:  "
        f"{len(balanced_labels):,}"
    )

    return (
        balanced_states,
        balanced_labels,
    )


# ============================================================
# CONVERT NORMALIZED STATE BACK TO ORIGINAL VALUES
# ============================================================

def state_to_feature_dictionary(
    normalized_state: np.ndarray,
    scaler,
) -> dict[str, float]:
    """
    Convert a standardized DQN state back to original
    CICIDS2017 feature values for the LLM.

    DQN:
        receives standardized values.

    LLM:
        receives original meaningful feature values.
    """

    if not hasattr(
        scaler,
        "feature_names_in_",
    ):
        raise ValueError(
            "Scaler does not contain feature names."
        )

    state_2d = np.asarray(
        normalized_state,
        dtype=np.float64,
    ).reshape(
        1,
        -1,
    )

    original_values = (
        scaler.inverse_transform(
            state_2d
        )[0]
    )

    feature_names = (
        scaler.feature_names_in_
    )

    return {
        str(feature_name): float(feature_value)
        for feature_name, feature_value
        in zip(
            feature_names,
            original_values,
        )
    }


# ============================================================
# LLM POTENTIAL
# ============================================================

def evaluate_potential(
    critic: LLMCritic,
    normalized_state: np.ndarray,
    scaler,
) -> float:
    """
    Ask the LLM Security Critic for Phi(s).
    """

    named_state = (
        state_to_feature_dictionary(
            normalized_state=normalized_state,
            scaler=scaler,
        )
    )

    return critic.evaluate(
        named_state
    )


# ============================================================
# TRAINING
# ============================================================

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
    Train either:

    1. Baseline DQN

        reward = R_env

    or

    2. LLM-guided DQN with PBRS

        reward =
            R_env
            + gamma * Phi(s')
            - Phi(s)
    """

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if target_update_interval <= 0:
        raise ValueError(
            "target_update_interval "
            "must be greater than zero."
        )

    if (
        max_steps_per_epoch is not None
        and max_steps_per_epoch <= 0
    ):
        raise ValueError(
            "max_steps_per_epoch "
            "must be greater than zero."
        )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_random_seed(
        seed
    )

    # --------------------------------------------------------
    # Load ALL original CICIDS2017 files
    # --------------------------------------------------------

    print(
        "\nLoading complete CICIDS2017 dataset..."
    )

    full_dataframe = load_all_csvs(
        "data"
    )

    # --------------------------------------------------------
    # Random 60 / 20 / 20 split
    # --------------------------------------------------------

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_random_data_split(
        full_dataframe,
        seed=seed,
    )

    # --------------------------------------------------------
    # Save exactly this split
    # --------------------------------------------------------

    save_data_split(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
    )

    # We no longer need the full combined dataframe.
    del full_dataframe

    # --------------------------------------------------------
    # Prepare TRAINING states
    #
    # StandardScaler is fitted only here.
    # --------------------------------------------------------

    print(
        "\nPreparing standardized "
        "training states..."
    )

    (
        train_states,
        train_labels,
        train_original_labels,
        scaler,
    ) = build_training_states(
        train_dataframe
    )

    del train_dataframe

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_path = (
        MODELS_DIR
        / "standard_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(
        f"Saved scaler to: {scaler_path}"
    )

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    train_states = np.asarray(
        train_states,
        dtype=np.float32,
    )

    train_labels = np.asarray(
        train_labels,
        dtype=np.int64,
    )

    print(
        "\nTraining data before balancing:"
    )

    print(
        f"States: {train_states.shape}"
    )

    print(
        f"Labels: {train_labels.shape}"
    )

    print(
        f"BENIGN: "
        f"{np.sum(train_labels == 0):,}"
    )

    print(
        f"ATTACK: "
        f"{np.sum(train_labels == 1):,}"
    )

    # --------------------------------------------------------
    # Balance TRAINING data only
    # --------------------------------------------------------

    (
        train_states,
        train_labels,
    ) = balance_training_data(
        train_states=train_states,
        train_labels=train_labels,
        seed=seed,
    )

    print(
        f"Training states: "
        f"{train_states.shape}"
    )

    print(
        f"Training labels: "
        f"{train_labels.shape}"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = NetworkEnvironment(
        states=train_states,
        labels=train_labels,
    )

    environment.action_space.seed(
        seed
    )

    environment.observation_space.seed(
        seed
    )

    state_dim = (
        environment
        .observation_space
        .shape[0]
    )

    action_dim = (
        environment.action_space.n
    )

    # --------------------------------------------------------
    # DQN Agent
    # --------------------------------------------------------

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    # --------------------------------------------------------
    # Optional LLM critic
    # --------------------------------------------------------

    llm_critic: Optional[
        LLMCritic
    ] = None

    reward_shaper: Optional[
        PotentialBasedRewardShaper
    ] = None

    if use_llm:

        print(
            "LLM-guided PBRS enabled."
        )

        llm_critic = LLMCritic(
            model="llama3"
        )

        reward_shaper = (
            PotentialBasedRewardShaper(
                gamma=gamma,
                scale=1.0,
            )
        )

    else:

        print(
            "Training baseline DQN "
            "using R_env only."
        )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    global_step = 0

    best_epoch_reward = float(
        "-inf"
    )

    reward_history: list[float] = []

    environment_reward_history: list[
        float
    ] = []

    shaping_reward_history: list[
        float
    ] = []

    loss_history: list[float] = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_seed = (
            seed
            + epoch
            - 1
        )

        state, reset_info = (
            environment.reset(
                seed=epoch_seed
            )
        )

        terminated = False
        truncated = False

        epoch_total_reward = 0.0

        epoch_environment_reward = (
            0.0
        )

        epoch_shaping_reward = 0.0

        epoch_losses: list[
            float
        ] = []

        step_in_epoch = 0

        # ----------------------------------------------------
        # Phi(s)
        # ----------------------------------------------------

        phi_current: Optional[
            float
        ] = None

        if use_llm:

            assert (
                llm_critic is not None
            )

            phi_current = (
                evaluate_potential(
                    critic=llm_critic,
                    normalized_state=state,
                    scaler=scaler,
                )
            )

        # ----------------------------------------------------
        # Environment loop
        # ----------------------------------------------------

        while (
            not terminated
            and not truncated
        ):

            action = (
                agent.select_action(
                    state=state,
                    explore=True,
                )
            )

            (
                next_state,
                environment_reward,
                terminated,
                truncated,
                info,
            ) = environment.step(
                action
            )

            done = (
                terminated
                or truncated
            )

            total_reward = float(
                environment_reward
            )

            shaping_reward = 0.0

            # ------------------------------------------------
            # LLM PBRS
            # ------------------------------------------------

            if use_llm:

                assert (
                    llm_critic
                    is not None
                )

                assert (
                    reward_shaper
                    is not None
                )

                assert (
                    phi_current
                    is not None
                )

                if done:

                    phi_next = 0.0

                else:

                    # IMPORTANT:
                    #
                    # Phi(s') must use NEXT_STATE,
                    # not the current state.
                    phi_next = (
                        evaluate_potential(
                            critic=llm_critic,
                            normalized_state=next_state,
                            scaler=scaler,
                        )
                    )

                reward_result = (
                    reward_shaper.calculate(
                        environment_reward=(
                            environment_reward
                        ),
                        phi_current=(
                            phi_current
                        ),
                        phi_next=(
                            phi_next
                        ),
                        done=done,
                    )
                )

                shaping_reward = float(
                    reward_result
                    .shaping_reward
                )

                total_reward = float(
                    reward_result
                    .total_reward
                )

                # Reuse Phi(s') during
                # the next iteration.
                phi_current = phi_next

            # ------------------------------------------------
            # Store transition
            # ------------------------------------------------

            agent.remember(
                state=state,
                action=action,
                reward=total_reward,
                next_state=next_state,
                done=done,
            )

            # ------------------------------------------------
            # DQN learning
            # ------------------------------------------------

            loss = agent.learn()

            if loss is not None:

                epoch_losses.append(
                    loss
                )

                loss_history.append(
                    loss
                )

            global_step += 1
            step_in_epoch += 1

            # ------------------------------------------------
            # Update target network
            # ------------------------------------------------

            if (
                global_step
                % target_update_interval
                == 0
            ):

                agent.update_target_network()

                print(
                    "Target network updated at "
                    f"global step "
                    f"{global_step:,}."
                )

            # ------------------------------------------------
            # Accumulate rewards
            # ------------------------------------------------

            epoch_total_reward += (
                total_reward
            )

            epoch_environment_reward += (
                environment_reward
            )

            epoch_shaping_reward += (
                shaping_reward
            )

            state = next_state

            # ------------------------------------------------
            # Optional step limit
            # ------------------------------------------------

            if (
                max_steps_per_epoch
                is not None
                and step_in_epoch
                >= max_steps_per_epoch
            ):

                truncated = True

        # ====================================================
        # END OF EPOCH
        # ====================================================

        average_loss = (
            float(
                np.mean(
                    epoch_losses
                )
            )
            if epoch_losses
            else float("nan")
        )

        reward_history.append(
            epoch_total_reward
        )

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
            f"\n  Average loss: "
            f"{average_loss:.6f}"
            f"\n  Epsilon: "
            f"{agent.current_epsilon:.6f}"
            f"\n  Replay-buffer size: "
            f"{len(agent.replay_buffer):,}"
        )

        # ----------------------------------------------------
        # Save best training checkpoint
        # ----------------------------------------------------

        if (
            epoch_total_reward
            > best_epoch_reward
        ):

            best_epoch_reward = (
                epoch_total_reward
            )

            model_name = (
                "best_llm_pbrs_dqn.pt"
                if use_llm
                else
                "best_baseline_dqn.pt"
            )

            model_path = (
                MODELS_DIR
                / model_name
            )

            agent.save(
                str(model_path)
            )

            print(
                f"Saved best model to: "
                f"{model_path}"
            )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    result_prefix = (
        "llm_pbrs"
        if use_llm
        else "baseline"
    )

    np.savetxt(
        RESULTS_DIR
        / f"{result_prefix}_reward_history.csv",
        np.asarray(
            reward_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    np.savetxt(
        RESULTS_DIR
        / (
            f"{result_prefix}"
            "_environment_reward_history.csv"
        ),
        np.asarray(
            environment_reward_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    np.savetxt(
        RESULTS_DIR
        / (
            f"{result_prefix}"
            "_shaping_reward_history.csv"
        ),
        np.asarray(
            shaping_reward_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    np.savetxt(
        RESULTS_DIR
        / f"{result_prefix}_loss_history.csv",
        np.asarray(
            loss_history,
            dtype=np.float64,
        ),
        delimiter=",",
    )

    print(
        "\nTraining completed."
    )

    print(
        f"Experiment seed: {seed}"
    )

    print(
        "\nNext:"
    )

    print(
        "Validation:"
    )

    print(
        "python src/evaluate.py "
        "--data-dir data_split/validation"
    )

    print(
        "\nFinal test:"
    )

    print(
        "python src/evaluate.py "
        "--data-dir data_split/test"
    )


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train a baseline or LLM-guided "
            "DQN for CICIDS2017 intrusion "
            "detection."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=(
            "Number of training epochs."
        ),
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help=(
            "Enable LLM Security Critic "
            "and potential-based reward shaping."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum environment steps "
            "per epoch."
        ),
    )

    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
        help=(
            "Environment steps between "
            "target-network updates."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help=(
            "Adam learning rate."
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "Reward discount factor."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Replay-buffer mini-batch size."
        ),
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help=(
            "Maximum replay-buffer size."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    arguments = (
        parse_arguments()
    )

    train(
        epochs=arguments.epochs,
        use_llm=arguments.use_llm,
        max_steps_per_epoch=(
            arguments.max_steps
        ),
        target_update_interval=(
            arguments
            .target_update_interval
        ),
        learning_rate=(
            arguments.learning_rate
        ),
        gamma=arguments.gamma,
        batch_size=(
            arguments.batch_size
        ),
        buffer_size=(
            arguments.buffer_size
        ),
        seed=arguments.seed,
    )
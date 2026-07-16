# src/evaluate.py

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from agent import DQNAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_test_csvs(data_dir: Path) -> pd.DataFrame:
    """
    Load and combine CSV files from a held-out test directory.
    """

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in: {data_dir}"
        )

    dataframes: list[pd.DataFrame] = []

    for csv_file in csv_files:
        print(f"Reading test file: {csv_file.name}")

        dataframe = pd.read_csv(csv_file)
        dataframe.columns = dataframe.columns.str.strip()

        if "Label" not in dataframe.columns:
            raise KeyError(
                f"'Label' column was not found in {csv_file.name}"
            )

        dataframes.append(dataframe)

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


def prepare_test_data(
    dataframe: pd.DataFrame,
    scaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean test features and transform them using the scaler
    previously fitted on the training data.

    Returns
    -------
    states:
        Standardized test state vectors.

    labels:
        Binary labels:
        0 = BENIGN
        1 = ATTACK
    """

    dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.str.strip()

    raw_labels = dataframe["Label"].astype(str).str.strip()

    binary_labels = np.where(
        raw_labels.str.upper() == "BENIGN",
        0,
        1,
    ).astype(np.int64)

    features = dataframe.drop(
        columns=["Label"]
    )

    # Preserve the same feature order used during training.
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(
            scaler.feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in features.columns
        ]

        extra_features = [
            feature
            for feature in features.columns
            if feature not in expected_features
        ]

        if missing_features:
            raise ValueError(
                "The test data are missing features used during "
                f"training: {missing_features}"
            )

        if extra_features:
            print(
                "Ignoring additional test features:",
                extra_features,
            )

        features = features[expected_features]

    # Convert every feature to a numeric value.
    features = features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Retain only rows whose features are all valid.
    valid_rows = features.notna().all(axis=1)

    removed_rows = int(
        (~valid_rows).sum()
    )

    if removed_rows > 0:
        print(
            f"Removed {removed_rows:,} rows containing "
            "NaN or infinite values."
        )

    features = features.loc[valid_rows]
    binary_labels = binary_labels[
        valid_rows.to_numpy()
    ]

    if len(features) == 0:
        raise ValueError(
            "No valid test rows remain after data cleaning."
        )

    states = scaler.transform(features)

    return (
        np.asarray(
            states,
            dtype=np.float32,
        ),
        np.asarray(
            binary_labels,
            dtype=np.int64,
        ),
    )


def calculate_false_positive_rate(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Compute false-positive rate:

        FPR = FP / (FP + TN)
    """

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    true_negatives = int(matrix[0, 0])
    false_positives = int(matrix[0, 1])

    denominator = (
        true_negatives
        + false_positives
    )

    if denominator == 0:
        return 0.0

    return false_positives / denominator


def evaluate(
    data_dir: Path,
    model_path: Path,
    scaler_path: Path,
    results_dir: Path,
    batch_size: int = 64,
) -> None:
    """
    Evaluate a trained DQN on held-out CICIDS2017 traffic.

    During evaluation:
    - epsilon exploration is disabled
    - replay memory is not used
    - neural-network weights are not updated
    """

    print(f"Loading scaler: {scaler_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file does not exist: {scaler_path}"
        )

    scaler = joblib.load(scaler_path)

    dataframe = load_test_csvs(
        data_dir=data_dir
    )

    states, true_labels = prepare_test_data(
        dataframe=dataframe,
        scaler=scaler,
    )

    state_dim = states.shape[1]
    action_dim = 2

    print(f"Test samples: {len(states):,}")
    print(f"State dimension: {state_dim}")
    print(
        f"Benign test samples: "
        f"{int((true_labels == 0).sum()):,}"
    )
    print(
        f"Attack test samples: "
        f"{int((true_labels == 1).sum()):,}"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=batch_size,
    )

    agent.load(str(model_path))

    # Evaluation mode disables training-specific behavior.
    agent.policy_network.eval()

    predictions = np.empty(
        len(states),
        dtype=np.int64,
    )

    print("Evaluating trained policy...")

    for index, state in enumerate(states):
        # explore=False means:
        # no random epsilon action.
        action = agent.select_action(
            state=state,
            explore=False,
        )

        # Action mapping:
        # 0 = Allow  -> predicted BENIGN
        # 1 = Inspect -> predicted ATTACK
        predictions[index] = action

        if (
            index > 0
            and index % 100_000 == 0
        ):
            print(
                f"Evaluated {index:,} "
                f"of {len(states):,} samples."
            )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predictions,
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    binary_f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    false_positive_rate = (
        calculate_false_positive_rate(
            true_labels=true_labels,
            predictions=predictions,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=[
            "BENIGN",
            "ATTACK",
        ],
        zero_division=0,
    )

    results = {
        "number_of_test_samples": int(
            len(states)
        ),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "attack_precision": float(
            precision
        ),
        "attack_recall": float(recall),
        "attack_f1": float(binary_f1),
        "macro_f1": float(macro_f1),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
    }

    print("\nEvaluation Results")
    print("==================")
    print(f"Accuracy:           {accuracy:.4f}")
    print(
        f"Balanced accuracy:  "
        f"{balanced_accuracy:.4f}"
    )
    print(f"Attack precision:   {precision:.4f}")
    print(f"Attack recall:      {recall:.4f}")
    print(f"Attack F1:          {binary_f1:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(
        f"False-positive rate:"
        f" {false_positive_rate:.4f}"
    )

    print("\nConfusion Matrix")
    print("================")
    print(matrix)

    print("\nClassification Report")
    print("=====================")
    print(report)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_file = (
        results_dir
        / "evaluation_metrics.json"
    )

    predictions_file = (
        results_dir
        / "evaluation_predictions.csv"
    )

    report_file = (
        results_dir
        / "classification_report.txt"
    )

    with results_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    prediction_dataframe = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_action": predictions,
            "true_name": np.where(
                true_labels == 0,
                "BENIGN",
                "ATTACK",
            ),
            "predicted_name": np.where(
                predictions == 0,
                "ALLOW",
                "INSPECT",
            ),
        }
    )

    prediction_dataframe.to_csv(
        predictions_file,
        index=False,
    )

    report_file.write_text(
        report,
        encoding="utf-8",
    )

    print(f"\nSaved metrics to: {results_file}")
    print(
        f"Saved predictions to: "
        f"{predictions_file}"
    )
    print(
        f"Saved classification report to: "
        f"{report_file}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DQN on held-out "
            "CICIDS2017 CSV files."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing held-out test CSV files."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            DEFAULT_MODELS_DIR
            / "best_baseline_dqn.pt"
        ),
        help="Path to the saved DQN checkpoint.",
    )

    parser.add_argument(
        "--scaler",
        type=Path,
        default=(
            DEFAULT_MODELS_DIR
            / "standard_scaler.joblib"
        ),
        help=(
            "Path to the StandardScaler fitted "
            "during training."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for evaluation output.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Agent batch size. It does not affect "
            "evaluation predictions."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    evaluate(
        data_dir=arguments.data_dir,
        model_path=arguments.model,
        scaler_path=arguments.scaler,
        results_dir=arguments.results_dir,
        batch_size=arguments.batch_size,
    )
```

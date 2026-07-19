# src/feature_extractor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_csvs


def clean_features_and_labels(
    df: pd.DataFrame,
):
    """
    Separate network-flow features and labels,
    remove invalid rows, and convert labels to binary.

    Labels:
        0 = BENIGN
        1 = ATTACK
    """

    df = df.copy()

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise KeyError(
            "'Label' column was not found in the dataframe."
        )

    # Separate labels from network-flow features
    labels = df["Label"].copy()

    features = df.drop(
        columns=["Label"]
    )

    # Convert feature columns to numeric values
    features = features.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinity with NaN
    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows containing missing/invalid feature values
    features = features.dropna()

    # Keep labels corresponding only to valid feature rows
    labels = labels.loc[
        features.index
    ]

    # Convert CICIDS2017 labels to binary labels:
    # BENIGN = 0
    # All attacks = 1
    binary_labels = labels.apply(
        lambda label:
        0
        if str(label).strip().upper() == "BENIGN"
        else 1
    )

    return (
        features,
        binary_labels.to_numpy(
            dtype=np.int64
        ),
    )


def build_training_states(
    df: pd.DataFrame,
):
    """
    Build state vectors for TRAINING data.

    The StandardScaler is fitted ONLY on training data.

    Returns:
        states
        labels
        scaler
    """

    features, labels = clean_features_and_labels(
        df
    )

    # Create and fit scaler using TRAINING data only
    scaler = StandardScaler()

    states = scaler.fit_transform(
        features
    )

    return (
        states.astype(np.float32),
        labels,
        scaler,
    )


def build_evaluation_states(
    df: pd.DataFrame,
    scaler: StandardScaler,
):
    """
    Build state vectors for VALIDATION or TEST data.

    Important:
    The scaler is NOT fitted again.

    It uses the scaler previously fitted on training data.
    """

    features, labels = clean_features_and_labels(
        df
    )

    # Make sure feature order matches training data
    if hasattr(
        scaler,
        "feature_names_in_"
    ):
        expected_features = list(
            scaler.feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in features.columns
        ]

        if missing_features:
            raise ValueError(
                "Validation/test data are missing "
                f"features: {missing_features}"
            )

        features = features[
            expected_features
        ]

    # IMPORTANT:
    # transform(), NOT fit_transform()
    states = scaler.transform(
        features
    )

    return (
        states.astype(np.float32),
        labels,
    )


if __name__ == "__main__":

    # ---------------------------------
    # TRAINING DATA
    # ---------------------------------

    train_df = load_all_csvs(
        "data/train"
    )

    train_states, train_labels, scaler = (
        build_training_states(
            train_df
        )
    )

    print("\nTraining Data")
    print("----------------")
    print(
        "States:",
        train_states.shape
    )
    print(
        "Labels:",
        train_labels.shape
    )
    print(
        "Scaler mean:",
        scaler.mean_
    )

    # ---------------------------------
    # VALIDATION DATA
    # ---------------------------------

    validation_df = load_all_csvs(
        "data/validation"
    )

    validation_states, validation_labels = (
        build_evaluation_states(
            validation_df,
            scaler
        )
    )

    print("\nValidation Data")
    print("----------------")
    print(
        "States:",
        validation_states.shape
    )
    print(
        "Labels:",
        validation_labels.shape
    )
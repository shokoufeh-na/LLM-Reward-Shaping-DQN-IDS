# src/feature_extractor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_features_and_labels(
    df: pd.DataFrame,
):
    """
    Clean CICIDS2017 data.

    Returns:
        features
        binary_labels
        original_labels

    Binary labels:
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

    # Keep original attack labels
    original_labels = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Separate network-flow features
    features = df.drop(
        columns=["Label"]
    )

    # Convert feature columns to numeric
    features = features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinity with NaN
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Find valid rows
    valid_mask = ~features.isna().any(axis=1)

    # Keep only valid rows
    features = features.loc[
        valid_mask
    ].copy()

    original_labels = original_labels.loc[
        valid_mask
    ].copy()

    # Binary labels
    binary_labels = (
        original_labels
        .str.upper()
        .ne("BENIGN")
        .astype(np.int64)
        .to_numpy()
    )

    return (
        features,
        binary_labels,
        original_labels.to_numpy(),
    )


def build_training_states(
    df: pd.DataFrame,
):
    """
    Build state vectors for TRAINING data.

    StandardScaler is fitted only on training data.

    Returns:
        states
        binary_labels
        original_labels
        scaler
    """

    (
        features,
        binary_labels,
        original_labels,
    ) = clean_features_and_labels(df)

    scaler = StandardScaler()

    states = scaler.fit_transform(
        features
    )

    return (
        states.astype(np.float32),
        binary_labels,
        original_labels,
        scaler,
    )


def build_evaluation_states(
    df: pd.DataFrame,
    scaler: StandardScaler,
):
    """
    Build state vectors for VALIDATION or TEST data.

    The scaler is NOT fitted again.
    """

    (
        features,
        binary_labels,
        original_labels,
    ) = clean_features_and_labels(df)

    # Make sure feature order matches training
    if hasattr(
        scaler,
        "feature_names_in_",
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

    states = scaler.transform(
        features
    )

    return (
        states.astype(np.float32),
        binary_labels,
        original_labels,
    )
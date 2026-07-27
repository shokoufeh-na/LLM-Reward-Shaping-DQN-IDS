# src/data_loader.py

from pathlib import Path

import pandas as pd


def load_all_csvs(
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load and combine all CICIDS2017 CSV files.

    Expected structure
    ------------------
    project/
    ├── data/
    │   ├── Monday-....csv
    │   ├── Tuesday-....csv
    │   ├── Wednesday-....csv
    │   ├── Thursday-....csv
    │   └── Friday-....csv
    └── src/
        └── data_loader.py

    Returns
    -------
    pd.DataFrame
        Combined CICIDS2017 dataframe.
    """

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    data_path = project_root / data_dir

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )

    # Read every CSV directly inside data/
    files = sorted(
        data_path.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}"
        )

    print(
        f"Found {len(files)} CSV files."
    )

    dataframes = []

    for file in files:

        print(f"\nReading: {file.name}")

        dataframe = pd.read_csv(file)

        # Remove spaces around CICIDS2017 column names
        dataframe.columns = (
            dataframe.columns.str.strip()
        )

        if "Label" not in dataframe.columns:
            raise KeyError(
                f"'Label' column was not found "
                f"in {file.name}"
            )

        print(
            f"Shape: {dataframe.shape}"
        )

        print(
            dataframe["Label"].value_counts()
        )

        dataframes.append(dataframe)

    # Combine all days
    data = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print("\n==============================")
    print("Complete CICIDS2017 Dataset")
    print("==============================")

    print(
        f"Combined shape: {data.shape}"
    )

    print("\nLabel distribution:")

    print(
        data["Label"].value_counts()
    )

    return data


if __name__ == "__main__":

    data = load_all_csvs("data")
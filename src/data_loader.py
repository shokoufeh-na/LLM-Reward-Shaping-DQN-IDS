from pathlib import Path

import pandas as pd


def load_all_csvs(data_dir: str = "data/train") -> pd.DataFrame:
    """
    Load and combine all CSV files from a project data subfolder.

    Example:
        load_all_csvs("data/train")
        load_all_csvs("data/validation")
        load_all_csvs("data/test")
    """

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / data_dir

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )

    files = sorted(data_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files were found in: {data_path}"
        )

    dataframes = []

    for file in files:
        print(f"Reading: {file.name}")

        dataframe = pd.read_csv(file)

        # CICIDS2017 column names sometimes contain spaces.
        dataframe.columns = dataframe.columns.str.strip()

        if "Label" not in dataframe.columns:
            raise KeyError(
                f"'Label' column was not found in {file.name}"
            )

        print("Shape:", dataframe.shape)
        print(dataframe["Label"].value_counts())

        dataframes.append(dataframe)

    data = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print("\nCombined data shape:", data.shape)
    print("\nData preview:")
    print(data.head())

    return data


if __name__ == "__main__":
    data = load_all_csvs("data/train")
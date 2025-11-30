import pandas as pd
from sklearn.model_selection import train_test_split

GCS_NEW_DATASET_PATH = "gs://aura_datasets_training_validation/new_dataset_with_target.csv"


def fix_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the new dataset
    """

    # Rename columns
    rename_map = {
        "people_count": "crowd_count",
        "lux_db": "light_lux",
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["noise_db", "light_lux", "crowd_count", "discomfort_level"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Scale crowd_count
    df["crowd_count"] = df["crowd_count"].astype(float) / 10.0

    # Make all floats
    df["noise_db"] = df["noise_db"].astype(float)
    df["light_lux"] = df["light_lux"].astype(float)
    df["discomfort_level"] = df["discomfort_level"].astype(float)

    return df[required]


def load_and_prepare_new_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load the new dataset from GCS, normalize it, and split into
    df_train and df_val.
    """

    df = pd.read_csv(GCS_NEW_DATASET_PATH)

    df = fix_dataset_schema(df)

    # Split into train / validation dataframes
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return df_train, df_val


if __name__ == "__main__":
    train_df, val_df = load_and_prepare_new_dataset()

import pandas as pd
from sklearn.model_selection import train_test_split

# GCS path for the new unified dataset
GCS_NEW_DATASET_PATH = "gs://aura_datasets_training_validation/new_dataset_with_target.csv"


def fix_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the new dataset so it matches the schema expected by the model:
      - noise_db
      - light_lux
      - crowd_count
      - discomfort_level

    This works using ONLY the new dataset (no old CSVs).
    """

    # 1. Rename new columns to match expected feature names
    rename_map = {
        "people_count": "crowd_count",
        "lux_db": "light_lux",
    }
    df = df.rename(columns=rename_map)

    # 2. Ensure required columns exist
    required = ["noise_db", "light_lux", "crowd_count", "discomfort_level"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # 3. Scale crowd_count down so it’s in a reasonable range for the model.
    #    New data has people_count around 90–150 → divide by 10 → ~9–15.
    df["crowd_count"] = df["crowd_count"].astype(float) / 10.0

    # 4. Enforce numeric types for all features/target
    df["noise_db"] = df["noise_db"].astype(float)
    df["light_lux"] = df["light_lux"].astype(float)
    df["discomfort_level"] = df["discomfort_level"].astype(float)

    # 5. Return only the columns the model expects, in correct order
    return df[required]


def load_and_prepare_new_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load the new dataset from GCS, normalize it, and split into
    df_train and df_val that can be used by gradient_boosting.py.
    """

    # 1. Load raw dataset from GCS
    df = pd.read_csv(GCS_NEW_DATASET_PATH)

    # 2. Normalize schema and scaling
    df = fix_dataset_schema(df)

    # 3. Split into train / validation dataframes
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        # regression target → no stratify
    )

    return df_train, df_val


if __name__ == "__main__":
    # Simple manual test: run this file directly to see shapes
    train_df, val_df = load_and_prepare_new_dataset()
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print(train_df.head())

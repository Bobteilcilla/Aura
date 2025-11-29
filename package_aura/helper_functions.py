'''
This file contains all helper functions used throughout the project.
This includes:
- download_data: To download original dataset for testing purposes
- discomfort_to_label: Maps the discomfort level to a verbally interpretable label
- upload_data_to_gcs: To upload model data to GCS
- load_model_from_gcs: To download model data from GCS
- load_dataset: To download the dataset from GCS
'''

# Imports
import pandas as pd
from google.cloud import storage
import pickle
import json
from datetime import datetime, UTC
from pathlib import Path
import os


def download_data():
    client = storage.Client()
    bucket = client.bucket(os.environ["DATA_BUCKET"])

    # Path to raw-data folder: Aura/raw-data/
    raw_data_dir = Path(__file__).resolve().parent.parent / "raw-data"
    raw_data_dir.mkdir(exist_ok=True)  # Create folder if missing

    files = {
    "AURA_aug_sep_60k.csv": "AURA_aug_sep_60k.csv",
    "AURA_validation_sep_12k.csv": "AURA_validation_sep_12k.csv",
}

    for blob_name, local_filename in files.items():

        local_path = raw_data_dir / local_filename

        print(f"Downloading gs://{bucket}/{blob_name}")
        print(f"→ {local_path}")

        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))

        print(f"Saved: {local_path}\n")

    return None



def discomfort_to_label(d: float) -> str:
    '''
    Turns numerical discomfort level to interpretable label.

    Args:
        d: Numerical discomfort_level as returned from model prediction
    '''
    if d <= 0.2:
        return "very_comfortable"
    elif d <= 0.4:
        return "comfortable"
    elif d <= 0.6:
        return "neutral"
    elif d <= 0.8:
        return "uncomfortable"
    else:
        return "stressed"



def upload_data_to_gcs(bucket_name, gcs_path, data):
    """
    Upload data to GCS.

    Args:
        bucket_name: GCS bucket name.
        gcs_path: Path inside bucket, e.g. "models/20250201-120000/logreg_pipeline.pkl".
        data: Bytes to upload (pickle.dumps output or JSON-encoded string).
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    blob.upload_from_string(data)

    print(f"Uploaded to gs://{bucket_name}/{gcs_path}")

    return None



def load_model_from_gcs(bucket_name, model_prefix, model_type):
    '''
    Loads latest data to a given folder (e.g. "logreg_model") from GCS.

    1. Connect to client
    2. List blobs
    3. Filter for blob with newest (i.e. highest) timestamp
    4. Load data from that blob

    Args:
        bucket_name: Name of the GCS bucket that holds the model data.
        model_prefix: Prefix for the model to identify the right bucket-folder.
        model_type: Type of the saved model, e.g. linreg

    Output:
        (pipeline, label_encoder, metadata)
    '''
    # 1. Connect to client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # 2. List blobs
    blobs = client.list_blobs(bucket_name, prefix=model_prefix)

    # 3. Filter for blob with newest (i.e. highest) timestamp
    timestamps = set()

    for blob in blobs:
        # e.g. log_reg_models/20251115-145418/logreg_pipeline.pkl
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[0] == model_prefix.rstrip("/"):
            timestamps.add(parts[1])

    latest_timestamp = max(timestamps)

    # 4. Load data from that blob
    model_path = f"{model_prefix}/{latest_timestamp}/{model_type}_pipeline.pkl"
    metadata_path = f"{model_prefix}/{latest_timestamp}/{model_type}_metadata.json"

    ## Load raw data
    model_bytes = bucket.blob(model_path).download_as_bytes()
    metadata_bytes = bucket.blob(metadata_path).download_as_bytes()

    # Encode pickle
    pipeline = pickle.loads(model_bytes)
    metadata = json.loads(metadata_bytes.decode("utf-8"))

    print("Model and metadata loaded successfully")

    return (pipeline, metadata)



def load_dataset():
    # 1. Fetch data
    df_train = pd.read_csv("gs://aura_datasets_training_validation/AURA_aug_sep_60k.csv")
    df_val = pd.read_csv("gs://aura_datasets_training_validation/AURA_validation_sep_12k.csv")

    # 2. Define X and y for training and validation set
    feature_cols = ["noise_db", "light_lux", "crowd_count"]
    target_col = "discomfort_level"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "download_data":
        download_data()

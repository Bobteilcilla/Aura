from google.cloud import storage
from pathlib import Path

# -------- CONFIG ---------

BUCKET_NAME = "aura_datasets_training_validation"

# Exact object names inside the bucket

FILES_TO_DOWNLOAD = {
    "new_dataset_with_target.csv": "new_dataset_with_target.csv",
}


def get_storage_client():
    """Get authenticated GCS client using ADC or env variable."""
    client = storage.Client()
    return client


def download_files():
    client = get_storage_client()
    bucket = client.bucket(BUCKET_NAME)

    # Path to raw-data folder: Aura/raw-data/
    raw_data_dir = Path(__file__).resolve().parent.parent / "raw-data"
    raw_data_dir.mkdir(exist_ok=True)  # Create folder if missing

    for blob_name, local_filename in FILES_TO_DOWNLOAD.items():

        local_path = raw_data_dir / local_filename

        print(f"Downloading gs://{BUCKET_NAME}/{blob_name}")
        print(f"→ {local_path}")

        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))

        print(f"Saved: {local_path}\n")


if __name__ == "__main__":
    download_files()

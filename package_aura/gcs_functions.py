# Imports
from google.cloud import storage
import pickle
import json
from datetime import datetime, UTC

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


def load_data_from_gcs(bucket_name, model_prefix, model_type):
    '''
    Loads latest data to a given folder (e.g. "logreg_model") from GCS.

    1. Connect to client
    2. List blobs
    3. Filter for blob with newest (i.e. highest) timestamp
    4. Load data from that blob

    Args:
        bucket_name: Name of the GCS bucket that holds the model data.
        model_prefix: Prefix for the model to identify the right bucket-folder.
        model_type: Type of the saved model, e.g. lin_reg

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

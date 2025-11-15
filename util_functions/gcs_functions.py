# Imports
from google.cloud import storage
import pickle
import json
from datetime import datetime, UTC

def upload_data_to_gcs(bucket_name, gcs_path, data, content_type=None):
    """
    Upload data to Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name.
        gcs_path: Path inside bucket, e.g. "models/20250201-120000/logreg_pipeline.pkl".
        data: Bytes to upload (pickle.dumps output or JSON-encoded string).
        content_type: Optional MIME type (e.g. "application/json").
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    if content_type:
        blob.upload_from_string(data, content_type=content_type)
    else:
        blob.upload_from_string(data)

    print(f"Uploaded to gs://{bucket_name}/{gcs_path}")

# ...existing code...
import os
import pickle
import json
from datetime import datetime, UTC

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from package_aura.gcs_functions import upload_data_to_gcs, load_data_from_gcs

def train_gradient_boosting_model():
    """
    Train a plain GradientBoostingRegressor on the training CSV in GCS
    and upload the pipeline + metadata to GCS under a versioned folder.
    """
    # 1. Fetch data from GCS
    df_train = pd.read_csv("gs://aura_datasets_training_validation/AURA_aug_sep_60k.csv")
    df_val = pd.read_csv("gs://aura_datasets_training_validation/AURA_validation_sep_12k.csv")

    # 2. Define X and y
    feature_cols = ["noise_db", "light_lux", "crowd_count"]
    target_col = "discomfort_level"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # 3. Preprocessing pipeline (no scaler needed for Gradient Boosting)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols)
        ],
        remainder="drop"
    )

    # 4. Plain Gradient Boosting model (no GridSearch)
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", gb),
    ])

    # 5. Fit on training data
    pipe.fit(X_train, y_train)

    # (Optional) quick evaluation on validation set
    val_pred = pipe.predict(X_val)
    mse_val = ((val_pred - y_val) ** 2).mean()
    print(f"Validation MSE: {mse_val:.6f}")

    # 6. Create metadata
    metadata = {
        "training_timestamp": datetime.now(UTC).isoformat(),
        "model_type": "gradient_boosting",
        "feature_names": feature_cols,
        "hyperparameters": gb.get_params(),  # all GB params
        "training_samples": int(len(X_train)),
        "validation_mse": float(mse_val),
    }

    # 7. Serialize and upload to GCS
    model_bytes = pickle.dumps(pipe)
    metadata_json = json.dumps(metadata, indent=2)

    bucket_name = os.environ["MODEL_BUCKET"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    gcs_folder = f"gb_models/{timestamp}"

    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/gb_pipeline.pkl",
        data=model_bytes
    )
    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/gb_metadata.json",
        data=metadata_json
    )

    print(f"Uploaded gradient boosting model to gs://{bucket_name}/{gcs_folder}/")
    return None


def gradient_boosting_predict(noise_db, light_lux, crowd_count):
    """
    Load latest GB model from GCS and predict discomfort_level for given inputs.
    Returns dict with rounded discomfort_score.
    """
    bucket_name = os.environ["MODEL_BUCKET"]
    model_prefix = "gb_models"
    model_type = "gb"

    input_df = pd.DataFrame([{
        "noise_db": noise_db,
        "light_lux": light_lux,
        "crowd_count": crowd_count
    }])

    pipeline, metadata = load_data_from_gcs(bucket_name, model_prefix, model_type)

    prediction = float(pipeline.predict(input_df)[0])
    prediction = max(0.0, min(1.0, prediction))

    return {
        "discomfort_score": round(prediction, 3),
        "metadata": metadata
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train_gb":
        train_gradient_boosting_model()

    elif len(sys.argv) > 1 and sys.argv[1] == "predict_gb":
        noise_db = float(sys.argv[2])
        light_lux = float(sys.argv[3])
        crowd_count = float(sys.argv[4])
        print(gradient_boosting_predict(noise_db, light_lux, crowd_count))

    else:
        print("Usage: python gradient_boosting_gcs.py [train_gb | predict_gb <noise> <light> <crowd>]")
# ...existing code...

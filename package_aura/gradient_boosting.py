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
from sklearn.model_selection import GridSearchCV

from package_aura.gcs_functions import upload_data_to_gcs, load_data_from_gcs

def train_gradient_boosting_model():
    """
    Train a GradientBoostingRegressor on the training CSV in GCS, perform a small grid search,
    and upload the best pipeline + metadata to GCS under a versioned folder.
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

    # 3. Preprocessing pipeline
    numeric_features = feature_cols
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )

    # 4. Model pipeline
    gb = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", gb)])

    # 5. Optional small grid search to find good hyperparameters (kept small to be practical)
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best CV (neg_mse):", grid_search.best_score_)

    # 6. Extract best pipeline
    best_pipe = grid_search.best_estimator_

    # 7. Create metadata
    metadata = {
        "training_timestamp": datetime.now(UTC).isoformat(),
        "model_type": "gradient_boosting",
        "feature_names": feature_cols,
        "hyperparameters": grid_search.best_params_,
        "training_samples": int(len(X_train))
    }

    # 8. Serialize and upload to GCS
    model_bytes = pickle.dumps(best_pipe)
    metadata_json = json.dumps(metadata, indent=2)

    bucket_name = os.environ["MODEL_BUCKET"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    gcs_folder = f"gb_models/{timestamp}"

    upload_data_to_gcs(bucket_name=bucket_name, gcs_path=f"{gcs_folder}/gb_pipeline.pkl", data=model_bytes)
    upload_data_to_gcs(bucket_name=bucket_name, gcs_path=f"{gcs_folder}/gb_metadata.json", data=metadata_json)

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
        "discomfort_score": round(prediction, 2),
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

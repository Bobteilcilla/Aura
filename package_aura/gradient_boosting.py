import os
import pickle
import json
from datetime import datetime, UTC

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from package_aura.gcs_functions import upload_data_to_gcs, load_data_from_gcs
from package_aura.prepare_new_dataset import load_and_prepare_new_dataset

def train_gradient_boosting_model():
    """
    1. Fetch data from GCP buckets
    2. Define X and y for training and validation set
    3. Build pipeline
    4. Create model
    5. Grid search for optimal parameters
    6. Evaluate best model on validation data
    7. Save pipeline and meta-data
    8. Push pipeline and meta-data to the cloud using util function
    """

    # 1. Fetch data
    df_train, df_val = load_and_prepare_new_dataset()

    # 2. Define X and y for training and validation set
    feature_cols = ["noise_db", "light_lux", "crowd_count"]
    target_col = "discomfort_level"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # 3. Build pipeline
    numeric_features = feature_cols
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )

    # 4. Base Gradient Boosting model
    gb = GradientBoostingRegressor(random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", gb),
        ]
    )

    # 5. Grid search for optimal parameters (kept relatively small)
    param_grid = {
        "clf__n_estimators": [200, 300],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
        "clf__subsample": [0.8, 0.9],
        "clf__min_samples_leaf": [1, 3],
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best CV neg_mae:", grid_search.best_score_)

    # 6. Extract best model and evaluate on validation set
    best_pipe = grid_search.best_estimator_

    val_pred = best_pipe.predict(X_val)

    mse_val = float(((val_pred - y_val) ** 2).mean())
    mae_val = float(mean_absolute_error(y_val, val_pred))
    rmse_val = float(np.sqrt(mse_val))
    r2_val = float(r2_score(y_val, val_pred))

    print(f"Validation MSE:  {mse_val:.6f}")
    print(f"Validation RMSE: {rmse_val:.6f}")
    print(f"Validation MAE:  {mae_val:.6f}")
    print(f"Validation R²:   {r2_val:.4f}")

    # 7. Define pipeline and meta-data to upload afterwards
    gb_model = best_pipe

    gb_metadata = {
        "gb_training_timestamp": datetime.now(UTC).isoformat(),
        "model_type": "gradient_boosting",
        "feature_names": feature_cols,
        "hyperparameters": grid_search.best_params_,
        "cv_neg_mae": float(grid_search.best_score_),
        "training_samples": int(len(X_train)),
        "validation_mse": mse_val,
        "validation_rmse": rmse_val,
        "validation_mae": mae_val,
        "validation_r2": r2_val,
    }

    # 8. Push pipeline and meta-data to the cloud
    model_bytes = pickle.dumps(gb_model)
    metadata_json = json.dumps(gb_metadata, indent=4)

    bucket_name = os.environ["MODEL_BUCKET"]

    # Create versioned folder in bucket
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    gcs_folder = f"gb_models/{timestamp}"

    # Upload files
    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/gb_pipeline.pkl",
        data=model_bytes,
    )

    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/gb_metadata.json",
        data=metadata_json,
    )

    print(f"All data uploaded to: gs://{bucket_name}/{gcs_folder}/")

    return None


def gradient_boosting_predict(noise_db, light_lux, crowd_count):
    """
    Load a Gradient Boosting model from GCS and make a prediction.

    Args:
        noise_db: Input for noise
        light_lux: Input for brightness
        crowd_count: Input for crowdedness

    Output:
        Returns discomfort_level between 0.0 and 1.0.
    """
    bucket_name = os.environ["MODEL_BUCKET"]
    model_prefix = "gb_models"
    model_type = "gb"

    input_df = pd.DataFrame(
        [
            {
                "noise_db": noise_db,
                "light_lux": light_lux,
                "crowd_count": crowd_count,
            }
        ]
    )

    pipeline, metadata = load_data_from_gcs(bucket_name, model_prefix, model_type)

    prediction = float(pipeline.predict(input_df)[0])

    # Make sure that prediction is in range 0 - 1
    prediction = max(0.0, min(1.0, prediction))

    return {
        "discomfort_score": round(prediction, 2)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train_gradient_boosting_model":
        train_gradient_boosting_model()

    elif len(sys.argv) > 1 and sys.argv[1] == "gradient_boosting_predict":
        noise_db = float(sys.argv[2])
        light_lux = float(sys.argv[3])
        crowd_count = int(sys.argv[4])

        print(gradient_boosting_predict(noise_db, light_lux, crowd_count))

    else:
        print(
            "No valid command given.\n"
            "Usage:\n"
            "  python gradient_boosting_gcs.py train_gradient_boosting_model\n"
            "  python gradient_boosting_gcs.py gradient_boosting_predict <noise> <light> <crowd>"
        )

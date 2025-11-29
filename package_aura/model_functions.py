# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from google.cloud import storage
import pickle

import os
from datetime import datetime, UTC
import json

from package_aura.helper_functions import upload_data_to_gcs, load_model_from_gcs, load_dataset


def train_linreg_model():
    '''
    1. Fetch data from GCP buckets and get X, y using helper function
    2. Build pipeline
    3. Create model
    4. Grid search for optimal parameters
    5. Extract best model
    6. Save pipeline, and meta-data
    7. Push pipeline, and meta-data to the cloud using util function
    '''

    # 1. Fetch data and get X, y
    X_train, y_train, X_val, y_val = load_dataset()
    feature_cols = ["noise_db", "light_lux", "crowd_count"]

    # 2. Build pipeline
    numeric_features = feature_cols
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )

    # 3. Create model
    ## LinReg model
    linreg = ElasticNet()

    ## Full pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", linreg)
    ])

    # 4. Grid search for optimal parameters
    param_grid = {
        "clf__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best CV neg_mae:", grid_search.best_score_)

    # 5. Extract best model
    best_pipe = grid_search.best_estimator_

    # 6. Define pipeline, label encoder, and meta-data to upload afterwards
    ## Model
    linreg_model = best_pipe

    ## Meta Data
    linreg_metadata = {
        "linreg_training_timestamp": datetime.now(UTC).isoformat(),
        "model_type": "linear_regression",
        "feature_names": feature_cols,
        "hyperparameters": grid_search.best_params_
    }

    # 7. Push pipeline, label encoder, and meta-data to the cloud
    model_bytes = pickle.dumps(linreg_model)
    metadata_json = json.dumps(linreg_metadata, indent=4)

    bucket_name = os.environ["MODEL_BUCKET"]

    ## Create versioned folder in bucket
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    gcs_folder = f"linreg_models/{timestamp}"

    ## Upload files
    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/linreg_pipeline.pkl",
        data=model_bytes
    )

    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/linreg_metadata.json",
        data=metadata_json,
    )

    print(f"All data uploaded to: gs://{bucket_name}/{gcs_folder}/")

    return None



def train_gradient_boosting_model():
    """
    1. Fetch data from GCP buckets and define X, y
    2. Build pipeline
    3. Create model
    4. Grid search for optimal parameters
    5. Evaluate best model on validation data
    6. Save pipeline and meta-data
    7. Push pipeline and meta-data to the cloud using util function
    """

    # 1. Fetch data
    X_train, y_train, X_val, y_val = load_dataset()
    feature_cols = ["noise_db", "light_lux", "crowd_count"]

    # 2. Build pipeline
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

    # 4. Grid search for optimal parameters (kept relatively small)
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

    # 5. Extract best model and evaluate on validation set
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

    # 6. Define pipeline and meta-data to upload afterwards
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

    # 7. Push pipeline and meta-data to the cloud
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




def model_predict(noise_db, light_lux, crowd_count, model_prefix):
    '''
    Load a LinReg model from GCS and make a prediction.

    Args:
        noise_db: Input for noise
        light_lux: Input for brightness
        crowd_count: Input for crowdedness
        model_prefix: Specify which model from GCS to use

    Output:
        Returns discomfort_level between 0.0 and 1.0.
    '''
    if model_prefix == "gb_models":
        model_type = "gb"
    elif model_prefix == "linreg_models":
        model_type = "linreg"
    else:
        print("Model not found")
        return None

    bucket_name = os.environ["MODEL_BUCKET"]

    input_df = pd.DataFrame([{
        "noise_db": noise_db,
        "light_lux": light_lux,
        "crowd_count": crowd_count
    }])

    pipeline, metadata = load_model_from_gcs(bucket_name, model_prefix, model_type)

    # Pipeline for preprocessing and prediction
    prediction = float(pipeline.predict(input_df)[0])

    # Make sure that prediction is in range 0 - 1
    prediction = max(0.0, min(1.0, prediction))

    return {
        "discomfort_score": round(prediction, 2)
    }


def model_evaluate(model_prefix):
    '''
    Load a model from GCS and get evaluation.

    Args:
        model_prefix: Model specific folder in model GCS bucket, e.g.:
            gb_models for gradient boosting model
            linreg_models for linear regression model
    '''
    # Map file-name to folder name
    if model_prefix == "gb_models":
        model_type = "gb"
    elif model_prefix == "linreg_models":
        model_type = "linreg"
    else:
        print("Model not found")
        return None

    # Get model
    bucket_name = os.environ["MODEL_BUCKET"]
    model_prefix = model_prefix

    model, metadata = load_model_from_gcs(bucket_name, model_prefix, model_type)

    # Get data
    df_val = pd.read_csv("gs://aura_datasets_training_validation/AURA_validation_sep_12k.csv")

    feature_cols = ["noise_db", "light_lux", "crowd_count"]
    target_col = "discomfort_level"

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # Get baseline
    baseline_value = y_val.mean()
    y_pred_baseline = np.full(shape=len(y_val), fill_value=baseline_value)

    # Baseline evaluation metrics
    mae_baseline = mean_absolute_error(y_val, y_pred_baseline)
    mse_baseline = mean_squared_error(y_val, y_pred_baseline)
    rmse_baseline = np.sqrt(mse_baseline)

    # Predict
    y_pred = model.predict(X_val)

    # Get evaluation metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    print(
        f'''
        Baseline model (Mean)
        MAE baseline: {mae_baseline:.4f},
        MSE baseline: {mse_baseline:.4f},
        RMSE baseline: {rmse_baseline:.4f},

        Model ({model_type})
        MAE: {mae:.4f},
        MSE: {mse:.4f},
        RMSE: {rmse:.4f},
        '''
    )
    return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train_linreg_model":
        train_linreg_model()

    elif len(sys.argv) > 1 and sys.argv[1] == "train_gradient_boosting_model":
        train_gradient_boosting_model()

    elif len(sys.argv) > 1 and sys.argv[1] == "model_predict":
        noise_db = float(sys.argv[2])
        light_lux = float(sys.argv[3])
        crowd_count = int(sys.argv[4])
        model_prefix = str(sys.argv[5])

        model_predict(noise_db, light_lux, crowd_count, model_prefix)

    elif len(sys.argv) > 1 and sys.argv[1] == "model_evaluate":
        model_prefix = sys.argv[2]
        model_evaluate(model_prefix)

    else:
        print("No valid command given.")

# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from google.cloud import storage
import pickle

import os
from datetime import datetime, UTC
import json

from package_aura.gcs_functions import upload_data_to_gcs, load_data_from_gcs


def train_linreg_model():
    '''
    1. Fetch data from GCP buckets
    2. Define X and y for training and validation set
    3. Build pipeline
    4. Create model
    5. Grid search for optimal parameters
    6. Extract best model
    7. Save pipeline, and meta-data
    8. Push pipeline, and meta-data to the cloud using util function
    '''

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

    # 3. Build pipeline
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

    # 4. Create model
    ## LinReg model
    lin_reg = ElasticNet()

    ## Full pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", lin_reg)
    ])

    # 5. Grid search for optimal parameters
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

    # 6. Extract best model
    best_pipe = grid_search.best_estimator_

    # 7. Define pipeline, label encoder, and meta-data to upload afterwards
    ## Model
    linreg_model = best_pipe

    ## Meta Data
    linreg_metadata = {
        "linreg_training_timestamp": datetime.now(UTC).isoformat(),
        "model_type": "linear_regression",
        "feature_names": feature_cols,
        "hyperparameters": grid_search.best_params_
    }

    # 8. Push pipeline, label encoder, and meta-data to the cloud
    model_bytes = pickle.dumps(linreg_model)
    metadata_json = json.dumps(linreg_metadata, indent=4)

    bucket_name = os.environ["MODEL_BUCKET"]

    ## Create versioned folder in bucket
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    gcs_folder = f"lin_reg_models/{timestamp}"

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


def linreg_model_predict(noise_db, light_lux, crowd_count):
    '''
    Load a LinReg model from GCS and make a prediction.

    Args:
        noise_db: Input for noise
        light_lux: Input for brightness
        crowd_count: Input for crowdedness

    Output:
        Returns discomfort_level between 0.0 and 1.0.
    '''
    bucket_name = os.environ["MODEL_BUCKET"]
    model_prefix = "lin_reg_models"
    model_type = "linreg"

    input_df = pd.DataFrame([{
        "noise_db": noise_db,
        "light_lux": light_lux,
        "crowd_count": crowd_count
    }])

    pipeline, metadata = load_data_from_gcs(bucket_name, model_prefix, model_type)

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
            lin_reg_models for linear regression model
    '''
    # Map file-name to folder name
    if model_prefix == "gb_models":
        model_type = "gb"
    elif model_prefix == "lin_reg_models":
        model_type = "linreg"
    else:
        print("Model not found")
        return None

    # Get model
    bucket_name = os.environ["MODEL_BUCKET"]
    model_prefix = model_prefix

    model, metadata = load_data_from_gcs(bucket_name, model_prefix, model_type)

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

    elif len(sys.argv) > 1 and sys.argv[1] == "linreg_model_predict":
        noise_db = float(sys.argv[2])
        light_lux = float(sys.argv[3])
        crowd_count = int(sys.argv[4])

        linreg_model_predict(noise_db, light_lux, crowd_count)

    elif len(sys.argv) > 1 and sys.argv[1] == "model_evaluate":
        model_prefix = sys.argv[2]
        model_evaluate(model_prefix)

    else:
        print("No valid command given.")

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from google.cloud import storage
import pickle

import os
from dotenv import load_dotenv
from datetime import datetime, UTC
import json

from util_functions.gcs_functions import upload_data_to_gcs

def train_logreg_model():
    '''
    1. Fetch data from GCP buckets
    2. Define X and y for training and validation set
    3. Label encode y's
    4. Build pipeline
    5. Create model
    6. Grid search for optimal parameters
    7. Extract best model
    8. Save pipeline, label encoder, and meta-data
    9. Push pipeline, label encoder, and meta-data to the cloud using util function
    '''

    # 1. Fetch data
    df_train = pd.read_csv("gs://aura_datasets_training_validation/AURA_aug_sep_60k.csv")
    df_val = pd.read_csv("gs://aura_datasets_training_validation/AURA_validation_sep_12k.csv")

    # 2. Define X and y for training and validation set
    feature_cols = ["noise_db", "light_lux", "crowd_count"]
    target_col = "comfort_label"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # 3. Label encode y's
    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    y_val_encoded = le.transform(y_val)

    # 4. Build pipeline
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

    # 5. Create model
    ## LogReg model
    log_reg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    )

    ## Full pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", log_reg)
    ])

    # 6. Grid search for optimal parameters
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train_encoded)

    print("Best params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    # 7. Extract best model
    best_pipe = grid_search.best_estimator_

    # 8. Define pipeline, label encoder, and meta-data to upload afterwards
    ## Model
    logreg_model = best_pipe

    ## Label Encoder
    label_encoder_logreg = le

    ## Meta Data
    logreg_metadata = {
        "logreg_training_timestamp": datetime.now(UTC).isoformat(),
        "model_type": "logistic_regression_multiclass",
        "feature_names": feature_cols,
        "target_classes": le.classes_.tolist(),
        "hyperparameters": grid_search.best_params_
    }

    # 9. Push pipeline, label encoder, and meta-data to the cloud
    model_bytes = pickle.dumps(logreg_model)
    encoder_bytes = pickle.dumps(label_encoder_logreg)
    metadata_json = json.dumps(logreg_metadata, indent=4)

    bucket_name = os.environ["MODEL_BUCKET"]

    ## Create versioned folder in bucket
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    gcs_folder = f"log_reg_models/{timestamp}"

    ## Upload files
    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/logreg_pipeline.pkl",
        data=model_bytes
    )

    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/logreg_label_encoder.pkl",
        data=encoder_bytes
    )

    upload_data_to_gcs(
        bucket_name=bucket_name,
        gcs_path=f"{gcs_folder}/logreg_metadata.json",
        data=metadata_json,
        content_type="application/json"
    )

    print(f"All data uploaded to: gs://{bucket_name}/{gcs_folder}/")

    return None





if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train_logreg_model":
        train_logreg_model()
    else:
        print("No valid command given. Try: train_logreg_model")

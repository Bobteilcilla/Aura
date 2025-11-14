from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


'''
Ready to be called from the frontend API to show the
discomfort level and comfort label based on user inputs.
The input parameters are:
- noise_db: float
- light_lux: float
- crowd_count: float
and the function returns:
- discomfort_level: float
- comfort_label: str

How to call in Streamlit API:

from pathlib import Path
import joblib
from package_aura.gradient_boosting import prediction_from_inputs

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"

model = joblib.load(MODEL_PATH)
noise = float(input_noise)
light = float(input_light)
crowd = int(input_crowd)

discomfort, label = prediction_from_inputs(noise, light, crowd)

st.metric("Discomfort Level", f"{discomfort:.3f}")
st.success(f"Comfort Label: {label}")
'''


# -----------------------------
# 1. Paths (relative to project)
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw-data"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)  # create folder if it doesn't exist
MODEL_PATH = MODEL_DIR / "gradient_boosting_model.pkl"

'''Model will be saved under Aura/models/gradient_boosting_model.pkl'''

train_path = RAW_DATA_DIR / "AURA_aug_sep_60k.csv"
val_path   = RAW_DATA_DIR / "AURA_validation_sep_12k.csv"

print(f"Train CSV:      {train_path}")
print(f"Validation CSV: {val_path}")


# -----------------------------
# 2. Load data
# -----------------------------

''' The training dataset and the validation dataset are two different CSV files
and have the following columns as features:
- noise_db: float
- light_lux: float
- crowd_count: float and the target column will be the discomfort_level: float
(0 to 1)'''

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)

feature_columns = ["noise_db", "light_lux", "crowd_count"]
target_columns   = "discomfort_level"

X_train = train_df[feature_columns]
y_train = train_df[target_columns]

X_val = val_df[feature_columns]
y_val = val_df[target_columns]


# ---------------------------------------
# 3. Model Training
# ---------------------------------------

''' Using sklearn's GradientBoostingRegressor with some example hyperparameters.
We can tune these as needed.The n_estimators is the number of boosting stages
to be run, the learning_rate shrinks the contribution of each tree by
learning_rate, max_depth is the maximum depth of the individual regression
estimators. The random_state is set for reproducibility.'''

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    max_features=None,
    random_state=42
)


model.fit(X_train, y_train)

# ------------------------------------------------
# 5. Saving trained model as a pickle file to disk
# ------------------------------------------------

joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# ------------------------------
# 5. Evaluate on validation set
# -------------------------------

''' Evaluate the model on the validation set using mean squared error (MSE)
and R² score.'''

y_val_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)
r2  = r2_score(y_val, y_val_pred)

print(f"\nValidation MSE: {mse:.4f}")
print(f"Validation R²:  {r2:.4f}")


# ---------------------------------------
# 6. Map discomfort_level → comfort_label
#    (you can tweak these thresholds)
# ---------------------------------------

def discomfort_to_label(d):
    """
    Mapping from the discomfort_level (0 to 1) to comfort_label.

      - <= 0.2        -> very_comfortable
      - 0.2–0.4       -> comfortable
      - 0.4–0.6       -> neutral
      - 0.6–0.8       -> uncomfortable
      - > 0.8         -> stressed
    """
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

# Add predicted discomfort & labels on validation set

''' Show some sample predictions on the validation set for testing purposes.'''

val_df["discomfort_pred"] = y_val_pred
val_df["comfort_pred"] = val_df["discomfort_pred"].apply(discomfort_to_label)

print("\nSample of predictions:")
print(val_df[["noise_db", "light_lux", "crowd_count",
              "discomfort_level", "discomfort_pred",
              "comfort_label", "comfort_pred"]].head())

# -------------------------------------------
# 7. Function for prediction from raw inputs
# -------------------------------------------

def prediction_from_inputs(noise_db, light_lux, crowd_count):
    """
    Given raw input values, predict discomfort_level and comfort_label.
    """
    input_df = pd.DataFrame({
        "noise_db": [noise_db],
        "light_lux": [light_lux],
        "crowd_count": [crowd_count],
    })

    discomfort_pred = model.predict(input_df)[0]
    comfort_label = discomfort_to_label(discomfort_pred)
    return discomfort_pred, comfort_label

# ---------------------------------------
# 8. Example: prediction from raw inputs
# ---------------------------------------

example = {
    "noise_db": 90.6,
    "light_lux": 900.0,
    "crowd_count": 19.0,
}

example_discomfort, example_label = prediction_from_inputs(
    example["noise_db"],
    example["light_lux"],
    example["crowd_count"],
)

print("\nExample input:", example)
print(f"Predicted discomfort_level: {example_discomfort:.3f}")
print(f"Predicted comfort label:    {example_label}")

'''Notes:
I have tried training the model with different hyperparameters and
the current settings (n_estimators=300, learning_rate=0.03, max_depth=6)
seem to give a reasonable balance between accuracy and overfitting.
You can further tune these hyperparameters based on validation performance'''

#----------------------------
# End of gradient_boosting.py
#----------------------------

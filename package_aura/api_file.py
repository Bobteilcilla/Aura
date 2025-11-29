from fastapi import FastAPI
import pickle

from package_aura.model_functions import model_predict

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"AURA is up and running!"}

# Prediction endpoint
@app.get("/predict")
def predict(noise_db, light_lux, crowd_count, model_prefix: str = "gb_models"):
    if model_prefix not in ("linreg_models", "gb_models"):
        print("Model not found:", model_prefix)
        raise ValueError("Model not found:", model_prefix)
        return None
    return model_predict(noise_db, light_lux, crowd_count, model_prefix)

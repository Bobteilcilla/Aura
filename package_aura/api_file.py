from fastapi import FastAPI
import pickle

from package_aura.hello_aura import hello_aura
from package_aura.linreg_model import linreg_model_predict
from package_aura.gradient_boosting import gradient_boosting_predict

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"AURA is up and running!"}

# Test endpoint
@app.get("/hello")
def hello():

    # Return greeting from hello_aura function
    return {"greeting": hello_aura()}

# Prediction endpoint
@app.get("/predict")
def predict(noise_db, light_lux, crowd_count, model_type: str = "linreg"):
    if model_type == "linreg":
        print("Received model_type:", model_type)
        return linreg_model_predict(noise_db, light_lux, crowd_count)
    elif model_type == "gb":
        print("Received model_type:", model_type)
        return gradient_boosting_predict(noise_db, light_lux, crowd_count)
    else:
        return {"error": "Invalid model_type. Choose 'linreg' or 'gb'."}

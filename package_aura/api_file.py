from fastapi import FastAPI
import pickle

from package_aura.hello_aura import hello_aura
from package_aura.logreg_model import logreg_model_predict

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
def predict(noise_db, light_lux, crowd_count):
    return logreg_model_predict(noise_db, light_lux, crowd_count)

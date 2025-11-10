from fastapi import FastAPI
import pickle

from package_aura.hello_aura import hello_aura

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"AURA is up and running!"}

# Prediction endpoint
@app.get("/hello")
def hello():

    # Return greeting from hello_aura function
    return {"greeting": hello_aura()}

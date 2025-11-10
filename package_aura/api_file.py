from fastapi import FastAPI
import pickle

from package_aura.hello_aura import hello_aura

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return hello_aura()

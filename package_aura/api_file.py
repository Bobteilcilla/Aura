from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from pathlib import Path
import base64
import io

from PIL import Image
from ultralytics import YOLO

from package_aura.hello_aura import hello_aura
from package_aura.linreg_model import linreg_model_predict
from package_aura.gradient_boosting import gradient_boosting_predict


# ---------------- #
# FASTAPI INSTANCE #
# ---------------- #

app = FastAPI()

# ---- CORS (needed for POST /yolo_crowd from Streamlit in browser) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # for dev; later you can restrict to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],         # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)


# ---------------------- #
# ROOT / TEST ENDPOINTS  #
# ---------------------- #

@app.get("/")
def root():
    return {"greeting": "AURA is up and running!"}


@app.get("/hello")
def hello():
    """Simple test endpoint using your existing hello_aura() function."""
    return {"greeting": hello_aura()}


# -------------------------- #
# PREDICTION ENDPOINT (/predict)
# -------------------------- #

@app.get("/predict")
def predict(noise_db: float, light_lux: float, crowd_count: float):
    """
    Main AURA prediction endpoint.

    Expects three floats:
      - noise_db   (e.g. 30–90)
      - light_lux  (e.g. 78–1200)
      - crowd_count (float from YOLO or your mapping)
    """
    return linreg_model_predict(float(noise_db), float(light_lux), float(crowd_count))


# ------------------------------ #
# YOLO CROWD ENDPOINT (/yolo_crowd)
# ------------------------------ #

# Path to your YOLO model in /app/models (because Docker does: COPY models models)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
# change filename if different

yolo_model = YOLO(str(MODEL_PATH))
print(f"✔️ YOLO model loaded from: {MODEL_PATH}")


class ImagePayload(BaseModel):
    image_base64: str  # "data:image/jpeg;base64,...."


@app.post("/yolo_crowd")
def yolo_crowd(payload: ImagePayload) -> Dict[str, object]:
    """
    Crowd detection endpoint using YOLO.
    Returns:
      - crowd_count (float)
      - image_base64: image with bounding boxes drawn
    """

    # --- Decode the incoming base64 image ---
    if "," in payload.image_base64:
        _, b64data = payload.image_base64.split(",", 1)
    else:
        b64data = payload.image_base64

    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # --- Run YOLO ---
    results = yolo_model(img, verbose=False)[0]

    # --- Count people (class 0 = person) ---
    people = [box for box in results.boxes if int(box.cls[0]) == 0]
    crowd_count = float(len(people))

    # --- Draw YOLO bounding boxes ---
    plotted_img = results.plot()  # numpy array with rendered boxes

    # Convert back to JPEG bytes
    pil_img = Image.fromarray(plotted_img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    # --- Return JSON with both values ---
    return {
        "crowd_count": crowd_count,
        "image_base64": "data:image/jpeg;base64," + img_b64
    }

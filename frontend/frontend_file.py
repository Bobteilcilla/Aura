# ----------------
# IMPORTS
import streamlit as st
import requests
from PIL import Image
from pathlib import Path
import base64



# ----------------
# BACKGROUND IMAGE
def set_background(img_path: str) -> None:
    """Set a full-page background image using inline CSS.

    Accepts a path to a local image. Safely no-ops if the file doesn't exist.
    """
    p = Path(img_path)
    if not p.exists():
        st.warning(f"Background image not found: {p.resolve()}")
        return

    ext = p.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".gif":
        mime = "image/gif"
    else:
        mime = "image/*"

    b64 = base64.b64encode(p.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        /* App background */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:{mime};base64,{b64}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }}
        /* Transparent header to let bg show through */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        /* Make main content transparent (optional) */
        .block-container {{
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply background image (file with spaces is handled safely)
set_background(str(Path(__file__).parent / "AURA_background_graphic.png"))



# ----------------
# API
API_URL = "https://aura-app-560310706773.europe-west1.run.app"



# ----------------
# TITLE

st.markdown(
    """
    <style>
    .custom-title {
        font-family: 'Arial', sans-serif !important;
        font-size: 10rem !important;
        font-weight: bold !important;
        text-align: center !important;
        color: #000000 !important;
        margin-top: -8rem !important;
        margin-bottom: -4rem !important;
    }
    </style>
    <h1 class="custom-title">AURA</h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='
        text-align: center;
        font-size: 1.93rem;
        margin-top: 0rem;
        margin-left: -1.6rem;
    '>
        ENVIRONMENT QUALITY CLASSIFIER
    </p>
    """,
    unsafe_allow_html=True
)




# ----------------
# INPUTS

# Title
st.write("Use the selectors below to choose the ambient sound, light intensity and crowd density. These are mapped to numeric features behind the scenes.")

# Environment Quality Prediction Section
sound = st.selectbox("Select Ambient Sound Level", options=["Low", "Medium", "High"])
light = st.selectbox("Select Light Intensity", options=["Low", "Medium", "High"])
crowd = st.selectbox("Select Crowd Density", options=["Low", "Medium", "High"])
feature_map = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

# Numeric mapping for the three categorical inputs
s_value = feature_map.get(sound, 0.0)
l_value = feature_map.get(light, 0.0)
c_value = feature_map.get(crowd, 0.0)

# Local Prediction Logic
def predict_environment_quality(s_value, l_value, c_value):
    """
    Accept numeric feature values and return a simple quality label.

    Current rule: sum of features > 1.5 -> good, else bad. TO BE REPLACED WITH MODEL!
    """
    total = s_value + l_value + c_value
    return "good" if total > 1.5 else "bad"



# ----------------
# STYLING

# Background styling
def get_styles(prediction):
    if prediction == "good":
        return {
            "bg_color": "#b6fcb6",
            "text_color": "#004400",
            "message": "✅ Environment is GOOD"
        }
    return {
        "bg_color": "#ffb6b6",
        "text_color": "#660000",
        "message": "⚠️ Environment is BAD"
    }
# Text styling
def apply_styles(styles):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {styles['bg_color']};
            color: {styles['text_color']};
            transition: background-color 1.0s ease;
        }}
        .custom-feedback {{
            font-weight: bold;
            font-size: 1.5rem;
            margin-top: 20px;
            padding: 0.5em 1em;
            border-radius: 0.5em;
            background-color: rgba(255, 255, 255, 0.5);
            display: inline-block;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='custom-feedback'>{styles['message']}</div>", unsafe_allow_html=True)


# Initialize or re-apply styles (default: white background)
if "styles" not in st.session_state:
    st.session_state["styles"] = {
        "bg_color": "#FFFFFF",
        "text_color": "#000000",
        "message": "",
    }
apply_styles(st.session_state["styles"])

# If inputs change after a prior prediction, fade back to white
current_inputs = {"sound": sound, "light": light, "crowd": crowd}
prev_inputs = st.session_state.get("prev_inputs")
if prev_inputs is None:
    st.session_state["prev_inputs"] = current_inputs
else:
    if current_inputs != prev_inputs:
        st.session_state["prev_inputs"] = current_inputs
        if st.session_state.get("styles"):
            st.session_state["styles"] = {
                "bg_color": "#FFFFFF",
                "text_color": "#000000",
                "message": "",
            }
            apply_styles(st.session_state["styles"])  # transition handles fade-out



# --- PREDICTIONS ---

# Local prediction
def main():

    st.write(f"Mapped values — sound: {s_value}, light: {l_value}, crowd: {c_value}")

    if st.button("Predict"):
        prediction = predict_environment_quality(s_value, l_value, c_value)
        styles = get_styles(prediction)
        st.session_state["styles"] = styles  # AI persist across reruns
        apply_styles(styles)

main()


# --- API ---

# Greetings response
st.title("API Test Response")

url = f"{API_URL}/hello"

response = requests.get(url).json()

if "greeting" in response:
        st.write(response["greeting"])
else:
        st.error("Error: 'greeting' key not found in the response.")

# API Prediction response
if st.button("API Predict"):
    params = {
        "sound": s_value,
        "light": l_value,
        "crowd": c_value
    }

    API_url = f"{API_URL}/predict"

    api_response = requests.get(API_url, params=params).json()

    if api_response.status_code == 200:
        api_prediction = api_response.get("prediction")
        st.success(f"API Prediction: {api_prediction}")
    else:
        st.error("Error in prediction. Please try again.")

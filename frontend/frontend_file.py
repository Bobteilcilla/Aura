# ------- #
# IMPORTS #
# ------- #

import streamlit as st
import requests
from PIL import Image
from pathlib import Path
import sys
import base64

#Add the root of the project (where package_aura is ) to the  sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent  # /mount/src/aura
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from package_aura.multiple_mapping import discomfort_to_label



# ---------- #
# PAGE WIDTH #
# ---------- #

st.set_page_config(layout="centered") 



# ---------------- #
# BACKGROUND IMAGE #
# ---------------- #

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



# --- #
# API #
# --- #

API_URL = "https://aura-app-560310706773.europe-west1.run.app"



# ----- #
# TITLE #
# ----- #

# Main Title
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

# Subtitle
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



# ----------------- #
# SELECTION SECTION #
# ----------------- #

# Title
st.write("Use the selectors below to choose the ambient sound, light intensity and crowd density. Press 'Predict' to classify the environment quality.")

# Environment Quality Prediction Selectors
sound = st.selectbox("How noisy is it?", options=["0–10 dB — Threshold of hearing, rustling leaves",
                                                            "10–20 dB — Very quiet room, whisper",
                                                            "20–40 dB — Soft whisper, library, quiet office",
                                                            "40–60 dB — Normal conversation, rainfall",
                                                            "60–80 dB — Busy street, vacuum cleaner",
                                                            "80–100 dB — Lawn mower, motorcycle, nightclub",
                                                            "100–140+ dB — Chainsaw, jet plane, fireworks, gunshot"])

light = st.selectbox("How bright is it?", options=["0–1 lux — Starlight, moonless night",
                                                        "1–10 lux — Full moon",
                                                        "10–100 lux — Twilight, dim indoor lighting",
                                                        "100–500 lux — Normal indoor lighting, office",
                                                        "500–5,000 lux — Bright office, overcast outdoor daylight",
                                                        "5,000–20,000 lux — Outdoor daylight (not direct sun)",
                                                        "20,000–100,000 lux — Direct sunlight (morning to midday)"])

crowd = st.selectbox("How many people are there?", options=["0–0.5 people/m² — Empty / Barely occupied",
                                                      "0.5–1.5 people/m² — Light crowd",
                                                      "1.5–3 people/m² — Moderate crowd",
                                                      "3–4.5 people/m² — Dense crowd",
                                                      "4.5–6 people/m² — Very dense crowd",
                                                      "6+ people/m² — Extreme density / Packed crowd"])

# Numeric mapping for the three categorical inputs
sound_feature_map = {
    "0–10 dB — Threshold of hearing, rustling leaves": 10,
    "10–20 dB — Very quiet room, whisper": 20,
    "20–40 dB — Soft whisper, library, quiet office": 40,
    "40–60 dB — Normal conversation, rainfall": 60,
    "60–80 dB — Busy street, vacuum cleaner": 80,
    "80–100 dB — Lawn mower, motorcycle, nightclub": 100,
    "100–140+ dB — Chainsaw, jet plane, fireworks, gunshot": 140
}

light_feature_map = {
    "0–1 lux — Starlight, moonless night": 1,
    "1–10 lux — Full moon": 10,
    "10–100 lux — Twilight, dim indoor lighting": 100,
    "100–500 lux — Normal indoor lighting, office": 500,
    "500–5,000 lux — Bright office, overcast outdoor daylight": 5000,
    "5,000–20,000 lux — Outdoor daylight (not direct sun)": 20000,
    "20,000–100,000 lux — Direct sunlight (morning to midday)": 100000
}

crowd_feature_map = {"0–0.5 people/m² — Empty / Barely occupied": 3.3,
                     "0.5–1.5 people/m² — Light crowd": 6.6,
                     "1.5–3 people/m² — Moderate crowd": 9.9,
                     "3–4.5 people/m² — Dense crowd": 13.2,
                     "4.5–6 people/m² — Very dense crowd": 16.5,
                     "6+ people/m² — Extreme density / Packed crowd": 20.0
}

# Numeric mapping for the three categorical inputs
s_value = sound_feature_map.get(sound, 0.0)
l_value = light_feature_map.get(light, 0.0)
c_value = crowd_feature_map.get(crowd, 0.0)



# ------------------ #
# PREDICTION STYLING #
# ------------------ #

# Background styling
def label_to_styles(label: str, score: float):
    label = label.lower()

    color_map = {
        "very_comfortable": ("#b6fcb6", "Environment is Very Comfortable"),
        "comfortable": ("#d8fcd1", "Environment is Comfortable"),
        "neutral": ("#fffab6", "Environment is Neutral"),
        "uncomfortable": ("#fdd9d9", "Environment is Uncomfortable"),
        "stressed": ("#ffb6b6", "Environment is Stressed"),
    }

    bg_color, message = color_map.get(label, ("#F2F2F2", "Unknown environment status"))
    score_line = f"score: {score:.2f}"
    
    return {
        "bg_color": bg_color,
        "text_color": "#000000",
        "message": message,
        "score": score_line
    }

# Text styling
def apply_button_styles():
    st.markdown("""
        <style>
        .stButton > button {
            font-weight: bold;
            font-size: 1.5rem;
            font-weight: bold;
            margin-left: 55px;
            padding: 0.6em 1.5em;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

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
            text-align: center;
            width: 100%;
        }}
        .score-line {{
            font-weight: bold;
            font-size: 1.5rem;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='custom-feedback'>{styles['message']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-line'>{styles['score']}</div>", unsafe_allow_html=True)

# Initialize or re-apply styles (default: grey background)
if "styles" not in st.session_state:
    st.session_state["styles"] = {
        "bg_color": "#F2F2F2",
        "text_color": "#000000",
        "message": "",
        "score": "",
    }
apply_styles(st.session_state["styles"])

# If inputs change after a prior prediction, fade back to grey
current_inputs = {"sound": sound, "light": light, "crowd": crowd}
prev_inputs = st.session_state.get("prev_inputs")
if prev_inputs is None:
    st.session_state["prev_inputs"] = current_inputs
else:
    if current_inputs != prev_inputs:
        st.session_state["prev_inputs"] = current_inputs
        if st.session_state.get("styles"):
            st.session_state["styles"] = {
                "bg_color": "#F2F2F2",
                "text_color": "#000000",
                "message": "",
                "score": "",
            }
            apply_styles(st.session_state["styles"])  # transition handles fade-out



# ----------- #
# PREDICTIONS #
# ----------- #

# API prediction

# Apply button styles
apply_button_styles()

# Set columns
col1, col2, col3 = st.columns([1, 3, 1])

# Prediction button
with col2:
    if st.button("TEST THE ENVIRONMENT"):
        params = {
            "noise_db": s_value,
            "light_lux": l_value,
            "crowd_count": c_value
        }

        URL = f"{API_URL}/predict"
        response = requests.get(URL, params=params)

        if response.status_code == 200:
            result = response.json()
            score = result.get("discomfort_score", None)
            
            if score is not None:
                label = discomfort_to_label(score)
                styles = label_to_styles(label, score)
                apply_styles(styles)
            else:
                st.warning("No discomfort score returned in the response.")
            
        else:
            st.error("Error in prediction. Please try again.")



# # --------------------- #
# # TESTING THE API CALLS #
# # --------------------- #

# # Greetings response
# st.title("API Test Response")

# url = f"{API_URL}/hello"

# response = requests.get(url).json()

# if "greeting" in response:
#         st.write(response["greeting"])
# else:
#         st.error("Error: 'greeting' key not found in the response.")

# st.space(size="medium")

# with st.expander("How does Aura work?"):
#     st.write('Explanation')

# st.space(size="medium")

# # Container for Inputs
# container = st.container()

# with container:

#     # Columns inside container to display inputs in one row
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         number_noise = st.number_input('How noisy is it?')

#     with col2:
#         number_light = st.number_input('How bright is it?')

#     with col3:
#         number_crowd = st.number_input('How many people are there?')

#     st.space(size="small")

#     # Colums to center button
#     button_col1, button_col2, button_col3 = st.columns([2, 1, 2])

#     # Place button in the center (i.e. second) column
#     with button_col2:
#         if st.button("Get prediction"):
#             st.write(f'''Noise level is {round(number_noise, 2)},
#                      brightness level is {round(number_light, 2)},
#                      crowdiness level is {round(number_crowd, 2)}''')

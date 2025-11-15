"""
Frontend script from Riccardo with simple authentication.
Next step, user and psw.
Next step, creating user profile track.

"""

import streamlit as st
import requests
from PIL import Image


# --- SIMPLE USER AUTHENTICATION (Name Only With Default) ---

def authenticate_user():
    """
    Dummy authentication: user must enter a name, otherwise 'Unknown user' is used.
    Username is stored in Streamlit session_state.
    """

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Show login only if user is not authenticated
    if not st.session_state.authenticated:
        st.write("### Please enter your name to continue:")
        username = st.text_input("Name:", key="username_input")

        if st.button("Enter"):
            # Assign default or user-provided name
            st.session_state.username = username.strip() if username.strip() else "Unknown user"

            st.session_state.authenticated = True
            st.success(f"Welcome, {st.session_state.username}!")

            # Reload UI after login
            st.experimental_rerun()

        # Stop the app before loading the other components
        st.stop()


# LOGO (shown even before authentication)
logo = Image.open("frontend/AURA Logo_Test.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=310)

# 🔒 Require authentication AFTER logo
authenticate_user()


# Change this URL to the one of your API
API_URL = "https://aura-app-560310706773.europe-west1.run.app"


# TITLE
st.title("AURA Project")
st.write(f"Environment Quality Classifier — Logged in as **{st.session_state.username}**")


# --- INPUTS ---

# Title
st.write("## Inputs")
st.write("Use the selectors below to choose the ambient sound, light intensity and crowd density. "
         "These are mapped to numeric features behind the scenes.")

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
def predict_environment_quality(s, l, c):
    """Simple placeholder rule: sum > 1.5 → good, else bad."""
    total = s + l + c
    return "good" if total > 1.5 else "bad"


# Background styling
def get_styles(prediction):
    if prediction == "good":
        return {"bg_color": "#b6fcb6", "text_color": "#004400", "message": "✅ Environment is GOOD"}
    return {"bg_color": "#ffb6b6", "text_color": "#660000", "message": "⚠️ Environment is BAD"}


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


# --- PREDICTIONS ---

def main():
    st.write(f"Mapped values — sound: {s_value}, light: {l_value}, crowd: {c_value}")

    if st.button("Predict"):
        prediction = predict_environment_quality(s_value, l_value, c_value)
        styles = get_styles(prediction)
        apply_styles(styles)


main()


# --- API ---

st.title("API Response")

# Greetings response
url = f"{API_URL}/hello"
response = requests.get(url).json()

if "greeting" in response:
    st.write(response["greeting"])
else:
    st.error("Error: 'greeting' key not found in the response.")


# API Prediction response
if st.button("API Predict"):
    params = {"sound": s_value, "light": l_value, "crowd": c_value}
    API_predict_url = f"{API_URL}/predict"

    api_response = requests.get(API_predict_url, params=params)

    if api_response.status_code == 200:
        api_prediction = api_response.json().get("prediction")
        st.success(f"API Prediction: {api_prediction}")
    else:
        st.error("Error in prediction. Please try again.")

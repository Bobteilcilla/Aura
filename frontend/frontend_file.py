 # IMPORTS
import streamlit as st
import requests
from PIL import Image



 # LOGO
logo = Image.open("frontend/AURA Logo_Test.png")
 # centre the logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=310)



 # Change this URL to the one of your API
API_URL = "https://aura-app-560310706773.europe-west1.run.app"



 # TITLE
st.title("AURA Project")
st.write("Environment Quality Classifier")


 # API RESPONSE
st.title("API Response")

url = f"{API_URL}/hello"

response = requests.get(url).json()

if "greeting" in response:
        st.write(response["greeting"])
else:
        st.error("Error: 'greeting' key not found in the response.")


 # INPUTS
st.write("## Inputs")
st.write("Use the selectors below to choose the ambient sound, light intensity and crowd density. These are mapped to numeric features behind the scenes.")

# Environment Quality Prediction Section
sound = st.selectbox("Select Ambient Sound Level", ["Low", "Medium", "High"])
light = st.selectbox("Select Light Intensity", ["Low", "Medium", "High"])
crowd = st.selectbox("Select Crowd Density", ["Low", "Medium", "High"])
feature_map = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

# Map selectbox categorical values to numeric features and prediction logic
def predict_environment_quality(s_value, l_value, c_value):
    """Accept numeric feature values and return a simple quality label.

    Current rule: sum of features > 1.5 -> good, else bad. TO BE REPLACED WITH MODEL!
    """
    total = s_value + l_value + c_value
    return "good" if total > 1.5 else "bad"


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

def main():

    # Numeric mapping for the three categorical inputs defined at top-level
    s_num = feature_map.get(sound, 0.0)
    l_num = feature_map.get(light, 0.0)
    c_num = feature_map.get(crowd, 0.0)

    st.write(f"Mapped values — sound: {s_num}, light: {l_num}, crowd: {c_num}")

    if st.button("Predict"):
        prediction = predict_environment_quality(s_num, l_num, c_num)
        styles = get_styles(prediction)
        apply_styles(styles)

main()

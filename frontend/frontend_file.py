import streamlit as st
import requests

 # Change this URL to the one of your API
API_URL = "https://aura-app-560310706773.europe-west1.run.app"

st.title("AURA Project")


url = f"{API_URL}/hello"

response = requests.get(url).json()
if "greeting" in response:
        st.write(response["greeting"])
else:
        st.error("Error: 'greeting' key not found in the response.")


# ---+---+---+---+---+---+---+---+---+---+--- #
# Environment Quality Prediction Section


# Dummy definitions

def predict_environment_quality(f1, f2, f3):
    return "good" if (f1 + f2 + f3) > 1.5 else "bad"

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
    st.title("Environment Quality Classifier")

    f1 = st.slider("Feature 1", 0.0, 1.0)
    f2 = st.slider("Feature 2", 0.0, 1.0)
    f3 = st.slider("Feature 3", 0.0, 1.0)

    if st.button("Predict"):
        prediction = predict_environment_quality(f1, f2, f3)
        styles = get_styles(prediction)
        apply_styles(styles)

main()

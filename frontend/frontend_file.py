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

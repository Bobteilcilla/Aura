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

st.space(size="medium")

with st.expander("How does Aura work?"):
    st.write('Explanation')

st.space(size="medium")

# Container for Inputs
container = st.container()

with container:

    # Columns inside container to display inputs in one row
    col1, col2, col3 = st.columns(3)

    with col1:
        number_noise = st.number_input('How noisy is it?')

    with col2:
        number_light = st.number_input('How bright is it?')

    with col3:
        number_crowd = st.number_input('How many people are there?')

    st.space(size="small")

    # Colums to center button
    button_col1, button_col2, button_col3 = st.columns([2, 1, 2])

    # Place button in the center (i.e. second) column
    with button_col2:
        if st.button("Get prediction"):
            st.write(f'''Noise level is {round(number_noise, 2)},
                     brightness level is {round(number_light, 2)},
                     crowdiness level is {round(number_crowd, 2)}''')

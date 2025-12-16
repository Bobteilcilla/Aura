# AURA - Adaptive Universal Recognition Assistant​

AURA is an application to help people with sensory hypersensitivities to better recognize challenging environments. AURA aims to facilitate self-empowerment in indivisuals with sensory hypersensibilities and to help them navigate their environment.

---

## Overview
Approximately 15% of the population struggle with sensory hypersensitivities, which is a kind of neurodiversity. These hypersensitivities can lead to overstimulation leading to overloads or shutdowns, as well an an overall reduction in quality of life.
AURA provides a way to assess the environment on one quick glance, helping individuals even if they are sensorically challenged.
AURA automatically capture the brightness, noise and crowdedness of a situation and turns those inputs into an understandable lable, using a gradient boosting model.

---

## Features
- Automatic capture of inputs (noise, brightness, crowdedness)
- Easy to understand output

---

## Future Features
- One of AURA greatest limitations is the training data used, which is synthetic data. Since the requirements for the data needed are highly specific, no fitting real world data could be identified. Therefore, the collection of such data is the most important improvement.
- Integration of the use of wearables instead of the webcam and headset to enable usage anywhere, anytime.
- Enabling a connection to smart home devices, making it possible for your home (e.g. the lighting) to automatically change according to your mood.
- Integrate a personal sensibility slider, i.e. letting the user decide on the weights of the different inputs according to their specific needs.
- Including stimming recommendations and tipps for users in challenging environments.

---

## Folder Description
frontend:
- Includes frontend files for the steamlit application

notebooks:
- Includes notebooks that were used to generate/evaluate data and to evaluate different models.

package_aura:
- Includes all package files, including:
  1. api_file.py: Definition of API-Endpoints
  2. helper_functions.py: Definition of all functions used to handle data up- and download from GCS
  3. model_functions.py: Definition of all functions used to train train, upload, and evaluate models as well as using those models to make predictions

---

## Link to Streamlit-Application:

https://xxmffajuo9yrdpcjuwfrr9.streamlit.app/

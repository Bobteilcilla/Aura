from package_aura.linreg_model import linreg_model_predict

# Example test values
noise_db = 10.1
light_lux = 20.2
crowd_count = 10.1

result = linreg_model_predict(noise_db, light_lux, crowd_count)
print("Prediction result:", result)
from package_aura.linreg_model import linreg_model_predict

# Example test values
noise = 10.1
light = 20.2
crowd = 10.1

result = linreg_model_predict(noise, light, crowd)
print("Prediction result:", result)
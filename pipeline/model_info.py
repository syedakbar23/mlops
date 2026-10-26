import os
import joblib
from datetime import datetime

# Project directory: MLOPS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model file path
model_path = os.path.join(BASE_DIR, "deployment", "deployment_model.pkl")

# Deployment information file path
info_path = os.path.join(BASE_DIR, "deployment", "deployment_info.txt")

# Load the existing model
model = joblib.load(model_path)

print("Deployment Model Loaded Successfully")

# Write deployment information
with open(info_path, "w", encoding="utf-8") as file:
    file.write("MODEL DEPLOYMENT INFORMATION\n\n")
    file.write("Model Name: Tuned Random Forest\n")
    file.write("Algorithm: Random Forest Regressor\n")
    file.write("Version: 1.0\n")
    file.write("Dataset: Student Performance Dataset\n")
    file.write("Target: G3\n")
    file.write("R² Score: Not provided\n")
    file.write(f"Created On: {datetime.now()}\n")

print("Deployment Information File Created Successfully")
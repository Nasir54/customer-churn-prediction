# test_api.py (updated with all required features)
import requests
import json
import os
# Complete sample data with ALL categorical features
sample_customer = {
    # Numerical features
    "tenure": 12,
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.2,
    
    # Categorical features (all that your model expects)
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

# Make prediction request
try:
    response = requests.post(
        'http://localhost:5001/predict',
        json=sample_customer
    )

    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Response:", json.dumps(response.json(), indent=2))
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response content:", response.text)
        
except Exception as e:
    print(f"Error: {e}")
    print("Make sure your Flask app is running on port 5001")
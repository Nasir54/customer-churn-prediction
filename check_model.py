import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'notebooks', 'churn_prediction_model.pkl')
print(f"Looking for model at: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

# Also check from Docker perspective
docker_model_path = "/app/notebooks/churn_prediction_model.pkl"
print(f"Docker path: {docker_model_path}")
print(f"Docker path exists: {os.path.exists(docker_model_path)}")

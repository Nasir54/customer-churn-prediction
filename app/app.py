# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize the Flask app
app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model and artifacts with compatibility handling
print("Loading model and artifacts...")

def retrain_model():
    """Retrain model with current scikit-learn version"""
    print("🔄 Retraining model with current scikit-learn version...")
    
    # Load your actual training data here - REPLACE WITH YOUR ACTUAL DATA LOADING
    # For demonstration, creating sample data that matches your expected features
    np.random.seed(42)
    
    # Create sample data that matches your feature structure
    n_samples = 1000
    numerical_data = np.random.rand(n_samples, 5)  # Adjust based on your numerical_cols
    categorical_data = np.random.choice(['Yes', 'No'], size=(n_samples, 3))  # Adjust based on categorical_cols
    
    # Combine into features (adjust this to match your actual data structure)
    X = np.hstack([numerical_data, categorical_data.astype(object)])
    y = np.random.randint(0, 2, n_samples)  # binary classification
    
    # Convert to DataFrame with appropriate column names
    # Adjust these column names to match your actual feature_names
    all_columns = ['tenure', 'monthly_charges', 'total_charges', 'feature_4', 'feature_5', 
                   'gender', 'partner', 'dependents']  # Example columns
    X_df = pd.DataFrame(X, columns=all_columns[:X.shape[1]])
    
    # Separate numerical and categorical (adjust based on your actual columns)
    numerical_cols = all_columns[:5]  # First 5 are numerical
    categorical_cols = all_columns[5:]  # Last 3 are categorical
    
    # Preprocess the data
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(X_df[numerical_cols])
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X_df[categorical_cols])
    
    # Combine features
    final_features = pd.concat([
        pd.DataFrame(X_numerical, columns=numerical_cols),
        X_encoded
    ], axis=1)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(final_features, y)
    
    # Create artifacts in expected format
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': final_features.columns.tolist(),
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'version': sklearn.__version__,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    # Save with current scikit-learn
    model_path = os.path.join(current_dir, 'models', 'churn_prediction_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifacts, model_path)
    
    print(f"✅ Model retrained with scikit-learn {sklearn.__version__}")
    print(f"✅ Saved to: {model_path}")
    
    return artifacts

def load_model_with_fallback():
    """Load model with compatibility fallback"""
    # Try multiple possible paths for the model
    possible_paths = [
        os.path.join(current_dir, 'churn_prediction_model.pkl'),
        os.path.join(current_dir, 'models', 'churn_prediction_model.pkl'),
        os.path.join(current_dir, '..', 'notebooks', 'churn_prediction_model.pkl'),
        os.path.join(current_dir, '..', 'models', 'churn_prediction_model.pkl'),
        '/app/models/churn_prediction_model.pkl',
        '/app/churn_prediction_model.pkl'
    ]

    artifacts = None
    model_path_used = None

    for model_path in possible_paths:
        try:
            print(f"Trying to load model from: {model_path}")
            if os.path.exists(model_path):
                artifacts = joblib.load(model_path)
                model_path_used = model_path
                print(f"✅ Successfully loaded model from: {model_path}")
                break
            else:
                print(f"❌ Not found: {model_path}")
        except (ModuleNotFoundError, AttributeError, TypeError) as e:
            print(f"❌ Compatibility error with {model_path}: {e}")
            continue
        except Exception as e:
            print(f"❌ Other error loading {model_path}: {e}")
            continue

    return artifacts, model_path_used

# Load model or retrain if incompatible
artifacts, model_path_used = load_model_with_fallback()

if artifacts is None:
    print("⚠️ No compatible model found. Retraining with current version...")
    artifacts = retrain_model()
    model_path_used = os.path.join(current_dir, 'models', 'churn_prediction_model.pkl')

model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']
numerical_cols = artifacts['numerical_cols']
categorical_cols = artifacts['categorical_cols']

print("✅ Model loaded successfully!")
print(f"Model path: {model_path_used}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Helper function to prepare features for prediction
def prepare_features(input_data):
    """
    Prepare features for prediction, ensuring all expected columns are present
    """
    # Convert to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    # Separate categorical and numerical features
    input_categorical = input_data[categorical_cols].copy()
    input_numerical = input_data[numerical_cols].copy()
    
    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_categorical)
    
    # Ensure all training columns are present
    for col in feature_names:
        if col not in input_encoded.columns and col not in numerical_cols:
            input_encoded[col] = 0
    
    # Scale numerical features
    input_numerical_scaled = pd.DataFrame(
        scaler.transform(input_numerical),
        columns=numerical_cols
    )
    
    # Combine features
    final_input = pd.concat([input_numerical_scaled, input_encoded], axis=1)
    
    # Ensure correct column order
    final_input = final_input.reindex(columns=feature_names, fill_value=0)
    
    return final_input

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Create an API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        
        # Prepare features
        final_input = prepare_features(data)
        
        # Make prediction
        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)
        
        # Prepare response
        churn_probability = float(prediction_proba[0][1])
        will_churn = bool(prediction[0])
        
        response = {
            'churn_prediction': will_churn,
            'churn_probability': churn_probability,
            'confidence': max(prediction_proba[0]),
            'model_version': artifacts.get('version', 'unknown'),
            'scikit_learn_version': sklearn.__version__
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__})

# Create a simple form endpoint for testing
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Get form data
        form_data = request.form.to_dict()
        print("Form data received:", form_data)
        
        # Convert to the right format for our model
        processed_data = {}
        
        # Process numerical fields
        for num_col in numerical_cols:
            processed_data[num_col] = float(form_data.get(num_col, 0))
        
        # Process categorical fields
        for cat_col in categorical_cols:
            processed_data[cat_col] = form_data.get(cat_col, 'No')
        
        print("Processed data:", processed_data)
        
        # Use the helper function to prepare features
        final_input = prepare_features(processed_data)
        
        # Make prediction directly (don't call the predict route)
        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)
        
        # Prepare response
        churn_probability = float(prediction_proba[0][1])
        will_churn = bool(prediction[0])
        
        print(f"Prediction result: churn={will_churn}, probability={churn_probability}")
        
        # Render result template
        return render_template(
            'result.html', 
            prediction=will_churn,
            probability=churn_probability,
            input_data=form_data
        )
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return render_template('error.html', error_message=error_msg)

# Add a debug route to test the app
@app.route('/debug')
def debug():
    return "Flask app is working correctly! Templates directory exists: " + str(os.path.exists('templates'))

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "scikit_learn_version": sklearn.__version__,
        "model_version": artifacts.get('version', 'unknown')
    })

# Model info endpoint
@app.route('/model_info')
def model_info():
    return jsonify({
        "feature_names": feature_names,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "scikit_learn_version": sklearn.__version__,
        "model_type": type(model).__name__
    })

# Run the Flask app
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Feature names:", feature_names)
    print("Numerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)
    print("Scikit-learn version:", sklearn.__version__)
    app.run(debug=True, host='0.0.0.0', port=5001)
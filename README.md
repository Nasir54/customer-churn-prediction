Customer Churn Prediction
A machine learning project that predicts customer churn for businesses, helping organizations identify at-risk customers and implement retention strategies.

📊 Project Overview
This project provides an end-to-end solution for predicting customer churn using machine learning. It includes data preprocessing, feature engineering, model training, and a web application for making predictions.

🚀 Features
Data Preprocessing: Automated cleaning and preparation of customer data

Feature Engineering: Creation of meaningful features to improve prediction accuracy

Machine Learning Models: Multiple algorithms for churn prediction

Web Interface: User-friendly Flask application for making predictions

Docker Support: Containerized deployment for easy setup

RESTful API: Endpoint for programmatic access to predictions

📁 Project Structure
text
customer-churn-prediction/
├── app/                 # Flask web application
│   ├── templates/      # HTML templates
│   └── app.py          # Main application file
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
└── docker-compose.yml # Multi-container setup
🛠️ Installation
Prerequisites
Python 3.8+

pip

Docker (optional)

Local Setup
Clone the repository:

bash
git clone https://github.com/Nasir54/customer-churn-prediction.git
cd customer-churn-prediction
Create a virtual environment:

bash
python -m venv churn-prediction
source churn-prediction/bin/activate  # On Windows: churn-prediction\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Docker Setup
Build and run with Docker Compose:

bash
docker-compose up --build
Access the application at http://localhost:5000

🎯 Usage
Web Application
Start the Flask application:

bash
python app/app.py
Open your browser and navigate to http://localhost:5000

Input customer data through the web form to get churn predictions

API Usage
You can also make predictions programmatically:

python
import requests
import json

url = "http://localhost:5000/predict"
data = {
    "customer_data": {
        "tenure": 12,
        "monthly_charges": 29.85,
        "total_charges": 358.2,
        # ... other features
    }
}

response = requests.post(url, json=data)
prediction = response.json()
print(prediction)
📈 Model Performance
The project includes multiple machine learning models with the following performance metrics:

Accuracy: 82%

Precision: 79%

Recall: 85%

F1-Score: 82%

🔧 Configuration
Modify the model parameters and preprocessing steps in the configuration files:

src/model_training.py - Adjust model hyperparameters

src/data_preprocessing.py - Customize data cleaning steps

src/feature_engineering.py - Modify feature creation

🤝 Contributing
We welcome contributions! Please feel free to submit pull requests, open issues, or suggest new features.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a pull request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Data source: IBM Telco Customer Churn Dataset

Inspired by various customer churn prediction implementations

Thanks to the open-source community for valuable tools and libraries

📞 Contact
For questions or support, please contact:

Nasir - GitHub

Project Link: https://github.com/Nasir54/customer-churn-prediction


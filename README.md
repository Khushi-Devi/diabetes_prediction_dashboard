# AI Diabetes Prediction Dashboard
A Machine Learning–Powered Interactive Health Prediction System

The AI Diabetes Prediction Dashboard is an interactive web application built with Streamlit that demonstrates the end-to-end machine learning workflow—from data exploration to model evaluation and real-time prediction. Using the PIMA Diabetes Dataset, the system trains multiple ML classifiers, compares their performance, and deploys the best-performing model for live inference.

=>Features

1. Data Exploration:-
Dataset preview, 
Summary statistics, 
Correlation heatmap, 
Outcome distribution

2. Model Comparison:-
Evaluates multiple ML models, 
Accuracy, Precision, Recall, F1-Score visualization, 
Highlights best-performing model, 
Performance comparison plots

3. Live Diabetes Prediction:-
User-friendly input form, 
Automatic data scaling using saved scaler.pkl, 
Predictions using best model stored as best_model.pkl, 
Displays probabilities and model confidence, 
Clean, modern UI with color-coded risk results

=> Machine Learning Workflow

1. Data preprocessing

2. Feature scaling

3. Model training (multiple algorithms)

4. Model evaluation and metric storage

5. Saving the best model and scaler using joblib

6. Deploying model inside Streamlit app 

## How to Run

Follow these steps to set up and run the project locally:

- **Clone the repository**
   ```bash
   git clone https://github.com/Khushi-Devi/diabetes_prediction_dashboard.git
   cd diabetes_prediction_dashboard


- Create a virtual environment (recommended)

python -m venv venv

source venv/bin/activate   # On Linux/Mac

venv\Scripts\activate      # On Windows

-pip install -r requirements.txt

-streamlit run app.py (run the dashboard)

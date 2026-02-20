import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report

# Define base path
BASE_DIR = '/home/ubuntu/nov25_bds_int_covid1'

# Define sub-directories
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data/processed/ML/test_data')

def predict(model_type, sampler_type):
    # Construct the filenames
    model_name = f"{model_type}_{sampler_type}.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    try:
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load data
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))

        # Process and predict
        X_new_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_new_scaled)

        # Print results
        clean_name = f"{model_type} {sampler_type}".replace('_', ' ')
        print(f"\n--- Results for: {clean_name.upper()} ---")
        print(classification_report(y_test, y_pred))

    except FileNotFoundError as e:
        print(f"Error: Could not find file at {e.filename}")

if __name__ == "__main__":
    predict(model_type='rf', sampler_type='none')
    predict(model_type='rf', sampler_type='smote')
    predict(model_type='rf', sampler_type='oversample')
    predict(model_type='rf', sampler_type='undersample')
    
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Configuration
RANDOM_STATE = 42
BASE_DIR = '/home/ubuntu/nov25_bds_int_covid1'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'data/processed/ML/train_data')

def train(model_type, sampler_type):
    # Load training data
    X_train = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'y_train.csv'))

    # Scale X_train and save scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # Sample data
    samplers = {
        'none': None,
        'smote': SMOTE(random_state=RANDOM_STATE),
        'oversample': RandomOverSampler(sampling_strategy='not majority', random_state=RANDOM_STATE),
        'undersample': RandomUnderSampler(sampling_strategy='majority', random_state=RANDOM_STATE)
    }
    
    sampler = samplers.get(sampler_type)
    if sampler:
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
    else:
        X_res, y_res = X_train_scaled, y_train

    # Random forest with cross-validation
    if model_type == 'rf':
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, 50],
            'min_samples_split': [2, 5]
        }
        print(f"Starting Grid Search for Random Forest with {sampler_type}...")
        
        # Cross-Validation with 5-folds 
        cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1_macro', n_jobs=2)
        grid_search.fit(X_res, y_res.values.ravel()) # values.ravel() ensures y is the correct shape
        
        model = grid_search.best_estimator_
        print(f"Best Params found: {grid_search.best_params_}")

    # Support vector classification with cross-validation
    elif model_type == 'svc':
        base_model = SVC(random_state=RANDOM_STATE)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': [0.001, 0.1, 0.5]
        }
        print(f"Starting Grid Search for SVC with {sampler_type}...")
        
        # Cross-Validation with 5-folds
        cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1_macro', n_jobs=2)
        grid_search.fit(X_res, y_res.values.ravel())
        
        model = grid_search.best_estimator_
        print(f"Best Params found: {grid_search.best_params_}")
    
    # Logistic regression
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=RANDOM_STATE)
        model.fit(X_res, y_res.values.ravel())
    
    else:
        print(f"Error: {model_type} not recognized.")
        return

    # Save model and sample data
    model_filename = f"{model_type}_{sampler_type}.pkl"
    save_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, save_path)
    
    print(f"Success: Final model saved as {model_filename}")

if __name__ == "__main__":
    train(model_type='rf', sampler_type='none')
    train(model_type='rf', sampler_type='smote')
    train(model_type='rf', sampler_type='oversample')
    train(model_type='rf', sampler_type='undersample')
    
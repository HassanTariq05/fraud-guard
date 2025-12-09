"""
Script to train and save the fraud detection models.
Run this once to create the pre-trained models.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

def train_and_save_models():
    """Train models on creditcard.csv and save them as pickle files."""
    
    print("Loading data...")
    df_raw = pd.read_csv('creditcard.csv')
    
    print("Balancing dataset...")
    fraud = df_raw[df_raw['Class'] == 1]
    non_fraud = df_raw[df_raw['Class'] == 0].sample(n=len(fraud), random_state=42)
    balanced_df = pd.concat([fraud, non_fraud])
    
    X = balanced_df.drop('Class', axis=1)
    y = balanced_df['Class']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Scaling features...")
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    X_train_processed = X_train.copy()
    X_train_processed['Scaled_Amount'] = scaler_amount.fit_transform(X_train[['Amount']])
    X_train_processed['Scaled_Time'] = scaler_time.fit_transform(X_train[['Time']])
    X_train_processed = X_train_processed.drop(['Amount', 'Time'], axis=1)
    
    X_test_processed = X_test.copy()
    X_test_processed['Scaled_Amount'] = scaler_amount.transform(X_test[['Amount']])
    X_test_processed['Scaled_Time'] = scaler_time.transform(X_test[['Time']])
    X_test_processed = X_test_processed.drop(['Amount', 'Time'], axis=1)
    
    feature_names = list(X_train_processed.columns)
    
    print("Training models...")
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    trained_models = {}
    all_metrics = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        y_pred = model.predict(X_test_processed)
        all_metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models
    print("Saving models...")
    with open('models/trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)
    
    with open('models/scaler_amount.pkl', 'wb') as f:
        pickle.dump(scaler_amount, f)
    
    with open('models/scaler_time.pkl', 'wb') as f:
        pickle.dump(scaler_time, f)
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    with open('models/metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)
    
    # Save sample data for the app
    sample_fraud = balanced_df[balanced_df['Class'] == 1].iloc[0].to_dict()
    sample_non_fraud = balanced_df[balanced_df['Class'] == 0].iloc[0].to_dict()
    
    samples = {
        'fraud': sample_fraud,
        'non_fraud': sample_non_fraud,
        'balanced_df': balanced_df
    }
    
    with open('models/samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
    
    print("✓ Models trained and saved successfully!")
    print(f"  - Trained models: {list(trained_models.keys())}")
    print(f"  - Feature names: {len(feature_names)} features")
    print(f"  - Saved to: models/")

if __name__ == "__main__":
    train_and_save_models()

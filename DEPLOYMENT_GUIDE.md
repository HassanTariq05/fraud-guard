# Credit Card Fraud Detection System - Pre-trained Model Deployment Guide

## Overview
This project uses pre-trained machine learning models to detect credit card fraud. The models are trained once and saved as pickle files, eliminating the need to load the heavy `creditcard.csv` file during deployment.

## Setup Instructions

### 1. First Time Setup (Training)
If you're setting up for the first time, you need to train the models:

```bash
# Install dependencies
pip install -r requirements.txt

# Train and save the pre-trained models
python train_model.py
```

This creates a `models/` directory with:
- `trained_models.pkl` - Three trained classifiers (Random Forest, Logistic Regression, XGBoost)
- `scaler_amount.pkl` - Scaler for transaction amounts
- `scaler_time.pkl` - Scaler for transaction times
- `feature_names.pkl` - List of feature names
- `metrics.pkl` - Model performance metrics
- `samples.pkl` - Sample transactions for demonstration

### 2. Running the Streamlit App
```bash
python -m streamlit run app.py
```

The app will load the pre-trained models from the `models/` directory in seconds, without needing the CSV file.

## Deployment to Streamlit Cloud

### Benefits of Pre-trained Models:
✅ **Lightweight** - Only pickle files (~50-200MB total) instead of the full 150MB+ CSV  
✅ **Fast Loading** - Models load in seconds  
✅ **Production Ready** - No training on deployment  
✅ **Easy Updates** - Retrain locally, push updated pickle files  

### Deployment Steps:

1. **Push to GitHub** (ensure `creditcard.csv` is in `.gitignore`):
   ```bash
   git add -A
   git commit -m "Add pre-trained models for deployment"
   git push
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Select the main branch and `app.py` as the entry point
   - Deploy!

The app will work perfectly without the CSV file since all models are pre-trained.

## Updating Models

If you want to retrain models with new data:

1. Place your updated dataset in the same format as `creditcard.csv`
2. Run `python train_model.py` (or modify it to use your new CSV)
3. The pickle files in `models/` will be updated
4. Push to GitHub and your Streamlit Cloud app will automatically use the new models

## File Structure
```
fraud-guard/
├── app.py                    # Main Streamlit app
├── train_model.py            # Training script (run once, commit pickle files)
├── models/                   # Pre-trained models (commit to Git)
│   ├── trained_models.pkl
│   ├── scaler_amount.pkl
│   ├── scaler_time.pkl
│   ├── feature_names.pkl
│   ├── metrics.pkl
│   └── samples.pkl
├── creditcard.csv           # EXCLUDED from Git (.gitignore)
├── requirements.txt
└── .gitignore
```

## Model Details

The system includes three ensemble models:
- **Random Forest Classifier** - Good for complex patterns
- **Logistic Regression** - Fast, interpretable baseline
- **XGBoost Classifier** - Highest performance with gradient boosting

All models are trained on a balanced dataset (equal fraud/non-fraud cases) for reliable fraud detection.

## Notes

- The `creditcard.csv` file is NOT needed for deployment (it's in `.gitignore`)
- Always commit the `models/` directory to Git
- Models are re-trained from scratch using `train_model.py` 
- For large-scale deployments, consider using a cloud database to manage model versioning

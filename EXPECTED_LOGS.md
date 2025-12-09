# Expected Logs Output

When the app deploys on Streamlit Cloud, you should see logs like this:

## ✅ SUCCESSFUL DEPLOYMENT LOGS

```
================================================================================
STREAMLIT APP STARTING
================================================================================
Python version: 3.x.x ...
Current working directory: /mount/src/fraud-guard
Script directory: /mount/src/fraud-guard
Path setup complete

Page config set successfully
Session state initialized
Calling load_pretrained_models()...

================================================================================
LOADING PRE-TRAINED MODELS
================================================================================
Script directory: /mount/src/fraud-guard
Models directory: /mount/src/fraud-guard/models
Models directory exists: True
Files in models directory: ['samples.pkl', 'scaler_amount.pkl', 'scaler_time.pkl', 'feature_names.pkl', 'metrics.pkl', 'trained_models.pkl']

Loading trained_models.pkl...
✓ Loaded trained_models: ['Random Forest', 'Logistic Regression', 'XGBoost']
Loading scaler_amount.pkl...
✓ Loaded scaler_amount
Loading scaler_time.pkl...
✓ Loaded scaler_time
Loading feature_names.pkl...
✓ Loaded 30 feature names
Loading metrics.pkl...
✓ Loaded metrics for ['Random Forest', 'Logistic Regression', 'XGBoost']
Loading samples.pkl...
✓ Loaded samples

================================================================================
ALL MODELS LOADED SUCCESSFULLY!
================================================================================

Models data returned. Loaded: True
Models loaded successfully, extracting variables...
All variables extracted successfully
Number of models: 3
Number of features: 30
Balanced DF shape: (4640, 20)
UI initialized successfully
```

## ❌ FAILED DEPLOYMENT LOGS (Common Issues)

### Issue 1: Models directory not found
```
Models directory: /mount/src/fraud-guard/models
Models directory exists: False
❌ Model file not found: [Errno 2] No such file or directory
MODEL LOADING FAILED: Model file not found
```
**Fix**: Make sure all files in `models/` directory are committed to git

### Issue 2: Corrupted pickle file
```
Loading trained_models.pkl...
❌ Error loading models: unpickling error
Traceback: ...pickle.UnpicklingError...
MODEL LOADING FAILED: Error loading models
```
**Fix**: Retrain models: `python train_model.py`

### Issue 3: Memory error
```
Loading trained_models.pkl...
❌ Error loading models: MemoryError: Unable to allocate
```
**Fix**: This means the model file is too large. Try reducing model complexity in `train_model.py`

### Issue 4: Missing dependency
```
Traceback (most recent call last):
  File "app.py", line X, in <module>
    import xgboost
ModuleNotFoundError: No module named 'xgboost'
```
**Fix**: Make sure all packages in `requirements.txt` are listed

## How to Check Logs on Streamlit Cloud

### Method 1: In-Browser Logs
1. Go to your app URL
2. Look for error message or "See logs" button
3. Click it to view console output

### Method 2: Streamlit Cloud Dashboard
1. Go to https://share.streamlit.io/
2. Click on your **fraud-guard** app
3. Look for **"Logs"** tab or panel on the right
4. Real-time logs will appear there

### Method 3: Direct Terminal View
Some deployments show logs directly in the deployment section

## Debugging Steps

1. **Check if models/ directory is in git:**
   ```bash
   git ls-files models/
   ```
   Should show all 6 pickle files

2. **Check file sizes:**
   ```bash
   ls -lh models/
   ```
   Total should be around 950KB

3. **Test locally with debug logging:**
   ```bash
   python -m streamlit run app.py --logger.level=debug
   ```

4. **View git log to confirm changes were pushed:**
   ```bash
   git log --oneline -5
   ```

## If App Still Fails After Seeing These Logs

1. Collect all console logs from Streamlit Cloud
2. Check for any Python version mismatches
3. Verify that all `.pkl` files are properly tracked:
   ```bash
   git check-ignore models/*.pkl
   ```
   (Should return nothing - means files are NOT ignored)

4. If models are being ignored, remove from .gitignore:
   ```bash
   # In .gitignore, make sure these lines are NOT present:
   # models/*.pkl
   # models/
   ```

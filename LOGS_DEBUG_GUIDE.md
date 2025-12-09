# How to View Streamlit Cloud Logs

## Option 1: Streamlit Cloud Console (Recommended)

1. Go to [Streamlit Cloud Dashboard](https://share.streamlit.io/)
2. Find your **fraud-guard** app
3. Click on the app name
4. Look for the **"Logs"** tab or button on the right side
5. You'll see real-time console output showing:
   - App startup process
   - Model loading status
   - Any errors that occur

## Option 2: View App Logs Directly

When the app fails to load on Streamlit Cloud:

1. The browser will show an error message
2. Click **"See logs"** or **"View logs"** button (if available)
3. This opens the live logs panel showing:
   ```
   ================================================================================
   STREAMLIT APP STARTING
   ================================================================================
   Python version: ...
   Current working directory: ...
   Script directory: ...
   Path setup complete
   ...
   Loading PRE-TRAINED MODELS
   ================================================================================
   Models directory: /mount/src/fraud-guard/models
   Models directory exists: True
   Files in models directory: [...]
   ...
   ```

## What to Look For

### ✅ Success Logs
```
================================================================================
ALL MODELS LOADED SUCCESSFULLY!
================================================================================
Number of models: 3
Number of features: 30
Balanced DF shape: (4640, 30)
UI initialized successfully
```

### ❌ Error Logs
Look for any of these error patterns:

1. **Missing models directory:**
   ```
   Models directory exists: False
   ❌ Model file not found
   ```

2. **Corrupted pickle file:**
   ```
   Error loading models: pickle.UnpicklingError...
   ```

3. **Memory issues:**
   ```
   MemoryError: Unable to allocate...
   ```

4. **Import errors:**
   ```
   ImportError: No module named...
   ```

## Sharing Logs with Support

If you need help debugging:

1. Copy all console logs from the Streamlit Cloud dashboard
2. Include:
   - The error message
   - Stack trace
   - File size information
   - Python version
   
Example:
```
[timestamp] STREAMLIT APP STARTING
[timestamp] Current working directory: /mount/src/fraud-guard
[timestamp] Models directory: /mount/src/fraud-guard/models
[timestamp] Files in models directory: [list of files]
[timestamp] ❌ ERROR: [detailed error message]
[timestamp] Traceback: [full traceback]
```

## Testing Locally Before Deployment

Run this command to simulate Streamlit logs locally:

```bash
python -m streamlit run app.py --logger.level=debug
```

This will show all debug messages including model loading details.

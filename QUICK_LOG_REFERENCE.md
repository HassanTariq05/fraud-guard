# 🔍 STREAMLIT LOGS - QUICK REFERENCE

## Where to Find Logs

### On Streamlit Cloud:
1. **Dashboard**: https://share.streamlit.io/ → Click your app → Look for **Logs** button
2. **In-App Error**: If app fails, you'll see error box → Click **"See logs"**
3. **Terminal/Console**: Real-time output appears as app loads

## What You'll See

### ✅ APP STARTING (First log line)
```
================================================================================
STREAMLIT APP STARTING
================================================================================
```

### ✅ MODELS LOADING (Next section)
```
================================================================================
LOADING PRE-TRAINED MODELS
================================================================================
Loading trained_models.pkl...
✓ Loaded trained_models: ['Random Forest', 'Logistic Regression', 'XGBoost']
...
```

### ✅ SUCCESS MESSAGE (If everything works)
```
================================================================================
ALL MODELS LOADED SUCCESSFULLY!
================================================================================
```

## Common Error Messages & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Models directory exists: False` | models/ not in git | `git add models/` then push |
| `unpickling error` | Corrupted pickle file | Run `python train_model.py` |
| `MemoryError` | Model too large | Reduce n_estimators in train_model.py |
| `No module named 'xgboost'` | Missing dependency | Check requirements.txt |
| `No such file or directory` | File not found | Verify all 6 .pkl files exist |

## Files That Must Be in Git

✅ These MUST be committed and pushed:
- `models/trained_models.pkl` (677KB)
- `models/samples.pkl` (248KB)
- `models/metrics.pkl` (4.1KB)
- `models/scaler_amount.pkl` (0.6KB)
- `models/scaler_time.pkl` (0.6KB)
- `models/feature_names.pkl` (0.2KB)

❌ These should NOT be committed:
- `creditcard.csv` (144MB - in .gitignore)
- `__pycache__/`
- `.local/`

## Testing Before Deployment

```bash
# Test locally with debug logs
python -m streamlit run app.py --logger.level=debug

# Check if models are tracked
git ls-files models/

# Check file sizes
ls -lh models/

# Check recent commits
git log --oneline -5
```

## If App Still Fails

1. **Copy all logs** from Streamlit Cloud
2. **Check .gitignore** - make sure models/ is NOT ignored
3. **Force add models**: `git add -f models/` (if ignored)
4. **Retrain locally**: `python train_model.py`
5. **Push to GitHub**: `git push`
6. **Trigger rebuild**: Go to Streamlit Cloud → Reboot app

## Quick Checklist

- [ ] All 6 pickle files exist in `models/`
- [ ] All 6 files are tracked in git (`git ls-files models/`)
- [ ] Files are not in `.gitignore`
- [ ] Latest commit pushed to GitHub
- [ ] App starts locally: `python -m streamlit run app.py`
- [ ] No import errors
- [ ] Python 3.7+ installed

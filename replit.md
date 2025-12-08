# Credit Card Fraud Detection System

## Overview

This is a machine learning application built with Streamlit that detects fraudulent credit card transactions. The system uses a Random Forest classifier trained on historical transaction data to predict whether a new transaction is fraudulent or legitimate. The application handles highly imbalanced data through undersampling and provides real-time predictions with performance metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Technology:** Streamlit web framework  
**Rationale:** Streamlit was chosen for rapid development of data science applications with minimal frontend code. It provides built-in widgets for user interaction and automatic UI rendering from Python code.

**Key Design Decisions:**
- Wide layout configuration to accommodate multiple columns and visualizations
- Caching strategy using `@st.cache_resource` to prevent retraining the model on every interaction
- Responsive design that works across different screen sizes

### Machine Learning Pipeline
**Problem:** Detect fraudulent credit card transactions from highly imbalanced dataset  
**Solution:** Random Forest Classifier with data balancing through undersampling

**Architectural Components:**
1. **Data Preprocessing**
   - StandardScaler for `Amount` and `Time` features to normalize scale
   - Undersampling of majority class (non-fraud) to match minority class (fraud) size
   - Rationale: Addresses class imbalance issue where fraud cases are rare (~0.17% typically)

2. **Feature Engineering**
   - Separate scalers for `Amount` and `Time` features
   - Original features replaced with scaled versions
   - V1-V28 features (PCA-transformed) used as-is from dataset

3. **Model Selection**
   - Random Forest Classifier chosen over Logistic Regression and XGBoost (considered in development)
   - Pros: Robust to overfitting, handles non-linear relationships, provides feature importance
   - Cons: Larger model size, slower predictions than linear models
   - Uses `n_jobs=-1` for parallel processing

4. **Train/Test Split**
   - 70/30 split with fixed random state for reproducibility
   - Stratification not explicitly set but data is pre-balanced

### Performance Monitoring
**Metrics Tracked:**
- Accuracy: Overall correctness
- Precision: Minimize false fraud alerts
- Recall: Catch actual fraud cases
- F1 Score: Balance between precision and recall
- Confusion Matrix: Detailed classification breakdown

**Design Choice:** Multiple metrics used because in fraud detection, both false positives (annoying customers) and false negatives (missing fraud) have real costs.

### Data Flow
1. Load creditcard.csv dataset
2. Balance dataset by undersampling non-fraud transactions
3. Split into train/test sets
4. Scale features independently
5. Train Random Forest model
6. Cache trained model and scalers
7. Accept user input for new transactions
8. Preprocess input using saved scalers
9. Generate prediction and confidence score

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for the UI
- **pandas**: Data manipulation and CSV loading
- **numpy**: Numerical operations and array handling
- **scikit-learn**: Machine learning framework providing:
  - StandardScaler for feature scaling
  - RandomForestClassifier for the ML model
  - train_test_split for data splitting
  - Performance metrics (accuracy, precision, recall, f1, confusion_matrix)

### Development Dependencies (Not in Production)
- **matplotlib**: Data visualization (used in attached prototype)
- **seaborn**: Statistical plotting (used in attached prototype)
- **xgboost**: Alternative classifier considered during development

### Data Requirements
- **creditcard.csv**: Required dataset containing transaction records
  - Must include features V1-V28 (PCA-transformed features)
  - Must include `Time`, `Amount`, and `Class` columns
  - Class column: 0 for legitimate, 1 for fraudulent transactions
  - Standard format from Kaggle Credit Card Fraud Detection dataset

### No External Services
This application runs entirely locally without external API calls, databases, or third-party service integrations. All data processing and model inference happens in-memory.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import pickle
import os
import io

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'custom_model' not in st.session_state:
    st.session_state.custom_model = None
    st.session_state.custom_scalers = None

if 'loaded_sample' not in st.session_state:
    st.session_state.loaded_sample = None

if 'prefill_sample' not in st.session_state:
    st.session_state.prefill_sample = False

@st.cache_resource
def load_pretrained_models():
    """Load pre-trained models from pickle files."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    
    if not os.path.exists(models_dir):
        st.error(f"❌ Models directory not found at {models_dir}. Please run train_model.py first.")
        st.stop()
    
    try:
        with open(os.path.join(models_dir, 'trained_models.pkl'), 'rb') as f:
            trained_models = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler_amount.pkl'), 'rb') as f:
            scaler_amount = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler_time.pkl'), 'rb') as f:
            scaler_time = pickle.load(f)
        
        with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        
        with open(os.path.join(models_dir, 'metrics.pkl'), 'rb') as f:
            all_metrics = pickle.load(f)
        
        with open(os.path.join(models_dir, 'samples.pkl'), 'rb') as f:
            samples = pickle.load(f)
        
        return trained_models, scaler_amount, scaler_time, feature_names, all_metrics, samples['fraud'], samples['non_fraud'], samples['balanced_df']
    
    except Exception as e:
        st.error(f"❌ Error loading pre-trained models: {str(e)}")
        st.stop()

# Load pre-trained models
trained_models, scaler_amount, scaler_time, feature_names, all_metrics, sample_fraud, sample_non_fraud, balanced_df = load_pretrained_models()

st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Single Prediction", 
    "📁 Batch Prediction",
    "📊 Model Comparison", 
    "📈 Visualizations",
    "📜 Prediction History",
    "🔄 Retrain Model",
    "📋 Sample Data"
])

with tab1:
    st.header("Enter Transaction Details")
    st.markdown("Fill in the transaction features below to predict if it's fraudulent or legitimate.")
    
    model_choice = st.selectbox(
        "Select Model for Prediction",
        list(trained_models.keys()),
        key="single_model_choice"
    )
    
    v_features = [f'V{i}' for i in range(1, 29)]
    
    if st.session_state.prefill_sample and st.session_state.loaded_sample:
        sample = st.session_state.loaded_sample
        st.session_state['time_input'] = float(sample['Time'])
        st.session_state['amount_input'] = float(sample['Amount'])
        for feature in v_features:
            st.session_state[feature] = float(sample[feature])
        st.session_state.prefill_sample = False
    
    loaded = st.session_state.loaded_sample
    
    st.subheader("Transaction Information")
    time_col, amount_col = st.columns(2)
    
    default_time = float(loaded['Time']) if loaded else 0.0
    default_amount = float(loaded['Amount']) if loaded else 0.0
    
    with time_col:
        time_value = st.number_input(
            "Time (seconds from first transaction)", 
            value=default_time, 
            min_value=0.0,
            format="%.2f",
            key="time_input",
            help="Time elapsed in seconds since the first transaction in the dataset"
        )
    
    with amount_col:
        amount_value = st.number_input(
            "Transaction Amount ($)", 
            value=default_amount, 
            min_value=0.0,
            format="%.2f",
            key="amount_input",
            help="The transaction amount in dollars"
        )
    
    st.markdown("---")
    st.subheader("PCA Features (V1-V28)")
    st.markdown("*These are anonymized features from PCA transformation*")
    
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    
    for i, feature in enumerate(v_features):
        default_val = float(loaded[feature]) if loaded else 0.0
        if i < 10:
            with col1:
                input_values[feature] = st.number_input(
                    feature, 
                    value=default_val, 
                    format="%.6f",
                    key=feature
                )
        elif i < 19:
            with col2:
                input_values[feature] = st.number_input(
                    feature, 
                    value=default_val, 
                    format="%.6f",
                    key=feature
                )
        else:
            with col3:
                input_values[feature] = st.number_input(
                    feature, 
                    value=default_val, 
                    format="%.6f",
                    key=feature
                )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn1:
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            scaled_amount = scaler_amount.transform([[amount_value]])[0][0]
            scaled_time = scaler_time.transform([[time_value]])[0][0]
            
            input_data = []
            for feature in feature_names:
                if feature == 'Scaled_Amount':
                    input_data.append(scaled_amount)
                elif feature == 'Scaled_Time':
                    input_data.append(scaled_time)
                else:
                    input_data.append(input_values[feature])
            
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            model = trained_models[model_choice]
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            result = "Fraudulent" if prediction == 1 else "Legitimate"
            confidence = probability[1] if prediction == 1 else probability[0]
            
            st.session_state.prediction_history.append({
                'Time': time_value,
                'Amount': amount_value,
                'Model': model_choice,
                'Prediction': result,
                'Confidence': f"{confidence*100:.2f}%"
            })
            
            st.markdown("---")
            if prediction == 1:
                st.error("🚨 **FRAUDULENT TRANSACTION DETECTED!**")
                st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
            else:
                st.success("✅ **Transaction is LEGITIMATE**")
                st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
    
    def load_fraud_sample():
        st.session_state.loaded_sample = sample_fraud
        st.session_state.prefill_sample = True
    
    def load_non_fraud_sample():
        st.session_state.loaded_sample = sample_non_fraud
        st.session_state.prefill_sample = True
    
    with col_btn2:
        st.button("📝 Load Sample Fraud", use_container_width=True, on_click=load_fraud_sample)
    
    with col_btn3:
        st.button("📝 Load Sample Non-Fraud", use_container_width=True, on_click=load_non_fraud_sample)

with tab2:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with multiple transactions to get predictions for all of them at once.")
    
    batch_model_choice = st.selectbox(
        "Select Model for Batch Prediction",
        list(trained_models.keys()),
        key="batch_model_choice"
    )
    
    st.markdown("""
    **Required CSV Format:**
    - Must contain columns: Time, Amount, V1-V28
    - Each row represents one transaction
    """)
    
    st.download_button(
        label="📥 Download Sample Template",
        data=balanced_df.drop('Class', axis=1).head(5).to_csv(index=False),
        file_name="batch_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'], key="batch_upload")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔮 Run Batch Prediction", type="primary"):
                required_cols = ['Time', 'Amount'] + v_features
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    batch_processed = batch_df.copy()
                    batch_processed['Scaled_Amount'] = scaler_amount.transform(batch_df[['Amount']])
                    batch_processed['Scaled_Time'] = scaler_time.transform(batch_df[['Time']])
                    batch_processed = batch_processed.drop(['Amount', 'Time'], axis=1)
                    
                    batch_processed = batch_processed[feature_names]
                    
                    model = trained_models[batch_model_choice]
                    predictions = model.predict(batch_processed)
                    probabilities = model.predict_proba(batch_processed)
                    
                    results_df = batch_df.copy()
                    results_df['Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]
                    results_df['Fraud_Probability'] = [f"{prob[1]*100:.2f}%" for prob in probabilities]
                    results_df['Legitimate_Probability'] = [f"{prob[0]*100:.2f}%" for prob in probabilities]
                    
                    st.markdown("---")
                    st.subheader("Batch Prediction Results")
                    
                    fraud_count = sum(predictions)
                    legit_count = len(predictions) - fraud_count
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric("Total Transactions", len(predictions))
                    with res_col2:
                        st.metric("Fraudulent", fraud_count)
                    with res_col3:
                        st.metric("Legitimate", legit_count)
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv_buffer = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_buffer,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
                    for _, row in results_df.iterrows():
                        st.session_state.prediction_history.append({
                            'Time': row['Time'],
                            'Amount': row['Amount'],
                            'Model': batch_model_choice,
                            'Prediction': row['Prediction'],
                            'Confidence': row['Fraud_Probability'] if row['Prediction'] == 'Fraudulent' else row['Legitimate_Probability']
                        })
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("Model Comparison Dashboard")
    st.markdown("Compare the performance of different machine learning models on the fraud detection task.")
    
    comparison_df = pd.DataFrame({
        'Model': list(all_metrics.keys()),
        'Accuracy': [all_metrics[m]['accuracy'] for m in all_metrics],
        'Precision': [all_metrics[m]['precision'] for m in all_metrics],
        'Recall': [all_metrics[m]['recall'] for m in all_metrics],
        'F1 Score': [all_metrics[m]['f1'] for m in all_metrics]
    })
    
    st.subheader("Performance Metrics Comparison")
    
    styled_df = comparison_df.copy()
    styled_df['Accuracy'] = styled_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    styled_df['Precision'] = styled_df['Precision'].apply(lambda x: f"{x*100:.2f}%")
    styled_df['Recall'] = styled_df['Recall'].apply(lambda x: f"{x*100:.2f}%")
    styled_df['F1 Score'] = styled_df['F1 Score'].apply(lambda x: f"{x*100:.2f}%")
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Visual Comparison")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = comparison_df[metric].values * 100
        bars = ax.bar(comparison_df['Model'], values, color=colors)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.subheader("Best Model Recommendation")
    
    best_model = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
    best_f1 = comparison_df['F1 Score'].max() * 100
    
    st.success(f"**Recommended Model: {best_model}** with F1 Score of {best_f1:.2f}%")
    st.markdown("""
    *The F1 Score is used as the primary metric for recommendation because it balances 
    precision (minimizing false fraud alerts) and recall (catching actual fraud cases).*
    """)

with tab4:
    st.header("Visualization Dashboard")
    
    viz_model = st.selectbox(
        "Select Model for Visualization",
        list(trained_models.keys()),
        key="viz_model_choice"
    )
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Confusion Matrix")
        
        cm = all_metrics[viz_model]['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        ax_cm.set_xlabel('Predicted', fontsize=12)
        ax_cm.set_ylabel('Actual', fontsize=12)
        ax_cm.set_title(f'Confusion Matrix - {viz_model}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()
        
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        **Matrix Interpretation:**
        - True Negatives (Correct Legitimate): **{tn}**
        - False Positives (False Alarms): **{fp}**
        - False Negatives (Missed Fraud): **{fn}**
        - True Positives (Caught Fraud): **{tp}**
        """)
    
    with col_viz2:
        st.subheader("Feature Importance")
        
        model = trained_models[viz_model]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(15)
            
            fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
            bars = ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db')
            ax_imp.set_xlabel('Importance Score', fontsize=12)
            ax_imp.set_title(f'Top 15 Feature Importance - {viz_model}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close()
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coefs
            }).sort_values('Importance', ascending=True).tail(15)
            
            fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
            bars = ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db')
            ax_imp.set_xlabel('Absolute Coefficient', fontsize=12)
            ax_imp.set_title(f'Top 15 Feature Importance - {viz_model}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close()
    
    st.markdown("---")
    st.subheader("Metrics Radar Chart")
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        all_metrics[viz_model]['accuracy'],
        all_metrics[viz_model]['precision'],
        all_metrics[viz_model]['recall'],
        all_metrics[viz_model]['f1']
    ]
    
    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]
    
    ax_radar.fill(angles, values_plot, color='#3498db', alpha=0.25)
    ax_radar.plot(angles, values_plot, color='#3498db', linewidth=2)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=11)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title(f'{viz_model} Performance', fontsize=14, fontweight='bold', pad=20)
    
    st.pyplot(fig_radar)
    plt.close()

with tab5:
    st.header("Prediction History")
    st.markdown("View all predictions made during this session.")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        hist_col1, hist_col2, hist_col3 = st.columns(3)
        
        total_preds = len(history_df)
        fraud_preds = len(history_df[history_df['Prediction'] == 'Fraudulent'])
        legit_preds = len(history_df[history_df['Prediction'] == 'Legitimate'])
        
        with hist_col1:
            st.metric("Total Predictions", total_preds)
        with hist_col2:
            st.metric("Fraudulent", fraud_preds)
        with hist_col3:
            st.metric("Legitimate", legit_preds)
        
        st.markdown("---")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        csv_history = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv_history,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions made yet. Make some predictions to see them here!")

with tab6:
    st.header("Model Retraining Interface")
    st.markdown("Upload a custom dataset to retrain the fraud detection model.")
    
    st.warning("""
    **Important Notes:**
    - Your dataset must have the same format as the original creditcard.csv
    - Required columns: Time, Amount, V1-V28, Class
    - Class should be 0 for legitimate and 1 for fraudulent transactions
    """)
    
    retrain_model_type = st.selectbox(
        "Select Model Type to Train",
        ['Random Forest', 'Logistic Regression', 'XGBoost'],
        key="retrain_model_type"
    )
    
    st.markdown("---")
    
    retrain_file = st.file_uploader("Upload Training Dataset", type=['csv'], key="retrain_upload")
    
    if retrain_file is not None:
        try:
            retrain_df = pd.read_csv(retrain_file)
            st.write("**Uploaded Dataset Preview:**")
            st.dataframe(retrain_df.head(), use_container_width=True)
            
            required_cols = ['Time', 'Amount', 'Class'] + v_features
            missing_cols = [col for col in required_cols if col not in retrain_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.success("Dataset format is valid!")
                
                class_dist = retrain_df['Class'].value_counts()
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    st.metric("Legitimate Transactions (0)", class_dist.get(0, 0))
                with dist_col2:
                    st.metric("Fraudulent Transactions (1)", class_dist.get(1, 0))
                
                balance_data = st.checkbox("Balance dataset using undersampling", value=True)
                test_size = st.slider("Test Set Size (%)", 10, 50, 30) / 100
                
                if st.button("🔄 Retrain Model", type="primary"):
                    with st.spinner("Training model... This may take a moment."):
                        if balance_data:
                            fraud = retrain_df[retrain_df['Class'] == 1]
                            non_fraud = retrain_df[retrain_df['Class'] == 0]
                            
                            if len(fraud) > 0 and len(non_fraud) > 0:
                                min_class = min(len(fraud), len(non_fraud))
                                fraud_sample = fraud.sample(n=min_class, random_state=42)
                                non_fraud_sample = non_fraud.sample(n=min_class, random_state=42)
                                retrain_df = pd.concat([fraud_sample, non_fraud_sample])
                        
                        X = retrain_df.drop('Class', axis=1)
                        y = retrain_df['Class']
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        custom_scaler_amount = StandardScaler()
                        custom_scaler_time = StandardScaler()
                        
                        X_train_processed = X_train.copy()
                        X_train_processed['Scaled_Amount'] = custom_scaler_amount.fit_transform(X_train[['Amount']])
                        X_train_processed['Scaled_Time'] = custom_scaler_time.fit_transform(X_train[['Time']])
                        X_train_processed = X_train_processed.drop(['Amount', 'Time'], axis=1)
                        
                        X_test_processed = X_test.copy()
                        X_test_processed['Scaled_Amount'] = custom_scaler_amount.transform(X_test[['Amount']])
                        X_test_processed['Scaled_Time'] = custom_scaler_time.transform(X_test[['Time']])
                        X_test_processed = X_test_processed.drop(['Amount', 'Time'], axis=1)
                        
                        if retrain_model_type == 'Random Forest':
                            custom_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                        elif retrain_model_type == 'Logistic Regression':
                            custom_model = LogisticRegression(random_state=42, max_iter=1000)
                        else:
                            custom_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
                        
                        custom_model.fit(X_train_processed, y_train)
                        
                        y_pred = custom_model.predict(X_test_processed)
                        
                        st.session_state.custom_model = custom_model
                        st.session_state.custom_scalers = (custom_scaler_amount, custom_scaler_time)
                        
                        st.success("Model trained successfully!")
                        
                        st.markdown("---")
                        st.subheader("Retrained Model Performance")
                        
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        
                        with perf_col1:
                            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
                        with perf_col2:
                            st.metric("Precision", f"{precision_score(y_test, y_pred)*100:.2f}%")
                        with perf_col3:
                            st.metric("Recall", f"{recall_score(y_test, y_pred)*100:.2f}%")
                        with perf_col4:
                            st.metric("F1 Score", f"{f1_score(y_test, y_pred)*100:.2f}%")
                        
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                   xticklabels=['Non-Fraud', 'Fraud'],
                                   yticklabels=['Non-Fraud', 'Fraud'])
                        ax_cm.set_xlabel('Predicted')
                        ax_cm.set_ylabel('Actual')
                        ax_cm.set_title(f'Confusion Matrix - Retrained {retrain_model_type}')
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                        plt.close()
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab7:
    st.header("Sample Data from Dataset")
    
    sample_type = st.radio(
        "Select sample type:",
        ["Mixed", "Fraud Only", "Non-Fraud Only"],
        horizontal=True
    )
    
    if sample_type == "Mixed":
        sample_data = balanced_df.sample(n=10, random_state=42)
    elif sample_type == "Fraud Only":
        sample_data = balanced_df[balanced_df['Class'] == 1].head(10)
    else:
        sample_data = balanced_df[balanced_df['Class'] == 0].head(10)
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Dataset Statistics (Balanced)")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric("Total Transactions", f"{len(balanced_df):,}")
    
    with stat_col2:
        fraud_count = len(balanced_df[balanced_df['Class'] == 1])
        st.metric("Fraud Cases", f"{fraud_count:,}")
    
    with stat_col3:
        non_fraud_count = len(balanced_df[balanced_df['Class'] == 0])
        st.metric("Legitimate Cases", f"{non_fraud_count:,}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Credit Card Fraud Detection System | Built with Streamlit, Scikit-learn & XGBoost"
    "</div>",
    unsafe_allow_html=True
)

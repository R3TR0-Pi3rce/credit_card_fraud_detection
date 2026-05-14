import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load Models
@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "xgboost.pkl")
    scaler_path = os.path.join("models", "robust_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_assets()

# UI Sidebar
st.sidebar.title("💳 Fraud Detection Panel")
menu = st.sidebar.radio("Navigation", ["Overview & Analytics", "Real-Time Prediction", "Batch Prediction", "Model Comparison"])

if menu == "Overview & Analytics":
    st.title("Dataset Overview & Analytics")
    st.write("Explore the highly imbalanced European Credit Card Fraud Dataset.")
    
    data_path = os.path.join("data", "creditcard.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class Distribution")
            class_counts = df['Class'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
            ax.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Transaction Amount Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df[df['Amount'] < 500]['Amount'], bins=50, kde=True, ax=ax2)
            st.pyplot(fig2)
            
        st.write("### Sample Data")
        st.dataframe(df.head(10))
    else:
        st.warning("Dataset `creditcard.csv` not found in `data/` folder.")

elif menu == "Real-Time Prediction":
    st.title("Real-Time Fraud Detection")
    st.write("Enter transaction details below to predict fraud likelihood.")
    
    if model is None or scaler is None:
        st.error("Model or Scaler not found. Please train models first.")
    else:
        # We will use sliders/number inputs for the 30 features
        st.write("### Transaction Features")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: time_val = st.number_input("Time", value=0.0)
        with col2: amount_val = st.number_input("Amount", value=0.0)
        
        st.write("**PCA Features (V1 - V28)**")
        v_features = []
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i-1) % 4]:
                v_features.append(st.number_input(f"V{i}", value=0.0))
        
        if st.button("Predict"):
            input_data = [time_val, amount_val] + v_features
            df_input = pd.DataFrame([input_data], columns=['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)])
            
            # Dynamically scale features
            expected_features = getattr(model, 'feature_names_in_', None)
            
            features_to_scale = []
            if 'Amount' in df_input.columns: features_to_scale.append('Amount')
            if expected_features is not None and 'scaled_time' not in expected_features:
                df_input.drop('Time', axis=1, inplace=True, errors='ignore')
            elif 'Time' in df_input.columns:
                features_to_scale.append('Time')
                
            if features_to_scale:
                scaled = scaler.transform(df_input[features_to_scale])
                df_input.drop(features_to_scale, axis=1, inplace=True)
                for i, col in enumerate(features_to_scale):
                    df_input.insert(i, f'scaled_{col.lower()}', scaled[:, i])
            
            # Reorder
            if expected_features is not None:
                df_input = df_input[expected_features]
            else:
                cols = df_input.columns.tolist()
                cols = [c for c in cols if 'scaled' in c] + [f'V{i}' for i in range(1, 29)]
                df_input = df_input[cols]
            
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]
            
            st.write("---")
            if prediction == 1:
                st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED!** (Confidence: {probability*100:.2f}%)")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION** (Confidence: {(1-probability)*100:.2f}%)")

elif menu == "Batch Prediction":
    st.title("Batch Prediction")
    st.write("Upload a CSV file with transactions to identify frauds.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        if model is None or scaler is None:
            st.error("Model or Scaler not found.")
        else:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} transactions.")
            
            if 'Class' in df.columns:
                df_eval = df.drop('Class', axis=1)
            else:
                df_eval = df.copy()
            
            try:
                # Scale
                expected_features = getattr(model, 'feature_names_in_', None)
                features_to_scale = []
                if 'Amount' in df_eval.columns: features_to_scale.append('Amount')
                if expected_features is not None and 'scaled_time' not in expected_features:
                    df_eval.drop('Time', axis=1, inplace=True, errors='ignore')
                elif 'Time' in df_eval.columns:
                    features_to_scale.append('Time')
                    
                if features_to_scale:
                    scaled = scaler.transform(df_eval[features_to_scale])
                    df_eval.drop(features_to_scale, axis=1, inplace=True)
                    for i, col in enumerate(features_to_scale):
                        df_eval.insert(i, f'scaled_{col.lower()}', scaled[:, i])
                
                if expected_features is not None:
                    df_eval = df_eval[expected_features]
                else:
                    cols = [c for c in df_eval.columns if 'scaled' in c] + [f'V{i}' for i in range(1, 29)]
                    df_eval = df_eval[cols]
                
                predictions = model.predict(df_eval)
                df['Prediction'] = predictions
                df['Status'] = df['Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Legitimate')
                
                st.write("### Results")
                fraud_count = df['Prediction'].sum()
                st.warning(f"Found {fraud_count} fraudulent transactions out of {len(df)}.")
                
                st.dataframe(df[['Time', 'Amount', 'Status']].head(50))
            except Exception as e:
                st.error(f"Error processing file. Please ensure it matches the Kaggle dataset schema. Details: {e}")

elif menu == "Model Comparison":
    st.title("Model Comparison & Metrics")
    st.write("Evaluation metrics generated by the `evaluate.py` script.")
    
    reports_dir = "reports"
    if os.path.exists(reports_dir):
        images = [f for f in os.listdir(reports_dir) if f.endswith('.png')]
        if len(images) == 0:
            st.info("No evaluation reports found. Run `evaluate.py` to generate them.")
        else:
            # Group by model
            models = ["logistic_regression", "random_forest", "xgboost", "isolation_forest", "autoencoder"]
            for mod in models:
                st.subheader(mod.replace('_', ' ').title())
                col1, col2 = st.columns(2)
                cm_img = os.path.join(reports_dir, f"{mod}_cm.png")
                roc_img = os.path.join(reports_dir, f"{mod}_roc.png")
                
                if mod == "autoencoder":
                    dist_img = os.path.join(reports_dir, "autoencoder_error_dist.png")
                    if os.path.exists(dist_img):
                        st.image(dist_img, caption="Reconstruction Error Distribution", use_column_width=True)
                else:
                    with col1:
                        if os.path.exists(cm_img): st.image(cm_img, caption="Confusion Matrix", use_column_width=True)
                    with col2:
                        if os.path.exists(roc_img): st.image(roc_img, caption="ROC Curve", use_column_width=True)
                st.write("---")
    else:
        st.info("Reports directory not found.")



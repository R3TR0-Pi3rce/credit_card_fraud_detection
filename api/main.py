from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time API for predicting fraudulent credit card transactions.",
    version="1.0.0"
)

# Load the best performing model (XGBoost) and Scaler
MODEL_PATH = os.path.join("models", "xgboost.pkl")
SCALER_PATH = os.path.join("models", "robust_scaler.pkl")

model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logging.info("Model and Scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model or scaler: {e}")

class TransactionInput(BaseModel):
    Time: float = 0.0
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}

@app.get("/health")
def health_check():
    if model is not None and scaler is not None:
        return {"status": "healthy"}
    return {"status": "unhealthy", "reason": "Model or Scaler not loaded"}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded.")
    
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([transaction.dict()])
        
        # Check expected features from model
        expected_features = getattr(model, 'feature_names_in_', None)
        
        # Scale Time and Amount dynamically
        features_to_scale = []
        if 'Amount' in data.columns: features_to_scale.append('Amount')
        # If the model doesn't expect scaled_time, we drop Time
        if expected_features is not None and 'scaled_time' not in expected_features:
            data.drop('Time', axis=1, inplace=True, errors='ignore')
        elif 'Time' in data.columns:
            features_to_scale.append('Time')
            
        if features_to_scale:
            scaled = scaler.transform(data[features_to_scale])
            data.drop(features_to_scale, axis=1, inplace=True)
            for i, col in enumerate(features_to_scale):
                data.insert(i, f'scaled_{col.lower()}', scaled[:, i])
        
        # Reorder to match model expected features if available
        if expected_features is not None:
            data = data[expected_features]

        # Predict
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        result = {
            "prediction": int(prediction),
            "status": "Fraudulent" if prediction == 1 else "Legitimate",
            "confidence": float(probability)
        }
        return result
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

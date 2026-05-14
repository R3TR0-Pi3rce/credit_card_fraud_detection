import os
import joblib
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocessing import load_data, preprocess_data, split_and_balance_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_autoencoder(input_dim):
    """Builds an Autoencoder model for anomaly detection."""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    data_path = os.path.join("data", "creditcard.csv")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}. Please place the Kaggle dataset there.")
        return

    # Load and preprocess
    df = load_data(data_path)
    df, scaler = preprocess_data(df)
    
    # Save the scaler for inference
    joblib.dump(scaler, os.path.join("models", "robust_scaler.pkl"))
    logging.info("Scaler saved to models/robust_scaler.pkl")

    # Split and balance
    X_train, X_test, y_train, y_test = split_and_balance_data(df)
    
    # Save test sets for evaluation script
    X_test.to_csv(os.path.join("data", "X_test.csv"), index=False)
    y_test.to_csv(os.path.join("data", "y_test.csv"), index=False)
    logging.info("Test sets saved for evaluation.")

    # 1. Logistic Regression
    logging.info("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join("models", "logistic_regression.pkl"))

    # 2. Random Forest
    logging.info("Training Random Forest...")
    # Limiting depth and estimators for academic scope/speed, can be increased.
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, os.path.join("models", "random_forest.pkl"))

    # 3. XGBoost
    logging.info("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join("models", "xgboost.pkl"))

    # 4. Isolation Forest
    logging.info("Training Isolation Forest...")
    iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    # Fit only on normal data for isolation forest
    X_train_normal = X_train[y_train == 0]
    iso_model.fit(X_train_normal)
    joblib.dump(iso_model, os.path.join("models", "isolation_forest.pkl"))

    # 5. Autoencoder
    logging.info("Training Autoencoder...")
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train Autoencoder only on normal transactions
    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=15,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    autoencoder.save(os.path.join("models", "autoencoder.h5"))
    logging.info("Autoencoder saved to models/autoencoder.h5")

    logging.info("All models trained and saved successfully.")

if __name__ == "__main__":
    main()

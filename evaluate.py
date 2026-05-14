import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_supervised(y_true, y_pred, y_prob, model_name):
    print(f"\\n{'='*40}\\n{model_name} Evaluation\\n{'='*40}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\\nClassification Report:\\n", classification_report(y_true, y_pred))
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join("reports", f"{model_name.replace(' ', '_').lower()}_cm.png"))
    plt.close()

    # Plot ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join("reports", f"{model_name.replace(' ', '_').lower()}_roc.png"))
        plt.close()

def main():
    X_test_path = os.path.join("data", "X_test.csv")
    y_test_path = os.path.join("data", "y_test.csv")
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        logging.error("Test data not found. Please run train.py first.")
        return

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)['Class'].values

    models = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    # Evaluate Supervised Models
    for name, filename in models.items():
        model_path = os.path.join("models", filename)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            evaluate_supervised(y_test, y_pred, y_prob, name)
        else:
            logging.warning(f"Model {name} not found.")

    # Evaluate Isolation Forest
    iso_path = os.path.join("models", "isolation_forest.pkl")
    if os.path.exists(iso_path):
        iso_model = joblib.load(iso_path)
        y_pred_iso = iso_model.predict(X_test)
        # Isolation Forest returns 1 for normal, -1 for anomaly. Convert to 0 normal, 1 anomaly
        y_pred_iso = np.where(y_pred_iso == 1, 0, 1)
        evaluate_supervised(y_test, y_pred_iso, None, "Isolation Forest")
    
    # Evaluate Autoencoder
    ae_path = os.path.join("models", "autoencoder.h5")
    if os.path.exists(ae_path):
        ae_model = load_model(ae_path)
        reconstructions = ae_model.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        
        # Determine threshold dynamically based on 95th percentile of MSE
        threshold = np.percentile(mse, 95)
        y_pred_ae = (mse > threshold).astype(int)
        evaluate_supervised(y_test, y_pred_ae, mse, "Autoencoder")
        
        # Plot reconstruction error
        plt.figure(figsize=(8,5))
        plt.hist(mse[y_test==0], bins=50, alpha=0.6, color='blue', label='Normal')
        plt.hist(mse[y_test==1], bins=50, alpha=0.6, color='red', label='Fraud')
        plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label='Threshold')
        plt.title('Autoencoder Reconstruction Error')
        plt.legend()
        plt.savefig(os.path.join("reports", "autoencoder_error_dist.png"))
        plt.close()

if __name__ == "__main__":
    main()

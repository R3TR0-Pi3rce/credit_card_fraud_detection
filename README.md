# Credit Card Fraud Detection Using Machine Learning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost%20%7C%20TensorFlow-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📌 Abstract
Financial fraud causes massive monetary losses worldwide. Traditional fraud detection systems struggle to identify rapidly evolving fraudulent patterns. This project builds an intelligent fraud detection system that analyzes transaction patterns and accurately classifies transactions as legitimate or fraudulent using multiple Machine Learning and Deep Learning algorithms.

## 📖 Introduction
Credit card fraud is one of the most critical issues in the financial sector. This project utilizes the **European Credit Card Fraud Dataset**, which is highly imbalanced, containing only a fraction of fraudulent transactions compared to legitimate ones. We address this using SMOTE (Synthetic Minority Over-sampling Technique) and compare various models to find the most effective approach.

## 🗂️ Dataset Information
- **Source**: Kaggle (European Credit Card Fraud Dataset)
- **Features**: Time, Amount, V1-V28 (PCA transformed), Class (0 for Legitimate, 1 for Fraud)

> Note: Ensure you place the `creditcard.csv` dataset inside the `data/` folder before training.

## 🏗️ System Architecture
1. **Data Preprocessing**: Scaling `Time` and `Amount`, handling outliers.
2. **Imbalance Handling**: Applying SMOTE to balance the classes.
3. **Model Training**: Logistic Regression, Random Forest, XGBoost, Isolation Forest, and Deep Autoencoders.
4. **Evaluation**: ROC-AUC, F1-Score, Precision, Recall, and Confusion Matrices.
5. **Deployment**: Real-time API built with FastAPI and an interactive dashboard using Streamlit.

## 🚀 Installation & Setup

### 1. Clone the repository and navigate to the folder
```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Place the `creditcard.csv` inside the `data/` folder.

### 4. Train the Models
Run the training pipeline to train models and generate evaluation metrics:
```bash
python train.py
```
*Trained models will be saved in `models/` and evaluation graphs in `reports/`.*

### 5. Run the FastAPI Server
The API will serve predictions based on the best-trained model (default: XGBoost).
```bash
uvicorn api.main:app --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### 6. Run the Streamlit Dashboard
The Streamlit app provides an intuitive UI for real-time predictions and analytics.
```bash
streamlit run dashboard.py
```

## 📊 Models Evaluated
- **Logistic Regression**: Baseline model.
- **Random Forest**: Ensemble method for non-linear patterns.
- **XGBoost**: High-performance gradient boosting tree.
- **Isolation Forest**: Unsupervised anomaly detection.
- **Autoencoder**: Deep learning reconstruction-error based anomaly detection.

## 🔮 Future Enhancements
- Implement real-time streaming data processing using Apache Kafka.
- Further refine Deep Learning models with Recurrent Neural Networks (LSTMs) for sequence analysis.
- Advanced Explainable AI (XAI) using full SHAP integration in the dashboard.

## 👨‍💻 Author
- Your Name / Organization

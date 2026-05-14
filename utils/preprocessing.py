import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset from the given filepath."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise e

def preprocess_data(df: pd.DataFrame):
    """Preprocesses the dataset: removes duplicates, scales Time and Amount."""
    # Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    # Drop 'id' if it exists (Kaggle 2023 dataset version)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Scale 'Time' and 'Amount' using RobustScaler (less prone to outliers)
    rob_scaler = RobustScaler()
    
    features_to_scale = []
    if 'Amount' in df.columns: features_to_scale.append('Amount')
    if 'Time' in df.columns: features_to_scale.append('Time')
    
    if features_to_scale:
        scaled_features = rob_scaler.fit_transform(df[features_to_scale])
        df.drop(features_to_scale, axis=1, inplace=True)
        for i, col in enumerate(features_to_scale):
            df.insert(i, f'scaled_{col.lower()}', scaled_features[:, i])
    
    logging.info("Data preprocessing completed: 'Time' and 'Amount' scaled if present.")
    return df, rob_scaler

def split_and_balance_data(df: pd.DataFrame):
    """Splits the data into train and test sets, and applies SMOTE on training data."""
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Splitting before SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info(f"Data split: Training size={X_train.shape[0]}, Test size={X_test.shape[0]}")
    
    # Apply SMOTE only on training data
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    logging.info(f"SMOTE applied. New training size={X_train_sm.shape[0]}, Class distribution:\\n{y_train_sm.value_counts()}")
    
    return X_train_sm, X_test, y_train_sm, y_test

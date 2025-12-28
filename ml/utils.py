"""
ML Utilities
============
Helper functions for data splitting, seeding, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score

SEED = 808

def get_splitter(df, target_col, test_size=0.2, val_size=0.2, random_state=SEED):
    """
    Splits data into Train, Validation, and Test sets.
    Default strategy: 60/20/20 split if test=0.2, val=0.2 (relative to total)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: Separating Test set
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Adjust validation size to be relative to the original dataset
    # If we want 20% of TOTAL for Val, and we have 80% left in Temp:
    # 0.2 / 0.8 = 0.25 (25% of the remaining temp data)
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state
    )
    
    print(f"Data Split Summary (Seed {random_state}):")
    print(f"  Train: {X_train.shape[0]} ({X_train.shape[0]/len(df):.1%})")
    print(f"  Val:   {X_val.shape[0]} ({X_val.shape[0]/len(df):.1%})")
    print(f"  Test:  {X_test.shape[0]} ({X_test.shape[0]/len(df):.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_classifier(y_true, y_pred, y_prob=None, model_name="Model"):
    """Returns a dictionary of classification metrics."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics["ROC AUC"] = None
    return metrics

def evaluate_regressor(y_true, y_pred, model_name="Model"):
    """Returns a dictionary of regression metrics."""
    return {
        "Model": model_name,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2 Score": r2_score(y_true, y_pred)
    }

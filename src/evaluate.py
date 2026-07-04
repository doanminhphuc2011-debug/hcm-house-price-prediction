"""
evaluate.py
------------
Extra evaluation metrics (R2, MAE, MSE) alongside RMSLE, and a small
helper to sweep XGBoost's learning_rate for the "additional metrics"
comparison in the notebook.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .models import RANDOM_STATE, rmsle


def full_metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
    }


def learning_rate_sweep(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    learning_rates=(0.01, 0.05, 0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """Train XGBoost at several learning rates and report R2/MAE/MSE/RMSLE
    on both train and test sets (reproduces notebook Section 9.1)."""
    rows = []
    for lr in learning_rates:
        model = XGBRegressor(
            learning_rate=lr, n_estimators=200, min_child_weight=2,
            subsample=1, colsample_bytree=0.8, reg_lambda=0.45, gamma=0.5,
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        )
        model.fit(X_train, y_train)
        m_train = full_metrics(y_train, model.predict(X_train))
        m_test = full_metrics(y_test, model.predict(X_test))
        rows.append({
            "learning_rate": lr,
            "r2_train": m_train["r2"], "r2_test": m_test["r2"],
            "mae_train": m_train["mae"], "mae_test": m_test["mae"],
            "mse_train": m_train["mse"], "mse_test": m_test["mse"],
            "rmsle_train": m_train["rmsle"], "rmsle_test": m_test["rmsle"],
        })
    return pd.DataFrame(rows)

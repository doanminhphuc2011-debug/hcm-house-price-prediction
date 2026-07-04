"""
models.py
----------
Model factories and training/evaluation helpers for the five models
benchmarked in the paper: Random Forest, XGBoost, LightGBM, Hybrid
Regression (equal-weight average), and Stacked Generalization.

The XGBoost hyperparameters are defined ONCE here (`XGB_TUNED_PARAMS`)
and reused everywhere a "tuned" XGBoost is needed, instead of being
retyped by hand in different notebook cells (which previously caused a
mismatch: the paper reports learning_rate=0.1 but one retraining cell
used learning_rate=0.05).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Hyperparameters (kept in one place -- see module docstring)
# ----------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=900, max_depth=20, min_samples_split=10,
    n_jobs=-1, random_state=RANDOM_STATE,
)

XGB_BASELINE_PARAMS = dict(
    learning_rate=0.1, n_estimators=200, min_child_weight=2,
    subsample=1, colsample_bytree=0.8, reg_lambda=0.45, gamma=0.5,
    n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
)

# Result of GridSearchCV in the paper (Sec. 3.2): this is the single
# source of truth for the "tuned" XGBoost used at every later stage.
XGB_TUNED_PARAMS = dict(
    learning_rate=0.1, n_estimators=400, max_depth=6,
    min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=0.45, gamma=0.5,
    n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
)

LGBM_PARAMS = dict(
    learning_rate=0.15, n_estimators=64, num_leaves=36,
    min_child_weight=2, colsample_bytree=0.8, reg_lambda=0.40,
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)

XGB_GRID = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6],
}


def rmsle(y_true, y_pred) -> float:
    """RMSLE on already-log1p-transformed targets is just RMSE."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def make_base_models() -> dict:
    """Fresh, unfitted instances of the three base models."""
    return {
        "Random Forest": RandomForestRegressor(**RF_PARAMS),
        "XGBoost": XGBRegressor(**XGB_BASELINE_PARAMS),
        "LightGBM": LGBMRegressor(**LGBM_PARAMS),
    }


def fit_eval(name: str, model, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit a model and return a result dict (matches the notebook's schema)."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    return {
        "model": model,
        "name": name,
        "train_rmsle": rmsle(y_tr, model.predict(X_tr)),
        "test_rmsle": rmsle(y_te, model.predict(X_te)),
        "time": round(time.time() - t0, 1),
    }


def blend(models: list, X: pd.DataFrame) -> np.ndarray:
    """Equal-weight average of predictions ('Hybrid Regression')."""
    return np.mean([m.predict(X) for m in models], axis=0)


def stacked_generalization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Two-level stacking: RF + LightGBM (level-1, out-of-fold) feeding a
    tuned XGBoost meta-learner (level-2). Prevents the meta-learner from
    seeing predictions on data its base learners were trained on.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    meta_train = np.zeros((len(X_train), 2))
    meta_test = np.zeros((len(X_test), 2))

    t0 = time.time()
    for tr_idx, val_idx in kf.split(X_train):
        X_tr_f, X_val_f = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr_f = y_train.iloc[tr_idx]

        base_learners = [
            RandomForestRegressor(**RF_PARAMS),
            LGBMRegressor(**LGBM_PARAMS),
        ]
        for i, base in enumerate(base_learners):
            base.fit(X_tr_f, y_tr_f)
            meta_train[val_idx, i] = base.predict(X_val_f)
            meta_test[:, i] += base.predict(X_test) / n_splits

    meta_model = XGBRegressor(**XGB_TUNED_PARAMS)
    meta_model.fit(meta_train, y_train)

    return {
        "model": meta_model,
        "name": "Stacked Generalization",
        "train_rmsle": rmsle(y_train, meta_model.predict(meta_train)),
        "test_rmsle": rmsle(y_test, meta_model.predict(meta_test)),
        "time": round(time.time() - t0, 1),
    }


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> GridSearchCV:
    """GridSearchCV over XGB_GRID. Kept for reproducibility / experimentation;
    the winning configuration is already hardcoded as XGB_TUNED_PARAMS so
    downstream code does not depend on rerunning this (slow) search.
    """
    from sklearn.metrics import make_scorer

    rmsle_scorer = make_scorer(lambda y, p: -rmsle(y, p), greater_is_better=True)
    grid = GridSearchCV(
        XGBRegressor(
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=0.45, gamma=0.5, n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=0,
        ),
        XGB_GRID,
        cv=cv,
        scoring=rmsle_scorer,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


def run_full_benchmark(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """Train all 5 models and return a results table sorted by test RMSLE.

    XGBoost is trained with XGB_TUNED_PARAMS directly (skipping the separate
    'baseline XGBoost' step) so results are consistent with the paper's
    headline numbers. Use `tune_xgboost` separately if you want to
    reproduce the GridSearchCV search itself.
    """
    models = make_base_models()
    models["XGBoost"] = XGBRegressor(**XGB_TUNED_PARAMS)

    results = [
        fit_eval(name, model, X_train, y_train, X_test, y_test)
        for name, model in models.items()
    ]

    fitted = {r["name"]: r["model"] for r in results}
    results.append({
        "model": None,
        "name": "Hybrid Regression",
        "train_rmsle": rmsle(
            y_train, blend(list(fitted.values()), X_train)
        ),
        "test_rmsle": rmsle(
            y_test, blend(list(fitted.values()), X_test)
        ),
        "time": 0.0,
    })

    results.append(stacked_generalization(X_train, y_train, X_test, y_test))

    results_df = (
        pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
        .sort_values("test_rmsle")
        .reset_index(drop=True)
    )
    return results_df, fitted

"""
features.py
------------
Feature construction for the modelling stage.

Both `target_encode` and `community_average_encode` are written so that
the encoding statistic is computed **exclusively from the training
fold** and then applied to the test fold. This is the fix for the data
leakage bug in the original notebook, where `communityAverage` was
computed on the full dataset (train + test combined) before splitting.
"""

from __future__ import annotations

import pandas as pd

BASE_FEATURES = [
    "Acreage", "Bedrooms", "Floors", "Parking",
    "width", "length", "frontage_ratio",
    "District", "Ward",
]

FEATURES_WITH_COMMUNITY_AVG = BASE_FEATURES + ["communityAverage"]


def target_encode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    col: str,
    y_train: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Encode a categorical column as the median target value per category,
    fit on the training fold only.

    Returns (encoded_train, encoded_test, mapping).
    Unseen categories in the test fold fall back to the training global median.
    """
    mapping = X_train.join(y_train).groupby(col)[y_train.name].median()
    global_median = y_train.median()
    enc_train = X_train[col].map(mapping).fillna(global_median)
    enc_test = X_test[col].map(mapping).fillna(global_median)
    return enc_train, enc_test, mapping


def community_average_encode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    price_per_m2_train: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Leak-free version of the `communityAverage` feature.

    communityAverage = median price-per-m2 of a Ward, computed only from
    the training rows. This must be called AFTER train_test_split, using
    only `price_per_m2_train` (aligned with X_train's index).

    Returns (train_values, test_values, mapping) where mapping can later
    be reused at inference time (see predict.py).
    """
    mapping = (
        X_train.join(price_per_m2_train.rename("price_per_m2"))
        .groupby("Ward")["price_per_m2"]
        .median()
    )
    global_median = price_per_m2_train.median()
    train_values = X_train["Ward"].map(mapping).fillna(global_median)
    test_values = X_test["Ward"].map(mapping).fillna(global_median)
    return train_values, test_values, mapping


def build_encoded_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    price_per_m2_train: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply target encoding to District/Ward (and optionally
    communityAverage) and return model-ready train/test feature frames.

    `price_per_m2_train` should be provided (aligned to X_train's index)
    only when `communityAverage` is part of X_train.columns.

    Returns (X_train_final, X_test_final, mappings) where `mappings` is a
    dict of {'district': Series, 'ward': Series, 'community_avg': Series|None}
    to be reused for inference.
    """
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()

    X_train_final["District_enc"], X_test_final["District_enc"], district_map = (
        target_encode(X_train, X_test, "District", y_train)
    )
    X_train_final["Ward_enc"], X_test_final["Ward_enc"], ward_map = target_encode(
        X_train, X_test, "Ward", y_train
    )

    community_map = None
    if "communityAverage" in X_train.columns:
        if price_per_m2_train is None:
            raise ValueError(
                "price_per_m2_train is required when communityAverage is a feature"
            )
        # Recompute communityAverage in a leak-free way (train-fold only),
        # overwriting any value that may have been pre-computed on the full data.
        train_vals, test_vals, community_map = community_average_encode(
            X_train, X_test, price_per_m2_train
        )
        X_train_final["communityAverage"] = train_vals
        X_test_final["communityAverage"] = test_vals

    X_train_final = X_train_final.drop(columns=["District", "Ward"])
    X_test_final = X_test_final.drop(columns=["District", "Ward"])

    mappings = {
        "district": district_map,
        "ward": ward_map,
        "community_avg": community_map,
    }
    return X_train_final, X_test_final, mappings

"""
train.py
---------
End-to-end training entry point:

    python train.py

Steps:
  1. Load + clean the raw dataset (src/data_processing.py)
  2. Split train/test, target-encode District/Ward (src/features.py)
  3. Benchmark 5 models (RF, XGBoost, LightGBM, Hybrid, Stacked) with the
     9 baseline features (src/models.py)
  4. Rebuild the feature set with a *leak-free* `communityAverage` and
     retrain the top 3 models
  5. Persist the final models + encoding maps to models/ with joblib,
     and write result tables to outputs/

Run `python train.py --help` for options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data_processing import clean, load_raw
from src.evaluate import learning_rate_sweep
from src.features import BASE_FEATURES, build_encoded_features
from src.models import XGB_TUNED_PARAMS, blend, rmsle, run_full_benchmark

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "data_nha_mat_dat_gop.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "houses_hcm_clean.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Train the HCM housing price models.")
    p.add_argument("--raw-path", type=Path, default=RAW_PATH)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--skip-lr-sweep", action="store_true",
                    help="skip the learning-rate sweep (saves time)")
    return p.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. load & clean
    print(f"loading raw data from {args.raw_path} ...")
    df_raw = load_raw(args.raw_path)
    df = clean(df_raw)
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"{len(df_raw):,} -> {len(df):,} rows after cleaning "
          f"(saved to {CLEAN_PATH})")

    # 2. baseline split + encoding (9 features)
    X = df[BASE_FEATURES].copy()
    y = df["log_price"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    X_train_enc, X_test_enc, base_maps = build_encoded_features(X_train, X_test, y_train)

    print("\n=== Baseline benchmark (9 features) ===")
    results_df, fitted_base = run_full_benchmark(X_train_enc, y_train, X_test_enc, y_test)
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUTS_DIR / "results_baseline.csv", index=False)

    # 3. learning-rate sweep (optional)
    if not args.skip_lr_sweep:
        print("\n=== XGBoost learning-rate sweep ===")
        lr_df = learning_rate_sweep(X_train_enc, y_train, X_test_enc, y_test)
        print(lr_df.round(5).to_string(index=False))
        lr_df.to_csv(OUTPUTS_DIR / "learning_rate_sweep.csv", index=False)

    # 4. communityAverage feature set (leak-free) 
    # Start from the 9 base features; communityAverage is filled in with a
    # placeholder here and then recomputed for real *after* the split,
    # using only the training fold's price_per_m2 (see build_encoded_features).
    X2 = df[BASE_FEATURES].copy()
    X2["communityAverage"] = 0.0  # placeholder, overwritten below
    y2 = df["log_price"].copy()
    price_per_m2 = df["price_per_m2"].copy()

    X2_train, X2_test, y2_train, y2_test, ppm2_train, _ = train_test_split(
        X2, y2, price_per_m2, test_size=args.test_size, random_state=args.random_state
    )
    X2_train_enc, X2_test_enc, comm_maps = build_encoded_features(
        X2_train, X2_test, y2_train, price_per_m2_train=ppm2_train
    )

    print("\n=== With leak-free communityAverage (10 features) ===")
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from src.models import RF_PARAMS, LGBM_PARAMS

    rf2 = RandomForestRegressor(**RF_PARAMS)
    xgb2 = XGBRegressor(**XGB_TUNED_PARAMS)
    lgbm2 = LGBMRegressor(**LGBM_PARAMS)

    for name, model in [("Random Forest", rf2), ("XGBoost", xgb2), ("LightGBM", lgbm2)]:
        model.fit(X2_train_enc, y2_train)

    comm_scores = {
        "XGBoost": rmsle(y2_test, xgb2.predict(X2_test_enc)),
        "LightGBM": rmsle(y2_test, lgbm2.predict(X2_test_enc)),
        "Hybrid Regression": rmsle(y2_test, blend([rf2, xgb2, lgbm2], X2_test_enc)),
    }
    baseline_scores = results_df.set_index("name")["test_rmsle"]
    cmp_df = pd.DataFrame({
        "before (9 feat.)": {k: baseline_scores[k] for k in comm_scores},
        "after (10 feat.)": comm_scores,
    })
    cmp_df["delta"] = cmp_df["after (10 feat.)"] - cmp_df["before (9 feat.)"]
    print(cmp_df.round(5).to_string())
    cmp_df.to_csv(OUTPUTS_DIR / "results_community_average.csv")

    # 5. persist models + mappings for inference
    joblib.dump(rf2, MODELS_DIR / "rf_community.pkl")
    joblib.dump(xgb2, MODELS_DIR / "xgb_community.pkl")
    joblib.dump(lgbm2, MODELS_DIR / "lgbm_community.pkl")
    joblib.dump(comm_maps, MODELS_DIR / "encoding_maps.pkl")
    joblib.dump(
        {"global_log_price_median": float(y2_train.median()),
         "global_price_per_m2_median": float(ppm2_train.median())},
        MODELS_DIR / "global_medians.pkl",
    )

    print(f"\nmodels + encoding maps saved to {MODELS_DIR}/")
    print("done.")


if __name__ == "__main__":
    main()

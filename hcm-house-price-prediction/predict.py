"""
predict.py
-----------
Load the persisted models + encoding maps from models/ (produced by
train.py) and predict a price for a single property, without needing to
retrain anything.

CLI usage:
    python predict.py --acreage 50 --bedrooms 4 --floors 6 --parking 1 \
        --width 5 --length 10 --district "Phu Tho Hoa" --ward "Tan Phu"

Or import `predict_price(...)` directly from another script/notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import blend

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def _load_artifacts():
    rf = joblib.load(MODELS_DIR / "rf_community.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_community.pkl")
    lgbm = joblib.load(MODELS_DIR / "lgbm_community.pkl")
    maps = joblib.load(MODELS_DIR / "encoding_maps.pkl")
    medians = joblib.load(MODELS_DIR / "global_medians.pkl")
    return rf, xgb, lgbm, maps, medians


def predict_price(
    acreage: float, bedrooms: int, floors: int, parking: int,
    width: float, length: float, district: str, ward: str,
) -> float:
    """Predict price (VND) for one property using the Hybrid Regression
    ensemble (RF + XGBoost + LightGBM, equal weight), matching the
    approach used in the paper's demo section.
    """
    rf, xgb, lgbm, maps, medians = _load_artifacts()

    district_map = maps["district"]
    ward_map = maps["ward"]
    community_map = maps["community_avg"]

    x = pd.DataFrame([{
        "Acreage": acreage,
        "Bedrooms": bedrooms,
        "Floors": floors,
        "Parking": parking,
        "width": width,
        "length": length,
        "frontage_ratio": width / length,
        "communityAverage": community_map.get(
            ward, medians["global_price_per_m2_median"]
        ),
        "District_enc": district_map.get(
            district, medians["global_log_price_median"]
        ),
        "Ward_enc": ward_map.get(
            ward, medians["global_log_price_median"]
        ),
    }])

    log_price = blend([rf, xgb, lgbm], x)[0]
    return float(np.expm1(log_price))


def _parse_args():
    p = argparse.ArgumentParser(description="Predict HCM house price.")
    p.add_argument("--acreage", type=float, required=True)
    p.add_argument("--bedrooms", type=int, required=True)
    p.add_argument("--floors", type=int, required=True)
    p.add_argument("--parking", type=int, choices=[0, 1], required=True)
    p.add_argument("--width", type=float, required=True)
    p.add_argument("--length", type=float, required=True)
    p.add_argument("--district", type=str, required=True)
    p.add_argument("--ward", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    price = predict_price(
        args.acreage, args.bedrooms, args.floors, args.parking,
        args.width, args.length, args.district, args.ward,
    )
    print(f"Predicted price: {price / 1e9:.2f} ty VND ({price:,.0f} VND)")

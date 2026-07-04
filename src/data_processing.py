"""
data_processing.py
-------------------
Load raw HCM housing listings and apply the cleaning / preprocessing
pipeline described in the paper (Section 2.3):

    1. remove records with Price < 500,000,000 VND or missing Area
    2. impute missing Floors with the column median
    3. clip Price to the 1st-99th percentile range
    4. parse the free-text `Area` column (e.g. "6x30m") into width/length
    5. derive frontage_ratio = width / length
    6. extract Ward from the Vietnamese Address string
    7. add log_price and price_per_m2

All functions are pure (no hidden global state) so they can be unit
tested and reused from both the notebook and `predict.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Regex patterns compiled once at import time
# ----------------------------------------------------------------------
_AREA_PATTERN = re.compile(r"(\d+\.?\d*)\s*[xX\u00d7]\s*(\d+\.?\d*)")
_WARD_PATTERN = re.compile(r"(Ph\u01b0\u1eddng|X\u00e3|Th\u1ecb tr\u1ea5n)\s+([\w\s]+?),")

MIN_VALID_PRICE = 500_000_000  # VND
WIDTH_RANGE = (1, 50)          # metres
LENGTH_RANGE = (1, 100)        # metres


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw scraped listings CSV."""
    return pd.read_csv(path)


def parse_area(value) -> tuple[float | None, float | None]:
    """Parse a dimension string like '6x30m' into (width, length).

    Returns (None, None) if the string cannot be parsed.
    """
    match = _AREA_PATTERN.search(str(value))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def extract_ward(address) -> str | None:
    """Extract 'Ph\u01b0\u1eddng X' / 'X\u00e3 Y' / 'Th\u1ecb tr\u1ea5n Z' from a Vietnamese address string."""
    match = _WARD_PATTERN.search(str(address))
    if not match:
        return None
    return f"{match.group(1)} {match.group(2).strip()}"


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to the raw dataframe.

    Returns a new, cleaned dataframe. Does not mutate the input.
    """
    df = df_raw.copy()

    # 1. Remove clearly-invalid prices and rows with no Area string
    df = df[df["Price(VND)"] >= MIN_VALID_PRICE]
    df = df.dropna(subset=["Area"])

    # 2. Impute missing Floors with the column median
    df["Floors"] = df["Floors"].fillna(df["Floors"].median())

    # 3. Clip Price to the 1st-99th percentile range
    p01, p99 = df["Price(VND)"].quantile([0.01, 0.99])
    df = df[df["Price(VND)"].between(p01, p99)]

    # 4. Parse Area -> width / length, drop unparsable / out-of-range rows
    parsed = df["Area"].apply(parse_area)
    df["width"] = parsed.apply(lambda t: t[0])
    df["length"] = parsed.apply(lambda t: t[1])
    df = df.dropna(subset=["width", "length"])
    df = df[
        df["width"].between(*WIDTH_RANGE) & df["length"].between(*LENGTH_RANGE)
    ]
    df["frontage_ratio"] = df["width"] / df["length"]

    # 5. Extract Ward from Address, drop rows where it can't be found
    df["Ward"] = df["Address"].apply(extract_ward)
    df = df.dropna(subset=["Ward"])

    # 6. Target + auxiliary feature
    df["log_price"] = np.log1p(df["Price(VND)"])
    df["price_per_m2"] = df["Price(VND)"] / df["Acreage"]

    return df.reset_index(drop=True)


def reduction_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Small helper reproducing Table 2 of the paper (rows removed per stage)."""
    return pd.DataFrame(
        {
            "stage": ["raw", "clean"],
            "rows": [len(df_raw), len(df_clean)],
        }
    )


if __name__ == "__main__":
    # Quick manual check: `python -m src.data_processing`
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "data_nha_mat_dat_gop.csv"
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "houses_hcm_clean.csv"

    df_raw = load_raw(raw_path)
    df_clean = clean(df_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"{len(df_raw):,} -> {len(df_clean):,} rows")
    print(f"log_price skewness: {df_clean['log_price'].skew():.4f}")
    print(f"saved to: {out_path}")

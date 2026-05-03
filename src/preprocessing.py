"""
preprocessing.py
================
Data cleaning, feature engineering, and sklearn Pipeline construction
for the Hotel Booking Cancellation Prediction project.

Key design decisions
--------------------
* reservation_status & reservation_status_date are DROPPED — they are
  direct data-leakage proxies for the target.
* company is DROPPED — ~94 % of original values were NaN.
* Categorical encoding uses OrdinalEncoder inside a ColumnTransformer
  so the pipeline is fully sklearn-compatible and picklable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ── Month ordering ──────────────────────────────────────────────────────────
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ── Columns to drop immediately (leakage / ID-like / near-zero variance) ───
DROP_COLS = [
    "reservation_status",       # direct leakage
    "reservation_status_date",  # direct leakage
    "company",                  # 94 % originally null
    "arrival_date_week_number", # redundant with month + day
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 – CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a fresh DataFrame."""
    df = df.copy()

    # 1. Remove exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[clean] Removed {before - len(df):,} duplicate rows")

    # 2. Fill nulls ──────────────────────────────────────────────────────────
    df["agent"]    = df["agent"].fillna(0)
    df["company"]  = df["company"].fillna(0)
    df["country"]  = df["country"].fillna("Unknown")
    df["children"] = df["children"].fillna(0)

    # 3. Type conversions (before zero-guest check) ─────────────────────────
    df["children"] = df["children"].astype(int)
    df["agent"]    = df["agent"].astype(int)

    # 4. Remove impossible bookings (zero guests) ───────────────────────────
    zero_guest_mask = (df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0)
    n_removed = zero_guest_mask.sum()
    df = df[~zero_guest_mask].copy()
    print(f"[clean] Removed {n_removed:,} zero-guest rows")

    # 5. Fix categorical "Undefined" values ─────────────────────────────────
    df["meal"]                 = df["meal"].replace("Undefined", "SC")
    df["market_segment"]       = df["market_segment"].replace("Undefined", "Online TA")
    df["distribution_channel"] = df["distribution_channel"].replace("Undefined", "TA/TO")

    # 6. Clip ADR outliers via IQR ──────────────────────────────────────────
    q1, q3 = df["adr"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["adr"] = df["adr"].clip(lower=lb, upper=ub)

    # 7. Convert reservation_status_date ────────────────────────────────────
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"], errors="coerce"
    )

    print(f"[clean] Clean dataset shape: {df.shape}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 – FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model signal."""
    df = df.copy()

    df["total_nights"]   = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"]   = df["adults"] + df["children"] + df["babies"]
    df["room_upgraded"]  = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    df["has_children"]   = ((df["children"] + df["babies"]) > 0).astype(int)
    df["revenue_per_night"] = np.where(
        df["total_nights"] > 0,
        df["adr"] * df["total_nights"],
        df["adr"],
    )
    df["is_long_stay"]   = (df["total_nights"] > 7).astype(int)

    # Month as ordered integer 1-12
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    df["month_num"] = df["arrival_date_month"].cat.codes + 1   # 1-12

    # Season
    season_map = {
        1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
        5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter",
    }
    df["season"] = df["month_num"].map(season_map)

    # Cancellation history ratio
    df["cancel_rate_history"] = (
        df["previous_cancellations"]
        / (df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1)
    )

    print(f"[feature_engineering] New shape: {df.shape}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 – SPLIT & PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_feature_lists(df: pd.DataFrame):
    """Return (numeric_features, categorical_features) after dropping leakage cols."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=["is_canceled", "arrival_date_month",
                           "reservation_status_date"], errors="ignore")

    num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_feats, cat_feats


def build_preprocessor(num_feats: list, cat_feats: list) -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,  num_feats),
        ("cat", categorical_transformer, cat_feats),
    ], remainder="drop")

    return preprocessor


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=["arrival_date_month", "reservation_status_date"],
                 errors="ignore")

    y = df["is_canceled"]
    X = df.drop(columns=["is_canceled"])

    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[prepare_data] Numeric features  : {len(num_feats)}")
    print(f"[prepare_data] Categorical features: {len(cat_feats)}")
    print(f"[prepare_data] Total features    : {len(num_feats) + len(cat_feats)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[prepare_data] Train: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor(num_feats, cat_feats)
    feature_names = num_feats + cat_feats

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    from data_loader import load_raw
    raw   = load_raw()
    clean_df  = clean(raw)
    feat_df   = engineer_features(clean_df)
    X_train, X_test, y_train, y_test, prep, feats = prepare_data(feat_df)
    print("Preprocessor ready. Features:", feats[:5], "...")

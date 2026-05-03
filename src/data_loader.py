"""
data_loader.py
==============
Loads and performs initial validation of the hotel_bookings dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hotel_bookings.csv"


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def basic_report(df: pd.DataFrame) -> None:
    """Print a concise summary of the raw dataset."""
    print("\n===== DATASET SUMMARY =====")
    print(f"Shape          : {df.shape}")
    print(f"Duplicates     : {df.duplicated().sum():,}")
    print(f"Target balance : \n{df['is_canceled'].value_counts(normalize=True).round(3)}")
    print("\nNull counts (only columns with nulls):")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string())
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("===========================\n")


if __name__ == "__main__":
    raw = load_raw()
    basic_report(raw)

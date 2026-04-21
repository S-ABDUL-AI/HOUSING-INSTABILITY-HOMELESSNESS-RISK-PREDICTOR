"""
Load housing / labor-market style features from CSV or generate a reproducible synthetic dataset.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLS = ["city", "median_rent", "median_income", "unemployment_rate", "eviction_rate", "risk_label"]


def make_synthetic_dataset(n_rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Synthetic panel: cities with correlated rent, income, unemployment, eviction, and a risk label.

    Labels follow a simple rule-based mapping so the classifier has learnable signal.
    """
    rng = np.random.default_rng(random_state)
    cities = [
        "Riverside Metro",
        "Lakeshore City",
        "Summit County",
        "Harbor District",
        "Prairie Township",
        "Highland Borough",
        "Oak Valley",
        "Northfield Corridor",
        "Southgate",
        "Cedar Plains",
    ]

    rows = []
    for _ in range(n_rows):
        city = rng.choice(cities)
        median_income = float(rng.uniform(22_000, 95_000))
        median_rent = float(rng.uniform(650, 3_200))
        # Rent burden proxy: rent / (monthly income); higher → stress
        monthly_income = max(median_income / 12.0, 500.0)
        rent_burden = median_rent / monthly_income
        unemployment_rate = float(rng.uniform(0.015, 0.20))
        eviction_rate = float(rng.uniform(0.005, 0.14))

        # Composite stress score → risk bucket (used as training label)
        stress = (
            0.45 * np.clip((rent_burden - 0.22) / 0.35, 0, 1.5)
            + 0.30 * (unemployment_rate / 0.20)
            + 0.25 * (eviction_rate / 0.12)
        )
        if stress > 0.85:
            label = "High"
        elif stress > 0.48:
            label = "Medium"
        else:
            label = "Low"

        rows.append(
            {
                "city": city,
                "median_rent": round(median_rent, 2),
                "median_income": round(median_income, 2),
                "unemployment_rate": round(unemployment_rate, 4),
                "eviction_rate": round(eviction_rate, 4),
                "risk_label": label,
            }
        )

    return pd.DataFrame(rows)


def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV; validates required columns."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Expected: {REQUIRED_COLS}")
    return df[REQUIRED_COLS].copy()


def ensure_dataset(df: Optional[pd.DataFrame], uploaded: Optional[bytes], use_synthetic: bool) -> pd.DataFrame:
    """Resolve dataset from upload, existing frame, or synthetic default."""
    if uploaded:
        return load_csv_bytes(uploaded)
    if df is not None and not use_synthetic:
        return df
    return make_synthetic_dataset()


def filter_by_cities(df: pd.DataFrame, cities: Optional[list]) -> pd.DataFrame:
    if not cities:
        return df
    return df[df["city"].isin(cities)].copy()

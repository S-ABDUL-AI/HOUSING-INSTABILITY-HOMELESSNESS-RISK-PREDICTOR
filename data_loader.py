"""
Load housing / labor-market style features from CSV or generate a reproducible synthetic dataset.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLS = ["city", "median_rent", "median_income", "unemployment_rate", "eviction_rate", "risk_label"]


def stress_score(
    median_rent: float,
    median_income: float,
    unemployment_rate: float,
    eviction_rate: float,
) -> float:
    """Scalar stress index (same rule as synthetic labels) for consistent training labels on hybrid data."""
    monthly_income = max(float(median_income) / 12.0, 500.0)
    rent_burden = float(median_rent) / monthly_income
    ue = float(unemployment_rate)
    ev = float(eviction_rate)
    return (
        0.45 * np.clip((rent_burden - 0.22) / 0.35, 0, 1.5)
        + 0.30 * (ue / 0.20)
        + 0.25 * (ev / 0.12)
    )


def risk_label_from_indicators(
    median_rent: float,
    median_income: float,
    unemployment_rate: float,
    eviction_rate: float,
) -> str:
    stress = stress_score(median_rent, median_income, unemployment_rate, eviction_rate)
    if stress > 0.85:
        return "High"
    if stress > 0.48:
        return "Medium"
    return "Low"


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
        unemployment_rate = float(rng.uniform(0.015, 0.20))
        eviction_rate = float(rng.uniform(0.005, 0.14))

        label = risk_label_from_indicators(median_rent, median_income, unemployment_rate, eviction_rate)

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

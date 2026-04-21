"""
Policy text, causal framing, budget simulation, and executive brief helpers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def recommendation_for_risk(risk: str) -> str:
    """Action-oriented line for policymakers by predicted risk band."""
    if risk == "High":
        return (
            "Increase rent subsidies, expand affordable housing programs, provide emergency assistance."
        )
    if risk == "Medium":
        return "Support job programs, stabilize rent increases."
    return "Monitor housing affordability trends."


def policy_causal_notes(row: pd.Series, ref: pd.DataFrame) -> List[str]:
    """
    Explain likely drivers vs regional reference (medians on the working dataset).

    High median_rent → affordability stress; high unemployment → income instability;
    high eviction_rate → tenure insecurity.
    """
    notes: List[str] = []
    rent_med = ref["median_rent"].median()
    inc_med = ref["median_income"].median()
    ue_med = ref["unemployment_rate"].median()
    ev_med = ref["eviction_rate"].median()

    burden = row["median_rent"] / max(row["median_income"] / 12.0, 1.0)
    burden_ref = rent_med / max(inc_med / 12.0, 1.0)

    if row["median_rent"] > rent_med * 1.12 or burden > burden_ref * 1.1:
        notes.append("Above-median rent / rent burden → affordability pressure.")
    if row["unemployment_rate"] > ue_med * 1.12:
        notes.append("Above-median unemployment → income instability.")
    if row["eviction_rate"] > ev_med * 1.12:
        notes.append("Above-median eviction filings → housing insecurity signal.")

    if not notes:
        notes.append("Drivers near regional medians — continue monitoring leading indicators.")
    return notes


def rank_cities_by_risk(df: pd.DataFrame, pred_col: str = "predicted_risk") -> pd.DataFrame:
    """Priority ranking: score cities by share of High + Medium and mean P(High) if present."""
    work = df.copy()
    score_map = {"Low": 0, "Medium": 2, "High": 5}
    work["_score"] = work[pred_col].map(score_map).fillna(0)
    grp = work.groupby("city", as_index=False).agg(
        rows=("city", "count"),
        risk_index=("_score", "mean"),
        high_share=(pred_col, lambda s: float((s == "High").mean())),
    )
    if "p_high" in work.columns:
        grp = grp.merge(
            work.groupby("city", as_index=False)["p_high"].mean().rename(columns={"p_high": "avg_p_high"}),
            on="city",
            how="left",
        )
    else:
        grp["avg_p_high"] = 0.0
    grp = grp.sort_values(["risk_index", "high_share", "avg_p_high"], ascending=False).reset_index(drop=True)
    grp.insert(0, "priority_rank", range(1, len(grp) + 1))
    return grp


def allocate_budget(budget_millions: float, city_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate allocating a fixed budget across cities, weighted by predicted **High** rows;
    if none, weight by **Medium+High** (elevated_rows).
    """
    if city_summary.empty or budget_millions <= 0:
        return pd.DataFrame(columns=["city", "high_rows", "elevated_rows", "allocation_m", "allocation_pct"])

    out = city_summary.copy()
    w = out["high_rows"].astype(float).clip(lower=0)
    if w.sum() <= 0:
        w = out["elevated_rows"].astype(float).clip(lower=0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=out.index)
    w = w / w.sum()
    out["allocation_m"] = (w * budget_millions).round(3)
    out["allocation_pct"] = (out["allocation_m"] / budget_millions * 100.0).round(2)
    return out


def build_city_high_counts(df: pd.DataFrame, pred_col: str = "predicted_risk") -> pd.DataFrame:
    """Rows per city with predicted High (for budget weights)."""
    high = df[df[pred_col] == "High"].groupby("city").size().rename("high_rows").reset_index()
    all_cities = pd.DataFrame({"city": sorted(df["city"].unique())})
    merged = all_cities.merge(high, on="city", how="left").fillna({"high_rows": 0})
    elevated = (
        df[df[pred_col].isin(["High", "Medium"])]
        .groupby("city")
        .size()
        .rename("elevated_rows")
        .reset_index()
    )
    merged = merged.merge(elevated, on="city", how="left").fillna({"elevated_rows": 0})
    return merged


def executive_policy_brief(
    df: pd.DataFrame,
    accuracy: float,
    pred_col: str,
    top_city: str,
    budget_table: pd.DataFrame,
) -> str:
    """Short, decision-ready summary."""
    n = len(df)
    high_n = int((df[pred_col] == "High").sum())
    med_n = int((df[pred_col] == "Medium").sum())
    low_n = int((df[pred_col] == "Low").sum())
    share_high = 100.0 * high_n / max(n, 1)

    alloc_hint = ""
    if not budget_table.empty and budget_table["allocation_m"].sum() > 0:
        top3 = budget_table.nlargest(3, "allocation_m")["city"].tolist()
        alloc_hint = f"Largest simulated allocations (first pass) concentrate in: {', '.join(top3)}."

    return (
        f"**Scope:** {n:,} sub-regional observations scored with a hold-out accuracy of **{accuracy:.1%}**. "
        f"**Risk mix:** {share_high:.1f}% predicted **High** ({high_n:,}), **Medium** {med_n:,}, **Low** {low_n:,}. "
        f"**Top pressure geography:** **{top_city}** shows the strongest concentration of elevated model risk—"
        f"use it as the anchor for site visits and program targeting. "
        f"{alloc_hint} "
        f"**Decision note:** Model output supports **prioritisation**; pair with local intake data and legal protections before funding moves."
    )


def compare_risk_counts(before: pd.Series, after: pd.Series) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Counts by risk label for before/after bars or tables."""
    return before.value_counts().to_dict(), after.value_counts().to_dict()

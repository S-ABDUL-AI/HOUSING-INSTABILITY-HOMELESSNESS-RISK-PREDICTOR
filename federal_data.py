"""
Hybrid federal data: HUD Fair Market Rents (public ArcGIS layer) + U.S. Census ACS 5-year (CBSA).

- **median_rent**: HUD 2-bedroom FMR (monthly $), published by HUD on ArcGIS Open Data.
- **median_income / unemployment_rate**: Census ACS table B19013 / B23025 at CBSA level.
- **eviction_rate**: Not published uniformly at CBSA in this pipeline; we use a **transparent proxy**
  tied to rent burden and unemployment (see ``eviction_stress_proxy``). For true eviction filings,
  merge jurisdiction-level court or Eviction Lab data offline.

If HUD or Census calls fail, ``app.py`` falls back to ``make_synthetic_dataset()``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from data_loader import REQUIRED_COLS, risk_label_from_indicators

# HUD GIS — Fair Market Rents feature layer (same service documented via ArcGIS item id 12d2516901f947b5bb4da4e780e35f07).
HUD_FMR_QUERY_BASE = (
    "https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/Fair_Market_Rents/FeatureServer/0/query"
)

_CBS_RE = re.compile(r"^METRO(\d{5})M\1$")

_DEFAULT_HEADERS = {
    "User-Agent": "HousingInstabilityRiskPredictor/1.0 (policy dashboard; +https://www.hud.gov)",
    "Accept": "application/json",
}


def parse_cbsa_from_fmr_code(fmr_code: str) -> str | None:
    """CBSA code embedded in HUD FMR_CODE for true MSAs, e.g. METRO10180M10180 -> 10180."""
    if not fmr_code or not isinstance(fmr_code, str):
        return None
    m = _CBS_RE.match(fmr_code.strip())
    return m.group(1) if m else None


def eviction_stress_proxy(median_rent: np.ndarray, median_income: np.ndarray, unemployment_rate: np.ndarray) -> np.ndarray:
    """
    Bounded proxy for housing insecurity / eviction pressure when court-level filings are not joined.

    Rises with rent-to-income stress and local unemployment (stylised, for model continuity only).
    """
    monthly = np.maximum(median_income.astype(float) / 12.0, 500.0)
    burden = median_rent.astype(float) / monthly
    ue = np.clip(unemployment_rate.astype(float), 0.001, 0.35)
    raw = 0.012 + 0.14 * np.clip(burden - 0.26, 0, 1.2) + 0.11 * (ue / 0.12)
    return np.clip(raw, 0.006, 0.22)


def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_DEFAULT_HEADERS)
    return s


def fetch_hud_fmr_msa_panel(session: requests.Session, max_msas: int = 320, timeout: int = 45) -> pd.DataFrame:
    """Pull HUD FMR rows for geographies whose names end with ' MSA' (standard metro FMR areas)."""
    where = "FMR_AREANAME LIKE '% MSA'"
    rows: List[Dict[str, Any]] = []
    offset = 0
    page = 500
    seen: set[str] = set()

    while len(rows) < max_msas:
        params = {
            "where": where,
            "outFields": "FMR_CODE,FMR_AREANAME,FMR_0BDR,FMR_1BDR,FMR_2BDR,FMR_3BDR,FMR_4BDR",
            "returnGeometry": "false",
            "resultRecordCount": str(page),
            "resultOffset": str(offset),
            "orderByFields": "FMR_CODE",
            "f": "json",
        }
        r = session.get(HUD_FMR_QUERY_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        feats = payload.get("features") or []
        if not feats:
            break
        for f in feats:
            a = f.get("attributes") or {}
            code = str(a.get("FMR_CODE") or "")
            cbsa = parse_cbsa_from_fmr_code(code)
            if not cbsa or cbsa in seen:
                continue
            seen.add(cbsa)
            rows.append(
                {
                    "cbsa": cbsa,
                    "city": str(a.get("FMR_AREANAME") or "").strip(),
                    "fmr_2br": float(a.get("FMR_2BDR") or 0),
                }
            )
            if len(rows) >= max_msas:
                break
        if not payload.get("exceededTransferLimit"):
            break
        offset += page

    return pd.DataFrame(rows)


def fetch_acs_cbsa_economics(
    cbsa_codes: List[str],
    year: str = "2022",
    session: requests.Session | None = None,
    chunk_size: int = 35,
    timeout: int = 45,
) -> pd.DataFrame:
    """
    ACS 5-year: median household income (B19013_001E), labour force (B23025_003E), unemployed (B23025_006E).

    Geography slug matches Census Data API examples: ``metropolitan statistical area/micropolitan statistical area``.
    """
    sess = session or _get_session()
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    geo_key = "metropolitan statistical area/micropolitan statistical area"
    key = (os.environ.get("CENSUS_API_KEY") or "").strip()
    out: List[Dict[str, Any]] = []

    for i in range(0, len(cbsa_codes), chunk_size):
        chunk = cbsa_codes[i : i + chunk_size]
        for_code = ",".join(chunk)
        params: Dict[str, str] = {
            "get": "NAME,B19013_001E,B23025_003E,B23025_006E",
            "for": f"{geo_key}:{for_code}",
        }
        if key:
            params["key"] = key
        r = sess.get(base, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 2:
            continue
        header, *body = data
        for row in body:
            rec = dict(zip(header, row))
            cbsa = rec.get(geo_key)
            if not cbsa:
                continue
            try:
                inc = float(rec.get("B19013_001E") or 0)
            except (TypeError, ValueError):
                inc = np.nan
            if inc < 0 or inc > 1_000_000:  # Census suppression sentinel
                inc = np.nan
            try:
                lf = float(rec.get("B23025_003E") or 0)
                unemp = float(rec.get("B23025_006E") or 0)
            except (TypeError, ValueError):
                lf, unemp = 0.0, 0.0
            ue_rate = float(unemp / lf) if lf > 0 else np.nan
            out.append(
                {
                    "cbsa": str(cbsa).zfill(5),
                    "census_name": rec.get("NAME"),
                    "median_income": inc,
                    "unemployment_rate": ue_rate,
                }
            )

    return pd.DataFrame(out)


def build_hybrid_dataset(max_msas: int = 280, acs_year: str = "2022") -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Assemble training frame from HUD FMR + Census ACS. Inner-joins on CBSA so every row has verifiable HUD rent
    and Census labour/income fields where the Census publishes estimates.
    """
    meta: Dict[str, Any] = {"hud_source": HUD_FMR_QUERY_BASE, "acs_dataset": f"ACS {acs_year} 5-year", "errors": []}
    sess = _get_session()

    try:
        hud = fetch_hud_fmr_msa_panel(sess, max_msas=max_msas)
    except Exception as exc:  # noqa: BLE001
        meta["errors"].append(f"hud_fmr:{exc}")
        raise

    if hud.empty:
        meta["errors"].append("hud_empty")
        raise RuntimeError("HUD FMR query returned no MSA rows.")

    meta["hud_msa_rows"] = int(len(hud))
    cbsa_list = [str(c).zfill(5) for c in hud["cbsa"].tolist()]

    try:
        acs = fetch_acs_cbsa_economics(cbsa_list, year=acs_year, session=sess)
    except Exception as exc:  # noqa: BLE001
        meta["errors"].append(f"census_acs:{exc}")
        raise

    meta["acs_rows"] = int(len(acs))
    merged = hud.merge(acs, on="cbsa", how="inner", suffixes=("", "_acs"))
    meta["joined_rows"] = int(len(merged))

    if merged.empty:
        meta["errors"].append("join_empty")
        raise RuntimeError("No overlap between HUD MSAs and Census CBSA codes.")

    merged["median_rent"] = merged["fmr_2br"].astype(float)
    # Fill rare missing income from rent using a neutral national rent-to-income prior (~30%).
    inc = merged["median_income"].astype(float)
    inc_filled = inc.where(inc.notna() & (inc > 0), merged["median_rent"] / 0.30 * 12.0)
    merged["median_income"] = inc_filled
    ue = merged["unemployment_rate"].astype(float)
    ue_filled = ue.where(ue.notna(), 0.055)
    merged["unemployment_rate"] = np.clip(ue_filled, 0.02, 0.22)

    merged["eviction_rate"] = eviction_stress_proxy(
        merged["median_rent"].values,
        merged["median_income"].values,
        merged["unemployment_rate"].values,
    )

    merged["risk_label"] = [
        risk_label_from_indicators(r, i, u, e)
        for r, i, u, e in zip(
            merged["median_rent"],
            merged["median_income"],
            merged["unemployment_rate"],
            merged["eviction_rate"],
        )
    ]

    out = merged[["city", "median_rent", "median_income", "unemployment_rate", "eviction_rate", "risk_label"]].copy()
    out["median_rent"] = out["median_rent"].round(2)
    out["median_income"] = out["median_income"].round(2)
    out["unemployment_rate"] = out["unemployment_rate"].round(4)
    out["eviction_rate"] = out["eviction_rate"].round(4)

    for col in REQUIRED_COLS:
        if col not in out.columns:
            raise RuntimeError(f"internal: missing {col}")

    meta["final_rows"] = int(len(out))
    return out, meta



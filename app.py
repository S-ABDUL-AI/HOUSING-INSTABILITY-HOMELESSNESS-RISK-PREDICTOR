"""
Housing Instability & Homelessness Risk Predictor — Streamlit entrypoint.

Trains a RandomForest on regional housing / labor signals and surfaces policy-ready views:
rankings, budget simulation, counterfactual rent / unemployment levers, and an executive brief.
"""

from __future__ import annotations

from datetime import datetime
import html
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import REQUIRED_COLS, filter_by_cities, load_csv_bytes, make_synthetic_dataset
from federal_data import build_hybrid_dataset
from modeling import adjust_for_simulation, feature_importance_series, predict_dataframe, train_random_forest
from policy import (
    allocate_budget,
    build_city_high_counts,
    compare_risk_counts,
    executive_policy_brief,
    policy_causal_notes,
    rank_cities_by_risk,
    recommendation_for_risk,
)

# ---------------------------------------------------------------------------
# Page & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Housing instability risk predictor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Source+Sans+3:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif !important; }
    .stApp { background: #f1f5f9; }
    .exec-hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e3a5f 55%, #0f172a 100%);
        color: #f8fafc;
        padding: 1.35rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        border: 1px solid #1e293b;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
    }
    .exec-hero h1 { color: #f8fafc !important; font-size: 1.55rem !important; margin: 0 0 0.5rem 0 !important; }
    .exec-hero p { color: #cbd5e1 !important; margin: 0 !important; line-height: 1.55 !important; font-size: 1.02rem !important; }
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.15rem 1.1rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .section-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        margin: 0 0 0.65rem 0;
    }
    div[data-testid="stMetricValue"] { color: #0f172a !important; }
    div[data-testid="stMetricLabel"] { color: #475569 !important; }
    .exec-snapshot {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.1rem 1.25rem 1.2rem 1.25rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.08);
        border-left: 5px solid #1d4ed8;
    }
    .exec-snapshot .snap-title {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
        color: #64748b; margin: 0 0 0.35rem 0;
    }
    .exec-snapshot .snap-headline {
        font-family: 'Inter', 'Source Sans 3', sans-serif;
        font-size: 1.22rem; font-weight: 800; color: #0f172a; margin: 0 0 0.75rem 0; line-height: 1.35;
    }
    .exec-snapshot .snap-rec {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
        padding: 0.85rem 1rem; margin-top: 0.5rem; color: #1e3a8a; font-size: 1.02rem; line-height: 1.5; font-weight: 600;
    }
    .tier-pill {
        display: inline-block;
        padding: 0.22rem 0.52rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-top: 0.28rem;
    }
    .tier-pill-high { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .tier-pill-medium { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .tier-pill-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .exec-snapshot div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #dbe7f3;
        border-radius: 10px;
        padding: 0.55rem 0.65rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=7200, show_spinner="Loading HUD + Census CBSA panel…")
def _cached_federal_hybrid(max_msas: int, acs_year: str) -> tuple:
    """Disk-backed cache so Streamlit reruns do not hammer HUD / Census."""
    return build_hybrid_dataset(max_msas=max_msas, acs_year=acs_year)


def _load_default_dataframe(max_msas: int = 280, acs_year: str = "2022") -> None:
    """
    Prefer live federal panel (HUD FMR + Census ACS). If anything fails or the join is too thin,
    fall back to the in-repo synthetic generator so the dashboard always runs.
    """
    try:
        df, meta = _cached_federal_hybrid(max_msas, acs_year)
        if len(df) < 40:
            raise ValueError(f"Only {len(df)} metro areas after HUD–Census join; need a larger overlap for training.")
        st.session_state.df_full = df
        st.session_state.data_provenance = {
            "mode": "hybrid",
            "detail": (
                "**Primary:** HUD Fair Market Rents (2‑BR) via HUD ArcGIS Open Data. "
                "**Joined:** U.S. Census Bureau ACS 5‑year median household income & unemployment rate by CBSA. "
                "**Eviction rate:** model proxy (not HUD court filings). "
                "**Labels (`risk_label`):** same transparent stress index used for the synthetic backup."
            ),
            **meta,
        }
    except Exception as exc:  # noqa: BLE001
        st.session_state.df_full = make_synthetic_dataset()
        st.session_state.data_provenance = {
            "mode": "synthetic_fallback",
            "detail": (
                "Using **synthetic backup** because the federal panel could not be built "
                f"(network, API change, or thin join). Error: `{exc}`"
            ),
            "error": str(exc),
        }


def _init_state() -> None:
    if "df_full" not in st.session_state:
        _load_default_dataframe()
    if "trained" not in st.session_state:
        st.session_state.trained = None
    if "predictions_df" not in st.session_state:
        st.session_state.predictions_df = None


def _hero() -> None:
    st.markdown(
        """
<div class="exec-hero">
  <h1>Housing instability & homelessness risk predictor</h1>
  <p>
    <b>At-a-glance:</b> the <b>Executive snapshot</b> below refreshes from your loaded panel (HUD + Census hybrid by default, or backup synthetic).
    <b>Train the model</b> in the sidebar to surface predicted risk, accuracy, and a portfolio-level <b>recommendation</b>—then scroll for rankings, budget simulation, and counterfactuals.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _portfolio_tier_and_recommendation(pred_df: pd.DataFrame | None, df_full: pd.DataFrame) -> tuple[str, str]:
    """Pick a region-sensitive tier + policy line for the snapshot."""
    if pred_df is not None and "predicted_risk" in pred_df.columns and len(pred_df):
        # Prefer probability-based tiering so the status reacts to regional changes
        # even when hard class labels are mostly Low.
        if "p_high" in pred_df.columns:
            mean_p_high = float(pred_df["p_high"].mean())
            if mean_p_high >= 0.55:
                return "High", recommendation_for_risk("High")
            if mean_p_high >= 0.30:
                return "Medium", recommendation_for_risk("Medium")
            return "Low", recommendation_for_risk("Low")

        s = pred_df["predicted_risk"]
        ph = float((s == "High").mean())
        pm = float((s == "Medium").mean())
        if ph >= 0.18:
            return "High", recommendation_for_risk("High")
        if ph + pm >= 0.45:
            return "Medium", recommendation_for_risk("Medium")
        return "Low", recommendation_for_risk("Low")

    # Label-based fallback (pre-train or no prediction frame).
    s = df_full["risk_label"]
    score = s.map({"Low": 0.0, "Medium": 1.0, "High": 2.0}).mean()
    if float(score) >= 1.20:
        return "High (label-based)", recommendation_for_risk("High")
    if float(score) >= 0.55:
        return "Medium (label-based)", recommendation_for_risk("Medium")
    return "Low (label-based)", recommendation_for_risk("Low")


def _render_executive_snapshot(
    full_df: pd.DataFrame,
    view_df: pd.DataFrame,
    pred_view: pd.DataFrame | None,
    prov: dict,
    trained,
    selected_region: str,
) -> None:
    """Dynamic KPIs + recommendation visible without scrolling."""
    mode = prov.get("mode", "—")
    mode_lbl = {
        "hybrid": "HUD FMR + Census ACS",
        "synthetic_fallback": "Synthetic backup",
        "synthetic_backup_manual": "Synthetic (manual)",
        "csv_upload": "Uploaded CSV",
    }.get(str(mode), str(mode))

    med_rent = float(view_df["median_rent"].median())
    med_inc = float(view_df["median_income"].median())
    med_ue = float(view_df["unemployment_rate"].median())
    med_ev = float(view_df["eviction_rate"].median())
    base_rent = float(full_df["median_rent"].median())
    base_inc = float(full_df["median_income"].median())
    base_ue = float(full_df["unemployment_rate"].median())
    base_ev = float(full_df["eviction_rate"].median())
    lbl_hi = float((view_df["risk_label"] == "High").mean())

    tier, rec_text = _portfolio_tier_and_recommendation(pred_view, view_df)
    tier_short = tier.replace(" (label-based)", "").strip()

    top_city = "—"
    if pred_view is not None and len(pred_view):
        rk = rank_cities_by_risk(pred_view)
        if not rk.empty:
            top_city = str(rk.iloc[0]["city"])

    region_safe = html.escape(selected_region)
    mode_safe = html.escape(mode_lbl)
    headline = (
        f"Scope: <strong>{region_safe}</strong> · {view_df['city'].nunique():,} regions · {len(view_df):,} rows · "
        f"Labels High: {lbl_hi:.0%} · Panel: {mode_safe}"
    )
    if pred_view is not None:
        ph = float((pred_view["predicted_risk"] == "High").mean())
        headline += f" · Model High: {ph:.0%}"
        if trained is not None:
            headline += f" · Accuracy {trained.accuracy:.0%}"
        headline += f" · Top pressure: {html.escape(top_city)}"

    st.markdown('<div class="exec-snapshot">', unsafe_allow_html=True)
    st.markdown('<p class="snap-title">Executive snapshot</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="snap-headline">{headline}</p>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric(
            "Median rent",
            f"${med_rent:,.0f}",
            delta=f"{med_rent - base_rent:+,.0f} vs all",
            delta_color="inverse",
        )
    with k2:
        st.metric(
            "Median household income",
            f"${med_inc:,.0f}",
            delta=f"{med_inc - base_inc:+,.0f} vs all",
            delta_color="normal",
        )
    with k3:
        st.metric(
            "Median unemployment",
            f"{med_ue:.1%}",
            delta=f"{(med_ue - base_ue) * 100:+.2f} pp vs all",
            delta_color="inverse",
        )
    with k4:
        st.metric(
            "Median eviction proxy",
            f"{med_ev:.1%}",
            delta=f"{(med_ev - base_ev) * 100:+.2f} pp vs all",
            delta_color="inverse",
        )
    is_high = tier_short == "High"
    is_medium = tier_short == "Medium"
    status_text = "Emergency" if is_high else ("Elevated monitoring" if is_medium else "No emergency")
    pill_class = "tier-pill-high" if is_high else ("tier-pill-medium" if is_medium else "tier-pill-low")

    with k5:
        if pred_view is not None and trained is not None:
            st.metric("Portfolio tier (model)", tier_short)
        else:
            st.metric("Portfolio tier (labels)", tier_short)
        st.markdown(
            f"<span class='tier-pill {pill_class}'>{html.escape(status_text)}</span>",
            unsafe_allow_html=True,
        )

    tier_display = html.escape(tier_short)
    rec_safe = html.escape(rec_text)
    st.markdown(
        f'<div class="snap-rec"><span style="color:#1e40af;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;">'
        f"RECOMMENDED ACTION · {tier_display}</span><br/>{rec_safe}</div>",
        unsafe_allow_html=True,
    )
    if pred_view is None:
        st.caption("Train the model in the sidebar to add **predicted** risk shares, accuracy, and geography-level recommendations in this strip.")
    st.markdown("</div>", unsafe_allow_html=True)


def _build_mckinsey_report_html(
    view_df: pd.DataFrame,
    pred_view: pd.DataFrame | None,
    selected_region: str,
    prov: dict,
    trained,
    budget_m: float,
) -> str:
    """Generate a polished, McKinsey-style insight report as standalone HTML."""
    mode = prov.get("mode", "—")
    mode_lbl = {
        "hybrid": "HUD FMR + Census ACS",
        "synthetic_fallback": "Synthetic backup",
        "synthetic_backup_manual": "Synthetic (manual)",
        "csv_upload": "Uploaded CSV",
    }.get(str(mode), str(mode))
    n_rows = len(view_df)
    n_regions = int(view_df["city"].nunique()) if n_rows else 0
    med_rent = float(view_df["median_rent"].median()) if n_rows else 0.0
    med_inc = float(view_df["median_income"].median()) if n_rows else 0.0
    med_ue = float(view_df["unemployment_rate"].median()) if n_rows else 0.0
    med_ev = float(view_df["eviction_rate"].median()) if n_rows else 0.0

    if pred_view is not None and len(pred_view):
        work = pred_view.copy()
        risk_col = "predicted_risk"
        high_share = float((work[risk_col] == "High").mean())
        med_share = float((work[risk_col] == "Medium").mean())
        low_share = float((work[risk_col] == "Low").mean())
        rank_df = rank_cities_by_risk(work, pred_col=risk_col).head(5)
    else:
        work = view_df.copy()
        work["predicted_risk"] = work["risk_label"]
        risk_col = "predicted_risk"
        high_share = float((work[risk_col] == "High").mean()) if len(work) else 0.0
        med_share = float((work[risk_col] == "Medium").mean()) if len(work) else 0.0
        low_share = float((work[risk_col] == "Low").mean()) if len(work) else 0.0
        rank_df = rank_cities_by_risk(work, pred_col=risk_col).head(5)

    top_city = str(rank_df.iloc[0]["city"]) if not rank_df.empty else "—"
    tier, recommendation = _portfolio_tier_and_recommendation(pred_view, view_df)
    tier_short = tier.replace(" (label-based)", "").strip()
    acc_text = f"{trained.accuracy:.1%}" if trained is not None else "N/A (model not trained this run)"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not rank_df.empty:
        rank_rows = "".join(
            f"<tr><td>{int(r.priority_rank)}</td><td>{html.escape(str(r.city))}</td>"
            f"<td>{float(r.high_share):.1%}</td><td>{float(r.risk_index):.2f}</td></tr>"
            for r in rank_df.itertuples(index=False)
        )
    else:
        rank_rows = "<tr><td colspan='4'>No rows available for ranking in selected scope.</td></tr>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Housing Instability Insight Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: #111827; margin: 28px; line-height: 1.45;
    }}
    .header {{ border-bottom: 3px solid #006b5f; padding-bottom: 10px; margin-bottom: 18px; }}
    .header h1 {{ margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.01em; }}
    .subtle {{ color: #4b5563; font-size: 13px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0 18px; }}
    .kpi {{ border: 1px solid #d1d5db; border-top: 3px solid #006b5f; border-radius: 6px; padding: 10px; }}
    .kpi .label {{ color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    h2 {{ font-size: 17px; margin: 18px 0 8px; color: #0f172a; }}
    .box {{ background: #f9fafb; border-left: 4px solid #006b5f; padding: 12px 14px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; font-size: 13px; text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 700; }}
    .footer {{ margin-top: 24px; font-size: 12px; color: #6b7280; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Housing Instability & Homelessness Risk — Insight Report</h1>
    <div class="subtle">Scope: <b>{html.escape(selected_region)}</b> · Data mode: {html.escape(mode_lbl)} · Generated: {generated_at}</div>
  </div>

  <h2>Executive Summary</h2>
  <div class="box">
    <b>Current tier: {html.escape(tier_short)}</b>. Priority action: {html.escape(recommendation)}<br/>
    Top pressure geography in this scope: <b>{html.escape(top_city)}</b>. Model hold-out accuracy: <b>{acc_text}</b>.
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="label">Median rent</div><div class="value">${med_rent:,.0f}</div></div>
    <div class="kpi"><div class="label">Median income</div><div class="value">${med_inc:,.0f}</div></div>
    <div class="kpi"><div class="label">Median unemployment</div><div class="value">{med_ue:.1%}</div></div>
    <div class="kpi"><div class="label">Median eviction proxy</div><div class="value">{med_ev:.1%}</div></div>
  </div>

  <h2>Risk Mix</h2>
  <p><b>High:</b> {high_share:.1%} · <b>Medium:</b> {med_share:.1%} · <b>Low:</b> {low_share:.1%} across {n_rows:,} rows / {n_regions:,} regions.</p>

  <h2>Top Priority Areas</h2>
  <table>
    <thead><tr><th>Rank</th><th>Region</th><th>High-risk share</th><th>Risk index</th></tr></thead>
    <tbody>{rank_rows}</tbody>
  </table>

  <h2>Policy Implication & Action</h2>
  <div class="box">
    <b>Implication:</b> Concentrated high-risk pockets indicate elevated near-term housing insecurity and potential pressure on homelessness response systems.<br/>
    <b>Action:</b> Use this tiering to target budget deployment (simulated budget: <b>${budget_m:.1f}M</b>), prioritize subsidy and stabilization interventions, and monitor labor-market shifts.
  </div>

  <div class="footer">
    One-page brief generated by the Housing Instability dashboard. This report is decision support, not a statutory determination.
  </div>
</body>
</html>"""


def _build_insight_report_pdf(
    view_df: pd.DataFrame,
    pred_view: pd.DataFrame | None,
    selected_region: str,
    prov: dict,
    trained,
    budget_m: float,
) -> bytes:
    """Generate a one-page PDF insight report with a clean consulting-style layout."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    mode = prov.get("mode", "—")
    mode_lbl = {
        "hybrid": "HUD FMR + Census ACS",
        "synthetic_fallback": "Synthetic backup",
        "synthetic_backup_manual": "Synthetic (manual)",
        "csv_upload": "Uploaded CSV",
    }.get(str(mode), str(mode))

    n_rows = len(view_df)
    n_regions = int(view_df["city"].nunique()) if n_rows else 0
    med_rent = float(view_df["median_rent"].median()) if n_rows else 0.0
    med_inc = float(view_df["median_income"].median()) if n_rows else 0.0
    med_ue = float(view_df["unemployment_rate"].median()) if n_rows else 0.0
    med_ev = float(view_df["eviction_rate"].median()) if n_rows else 0.0

    if pred_view is not None and len(pred_view):
        work = pred_view.copy()
        risk_col = "predicted_risk"
    else:
        work = view_df.copy()
        work["predicted_risk"] = work["risk_label"]
        risk_col = "predicted_risk"

    high_share = float((work[risk_col] == "High").mean()) if len(work) else 0.0
    med_share = float((work[risk_col] == "Medium").mean()) if len(work) else 0.0
    low_share = float((work[risk_col] == "Low").mean()) if len(work) else 0.0

    rank_df = rank_cities_by_risk(work, pred_col=risk_col).head(5)
    top_city = str(rank_df.iloc[0]["city"]) if not rank_df.empty else "—"
    tier, recommendation = _portfolio_tier_and_recommendation(pred_view, view_df)
    tier_short = tier.replace(" (label-based)", "").strip()
    acc_text = f"{trained.accuracy:.1%}" if trained is not None else "N/A (model not trained this run)"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    margin = 0.6 * inch
    y = h - margin

    def section_title(text: str, yy: float) -> float:
        c.setFillColor(colors.HexColor("#0F172A"))
        c.setFont("Helvetica-Bold", 10.8)
        c.drawString(margin, yy, text.upper())
        return yy - 12

    # Header and governing thought (Pyramid principle: answer first)
    c.setStrokeColor(colors.HexColor("#0B5D53"))
    c.setLineWidth(3)
    c.line(margin, y - 4, w - margin, y - 4)
    y -= 20
    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 17)
    c.drawString(margin, y, "Housing Instability & Homelessness Risk - Insight Report")
    y -= 14
    c.setFont("Helvetica", 8.8)
    c.setFillColor(colors.HexColor("#4B5563"))
    c.drawString(margin, y, f"Scope: {selected_region} | Data mode: {mode_lbl} | Generated: {generated_at}")
    y -= 18

    c.setFillColor(colors.HexColor("#F8FAFC"))
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.roundRect(margin, y - 56, w - 2 * margin, 54, 5, stroke=1, fill=1)
    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 11.2)
    c.drawString(margin + 8, y - 16, f"Governing thought: {tier_short} risk tier in scope '{selected_region}' requires targeted intervention.")
    c.setFont("Helvetica", 9.2)
    c.drawString(margin + 8, y - 31, f"Primary recommendation: {recommendation}")
    c.drawString(margin + 8, y - 45, f"Top pressure geography: {top_city} | Model accuracy: {acc_text}")
    y -= 68

    # Situation / Complication / Answer (SCQA-inspired)
    y = section_title("Situation", y)
    c.setFont("Helvetica", 9.2)
    c.drawString(margin, y, f"The current scope covers {n_rows:,} observations across {n_regions:,} regions.")
    c.drawString(margin, y - 12, f"Median conditions: rent ${med_rent:,.0f}, income ${med_inc:,.0f}, unemployment {med_ue:.1%}, eviction proxy {med_ev:.1%}.")
    y -= 28

    y = section_title("Complication", y)
    c.setFont("Helvetica", 9.2)
    c.drawString(
        margin,
        y,
        f"Risk concentration is uneven (High {high_share:.1%}, Medium {med_share:.1%}, Low {low_share:.1%}), creating pressure on limited response capacity.",
    )
    y -= 16

    y = section_title("Answer and impact", y)
    c.setFont("Helvetica", 9.2)
    c.drawString(
        margin,
        y,
        f"Prioritize top geographies and deploy the simulated ${budget_m:.1f}M budget toward prevention, stabilization, and emergency support pathways.",
    )
    y -= 20

    # KPI exhibits strip
    kpi_labels = ["Median rent", "Median income", "Median unemployment", "Median eviction proxy"]
    kpi_values = [f"${med_rent:,.0f}", f"${med_inc:,.0f}", f"{med_ue:.1%}", f"{med_ev:.1%}"]
    kpi_w = (w - 2 * margin - 18) / 4
    x0 = margin
    for i in range(4):
        x = x0 + i * (kpi_w + 6)
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.roundRect(x, y - 44, kpi_w, 42, 4, stroke=1, fill=1)
        c.setFont("Helvetica", 8.2)
        c.setFillColor(colors.HexColor("#6B7280"))
        c.drawString(x + 6, y - 14, kpi_labels[i])
        c.setFont("Helvetica-Bold", 14.5)
        c.setFillColor(colors.HexColor("#0F172A"))
        c.drawString(x + 6, y - 31, kpi_values[i])
    y -= 56

    # Prioritized actions table
    y = section_title("Priority actions (top 5 regions)", y)
    c.setFont("Helvetica-Bold", 8.8)
    c.setFillColor(colors.HexColor("#374151"))
    c.drawString(margin, y, "Rank")
    c.drawString(margin + 35, y, "Region")
    c.drawString(w - margin - 165, y, "High-risk share")
    c.drawString(w - margin - 75, y, "Risk index")
    y -= 5
    c.setStrokeColor(colors.HexColor("#E5E7EB"))
    c.line(margin, y, w - margin, y)
    y -= 10
    c.setFont("Helvetica", 8.6)
    c.setFillColor(colors.HexColor("#111827"))
    if rank_df.empty:
        c.drawString(margin, y, "No rows available for ranking in selected scope.")
        y -= 12
    else:
        for r in rank_df.itertuples(index=False):
            c.drawString(margin, y, str(int(r.priority_rank)))
            c.drawString(margin + 35, y, str(r.city)[:52])
            c.drawRightString(w - margin - 98, y, f"{float(r.high_share):.1%}")
            c.drawRightString(w - margin, y, f"{float(r.risk_index):.2f}")
            y -= 11
            if y < margin + 36:
                break

    # Footer
    c.setFillColor(colors.HexColor("#6B7280"))
    c.setFont("Helvetica", 8.3)
    c.drawString(
        margin,
        margin - 2,
        "One-page insight brief generated by the Housing Instability dashboard. Decision support, not a statutory determination.",
    )
    c.save()
    return buf.getvalue()


_init_state()
_hero()

# ---------------------------------------------------------------------------
# Sidebar — data, filter, train, budget
# ---------------------------------------------------------------------------
st.sidebar.header("Workspace")
prov = st.session_state.get("data_provenance") or {}
with st.sidebar.expander("How to use this app", expanded=False):
    st.markdown(
        """
1. Choose a region scope (or leave **All regions**) from **Region (display only)**.
2. Click **Train / refresh model** to update predictions and snapshot KPIs.
3. Review **Priority ranking**, **Budget simulation**, and **Counterfactual simulation**.
4. Download the one-page **Insight report** from the **Insight report** card below the Executive snapshot.
"""
    )
if prov.get("mode") == "hybrid":
    st.sidebar.success("Data: **HUD FMR + Census ACS** (hybrid)")
elif prov.get("mode") == "synthetic_fallback":
    st.sidebar.warning("Data: **synthetic backup** (federal panel unavailable)")
elif prov.get("mode") == "csv_upload":
    st.sidebar.info("Data: **uploaded CSV**")
with st.sidebar.expander("Data lineage & HUD / Census notes", expanded=False):
    st.markdown(prov.get("detail") or "_No lineage recorded._")
    if prov.get("mode") == "hybrid":
        st.caption(
            f"HUD MSA rows pulled: **{prov.get('hud_msa_rows', '—')}** · "
            f"Census matches: **{prov.get('acs_rows', '—')}** · "
            f"Joined training rows: **{prov.get('joined_rows', '—')}** · "
            f"Final rows: **{prov.get('final_rows', '—')}**"
        )
    st.markdown(
        "**Sources:** [HUD Fair Market Rents (ArcGIS)](https://hudgis-hud.opendata.arcgis.com/datasets/12d2516901f947b5bb4da4e780e35f07) · "
        "[Census ACS 5-year](https://www.census.gov/data/developers/data-sets/acs-5year.html). "
        "Optional `CENSUS_API_KEY` environment variable raises Census rate limits. "
        "For token-based HUD User FMR/IL endpoints, see [HUD User API](https://www.huduser.gov/portal/dataset/fmr-api.html)."
    )

with st.sidebar.expander("Upload CSV (optional)", expanded=False):
    up = st.sidebar.file_uploader(
        "CSV file",
        type=["csv"],
        label_visibility="collapsed",
        help="Must include: " + ", ".join(REQUIRED_COLS),
        key="csv_uploader_sidebar",
    )
    if up is not None:
        try:
            st.session_state.df_full = load_csv_bytes(up.getvalue())
            st.session_state.trained = None
            st.session_state.predictions_df = None
            st.session_state.data_provenance = {
                "mode": "csv_upload",
                "detail": "Using columns from your uploaded file (overrides federal hybrid until you reload it).",
            }
            st.success("CSV loaded — train the model to refresh scores.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load CSV: {exc}")

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button(
        "Reload federal data",
        use_container_width=True,
        help="Clears cache and tries HUD + Census again.",
        key="btn_reload_federal",
    ):
        _cached_federal_hybrid.clear()
        for k in ("df_full", "trained", "predictions_df", "data_provenance"):
            st.session_state.pop(k, None)
        _load_default_dataframe()
        st.rerun()
with c2:
    if st.button("Synthetic backup", use_container_width=True, key="btn_synthetic_backup"):
        st.session_state.df_full = make_synthetic_dataset()
        st.session_state.trained = None
        st.session_state.predictions_df = None
        st.session_state.data_provenance = {
            "mode": "synthetic_backup_manual",
            "detail": "Synthetic dataset loaded manually (for demos or when you want offline-only rows).",
        }
        st.rerun()

_all_regions = sorted(st.session_state.df_full["city"].unique().tolist())
_REGION_ALL = "(All regions)"
_region_options = [_REGION_ALL] + _all_regions
with st.sidebar.expander("Region (display only)", expanded=False):
    _picked = st.sidebar.selectbox(
        "Focus charts & tables on one region, or show all",
        options=_region_options,
        index=0,
        help="Training always uses the **full** dataset. This only narrows what you **see** in the main workspace—one control, no tag list.",
        key="region_display_select",
    )
    city_filter: list[str] | None = None if _picked == _REGION_ALL else [_picked]
    selected_region_label = "All regions" if _picked == _REGION_ALL else str(_picked)
    st.caption(f"**{len(_all_regions):,}** regions in this panel — choose **All** or one metro to drill in.")

budget_m = st.sidebar.number_input("Simulation budget ($ millions)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("Model")
if st.sidebar.button("Train / refresh model", type="primary", use_container_width=True):
    with st.spinner("Training RandomForest…"):
        try:
            tm = train_random_forest(st.session_state.df_full)
            st.session_state.trained = tm
            X = st.session_state.df_full.drop(columns=["risk_label"])
            preds, p_high = predict_dataframe(tm, X)
            out = st.session_state.df_full.copy()
            out["predicted_risk"] = preds
            out["p_high"] = p_high
            out["recommendation"] = [recommendation_for_risk(r) for r in preds]
            st.session_state.predictions_df = out
            st.sidebar.success(f"Hold-out accuracy: **{tm.accuracy:.1%}**")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Training failed: {exc}")

st.sidebar.divider()
st.sidebar.caption("**Developed by:** Sherriff Abdul-Hamid")

# ---------------------------------------------------------------------------
# Executive snapshot + dataset (main)
# ---------------------------------------------------------------------------
df_full = st.session_state.df_full
pred_df: pd.DataFrame | None = st.session_state.predictions_df
view_df = filter_by_cities(df_full, city_filter)
pred_view_for_snapshot = filter_by_cities(pred_df, city_filter) if pred_df is not None else None

_render_executive_snapshot(
    full_df=df_full,
    view_df=view_df,
    pred_view=pred_view_for_snapshot,
    prov=prov,
    trained=st.session_state.trained,
    selected_region=selected_region_label,
)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Insight report")
st.caption("Download location: use the button below.")
report_scope_slug = selected_region_label.replace(" ", "_").replace(",", "").replace("/", "-")
try:
    report_pdf = _build_insight_report_pdf(
        view_df=view_df,
        pred_view=pred_view_for_snapshot,
        selected_region=selected_region_label,
        prov=prov,
        trained=st.session_state.trained,
        budget_m=float(budget_m),
    )
    st.download_button(
        "Download insight report (.pdf)",
        data=report_pdf,
        file_name=f"housing_insight_report_{report_scope_slug}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
except ModuleNotFoundError as exc:
    # Keep the app usable on environments where optional PDF libs were not installed yet.
    st.warning(
        f"PDF export is unavailable on this deployment ({exc}). "
        "Use the HTML fallback below until dependencies are refreshed."
    )
    report_html_fallback = _build_mckinsey_report_html(
        view_df=view_df,
        pred_view=pred_view_for_snapshot,
        selected_region=selected_region_label,
        prov=prov,
        trained=st.session_state.trained,
        budget_m=float(budget_m),
    )
    st.download_button(
        "Download insight report (.html fallback)",
        data=report_html_fallback,
        file_name=f"housing_insight_report_{report_scope_slug}.html",
        mime="text/html",
        use_container_width=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Dataset preview", expanded=False):
    st.caption(
        "Training uses **all** rows in the loaded dataset; the table below follows the **Region (display only)** control in the sidebar. "
        "**Hybrid default:** HUD 2‑BR FMR + Census ACS CBSA fields unless you uploaded CSV or use synthetic backup—see **Data lineage**."
    )
    st.dataframe(view_df.head(40), use_container_width=True, height=320)

if pred_df is None:
    st.info("**Next step:** use **Train / refresh model** in the sidebar to generate predictions, charts, policy layers, and simulations.")
    st.stop()

# Narrow predictions for display
pred_view = filter_by_cities(pred_df, city_filter)

# ---------------------------------------------------------------------------
# Predictions table
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Model outputs</p>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Predictions & policy actions")
show_cols = [
    "city",
    "median_rent",
    "median_income",
    "unemployment_rate",
    "eviction_rate",
    "risk_label",
    "predicted_risk",
    "p_high",
    "recommendation",
]
_pv = pred_view[show_cols].copy()
_pv["p_high"] = _pv["p_high"].round(4)
st.dataframe(_pv, use_container_width=True, height=320)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Diagnostics</p>', unsafe_allow_html=True)
c_left, c_right = st.columns(2, gap="large")
with c_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Risk mix by city (predicted)")
    chart_df = (
        pred_view.groupby(["city", "predicted_risk"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .pivot(index="city", columns="predicted_risk", values="count")
        .fillna(0)
        .sort_index()
    )
    for col in ["High", "Medium", "Low"]:
        if col not in chart_df.columns:
            chart_df[col] = 0
    chart_df = chart_df[["Low", "Medium", "High"]]
    st.bar_chart(chart_df, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Feature importance (global)")
    tm = st.session_state.trained
    if tm is not None:
        imp = feature_importance_series(tm)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        imp.plot(kind="barh", ax=ax, color="#1d4ed8")
        ax.set_xlabel("Importance")
        ax.set_title("Top drivers in the fitted forest")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Policy layer — row recommendations already in table; causal panel
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Policy layer</p>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Causal framing (rule-based, transparent)")
st.markdown(
    """
- **High median rent** relative to income → **affordability stress** (payment-to-income pressure).
- **High unemployment rate** → **income instability** and slower rent recovery.
- **High eviction rate** (column) → on the **federal hybrid** panel this is a **stress proxy** derived from rent burden
  and unemployment (not HUD eviction filings); treat it as a model input, not a court statistic.
"""
)
sample = pred_view.head(12).copy()
sample["drivers"] = sample.apply(lambda r: "; ".join(policy_causal_notes(r, df_full)), axis=1)
st.dataframe(sample[["city", "median_rent", "unemployment_rate", "eviction_rate", "predicted_risk", "drivers"]], use_container_width=True, height=260)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Priority ranking & top geography
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Priority ranking & budget</p>', unsafe_allow_html=True)
rank_full = rank_cities_by_risk(pred_df)
city_summary = build_city_high_counts(pred_df)
alloc = allocate_budget(budget_m, city_summary)

top_city = rank_full.iloc[0]["city"] if not rank_full.empty else "—"
top_row = rank_full.iloc[0] if not rank_full.empty else None

p1, p2 = st.columns((1.1, 1.0), gap="large")
with p1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Priority ranking (model-led)")
    st.dataframe(rank_full, use_container_width=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

with p2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top priority area")
    if top_row is not None:
        st.metric("City", str(top_city))
        st.metric("High-risk share (rows)", f"{float(top_row['high_share']):.1%}")
        st.metric("Composite risk index", f"{float(top_row['risk_index']):.2f}")
    else:
        st.write("No rows to rank.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader(f"Budget simulation — ${budget_m:.1f}M to geographies (weighted by predicted High rows)")
alloc_show = alloc[alloc["allocation_m"] > 0].sort_values("allocation_m", ascending=False)
st.dataframe(alloc_show, use_container_width=True, height=220)
st.caption("Weights use **High** counts first; if a city has zero High rows in-sample, weight falls through to **Medium+High** mass.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Executive policy brief
# ---------------------------------------------------------------------------
brief = executive_policy_brief(
    pred_df,
    float(st.session_state.trained.accuracy) if st.session_state.trained else 0.0,
    "predicted_risk",
    str(top_city),
    alloc_show,
)
st.markdown('<p class="section-title">Executive policy brief</p>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(brief)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Before vs after simulation (rent / unemployment levers)
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Counterfactual simulation</p>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
tm = st.session_state.trained
st.subheader("Before vs after — rent & unemployment levers")
st.caption(
    "Applies **percentage adjustments** to the full feature matrix, then re-scores with the **already trained** model. "
    "This is a stylised stress test, not a forecast of programme impacts."
)
sim1, sim2 = st.columns(2)
with sim1:
    rent_adj = st.slider("Median rent change (%)", -25.0, 25.0, 0.0, help="Negative reduces rent (subsidy / cap effect).")
with sim2:
    ue_adj = st.slider("Unemployment rate change (%)", -40.0, 40.0, 0.0, help="Negative improves labor market (relative to baseline).")

feat_cols = ["city", "median_rent", "median_income", "unemployment_rate", "eviction_rate"]
X_feat = pred_df[feat_cols].copy()

before_labels, _ = predict_dataframe(tm, X_feat)
before_s = pd.Series(before_labels, name="before")
X_after = adjust_for_simulation(X_feat, rent_adj, ue_adj)
after_labels, _ = predict_dataframe(tm, X_after)
after_s = pd.Series(after_labels, name="after")

b_counts, a_counts = compare_risk_counts(before_s, after_s)
b_df = pd.DataFrame({"risk": ["Low", "Medium", "High"], "before": [b_counts.get(k, 0) for k in ["Low", "Medium", "High"]]})
a_df = pd.DataFrame({"risk": ["Low", "Medium", "High"], "after": [a_counts.get(k, 0) for k in ["Low", "Medium", "High"]]})
cmp = b_df.merge(a_df, on="risk")

s1, s2, s3 = st.columns(3)
with s1:
    st.metric("High-risk rows (before)", int(b_counts.get("High", 0)))
with s2:
    st.metric("High-risk rows (after)", int(a_counts.get("High", 0)))
with s3:
    delta_h = int(a_counts.get("High", 0)) - int(b_counts.get("High", 0))
    st.metric("Change in High rows", f"{delta_h:+d}")

st.dataframe(cmp.set_index("risk"), use_container_width=True)
st.bar_chart(cmp.set_index("risk"))
st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Repository layout: `data_loader.py` (I/O), `modeling.py` (sklearn pipeline), `policy.py` (briefs & simulations), `app.py` (UI)."
)

"""
Housing Instability & Homelessness Risk Predictor — Streamlit entrypoint.

Trains a RandomForest on regional housing / labor signals and surfaces policy-ready views:
rankings, budget simulation, counterfactual rent / unemployment levers, and an executive brief.
"""

from __future__ import annotations

import html

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
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
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
    .exec-snapshot .snap-headline { font-size: 1.2rem; font-weight: 700; color: #0f172a; margin: 0 0 0.75rem 0; line-height: 1.35; }
    .exec-snapshot .snap-rec {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
        padding: 0.85rem 1rem; margin-top: 0.5rem; color: #1e3a8a; font-size: 1.02rem; line-height: 1.5; font-weight: 600;
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
    """Pick a single tier + policy line for the headline (predicted risk if available, else training labels)."""
    if pred_df is not None and "predicted_risk" in pred_df.columns:
        s = pred_df["predicted_risk"]
        ph = float((s == "High").mean())
        pm = float((s == "Medium").mean())
        if ph >= 0.10:
            return "High", recommendation_for_risk("High")
        if ph + pm >= 0.38:
            return "Medium", recommendation_for_risk("Medium")
        return "Low", recommendation_for_risk("Low")
    s = df_full["risk_label"]
    ph = float((s == "High").mean())
    pm = float((s == "Medium").mean())
    if ph >= 0.10:
        return "High (label-based)", recommendation_for_risk("High")
    if ph + pm >= 0.38:
        return "Medium (label-based)", recommendation_for_risk("Medium")
    return "Low (label-based)", recommendation_for_risk("Low")


def _render_executive_snapshot(
    df_full: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    prov: dict,
    trained,
) -> None:
    """Dynamic KPIs + recommendation visible without scrolling."""
    mode = prov.get("mode", "—")
    mode_lbl = {
        "hybrid": "HUD FMR + Census ACS",
        "synthetic_fallback": "Synthetic backup",
        "synthetic_backup_manual": "Synthetic (manual)",
        "csv_upload": "Uploaded CSV",
    }.get(str(mode), str(mode))

    med_rent = float(df_full["median_rent"].median())
    med_inc = float(df_full["median_income"].median())
    med_ue = float(df_full["unemployment_rate"].median())
    med_ev = float(df_full["eviction_rate"].median())
    lbl_hi = float((df_full["risk_label"] == "High").mean())

    tier, rec_text = _portfolio_tier_and_recommendation(pred_df, df_full)
    tier_short = tier.replace(" (label-based)", "").strip()

    top_city = "—"
    if pred_df is not None and len(pred_df):
        rk = rank_cities_by_risk(pred_df)
        if not rk.empty:
            top_city = str(rk.iloc[0]["city"])

    headline = (
        f"{df_full['city'].nunique():,} regions · {len(df_full):,} rows · "
        f"Labels High: {lbl_hi:.0%} · Panel: {mode_lbl}"
    )
    if pred_df is not None:
        ph = float((pred_df["predicted_risk"] == "High").mean())
        headline += f" · Model High: {ph:.0%}"
        if trained is not None:
            headline += f" · Accuracy {trained.accuracy:.0%}"
        headline += f" · Top pressure: {top_city}"

    st.markdown('<div class="exec-snapshot">', unsafe_allow_html=True)
    st.markdown('<p class="snap-title">Executive snapshot</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="snap-headline">{html.escape(headline)}</p>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Median rent (2-BR FMR or panel)", f"${med_rent:,.0f}")
    with k2:
        st.metric("Median household income", f"${med_inc:,.0f}")
    with k3:
        st.metric("Median unemployment rate", f"{med_ue:.1%}")
    with k4:
        st.metric("Median eviction proxy", f"{med_ev:.1%}")
    with k5:
        if pred_df is not None and trained is not None:
            st.metric("Portfolio tier (model)", tier_short)
        else:
            st.metric("Portfolio tier (labels)", tier_short)

    tier_display = html.escape(tier_short)
    rec_safe = html.escape(rec_text)
    st.markdown(
        f'<div class="snap-rec"><span style="color:#1e40af;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;">'
        f"RECOMMENDED ACTION · {tier_display}</span><br/>{rec_safe}</div>",
        unsafe_allow_html=True,
    )
    if pred_df is None:
        st.caption("Train the model in the sidebar to add **predicted** risk shares, accuracy, and geography-level recommendations in this strip.")
    st.markdown("</div>", unsafe_allow_html=True)


_init_state()
_hero()

# ---------------------------------------------------------------------------
# Sidebar — data, filter, train, budget
# ---------------------------------------------------------------------------
st.sidebar.header("Workspace")
prov = st.session_state.get("data_provenance") or {}
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

# ---------------------------------------------------------------------------
# Executive snapshot + dataset (main)
# ---------------------------------------------------------------------------
df_full = st.session_state.df_full
pred_df: pd.DataFrame | None = st.session_state.predictions_df
view_df = filter_by_cities(df_full, city_filter)

_render_executive_snapshot(df_full, pred_df, prov, st.session_state.trained)

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

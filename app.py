"""
Housing Instability & Homelessness Risk Predictor — Streamlit entrypoint.

Trains a RandomForest on regional housing / labor signals and surfaces policy-ready views:
rankings, budget simulation, counterfactual rent / unemployment levers, and an executive brief.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import REQUIRED_COLS, filter_by_cities, load_csv_bytes, make_synthetic_dataset
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
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "df_full" not in st.session_state:
        st.session_state.df_full = make_synthetic_dataset()
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
    Executive dashboard for <b>regional prioritisation</b>: upload sub-regional observations (or use the demo dataset),
    train a transparent classifier, and translate model outputs into <b>funding signals</b>, <b>priority geographies</b>,
    and <b>counterfactual levers</b> (rent relief and employment shocks). Outputs support decisions—they are not a substitute
    for statutory homelessness definitions or lived-experience intake data.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


_init_state()
_hero()

# ---------------------------------------------------------------------------
# Sidebar — data, filter, train, budget
# ---------------------------------------------------------------------------
st.sidebar.header("Workspace")
up = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"], help="Must include: " + ", ".join(REQUIRED_COLS))
if up is not None:
    try:
        st.session_state.df_full = load_csv_bytes(up.getvalue())
        st.session_state.trained = None
        st.session_state.predictions_df = None
        st.sidebar.success("CSV loaded — train the model to refresh scores.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not load CSV: {exc}")

if st.sidebar.button("Reset to synthetic demo data", use_container_width=True):
    st.session_state.df_full = make_synthetic_dataset()
    st.session_state.trained = None
    st.session_state.predictions_df = None
    st.rerun()

all_cities = sorted(st.session_state.df_full["city"].unique().tolist())
city_filter = st.sidebar.multiselect(
    "City filter (display)",
    options=all_cities,
    default=all_cities,
    help="Filters tables and charts in the main workspace. Training always uses the full loaded dataset.",
)

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
# Main metrics (dataset-level)
# ---------------------------------------------------------------------------
df_full: pd.DataFrame = st.session_state.df_full
view_df = filter_by_cities(df_full, city_filter)
pred_df: pd.DataFrame | None = st.session_state.predictions_df

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Observations (full dataset)", f"{len(df_full):,}")
with m2:
    st.metric("Geographies (cities)", f"{df_full['city'].nunique()}")
with m3:
    if pred_df is not None:
        st.metric("Predicted high-risk share", f"{(pred_df['predicted_risk'] == 'High').mean():.1%}")
    else:
        st.metric("Predicted high-risk share", "—", help="Train the model to compute.")
with m4:
    if st.session_state.trained is not None:
        st.metric("Hold-out accuracy", f"{st.session_state.trained.accuracy:.1%}")
    else:
        st.metric("Hold-out accuracy", "—")

# ---------------------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Dataset preview</p>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.caption("Filtered view respects the sidebar city filter; training uses **all** rows in the loaded dataset.")
st.dataframe(view_df.head(25), use_container_width=True, height=280)
st.markdown("</div>", unsafe_allow_html=True)

if pred_df is None:
    st.info("Train the model from the sidebar to unlock predictions, charts, policy layers, and simulations.")
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
- **High eviction rate** → **housing insecurity** and churn in the rental stock.
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

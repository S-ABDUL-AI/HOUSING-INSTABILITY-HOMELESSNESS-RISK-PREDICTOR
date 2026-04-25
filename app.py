"""
Impact Allocation Engine — Streamlit entrypoint.

Evidence-leaning cost-effectiveness with explicit uncertainty and scenario controls.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Align chart text with UI sans (Inter loads via CSS; matplotlib falls back if needed).
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    }
)

from data_loader import DEFAULT_REMOTE_CSV, load_interventions_csv
from modeling import allocate_budget, calculate_scores, compare_budget_scale, scenario_effect_multiplier
from policy import allocation_insights, executive_summary_strip, funding_brief_markdown
from report_html import build_mckinsey_style_report_html

st.set_page_config(
    page_title="Impact Allocation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
<style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");

    :root {
        --iae-font-ui: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    html, body,
    .stApp,
    .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stSidebar"],
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stVerticalBlock"],
    .stApp label,
    .stApp p,
    .stApp li,
    .stApp small,
    .stApp input,
    .stApp textarea,
    .stApp button,
    .stApp [data-baseweb="select"],
    .stApp [data-baseweb="input"] {
        font-family: var(--iae-font-ui) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Headings: same Inter family as body — hierarchy from weight + size, not a second typeface */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        font-family: var(--iae-font-ui) !important;
        color: #0f172a !important;
    }

    .stApp h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
        font-weight: 800 !important;
        letter-spacing: -0.045em !important;
    }
    .stApp h2 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }
    .stApp h3 {
        font-size: 1.12rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    .stApp h4 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }
    .stApp h5, .stApp h6 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    .stApp details > summary {
        font-family: var(--iae-font-ui) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }

    div[data-testid="stMetricLabel"] {
        font-family: var(--iae-font-ui) !important;
        font-weight: 600 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #64748b !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--iae-font-ui) !important;
        font-weight: 700 !important;
        font-feature-settings: "tnum" 1, "lnum" 1;
        color: #0f172a !important;
    }

    .block-container { padding-top: 1rem !important; }

    /* Keep the control panel readable on wide layouts (does not block Streamlit’s collapse control) */
    section[data-testid="stSidebar"] {
        min-width: 18rem !important;
    }
    /* Vertical “card” panels inside the sidebar (Streamlit bordered containers) */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08) !important;
        border-color: #e5e7eb !important;
        background-color: #ffffff !important;
        margin-bottom: 0.65rem !important;
    }

    /* Compact Download Report control (tighter than default Streamlit button chrome) */
    div[data-testid="stDownloadButton"] button {
        font-size: 0.78rem !important;
        padding: 0.22rem 0.55rem !important;
        min-height: 2rem !important;
        line-height: 1.2 !important;
        border-radius: 8px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_interventions(remote_url: str) -> tuple:
    """Cache CSV load so reruns do not refetch constantly."""
    df, src = load_interventions_csv(remote_url=remote_url or None)
    return df, src


st.title("Impact Allocation Engine")

# ---------------------------------------------------------------------------
# Sidebar — vertical card-style sections (bordered containers, stacked top to bottom)
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.container(border=True):
        st.markdown("##### Guide")
        with st.expander("How to use", expanded=False):
            st.markdown(
                """
1. **Data** — Keep the default CSV URL or paste a raw link to your own `interventions`-style file.
2. **Scope** — Choose **Select Region**, then open **Select Interventions** to add or remove programs.
3. **Budget & scenario** — Set **Total budget** and **Scenario** (Base / Optimistic / Pessimistic).
4. **Stress-test** — Move **Sensitivity** sliders (and optional **Moral weights**) to see how scores and allocations shift.
5. **Decide** — Read **Executive summary** (use **Download Report** in its top-right), then metrics/charts, allocation, and policy notes.
"""
            )

    with st.container(border=True):
        st.markdown("##### Data")
        remote = st.text_input(
            "Raw CSV URL (optional)",
            value=os.environ.get("IMPACT_CSV_URL", DEFAULT_REMOTE_CSV),
            help="Defaults to GitHub raw `interventions.csv` once pushed to the repo root.",
        )

_effective_url = remote.strip() or DEFAULT_REMOTE_CSV
df0, _ = cached_interventions(_effective_url)

with st.sidebar:
    with st.container(border=True):
        st.markdown("##### Scope")
        regions = sorted(df0["region"].unique().tolist())
        region_pick = st.selectbox("Select Region", ["All regions"] + regions)
        ids_default = df0["intervention_id"].tolist()
        _name_by_id = df0.set_index("intervention_id")["intervention_name"].to_dict()
        with st.expander("Select Interventions", expanded=False):
            id_pick = st.multiselect(
                "Select Interventions",
                options=ids_default,
                default=ids_default,
                format_func=lambda i: f"{i} — {_name_by_id.get(i, '')}",
                label_visibility="collapsed",
            )

    with st.container(border=True):
        st.markdown("##### Budget & scenario")
        total_budget = st.slider("Total budget (USD)", 10_000, 2_000_000, 200_000, step=5_000)
        scenario = st.selectbox("Scenario", ["Base", "Optimistic", "Pessimistic"], index=0)

    with st.container(border=True):
        st.markdown("##### Sensitivity")
        cost_adj = st.slider("Cost multiplier (all rows)", 0.5, 2.0, 1.0, 0.05)
        effect_adj = st.slider("Effectiveness multiplier (effect_size)", 0.5, 1.5, 1.0, 0.05)

    with st.container(border=True):
        st.markdown("##### Options")
        with st.expander("Moral weights (optional)", expanded=False):
            life_weight = st.slider("Weight: lives / life-saving programs", 0.5, 2.0, 1.0, 0.05)
            income_weight = st.slider("Weight: income gains", 0.0, 1.0, 0.10, 0.02)
        malaria_double = st.checkbox(
            "Double cost for Malaria Bed Nets (id=1)",
            value=False,
            help="Applies a 2× multiplier to cost only for Malaria Bed Nets when that intervention is in scope.",
        )

    st.caption("**Developed by:** Sherriff Abdul-Hamid")

# ---------------------------------------------------------------------------
# Filter + score
# ---------------------------------------------------------------------------
df = df0[df0["intervention_id"].isin(id_pick)].copy()
if region_pick != "All regions":
    df = df[df["region"] == region_pick].copy()

if df.empty:
    st.warning("No rows after filters — broaden region or select interventions.")
    st.stop()

per_row_cost = pd.Series(1.0, index=df.index)
if malaria_double:
    per_row_cost.loc[df["intervention_id"] == 1] = 2.0

scored = calculate_scores(
    df,
    cost_adj=cost_adj,
    effect_adj=effect_adj,
    scenario=scenario,  # type: ignore[arg-type]
    life_weight=life_weight,
    income_weight=income_weight,
    per_row_cost_mult=per_row_cost,
)
alloc = allocate_budget(scored, total_budget)

report_html = build_mckinsey_style_report_html(
    alloc,
    float(total_budget),
    scenario,
    region_pick,
    cost_adj,
    effect_adj,
)

# ---------------------------------------------------------------------------
# Executive summary (portfolio / sponsor readers — quantitative detail stays below)
# ---------------------------------------------------------------------------
_ex = executive_summary_strip(alloc, float(total_budget), scenario, region_pick)
with st.container(border=True):
    _hdr_l, _hdr_r = st.columns((6, 1))
    with _hdr_l:
        st.markdown("#### Executive summary")
        st.caption("Answer first readout, modeled recommendation only")
    with _hdr_r:
        st.download_button(
            "Download Report",
            data=report_html.encode("utf-8"),
            file_name="impact_allocation_funding_report.html",
            mime="text/html",
            key="iae_download_report",
            use_container_width=False,
            type="secondary",
            help="Concise HTML brief: headline, findings, and appendix table. Open in a browser or import into Word.",
        )
    es_l, es_r = st.columns((3, 1))
    with es_l:
        st.markdown(_ex["headline"])
        st.caption(_ex["caption"])
        for _b in _ex["bullets"]:
            st.markdown(f"- {_b}")
    with es_r:
        st.metric("Programs in view", _ex["n_programs"])
        st.metric("Top share of budget", f"{_ex['top_pct']:.0f}%")

with st.expander("Problem statement", expanded=False):
    st.markdown(
        """
NGOs and foundations must divide limited budgets across many programs, each with different costs, evidence
quality, and room to scale—often under shifting assumptions about effectiveness and unit costs. Without a
transparent way to compare options and stress-test those assumptions, funding decisions can drift from impact
or become hard to explain to boards and partners.
"""
    )

with st.expander("Solution", expanded=False):
    st.markdown(
        """
**Impact Allocation Engine** turns your intervention panel into a comparable score for each program—blending
expected impact, evidence strength, uncertainty, funding headroom, and scalability—then splits a fixed budget in
proportion to those scores. You can filter by region and program set, stress-test costs and effectiveness,
run optimistic / base / pessimistic cases, and export a structured funding brief so trade-offs are visible before
capital is committed.
"""
    )

with st.expander("Methodology", expanded=False):
    st.markdown(
        """
This tool applies cost-effectiveness logic, explicit uncertainty discounts, and scenario analysis to approximate
how a funder might allocate a fixed budget across interventions.

It is inspired by evidence-first approaches in global health and development (for example, the transparency goals
behind organizations like [GiveWell](https://www.givewell.org/)), but it is not an official GiveWell model and should
not be treated as their published cost-effectiveness estimates.
"""
    )

# ---------------------------------------------------------------------------
# Executive metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Interventions in scope", len(alloc))
with m2:
    st.metric("Scenario effect multiplier", f"{scenario_effect_multiplier(scenario):.2f}×")
with m3:
    st.metric("Total allocation", f"${alloc['allocation'].sum():,.0f}")
with m4:
    top_name = alloc.sort_values("final_score", ascending=False).iloc[0]["intervention_name"]
    st.metric("Top ranked", top_name[:28] + ("…" if len(top_name) > 28 else ""))

# ---------------------------------------------------------------------------
# What-if strips
# ---------------------------------------------------------------------------
st.subheader("What-if analyses")
w1, w2 = st.columns(2)
with w1:
    st.markdown("**What happens if budget increases by 50%?**")
    cmp_budget = compare_budget_scale(scored, float(total_budget), 1.5)
    st.dataframe(cmp_budget, use_container_width=True, height=220)
with w2:
    st.markdown("**Sensitivity snapshot (current sliders)**")
    st.dataframe(
        alloc[
            [
                "intervention_name",
                "adjusted_effect_size",
                "adjusted_cost",
                "expected_impact",
                "cost_effectiveness",
                "uncertainty_adjusted",
                "final_score",
                "allocation",
            ]
        ].round(6),
        use_container_width=True,
        height=260,
    )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Allocation (USD)")
    plot_df = alloc.sort_values("allocation", ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(plot_df["intervention_name"], plot_df["allocation"], color="#0B5D53")
    ax.set_xlabel("Allocated USD")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with c2:
    st.subheader("Final score ranking")
    rnk = alloc.sort_values("final_score", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.8))
    ax2.barh(rnk["intervention_name"], rnk["final_score"], color="#1D4ED8")
    ax2.set_xlabel("Final score")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ---------------------------------------------------------------------------
# Tables + policy
# ---------------------------------------------------------------------------
st.subheader("Allocation table")
show_cols = [
    "intervention_name",
    "region",
    "final_score",
    "allocation",
    "funding_gap",
    "scalability",
    "evidence_strength",
    "uncertainty_level",
]
st.dataframe(alloc[show_cols].sort_values("final_score", ascending=False), use_container_width=True, height=280)

st.subheader("Policy insights (why these allocations)")
for b in allocation_insights(alloc):
    st.markdown(f"- {b}")

st.markdown("## Funding recommendation report")
st.markdown(funding_brief_markdown(alloc, total_budget, scenario))

st.markdown("**Note**")
st.caption(
    "Figures are illustrative; they should be interpreted alongside qualitative judgment, "
    "implementation risk, and partner capacity."
)

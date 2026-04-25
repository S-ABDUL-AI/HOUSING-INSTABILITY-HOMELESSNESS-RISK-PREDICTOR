# Housing Stability Risk Monitor
### HUD & Census Risk Predictor for Government Program Officers

**Built by Sherriff Abdul-Hamid**  
Product leader specializing in government digital services, safety net benefits delivery,  
and decision-support tools for underserved communities.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)

---

## The Problem This Solves

> *Which communities are at greatest risk of housing instability — and what should program officers do about it?*

Housing program officers, HUD grantees, and homelessness prevention coordinators face a critical challenge: identifying which communities are most at risk *before* residents reach crisis point. Reactive intervention is expensive and often too late. This tool gives them a structured, evidence-based framework for **proactive** resource prioritisation — using HUD Fair Market Rent data and Census ACS indicators across 240+ regions.

---

## What This Tool Produces

| Output | Description |
|---|---|
| **Executive Snapshot** | 5 KPI cards: regions analysed, high-risk count, median rent, unemployment, model accuracy |
| **Policy Brief** | Three-part decision summary: Risk · Implication · Action Now |
| **Budget Simulation** | Simulates how a user-defined budget ($M) flows across risk bands |
| **Regional Rankings** | Top 20 highest-risk regions visualised by composite risk score |
| **Indicator Scatter** | Cost burden vs eviction rate — identify the most vulnerable quadrant |
| **Regional Insight Cards** | Plain-language briefing + recommended action per high-risk region |
| **ML Risk Prediction** | Random Forest classifier predicts risk band with ~79% hold-out accuracy |
| **McKinsey PDF Report** | 6-page downloadable report ready for programme directors and HUD grantees |
| **CSV Export** | Full 240-region panel for integration with agency data systems |

---

## Risk Score Methodology

```
risk_score = (cost_burden × 0.30) + (income_index × 0.25) +
             (unemployment × 0.20) + (eviction_rate × 0.15) +
             (crowding_rate × 0.10)
```

Scaled 0–100. Priority bands:

| Band | Score | Criteria | Action |
|---|---|---|---|
| **High** | 67–100 | Cost burden > 40%; eviction > 5%; unemployment > 7% | Emergency rehousing & rental assistance |
| **Medium** | 34–66 | 1–2 elevated indicators; affordability pressure emerging | SNAP outreach & eviction prevention |
| **Low** | 0–33 | All indicators within moderate range | Monitor & maintain preventive access |

---

## Input Data Fields

| Field | Type | Source |
|---|---|---|
| `region` | string | County or region identifier |
| `median_rent` | float | HUD Fair Market Rents (annual) |
| `median_income` | float | Census ACS 5-year — median household income |
| `unemployment` | float | Census ACS 5-year — unemployment rate (%) |
| `eviction_rate` | float | Princeton Eviction Lab — eviction filing rate (%) |
| `crowding_rate` | float | Census ACS 5-year — household crowding rate |
| `cost_burden` | float | Derived: (median_rent × 12) / median_income |

---

## Repository Structure

```
├── app.py                          # Main Streamlit application
├── housing_report_generator.py     # McKinsey-style PDF report engine
├── requirements.txt                # Runtime dependencies
└── README.md                       # This file
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/S-ABDUL-AI/HOUSING-STABILITY-RISK-MONITOR.git
cd HOUSING-STABILITY-RISK-MONITOR

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

**Requirements:** `streamlit` · `pandas` · `numpy` · `plotly` · `scikit-learn` · `reportlab` · `pypdf` · `matplotlib`

---

## McKinsey-Style PDF Report

Clicking **Generate Report PDF** produces a 6-page downloadable briefing document:

| Page | Content |
|---|---|
| 1 | Cover — navy/gold, geometric blocks, report date, author credentials |
| 2 | Executive snapshot — 5 KPI cards + risk band distribution chart |
| 3 | Policy brief — Risk · Implication · Action (colour-coded cards) + budget simulation table |
| 4 | Exhibit 1 — Top 15 regional risk rankings (horizontal bar chart) |
| 5 | Exhibit 2 — Cost burden vs eviction rate scatter + top 20 rankings table |
| 6 | Regional insight cards (top 6) + methodology, limitations, and data sources |

---

## Deployment

Deployed on Streamlit Community Cloud.  
Live demo: [housing-stability-risk-monitor.streamlit.app](https://your-app-url.streamlit.app/)

---

## Scope Note

> All data shown in this application is **illustrative sample data** in demo mode.  
> Any real housing programme allocation decision must be paired with official HUD data,  
> Census Bureau statistics, legal review, and agency approval processes.  
> This tool is designed to support — not replace — government decision-making.

---

## About the Author

**Sherriff Abdul-Hamid** is a product leader and data scientist specializing in government digital services, safety net benefits delivery, and decision-support tools for underserved communities.

- Former Founder & CEO, Poverty 360 — 25,000+ beneficiaries served across West Africa
- Partnered with Ghana's National Health Insurance Authority to enroll 1,250 vulnerable individuals into national health coverage
- Directed $200M+ in resource allocation decisions for USAID, UNDP, and UKAID-funded programs
- **Obama Foundation Leaders Award** — Top 1.3% globally, 2023
- **Mandela Washington Fellow** — Top 0.3%, U.S. Department of State, 2018
- Harvard Business School · Senior Executive Program

**Connect:** [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) · [Portfolio](https://share.streamlit.io/user/s-abdul-ai)

---

## Related Projects

| Project | Description |
|---|---|
| [Public Budget Allocation Tool](https://smart-resource-allocation-dashboard-eudzw5r2f9pbu4qyw3psez.streamlit.app) | Need-based government budget distribution across regions with ministerial brief |
| [Medicaid & Healthcare Access Risk Monitor](https://chpghrwawmvddoquvmniwm.streamlit.app) | ML-powered healthcare coverage gap analysis across all 50 US states |
| [Safety Net Risk Monitor](https://povertyearlywarningsystem-7rrmkktbi7bwha2nna8gk7.streamlit.app) | SNAP and food security vulnerability targeting for program officers |
| [GovFund Allocation Engine](https://impact-allocation-engine-ahxxrbgwmvyapwmifahk2b.streamlit.app) | Cost-effectiveness decision tool for public health funders |
| [Global Vaccination Coverage Explorer](https://worldvaccinationcoverage-etl-ftvwbikifyyx78xyy2j3zv.streamlit.app) | WHO vaccination data across 190+ countries for public health managers |

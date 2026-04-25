# Housing Stability Risk Monitor

Housing Stability Risk Monitor is a production-style Streamlit app for public sector and social policy teams to identify where housing stress is rising, prioritize places with elevated instability risk, and run budget and counterfactual simulations for policy planning.

## Why this app exists

Housing instability is often detected too late, after financial stress has already translated into evictions and service pressure. This app helps teams move earlier by combining affordability, labor, and stress indicators into an operational risk view.

## Primary users

- State and city housing agencies
- Homelessness prevention and benefits delivery teams
- Program managers in social protection and safety net operations
- Policy and strategy teams advising government service delivery

## Core capabilities

- Executive snapshot with board-readable recommendation
- Hybrid data option (HUD FMR + Census ACS) with synthetic fallback
- Region filtering for decision-focused analysis
- RandomForest model training and holdout accuracy readout
- Predicted risk mix and policy action recommendations
- Priority ranking by geography
- Budget allocation simulation by risk intensity
- Counterfactual testing for rent and unemployment shocks
- Downloadable insight report in PDF

## Product logic

The monitor combines observed housing and labor signals and then:

1. Trains a supervised model on labeled risk tiers.
2. Estimates risk class and high-risk probability by geography.
3. Aggregates geographic pressure into priority rankings.
4. Simulates budget distribution weighted by predicted stress.
5. Tests policy-relevant counterfactuals (rent and unemployment adjustments).

## Files delivered

- `app.py` — complete Streamlit app file
- `housing_stability_risk_monitor_README.md` — portfolio README for this app
- `requirements.txt` — runtime dependencies for local and cloud deployment

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data behavior

- **Default mode:** attempts a federal hybrid panel (HUD FMR + Census ACS).
- **Fallback mode:** synthetic backup dataset if APIs fail or overlap is too thin.
- **Optional upload:** custom CSV input for local scenario work.

## Deployment notes

- This app is designed to remain usable under partial data failure by automatically falling back to synthetic data.
- PDF report export requires the report dependencies in `requirements.txt`.

## Disclaimer

This tool supports policy prioritization and operational planning. It does not replace local program design diligence, legal review, or implementation feasibility assessment.


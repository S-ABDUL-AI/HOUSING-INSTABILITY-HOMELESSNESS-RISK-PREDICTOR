"""
Train / score RandomForest risk model with city + numeric features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}
RISK_ORDER_INV = {0: "Low", 1: "Medium", 2: "High"}


@dataclass
class TrainedModel:
    pipeline: Pipeline
    accuracy: float
    feature_names: List[str]
    label_classes: np.ndarray


def _build_pipeline() -> Pipeline:
    """Numeric scaling + city one-hot + RandomForest."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["median_rent", "median_income", "unemployment_rate", "eviction_rate"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", pre), ("rf", clf)])


def train_random_forest(df: pd.DataFrame, test_size: float = 0.25) -> TrainedModel:
    """Fit pipeline; returns hold-out accuracy for dashboard display."""
    data = df.copy()
    y = data["risk_label"].map(RISK_ORDER).astype(int)
    X = data.drop(columns=["risk_label"])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    prep: ColumnTransformer = pipe.named_steps["prep"]
    feature_names = list(prep.get_feature_names_out())

    return TrainedModel(
        pipeline=pipe,
        accuracy=acc,
        feature_names=feature_names,
        label_classes=pipe.named_steps["rf"].classes_,
    )


def predict_dataframe(tm: TrainedModel, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred_labels_str, proba_high)."""
    probs = tm.pipeline.predict_proba(X)
    classes = tm.pipeline.named_steps["rf"].classes_
    idx_high = int(np.where(classes == RISK_ORDER["High"])[0][0]) if RISK_ORDER["High"] in classes else -1
    pred_idx = tm.pipeline.predict(X)
    labels = np.array([RISK_ORDER_INV[int(i)] for i in pred_idx])
    proba_high = probs[:, idx_high] if idx_high >= 0 else np.zeros(len(X))
    return labels, proba_high


def feature_importance_series(tm: TrainedModel) -> pd.Series:
    """Named importances aligned to expanded feature space."""
    rf: RandomForestClassifier = tm.pipeline.named_steps["rf"]
    imp = rf.feature_importances_
    names = tm.feature_names
    if len(names) != len(imp):
        # Fallback if sklearn version differs
        names = [f"f{i}" for i in range(len(imp))]
    s = pd.Series(imp, index=names).sort_values(ascending=True)
    return s.tail(20)  # top 20 for chart readability


def adjust_for_simulation(X: pd.DataFrame, rent_pct: float, unemployment_pct: float) -> pd.DataFrame:
    """
    Core simulation hook: scale rent and unemployment for counterfactual scoring.

    rent_pct: negative means lower rent (e.g. -5 → 5% reduction).
    unemployment_pct: negative means lower unemployment.
    """
    out = X.copy()
    out["median_rent"] = out["median_rent"] * (1.0 + rent_pct / 100.0)
    out["median_income"] = out["median_income"]  # unchanged unless we add slider later
    out["unemployment_rate"] = np.clip(out["unemployment_rate"] * (1.0 + unemployment_pct / 100.0), 0.001, 0.35)
    out["eviction_rate"] = out["eviction_rate"]
    return out

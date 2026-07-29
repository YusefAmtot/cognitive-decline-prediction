# src/models/models.py
"""Candidate models for predicting an individual's future decline rate from
demographics + whatever personal longitudinal history they have so far.

Ground truth (`{domain}_slope`) only exists for the simulated cohort (see
notebooks/00_generate_simulated_data.ipynb -> data/simulated/true_slopes.csv),
so training/evaluation happens there; real CLI participants get a
personalized *estimate* via the mixed-effects route in
src/longitudinal/prediction.py instead, since they have no ground truth to
fit an ML model against.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def get_naive_baseline() -> DummyRegressor:
    """Predicts the population mean slope for everyone - the floor any real model must beat."""
    return DummyRegressor(strategy="mean")


def build_ridge_model(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
    ])


def build_rf_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE
    )


def build_gbr_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
    )


def get_candidate_models() -> dict:
    return {
        "baseline_mean": get_naive_baseline(),
        "ridge": build_ridge_model(),
        "random_forest": build_rf_model(),
        "gbr": build_gbr_model(),
    }


def build_feature_matrix(
    slope_table: pd.DataFrame,
    trend_table: pd.DataFrame,
    true_slopes_df: pd.DataFrame,
    domain: str = "memory",
    id_col: str = "participant_id",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y, groups) for predicting `{domain}_slope`.

    `slope_table`/`trend_table` should come from extract_slopes()/
    build_trend_features() run on a TRUNCATED history (e.g. only each
    participant's first 2 visits) to simulate a new user with minimal data -
    that's what makes `has_own_slope`/`{domain}_n_obs` meaningful cold-start
    features rather than leaking full-history information into training.

    Demographics (age/education_years/sex) come from true_slopes_df, which
    only exists for the simulated cohort - ground truth is required to train
    or evaluate these models, so this path never runs on real CLI sessions.
    """
    merged = (
        true_slopes_df.merge(slope_table, on=id_col, how="left")
        .merge(trend_table, on=id_col, how="left")
    )

    merged["sex_encoded"] = (merged["sex"] == "M").astype(int)

    slope_col = f"{domain}_slope_ols"
    merged["has_own_slope"] = merged[slope_col].notna().astype(int) if slope_col in merged else 0

    feature_cols = [
        "age", "education_years", "sex_encoded", "has_own_slope",
        f"{domain}_slope_ols", f"{domain}_n_obs", f"{domain}_resid_std", f"{domain}_baseline_value",
    ]
    feature_cols = [c for c in feature_cols if c in merged.columns]

    X = merged[feature_cols].copy()
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    y = merged[f"{domain}_slope"]
    groups = merged[id_col]

    return X, y, groups


def project_future_score(current_score: float, predicted_slope: float, months_ahead: float) -> float:
    """Linearly project a future score from a current score and a decline rate."""
    return current_score + predicted_slope * months_ahead

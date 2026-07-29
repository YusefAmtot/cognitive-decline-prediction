# src/evaluation/evaluation.py
"""Evaluation harness comparing the naive baseline, mixed-effects slopes, and
ML models on genuinely held-out data.

Two complementary modes are needed because a single GroupKFold-over-
participants scheme can't fairly evaluate MixedLM (statsmodels has no native
out-of-sample BLUP prediction for a group unseen during fitting):

- `evaluate_cross_participant`: GroupKFold on participant_id, for models
  trained on demographics + cold-start features (Ridge/RF/GBR/naive).
- `evaluate_temporal_holdout`: fit slopes on all-but-last visit(s) per
  participant (their own data stays in the fit, as a real deployed system
  would do), project forward, compare to the actual held-out visit.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.longitudinal.slope_extraction import (
    mixedlm_slope_per_subject,
    ols_slope_per_subject,
    theilsen_slope_per_subject,
)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "n": int(len(y_true)),
    }


def evaluate_cross_participant(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    models: Dict[str, object],
    n_splits: int = 5,
) -> pd.DataFrame:
    """GroupKFold cross-validation (no participant appears in both train/test of a fold)."""
    n_splits = min(n_splits, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    for name, model in models.items():
        fold_preds, fold_true = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            fitted = clone(model)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_preds.append(fitted.predict(X.iloc[test_idx]))
            fold_true.append(y.iloc[test_idx].to_numpy())

        metrics = compute_metrics(np.concatenate(fold_true), np.concatenate(fold_preds))
        rows.append({"model": name, "mode": "cross_participant", **metrics})

    return pd.DataFrame(rows)


def evaluate_temporal_holdout(
    long_df: pd.DataFrame,
    value_col: str,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    covariates: Sequence[str] = ("age_baseline", "education_years"),
    holdout_last_n: int = 1,
) -> pd.DataFrame:
    """Fit slope estimators on all-but-last visit(s), project forward, score against actual.

    Only participants with at least `holdout_last_n + 2` visits are eligible
    (need >=2 training visits for a slope, plus the held-out visit(s)).
    """
    df = long_df.dropna(subset=[time_col, value_col]).sort_values([id_col, time_col])
    eligible = df.groupby(id_col).filter(lambda g: len(g) >= holdout_last_n + 2)

    if eligible.empty:
        return pd.DataFrame(columns=["model", "mode", "mae", "rmse", "r2", "n"])

    train_parts, test_parts = [], []
    for _, group in eligible.groupby(id_col):
        train_parts.append(group.iloc[:-holdout_last_n])
        test_parts.append(group.iloc[-holdout_last_n:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    ols = ols_slope_per_subject(train_df, id_col, time_col, value_col)
    theilsen = theilsen_slope_per_subject(train_df, id_col, time_col, value_col)
    try:
        mixedlm = mixedlm_slope_per_subject(train_df, time_col, value_col, covariates, id_col)
    except Exception:
        mixedlm = pd.DataFrame({id_col: train_df[id_col].unique()})

    slope_table = ols.merge(theilsen, on=id_col, how="outer").merge(mixedlm, on=id_col, how="outer")
    test = test_df.merge(slope_table, on=id_col, how="left")

    specs = [
        ("ols", f"{value_col}_slope_ols", f"{value_col}_intercept_ols"),
        ("theilsen", f"{value_col}_slope_theilsen", f"{value_col}_intercept_theilsen"),
        ("mixedlm_blup", f"{value_col}_slope_mixedlm", f"{value_col}_intercept_mixedlm"),
        ("mixedlm_population_only", f"{value_col}_slope_population", f"{value_col}_intercept_mixedlm"),
    ]

    rows = []
    for name, slope_col, intercept_col in specs:
        if slope_col not in test.columns or intercept_col not in test.columns:
            continue
        valid = test.dropna(subset=[slope_col, intercept_col, value_col, time_col])
        if valid.empty:
            continue
        projected = valid[intercept_col] + valid[slope_col] * valid[time_col]
        metrics = compute_metrics(valid[value_col], projected)
        rows.append({"model": name, "mode": "temporal_holdout", **metrics})

    return pd.DataFrame(rows)


def compare_all(
    cross_participant_results: pd.DataFrame,
    temporal_holdout_results: pd.DataFrame,
) -> pd.DataFrame:
    """Single tidy comparison table across every model/evaluation mode."""
    return pd.concat([cross_participant_results, temporal_holdout_results], ignore_index=True).sort_values(
        ["mode", "mae"]
    ).reset_index(drop=True)

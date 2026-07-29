# src/longitudinal/trends.py
"""Additional per-subject longitudinal features beyond a linear slope:
trend variability, curvature (accelerating/decelerating decline), and
missingness/dropout pattern - all useful inputs to the ML models in
src/models/models.py alongside the slopes from slope_extraction.py.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_visit_variability(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    value_col: str = "memory",
) -> pd.DataFrame:
    """Std of residuals around each participant's own linear trend.

    Captures noisy testing/erratic performance independent of the trend
    itself (a fast decliner who tests consistently differs from a slow
    decliner who fluctuates wildly).
    """
    rows = []
    for pid, group in df.dropna(subset=[time_col, value_col]).groupby(id_col):
        x = group[time_col].to_numpy(dtype=float)
        y = group[value_col].to_numpy(dtype=float)

        if len(x) < 2 or np.unique(x).size < 2:
            rows.append({id_col: pid, f"{value_col}_resid_std": np.nan})
            continue

        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan
        rows.append({id_col: pid, f"{value_col}_resid_std": resid_std})

    return pd.DataFrame(rows)


def compute_curvature(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    value_col: str = "memory",
    min_obs: int = 3,
) -> pd.DataFrame:
    """Quadratic-term coefficient per participant: is decline accelerating (negative)
    or decelerating (positive)? NaN if fewer than `min_obs` distinct visits.
    """
    rows = []
    for pid, group in df.dropna(subset=[time_col, value_col]).groupby(id_col):
        x = group[time_col].to_numpy(dtype=float)
        y = group[value_col].to_numpy(dtype=float)

        if len(x) < min_obs or np.unique(x).size < min_obs:
            rows.append({id_col: pid, f"{value_col}_curvature": np.nan})
            continue

        coeffs = np.polyfit(x, y, 2)
        rows.append({id_col: pid, f"{value_col}_curvature": float(coeffs[0])})

    return pd.DataFrame(rows)


def compute_missingness_features(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    expected_grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Visit-count/adherence/gap features summarizing each participant's
    observation pattern - dropout in the simulated cohort grows with time
    (MNAR), so these are informative predictive features here, though that
    wouldn't automatically transfer to a real clinical dropout mechanism
    (see docs/limitations.md).
    """
    rows = []
    expected_n = len(expected_grid) if expected_grid is not None else None

    for pid, group in df.dropna(subset=[time_col]).groupby(id_col):
        times = np.sort(group[time_col].to_numpy(dtype=float))
        n_visits = len(times)
        adherence_rate = (n_visits / expected_n) if expected_n else np.nan
        gaps = np.diff(times) if n_visits > 1 else np.array([])

        rows.append({
            id_col: pid,
            "n_visits": n_visits,
            "adherence_rate": adherence_rate,
            "max_gap": float(gaps.max()) if gaps.size else np.nan,
            "mean_gap": float(gaps.mean()) if gaps.size else np.nan,
            "last_visit_month": float(times[-1]) if n_visits else np.nan,
        })

    return pd.DataFrame(rows)


def compute_baseline_to_last_delta(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    value_col: str = "memory",
) -> pd.DataFrame:
    """Simple non-model feature: raw change from first to last observed visit."""
    rows = []
    for pid, group in df.dropna(subset=[time_col, value_col]).sort_values(time_col).groupby(id_col):
        baseline_value = float(group[value_col].iloc[0])
        last_value = float(group[value_col].iloc[-1])
        elapsed = float(group[time_col].iloc[-1] - group[time_col].iloc[0])

        raw_delta = last_value - baseline_value
        raw_delta_per_month = raw_delta / elapsed if elapsed > 0 else np.nan

        rows.append({
            id_col: pid,
            f"{value_col}_baseline_value": baseline_value,
            f"{value_col}_last_value": last_value,
            f"{value_col}_raw_delta": raw_delta,
            f"{value_col}_raw_delta_per_month": raw_delta_per_month,
        })

    return pd.DataFrame(rows)


def build_trend_features(
    df: pd.DataFrame,
    domains: Sequence[str] = ("memory", "attention", "language"),
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    expected_grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Merge missingness features (computed once) with per-domain variability,
    curvature, and baseline-to-last-delta features into one wide table keyed
    by participant_id, joinable with slope_extraction.extract_slopes' output.
    """
    out = compute_missingness_features(df, id_col, time_col, expected_grid)

    for domain in domains:
        variability = compute_visit_variability(df, id_col, time_col, domain)
        curvature = compute_curvature(df, id_col, time_col, domain)
        delta = compute_baseline_to_last_delta(df, id_col, time_col, domain)

        domain_features = variability.merge(curvature, on=id_col, how="outer").merge(delta, on=id_col, how="outer")
        out = out.merge(domain_features, on=id_col, how="outer")

    return out

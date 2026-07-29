# src/longitudinal/slope_extraction.py
"""Per-subject decline-rate (slope) extraction.

Provides two simple per-subject baselines (OLS, Theil-Sen) and a population-
pooled mixed-effects model (statsmodels MixedLM, random intercept + random
slope per participant). The mixed model's BLUP slopes shrink individual
estimates toward the population trend in proportion to how little data a
given participant has - this is what makes decline-rate estimates for
sparsely-observed participants (heavy dropout, or CLI users with only a
couple of sessions) more statistically robust than an isolated per-subject
regression.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import theilslopes


def ols_slope_per_subject(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    value_col: str = "memory",
) -> pd.DataFrame:
    """Fit an independent OLS slope per participant. NaN if fewer than 2 distinct visits."""
    rows = []
    for pid, group in df.dropna(subset=[time_col, value_col]).groupby(id_col):
        x = group[time_col].to_numpy(dtype=float)
        y = group[value_col].to_numpy(dtype=float)
        n = len(x)

        if n < 2 or np.unique(x).size < 2:
            rows.append({
                id_col: pid,
                f"{value_col}_slope_ols": np.nan,
                f"{value_col}_intercept_ols": np.nan,
                f"{value_col}_n_obs": n,
                f"{value_col}_r2": np.nan,
            })
            continue

        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append({
            id_col: pid,
            f"{value_col}_slope_ols": float(slope),
            f"{value_col}_intercept_ols": float(intercept),
            f"{value_col}_n_obs": n,
            f"{value_col}_r2": r2,
        })

    return pd.DataFrame(rows)


def theilsen_slope_per_subject(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    value_col: str = "memory",
) -> pd.DataFrame:
    """Fit a robust Theil-Sen slope per participant (resists dropout-driven gaps/outliers)."""
    rows = []
    for pid, group in df.dropna(subset=[time_col, value_col]).groupby(id_col):
        x = group[time_col].to_numpy(dtype=float)
        y = group[value_col].to_numpy(dtype=float)

        if len(x) < 2 or np.unique(x).size < 2:
            rows.append({
                id_col: pid,
                f"{value_col}_slope_theilsen": np.nan,
                f"{value_col}_intercept_theilsen": np.nan,
                f"{value_col}_lo": np.nan,
                f"{value_col}_hi": np.nan,
            })
            continue

        slope, intercept, lo, hi = theilslopes(y, x)
        rows.append({
            id_col: pid,
            f"{value_col}_slope_theilsen": float(slope),
            f"{value_col}_intercept_theilsen": float(intercept),
            f"{value_col}_lo": float(lo),
            f"{value_col}_hi": float(hi),
        })

    return pd.DataFrame(rows)


def fit_mixedlm(
    df: pd.DataFrame,
    time_col: str = "visit_month",
    value_col: str = "memory",
    covariates: Sequence[str] = ("age_baseline", "education_years"),
    group_col: str = "participant_id",
):
    """Fit a random-intercept + random-slope mixed model, with convergence fallbacks.

    Tries lbfgs, then powell, then falls back to a random-intercept-only model
    if the richer random-slope model fails to converge - a real risk here
    since simulated visit dropout probability grows with time, leaving many
    participants with very few observations.
    """
    required = [time_col, value_col, group_col, *covariates]
    clean = df.dropna(subset=required)

    formula = f"{value_col} ~ {time_col} + " + " + ".join(covariates)

    def _try_fit(re_formula: str):
        model = smf.mixedlm(formula, data=clean, groups=clean[group_col], re_formula=re_formula)
        for method in ("lbfgs", "powell"):
            try:
                result = model.fit(reml=True, method=method)
                if result.converged:
                    return result
            except Exception:
                continue
        return None

    result = _try_fit(f"~{time_col}")
    if result is not None:
        return result

    # Fallback: random intercept only (no random slope term).
    model = smf.mixedlm(formula, data=clean, groups=clean[group_col], re_formula="~1")
    return model.fit(reml=True, method="lbfgs")


def mixedlm_slope_per_subject(
    df: pd.DataFrame,
    time_col: str = "visit_month",
    value_col: str = "memory",
    covariates: Sequence[str] = ("age_baseline", "education_years"),
    group_col: str = "participant_id",
) -> pd.DataFrame:
    """Extract per-subject BLUP slopes from a fitted MixedLM.

    Returns {value_col}_slope_mixedlm (population + subject-specific random
    effect), {value_col}_intercept_mixedlm, {value_col}_slope_population
    (fixed effect only, same for every row), and `converged`.
    """
    result = fit_mixedlm(df, time_col, value_col, covariates, group_col)

    fe_slope = float(result.fe_params[time_col])
    fe_intercept = float(result.fe_params.get("Intercept", 0.0))
    converged = bool(getattr(result, "converged", True))

    rows = []
    for pid, re in result.random_effects.items():
        re_index = list(re.index)
        intercept_terms = [t for t in re_index if t != time_col]
        re_intercept = float(re[intercept_terms[0]]) if intercept_terms else 0.0
        re_slope = float(re[time_col]) if time_col in re_index else 0.0

        rows.append({
            group_col: pid,
            f"{value_col}_slope_mixedlm": fe_slope + re_slope,
            f"{value_col}_intercept_mixedlm": fe_intercept + re_intercept,
            f"{value_col}_slope_population": fe_slope,
            f"{value_col}_converged": converged,
        })

    return pd.DataFrame(rows)


def extract_slopes(
    df: pd.DataFrame,
    domains: Sequence[str] = ("memory", "attention", "language"),
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    covariates: Sequence[str] = ("age_baseline", "education_years"),
) -> pd.DataFrame:
    """Extract OLS + Theil-Sen + MixedLM slope features for every domain.

    Works identically on raw simulated columns (memory/attention/language) or
    on normed z-score columns (memory_z/attention_z/language_z) once
    src/tasks/norms.py has been applied - this is the single entry point
    notebooks 02/03 should call.
    """
    out = None

    for domain in domains:
        ols = ols_slope_per_subject(df, id_col, time_col, domain)
        theilsen = theilsen_slope_per_subject(df, id_col, time_col, domain)

        try:
            mixedlm = mixedlm_slope_per_subject(df, time_col, domain, covariates, id_col)
        except Exception:
            mixedlm = pd.DataFrame({id_col: df[id_col].unique()})

        domain_features = ols.merge(theilsen, on=id_col, how="outer").merge(mixedlm, on=id_col, how="outer")
        out = domain_features if out is None else out.merge(domain_features, on=id_col, how="outer")

    return out if out is not None else pd.DataFrame({id_col: df[id_col].unique()})

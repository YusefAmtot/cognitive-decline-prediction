# src/tasks/norms.py
"""Demographic-adjusted normative (z-score) scoring for standardized tests.

Two independent reference samples are used, because the simulated
longitudinal cohort (MMSE-like point scales) and the CLI test battery
(word-recall counts, reaction time, percent-correct) are not on comparable
raw scales:

1. The simulated cohort's own baseline (visit_month == 0) rows, used to norm
   `memory` / `attention` / `language`.
2. A small FABRICATED cross-sectional reference sample for the CLI battery
   (see `generate_task_norm_reference`). This is illustrative only, not a
   validated clinical norm - see docs/limitations.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DEFAULT_PREDICTORS = ("age_baseline", "education_years")


@dataclass
class NormModel:
    predictors: Sequence[str]
    coef_: Dict[str, float] = field(default_factory=dict)
    intercept_: float = 0.0
    resid_std_: float = 1.0

    def predict(self, age_baseline: float, education_years: float) -> float:
        values = {"age_baseline": age_baseline, "education_years": education_years}
        return self.intercept_ + sum(self.coef_[p] * values[p] for p in self.predictors)


def fit_norms(
    reference_df: pd.DataFrame,
    score_col: str,
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
) -> NormModel:
    """Fit a regression-based normative model: score_col ~ predictors.

    Mirrors real neuropsych norming: predicted score is a linear function of
    demographics, and deviations from that prediction are expressed as a
    z-score using the residual standard deviation as the denominator.
    """
    clean = reference_df.dropna(subset=[*predictors, score_col])
    if len(clean) < 5:
        raise ValueError(
            f"Need at least 5 reference rows to fit norms for '{score_col}', got {len(clean)}."
        )

    X = clean[list(predictors)].to_numpy(dtype=float)
    y = clean[score_col].to_numpy(dtype=float)

    reg = LinearRegression().fit(X, y)
    residuals = y - reg.predict(X)
    resid_std = float(np.std(residuals, ddof=1)) or 1.0

    return NormModel(
        predictors=tuple(predictors),
        coef_={p: float(c) for p, c in zip(predictors, reg.coef_)},
        intercept_=float(reg.intercept_),
        resid_std_=resid_std,
    )


def zscore(raw_score: float, age_baseline: float, education_years: float, norm: NormModel) -> float:
    """Convert a raw score to a demographically-adjusted z-score using `norm`."""
    if raw_score is None or pd.isna(raw_score):
        return float("nan")
    predicted = norm.predict(age_baseline, education_years)
    return (raw_score - predicted) / norm.resid_std_


def fit_domain_norms(
    sim_df: pd.DataFrame,
    domains: Sequence[str] = ("memory", "attention", "language"),
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
) -> Dict[str, NormModel]:
    """Fit one NormModel per simulated domain using baseline-visit rows only."""
    baseline = sim_df[sim_df["visit_month"] == 0]
    return {domain: fit_norms(baseline, domain, predictors) for domain in domains}


def apply_domain_zscores(
    sim_df: pd.DataFrame,
    domain_norms: Dict[str, NormModel],
) -> pd.DataFrame:
    """Add `{domain}_z` columns to every row of the simulated longitudinal df."""
    df = sim_df.copy()
    for domain, norm in domain_norms.items():
        df[f"{domain}_z"] = [
            zscore(raw, age, edu, norm)
            for raw, age, edu in zip(df[domain], df["age_baseline"], df["education_years"])
        ]
    return df


# ---------------------------------------------------------------------------
# CLI test-battery normative reference (fabricated, illustrative only)
# ---------------------------------------------------------------------------

TASK_SCORE_COLUMNS = (
    "memory_immediate_score",
    "memory_delayed_score",
    "reaction_speed_score",
    "multidomain_percent",
)


def generate_task_norm_reference(n: int = 500, seed: int | None = 42) -> pd.DataFrame:
    """Simulate a cross-sectional normative reference sample for the CLI battery.

    NOT real normative data - a fabricated, illustrative distribution used so
    the CLI battery's raw scores can be demographically z-scored the same way
    the simulated cohort's domain scores are. See docs/limitations.md.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(72, 8, size=n)
    education_years = np.clip(rng.normal(13, 2.5, size=n), 8, 20)

    age_effect = (age - 72) * -0.05
    edu_effect = (education_years - 13) * 0.15

    memory_immediate_score = np.clip(
        rng.normal(6, 1.5, size=n) + age_effect + edu_effect, 0, 10
    )
    memory_delayed_score = np.clip(
        rng.normal(5, 1.8, size=n) + age_effect + edu_effect, 0, 10
    )

    # Reaction speed score = 1000/avg_rt_ms; older age -> slower -> lower speed score.
    avg_rt_ms = np.clip(rng.normal(550, 90, size=n) - age_effect * 20 - edu_effect * 5, 250, 1200)
    reaction_speed_score = 1000.0 / avg_rt_ms

    multidomain_percent = np.clip(
        rng.normal(0.75, 0.15, size=n) + age_effect * 0.01 + edu_effect * 0.01, 0.0, 1.0
    )

    return pd.DataFrame({
        "age_baseline": age,
        "education_years": education_years.astype(int),
        "memory_immediate_score": memory_immediate_score,
        "memory_delayed_score": memory_delayed_score,
        "reaction_speed_score": reaction_speed_score,
        "multidomain_percent": multidomain_percent,
    })


def fit_task_norms(
    reference_df: pd.DataFrame | None = None,
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
) -> Dict[str, NormModel]:
    """Fit one NormModel per CLI subtest score against the (fabricated) reference sample."""
    if reference_df is None:
        reference_df = generate_task_norm_reference()
    return {col: fit_norms(reference_df, col, predictors) for col in TASK_SCORE_COLUMNS}


def zscore_task_outputs(
    task_scores: Dict[str, float],
    age_baseline: float,
    education_years: float,
    task_norms: Dict[str, NormModel],
) -> Dict[str, float]:
    """Z-score each raw CLI subtest score in `task_scores` against `task_norms`."""
    return {
        f"{col}_z": zscore(task_scores.get(col), age_baseline, education_years, norm)
        for col, norm in task_norms.items()
    }


# Weights mapping CLI subtests onto the simulated cohort's domain vocabulary.
# `language_z` has no CLI equivalent and is always NaN for real participants -
# downstream composite/renorm logic must treat missing domains as absent,
# not zero.
MEMORY_SUBTEST_WEIGHTS = {"memory_immediate_score_z": 0.45, "memory_delayed_score_z": 0.55}
ATTENTION_SUBTEST_WEIGHTS = {"reaction_speed_score_z": 0.5, "multidomain_percent_z": 0.5}


def combine_real_to_domains(task_z_scores: Dict[str, float]) -> Dict[str, float]:
    """Map CLI subtest z-scores onto the simulated cohort's domain vocabulary.

    memory_z = weighted z of immediate + delayed recall.
    attention_z = weighted z of reaction speed + multidomain screen.
    language_z = NaN (no CLI language subtest exists).
    """

    def _weighted(weights: Dict[str, float]) -> float:
        present = {k: w for k, w in weights.items() if not pd.isna(task_z_scores.get(k, float("nan")))}
        if not present:
            return float("nan")
        wsum = sum(present.values())
        return sum(task_z_scores[k] * w for k, w in present.items()) / wsum

    return {
        "memory_z": _weighted(MEMORY_SUBTEST_WEIGHTS),
        "attention_z": _weighted(ATTENTION_SUBTEST_WEIGHTS),
        "language_z": float("nan"),
    }

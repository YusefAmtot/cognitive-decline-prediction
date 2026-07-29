import numpy as np
import pandas as pd
import pytest

from src.longitudinal.slope_extraction import (
    fit_mixedlm,
    mixedlm_slope_per_subject,
    ols_slope_per_subject,
    theilsen_slope_per_subject,
)


def _exact_linear_df():
    """5 participants, exact linear relationship, zero noise."""
    rows = []
    true_slopes = {"P1": -1.0, "P2": 0.5, "P3": 2.0, "P4": -0.25, "P5": 0.1}
    for pid, slope in true_slopes.items():
        intercept = 10.0
        for t in [0, 1, 2, 3, 4]:
            rows.append({"participant_id": pid, "visit_month": t, "value": intercept + slope * t})
    return pd.DataFrame(rows), true_slopes


def test_ols_slope_per_subject_recovers_exact_slope():
    df, true_slopes = _exact_linear_df()
    result = ols_slope_per_subject(df, value_col="value")

    for _, row in result.iterrows():
        expected = true_slopes[row["participant_id"]]
        assert row["value_slope_ols"] == pytest.approx(expected, abs=1e-9)
        assert row["value_r2"] == pytest.approx(1.0, abs=1e-9)


def test_ols_slope_per_subject_nan_for_single_observation():
    df = pd.DataFrame({"participant_id": ["P1"], "visit_month": [0], "value": [10.0]})
    result = ols_slope_per_subject(df, value_col="value")
    assert result.loc[0, "value_n_obs"] == 1
    assert np.isnan(result.loc[0, "value_slope_ols"])


def test_theilsen_nan_for_single_observation():
    df = pd.DataFrame({"participant_id": ["P1"], "visit_month": [0], "value": [10.0]})
    result = theilsen_slope_per_subject(df, value_col="value")
    assert np.isnan(result.loc[0, "value_slope_theilsen"])


def test_theilsen_more_robust_to_outlier_than_ols():
    true_slope = -0.5
    intercept = 20.0
    times = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    values = intercept + true_slope * times
    values_with_outlier = values.copy()
    values_with_outlier[3] += 15.0  # single large outlier

    df = pd.DataFrame({
        "participant_id": ["P1"] * len(times),
        "visit_month": times,
        "value": values_with_outlier,
    })

    ols = ols_slope_per_subject(df, value_col="value")
    theilsen = theilsen_slope_per_subject(df, value_col="value")

    ols_error = abs(ols.loc[0, "value_slope_ols"] - true_slope)
    theilsen_error = abs(theilsen.loc[0, "value_slope_theilsen"] - true_slope)

    assert theilsen_error < ols_error


def _pooled_mixedlm_df(seed=0):
    """A well-behaved pooled dataset: 30 noisy subjects (to estimate variance
    components) plus a LOW_OBS subject (2 clean visits) and a HIGH_OBS
    subject (20 clean visits) sharing the same large true random effect, far
    from the population slope - so BLUP shrinkage should be clearly visible
    and the two subjects' shrinkage amounts comparable.
    """
    rng = np.random.default_rng(seed)
    population_slope = -0.10
    population_intercept = 28.0
    tau = 0.05
    sigma = 0.3

    rows = []
    times_full = np.array([0, 6, 12, 18, 24, 30], dtype=float)
    for i in range(30):
        pid = f"P{i:03d}"
        age = rng.normal(72, 6)
        edu = rng.normal(13, 2.5)
        subject_slope = population_slope + rng.normal(0, tau)
        for t in times_full:
            value = population_intercept + subject_slope * t + rng.normal(0, sigma)
            rows.append({
                "participant_id": pid, "visit_month": t,
                "age_baseline": age, "education_years": edu, "value": value,
            })

    extreme_slope = population_slope + 5 * tau

    low_pid = "LOW_OBS"
    for t in [0.0, 6.0]:
        rows.append({
            "participant_id": low_pid, "visit_month": t,
            "age_baseline": 72.0, "education_years": 13.0,
            "value": population_intercept + extreme_slope * t,
        })

    high_pid = "HIGH_OBS"
    for t in np.arange(0, 60, 3):
        rows.append({
            "participant_id": high_pid, "visit_month": float(t),
            "age_baseline": 72.0, "education_years": 13.0,
            "value": population_intercept + extreme_slope * t,
        })

    return pd.DataFrame(rows), low_pid, high_pid


def test_mixedlm_converges_with_random_slope():
    df, _, _ = _pooled_mixedlm_df()
    result = fit_mixedlm(
        df, time_col="visit_month", value_col="value",
        covariates=("age_baseline", "education_years"), group_col="participant_id",
    )
    assert result.converged
    first_group_re = next(iter(result.random_effects.values()))
    assert "visit_month" in first_group_re.index


def test_mixedlm_blup_shrinks_low_obs_subject_more_than_high_obs_subject():
    df, low_pid, high_pid = _pooled_mixedlm_df()

    blup = mixedlm_slope_per_subject(
        df, time_col="visit_month", value_col="value",
        covariates=("age_baseline", "education_years"), group_col="participant_id",
    )
    assert blup["value_converged"].all()

    ols = ols_slope_per_subject(df, value_col="value")
    population_slope = blup["value_slope_population"].iloc[0]

    low_ols = ols.loc[ols["participant_id"] == low_pid, "value_slope_ols"].iloc[0]
    low_blup = blup.loc[blup["participant_id"] == low_pid, "value_slope_mixedlm"].iloc[0]
    high_ols = ols.loc[ols["participant_id"] == high_pid, "value_slope_ols"].iloc[0]
    high_blup = blup.loc[blup["participant_id"] == high_pid, "value_slope_mixedlm"].iloc[0]

    # Both subjects share the same extreme true slope, far from the
    # population - BLUP should pull the sparsely-observed subject's estimate
    # toward the population slope more than it pulls the richly-observed one.
    low_shrinkage = abs(low_ols - population_slope) - abs(low_blup - population_slope)
    high_shrinkage = abs(high_ols - population_slope) - abs(high_blup - population_slope)

    assert low_shrinkage > 0
    assert low_shrinkage > high_shrinkage

# src/visualisation/visualisation.py
"""Shared plotting helpers for notebooks 01-07."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sample_trajectories(
    df: pd.DataFrame,
    value_col: str = "memory",
    n: int = 5,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    seed: int = 0,
    ax=None,
):
    ax = ax or plt.gca()
    sample_ids = df[id_col].drop_duplicates().sample(n, random_state=seed)

    for pid in sample_ids:
        sub = df[df[id_col] == pid].sort_values(time_col)
        ax.plot(sub[time_col], sub[value_col], marker="o", label=str(pid))

    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"Sample {value_col} trajectories")
    ax.legend(title=id_col, fontsize="small")
    ax.grid(True)
    return ax


def plot_slope_shrinkage(slope_table: pd.DataFrame, domain: str = "memory", ax=None):
    """Scatter of independent OLS slope vs. MixedLM BLUP slope, colored by
    n_obs - visually shows shrinkage: participants with few visits cluster
    toward the population line, participants with many visits stay closer to
    their own OLS estimate.
    """
    ax = ax or plt.gca()
    ols_col, blup_col, n_col, pop_col = (
        f"{domain}_slope_ols", f"{domain}_slope_mixedlm", f"{domain}_n_obs", f"{domain}_slope_population",
    )
    valid = slope_table.dropna(subset=[ols_col, blup_col])

    sc = ax.scatter(valid[ols_col], valid[blup_col], c=valid.get(n_col), cmap="viridis", alpha=0.7)
    lims = [min(valid[ols_col].min(), valid[blup_col].min()), max(valid[ols_col].max(), valid[blup_col].max())]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x")

    if pop_col in valid.columns:
        pop_slope = valid[pop_col].iloc[0]
        ax.axhline(pop_slope, color="red", linestyle=":", label="population slope")

    ax.set_xlabel(f"{domain} OLS slope (independent per-subject)")
    ax.set_ylabel(f"{domain} MixedLM BLUP slope (pooled)")
    ax.set_title(f"Shrinkage: {domain} slope, OLS vs. MixedLM")
    plt.colorbar(sc, ax=ax, label="n_obs")
    ax.legend()
    return ax


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = "mae", ax=None):
    ax = ax or plt.gca()
    data = comparison_df.sort_values(metric)
    ax.bar(data["model"] + " (" + data["mode"] + ")", data[metric])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model comparison - {metric.upper()} (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_feature_importance(feature_names, importances, ax=None):
    ax = ax or plt.gca()
    order = np.argsort(importances)
    ax.barh(np.array(feature_names)[order], np.array(importances)[order])
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    return ax

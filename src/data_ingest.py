# src/data_ingest.py
"""Top-level convenience loaders for notebooks: combine preprocessing +
normative scoring into ready-to-use, demographically z-scored longitudinal
data - both the simulated cohort and any real CLI sessions recorded so far.
"""
from __future__ import annotations

import os

import pandas as pd

from src.preprocessing import (
    CANONICAL_COLUMNS,
    load_data,
    merge_simulated_and_real,
    normalize_sessions_to_long,
    validate_longitudinal,
)
from src.tasks.norms import apply_domain_zscores, fit_domain_norms, fit_task_norms
from src.sessions.persistence import DEFAULT_PARTICIPANTS_PATH, DEFAULT_SESSIONS_LOG_PATH

DEFAULT_SIM_PATH = "data/simulated/longitudinal_simulated.csv"


def load_simulated_longitudinal(path: str = DEFAULT_SIM_PATH):
    """Load, validate, and demographically norm the simulated cohort.

    Returns (normed_df, domain_norms) - domain_norms is reused whenever a
    real participant's scores need to be expressed on the same scale.
    """
    df = validate_longitudinal(load_data(path))
    domain_norms = fit_domain_norms(df)
    return apply_domain_zscores(df, domain_norms), domain_norms


def load_real_sessions_normed(
    sessions_path: str = DEFAULT_SESSIONS_LOG_PATH,
    participants_path: str = DEFAULT_PARTICIPANTS_PATH,
) -> pd.DataFrame:
    """Load every persisted CLI session across all participants, normed onto
    the same memory_z/attention_z/language_z schema as the simulated cohort.
    Returns an empty canonical-schema frame if no sessions exist yet.
    """
    if not (os.path.exists(sessions_path) and os.path.exists(participants_path)):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    sessions = pd.read_csv(sessions_path, dtype={"participant_id": str})
    participants = pd.read_csv(participants_path, dtype={"participant_id": str}).set_index("participant_id")

    task_norms = fit_task_norms()
    return normalize_sessions_to_long(sessions, participants, task_norms)


def load_combined_longitudinal(
    sim_path: str = DEFAULT_SIM_PATH,
    sessions_path: str = DEFAULT_SESSIONS_LOG_PATH,
    participants_path: str = DEFAULT_PARTICIPANTS_PATH,
) -> pd.DataFrame:
    """The full analysis-ready dataset: simulated cohort + any real sessions,
    both on the shared memory_z/attention_z/language_z schema.
    """
    sim_normed, _ = load_simulated_longitudinal(sim_path)
    real_normed = load_real_sessions_normed(sessions_path, participants_path)

    if real_normed.empty:
        out = sim_normed.copy()
        out["source"] = "simulated"
        return out[CANONICAL_COLUMNS]

    return merge_simulated_and_real(sim_normed, real_normed)

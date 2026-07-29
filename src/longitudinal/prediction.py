# src/longitudinal/prediction.py
"""Connects real CLI session history to the slope-extraction pipeline used
for the simulated cohort, producing a personalized decline-rate estimate.

A single participant's own sessions can't be fit with MixedLM in isolation -
a mixed-effects model needs many groups to estimate between-participant
variance - so real sessions are normed and merged into the simulated cohort
before fitting, then that participant's row is pulled back out. This is what
turns "population-level model" into "personalized to this person."
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import pandas as pd

from src.preprocessing import merge_simulated_and_real, normalize_sessions_to_long, validate_longitudinal
from src.tasks.norms import apply_domain_zscores, fit_domain_norms, fit_task_norms
from src.longitudinal.slope_extraction import extract_slopes
from src.sessions.persistence import DEFAULT_SESSIONS_LOG_PATH, MIN_SESSIONS_FOR_PERSONALIZATION, load_participant_sessions

DEFAULT_SIM_PATH = "data/simulated/longitudinal_simulated.csv"


def estimate_personalized_decline(
    participant_id: str,
    profile: Dict[str, Any],
    sim_path: str = DEFAULT_SIM_PATH,
    sessions_path: str = DEFAULT_SESSIONS_LOG_PATH,
    domains: Sequence[str] = ("memory_z", "attention_z"),
) -> Optional[pd.DataFrame]:
    """Return this participant's decline-rate estimate row, or None if they
    don't have MIN_SESSIONS_FOR_PERSONALIZATION real sessions yet.

    `language_z` is intentionally excluded from the default domains - the CLI
    battery has no language subtest, so it would always be NaN for real
    participants (see src/tasks/norms.py:combine_real_to_domains).
    """
    sessions = load_participant_sessions(participant_id, sessions_path)
    if len(sessions) < MIN_SESSIONS_FOR_PERSONALIZATION:
        return None

    sim_df = validate_longitudinal(pd.read_csv(sim_path))
    domain_norms = fit_domain_norms(sim_df)
    sim_normed = apply_domain_zscores(sim_df, domain_norms)

    task_norms = fit_task_norms()
    participants_df = pd.DataFrame([{**profile, "participant_id": participant_id}]).set_index("participant_id")
    real_normed = normalize_sessions_to_long(sessions, participants_df, task_norms)

    combined = merge_simulated_and_real(sim_normed, real_normed)
    slopes = extract_slopes(combined, domains=domains)

    participant_row = slopes[slopes["participant_id"] == participant_id]
    return participant_row if not participant_row.empty else None

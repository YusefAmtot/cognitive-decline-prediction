# src/sessions/persistence.py
"""Persists CLI test-battery sessions per participant so repeated play.py runs
build up a real longitudinal trajectory, feeding the same slope-extraction /
prediction pipeline used for the simulated cohort (see src/longitudinal/).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

DEFAULT_PARTICIPANTS_PATH = "data/sessions/participants.csv"
DEFAULT_SESSIONS_LOG_PATH = "data/sessions/sessions_log.csv"

PARTICIPANT_COLUMNS = [
    "participant_id", "age_baseline", "education_years", "sex", "first_session_timestamp",
]

SESSION_COLUMNS = [
    "session_id", "participant_id", "session_timestamp",
    "memory_immediate_score", "memory_immediate_max",
    "memory_delayed_score", "memory_delayed_max",
    "reaction_avg_rt_ms", "reaction_speed_score", "reaction_accuracy",
    "multidomain_score", "multidomain_max", "multidomain_percent",
    "composite_raw_score",
]

MIN_SESSIONS_FOR_PERSONALIZATION = 3


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def find_participant_profile(
    participant_id: str,
    participants_path: str = DEFAULT_PARTICIPANTS_PATH,
) -> Dict[str, Any] | None:
    """Look up an existing participant's demographic profile, or None if new."""
    if not Path(participants_path).exists():
        return None

    participants = pd.read_csv(participants_path, dtype={"participant_id": str})
    existing = participants[participants["participant_id"] == participant_id]
    return existing.iloc[0].to_dict() if not existing.empty else None


def save_participant_profile(
    participant_id: str,
    age_baseline: float,
    education_years: int,
    sex: str,
    participants_path: str = DEFAULT_PARTICIPANTS_PATH,
) -> Dict[str, Any]:
    """Create and persist a new participant's demographic profile.

    Shared by both the CLI (src/play.py) and the web app (webapp/app.py) -
    neither should call this for a participant_id that already has a profile.
    """
    _ensure_parent(participants_path)

    if Path(participants_path).exists():
        participants = pd.read_csv(participants_path, dtype={"participant_id": str})
    else:
        participants = pd.DataFrame(columns=PARTICIPANT_COLUMNS)

    profile = {
        "participant_id": participant_id,
        "age_baseline": age_baseline,
        "education_years": education_years,
        "sex": sex,
        "first_session_timestamp": datetime.now().isoformat(),
    }

    participants = pd.concat([participants, pd.DataFrame([profile])], ignore_index=True)
    participants.to_csv(participants_path, index=False)

    return profile


def get_or_create_participant_profile(
    participant_id: str,
    participants_path: str = DEFAULT_PARTICIPANTS_PATH,
    prompt_fn=input,
    print_fn=print,
) -> Dict[str, Any]:
    """Look up (or create, prompting once) a participant's demographic profile.

    Age/education/sex are required for demographic norming (src/tasks/norms.py)
    but the original play.py only ever asked for a participant_id - this closes
    that gap. Demographics are only collected the first time a participant_id
    is seen; every later call just returns the stored row.
    """
    existing = find_participant_profile(participant_id, participants_path)
    if existing is not None:
        return existing

    print_fn("\nFirst time seeing this participant ID - a few details are needed for personalized scoring.")
    age_baseline = float(prompt_fn("Age: ").strip())
    education_years = int(prompt_fn("Years of education: ").strip())
    sex = prompt_fn("Sex (M/F): ").strip().upper()

    return save_participant_profile(participant_id, age_baseline, education_years, sex, participants_path)


def _flatten_session_outputs(participant_id: str, outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    reaction = outputs.get("reaction_simple", outputs.get("reaction_go_nogo", {}))
    memory_immediate = outputs.get("memory_immediate_recall", {})
    memory_delayed = outputs.get("memory_delayed_recall", {})
    multidomain = outputs.get("multidomain_screen", {})
    composite = outputs.get("composite", {})

    return {
        "participant_id": participant_id,
        "session_timestamp": datetime.now().isoformat(),
        "memory_immediate_score": memory_immediate.get("score"),
        "memory_immediate_max": memory_immediate.get("max_score"),
        "memory_delayed_score": memory_delayed.get("score"),
        "memory_delayed_max": memory_delayed.get("max_score"),
        "reaction_avg_rt_ms": reaction.get("metrics", {}).get("avg_rt_ms"),
        "reaction_speed_score": reaction.get("metrics", {}).get("speed_score"),
        "reaction_accuracy": reaction.get("metrics", {}).get("accuracy"),
        "multidomain_score": multidomain.get("score"),
        "multidomain_max": multidomain.get("max_score"),
        "multidomain_percent": multidomain.get("metrics", {}).get("percent"),
        "composite_raw_score": composite.get("score"),
    }


def append_session(
    participant_id: str,
    outputs: Dict[str, Dict[str, Any]],
    sessions_path: str = DEFAULT_SESSIONS_LOG_PATH,
) -> Dict[str, Any]:
    """Flatten and append one completed CLI session to the sessions log."""
    _ensure_parent(sessions_path)

    if Path(sessions_path).exists():
        log = pd.read_csv(sessions_path, dtype={"participant_id": str})
    else:
        log = pd.DataFrame(columns=SESSION_COLUMNS)

    row = _flatten_session_outputs(participant_id, outputs)
    row["session_id"] = f"{participant_id}-{len(log[log['participant_id'] == participant_id]) + 1}" \
        if not log.empty else f"{participant_id}-1"

    log = pd.concat([log, pd.DataFrame([row])[SESSION_COLUMNS]], ignore_index=True)
    log.to_csv(sessions_path, index=False)

    return row


def load_participant_sessions(
    participant_id: str,
    sessions_path: str = DEFAULT_SESSIONS_LOG_PATH,
) -> pd.DataFrame:
    """Load all persisted sessions for one participant, oldest first."""
    if not Path(sessions_path).exists():
        return pd.DataFrame(columns=SESSION_COLUMNS)

    log = pd.read_csv(sessions_path, dtype={"participant_id": str})
    subset = log[log["participant_id"] == participant_id].copy()
    subset["session_timestamp"] = pd.to_datetime(subset["session_timestamp"])
    return subset.sort_values("session_timestamp").reset_index(drop=True)


def count_sessions(participant_id: str, sessions_path: str = DEFAULT_SESSIONS_LOG_PATH) -> int:
    return len(load_participant_sessions(participant_id, sessions_path))


def has_enough_sessions_for_personalization(
    participant_id: str, sessions_path: str = DEFAULT_SESSIONS_LOG_PATH
) -> bool:
    return count_sessions(participant_id, sessions_path) >= MIN_SESSIONS_FOR_PERSONALIZATION

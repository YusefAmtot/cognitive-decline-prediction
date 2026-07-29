# src/preprocessing.py
from __future__ import annotations

from typing import Iterable, Sequence
import pandas as pd

REQUIRED_SIM_COLUMNS = [
    "participant_id", "visit_month", "age_baseline",
    "education_years", "sex", "memory", "attention", "language",
]

# Columns every long-format record (simulated or real) must carry once
# normed z-scores have been attached, so the two sources are genuinely
# comparable downstream (slope extraction, trend features, modeling).
CANONICAL_COLUMNS = [
    "participant_id", "visit_month", "age_baseline", "education_years", "sex",
    "memory_z", "attention_z", "language_z", "source",
]

VALID_SEX_VALUES = {"M", "F"}


def load_data(path: str = "data/simulated/longitudinal_simulated.csv") -> pd.DataFrame:
    """Load the raw simulated longitudinal CSV and coerce dtypes."""
    df = pd.read_csv(path)
    if "visit_month" in df.columns:
        df["visit_month"] = df["visit_month"].astype(float)
    if "education_years" in df.columns:
        df["education_years"] = df["education_years"].astype(int)
    return df


def validate_longitudinal(
    df: pd.DataFrame,
    required_cols: Sequence[str] = REQUIRED_SIM_COLUMNS,
) -> pd.DataFrame:
    """Validate a long-format longitudinal dataframe and return it sorted/deduped.

    Raises ValueError on missing columns, invalid `sex` values, nulls in the
    identifying columns, or duplicate (participant_id, visit_month) rows.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["participant_id"].isnull().any():
        raise ValueError("Null participant_id values found.")
    if df["visit_month"].isnull().any():
        raise ValueError("Null visit_month values found.")

    if "sex" in df.columns:
        bad_sex = set(df["sex"].dropna().unique()) - VALID_SEX_VALUES
        if bad_sex:
            raise ValueError(f"Invalid sex values found: {bad_sex}")

    if "age_baseline" in df.columns:
        out_of_range = df[(df["age_baseline"] < 40) | (df["age_baseline"] > 100)]
        if not out_of_range.empty:
            raise ValueError(
                f"age_baseline outside plausible bounds (40-100) for "
                f"{len(out_of_range)} row(s)."
            )

    dup_mask = df.duplicated(subset=["participant_id", "visit_month"], keep=False)
    if dup_mask.any():
        raise ValueError(
            "Duplicate (participant_id, visit_month) rows found: "
            f"{df.loc[dup_mask, ['participant_id', 'visit_month']].to_dict('records')}"
        )

    return df.sort_values(["participant_id", "visit_month"]).reset_index(drop=True)


def sessions_wide_to_long(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape data/sessions/sessions_log.csv rows into the canonical long schema.

    `visit_month` is derived per participant as the number of months elapsed
    since that participant's first recorded session.
    """
    df = sessions_df.copy()
    df["session_timestamp"] = pd.to_datetime(df["session_timestamp"])

    first_ts = df.groupby("participant_id")["session_timestamp"].transform("min")
    elapsed_days = (df["session_timestamp"] - first_ts).dt.total_seconds() / 86400.0
    df["visit_month"] = elapsed_days / 30.44

    return df.sort_values(["participant_id", "visit_month"]).reset_index(drop=True)


def truncate_to_early_visits(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "visit_month",
    n_visits: int = 2,
) -> pd.DataFrame:
    """Keep only each participant's earliest `n_visits` visits.

    Used to build cold-start features (simulating a brand-new user with
    minimal history) for src/models/models.py:build_feature_matrix, so
    training doesn't leak full-history information the model wouldn't
    actually have for a new participant.
    """
    sorted_df = df.sort_values([id_col, time_col])
    return sorted_df.groupby(id_col, group_keys=False).head(n_visits).reset_index(drop=True)


def normalize_sessions_to_long(
    sessions_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    task_norms,
) -> pd.DataFrame:
    """Convert raw sessions_log.csv rows (any number of participants) into the
    canonical normed long schema, using each session's own participant profile
    for demographic z-scoring.

    `participants_df` must be indexed by participant_id (see
    src/sessions/persistence.py). Sessions whose participant_id has no
    matching profile are silently skipped.
    """
    from src.tasks.norms import combine_real_to_domains, zscore_task_outputs

    rows = []
    for _, session in sessions_df.iterrows():
        pid = session["participant_id"]
        if pid not in participants_df.index:
            continue
        profile = participants_df.loc[pid]

        task_scores = {
            "memory_immediate_score": session.get("memory_immediate_score"),
            "memory_delayed_score": session.get("memory_delayed_score"),
            "reaction_speed_score": session.get("reaction_speed_score"),
            "multidomain_percent": session.get("multidomain_percent"),
        }
        z_scores = zscore_task_outputs(
            task_scores, profile["age_baseline"], profile["education_years"], task_norms
        )
        domains = combine_real_to_domains(z_scores)

        rows.append({
            "participant_id": pid,
            "session_timestamp": session["session_timestamp"],
            "age_baseline": profile["age_baseline"],
            "education_years": profile["education_years"],
            "sex": profile["sex"],
            **domains,
        })

    if not rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    return sessions_wide_to_long(pd.DataFrame(rows))


def merge_simulated_and_real(
    sim_long: pd.DataFrame,
    real_long: pd.DataFrame,
    columns: Iterable[str] = CANONICAL_COLUMNS,
) -> pd.DataFrame:
    """Concatenate normed simulated and real longitudinal data on the shared schema.

    Both inputs must already carry `memory_z`/`attention_z`/`language_z`
    columns (see src/tasks/norms.py) - raw scales differ between the
    simulated cohort and the CLI battery and are not compared directly.
    """
    columns = list(columns)

    sim = sim_long.copy()
    real = real_long.copy()
    sim["source"] = "simulated"
    real["source"] = "real"

    for name, frame in (("sim_long", sim), ("real_long", real)):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing canonical columns: {missing}")

    merged = pd.concat([sim[columns], real[columns]], ignore_index=True)
    return merged.sort_values(["participant_id", "visit_month"]).reset_index(drop=True)

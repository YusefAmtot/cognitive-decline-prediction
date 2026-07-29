import pandas as pd
import pytest

from src.preprocessing import (
    CANONICAL_COLUMNS,
    merge_simulated_and_real,
    sessions_wide_to_long,
    validate_longitudinal,
)


def _valid_long_df():
    return pd.DataFrame({
        "participant_id": ["P0001", "P0001", "P0002"],
        "visit_month": [0, 6, 0],
        "age_baseline": [70.0, 70.0, 65.0],
        "education_years": [12, 12, 16],
        "sex": ["M", "M", "F"],
        "memory": [28.0, 27.5, 29.0],
        "attention": [90.0, 88.0, 92.0],
        "language": [45.0, 44.0, 46.0],
    })


def test_validate_longitudinal_raises_on_missing_column():
    df = _valid_long_df().drop(columns=["memory"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_longitudinal(df)


def test_validate_longitudinal_raises_on_bad_sex():
    df = _valid_long_df()
    df.loc[0, "sex"] = "X"
    with pytest.raises(ValueError, match="Invalid sex values"):
        validate_longitudinal(df)


def test_validate_longitudinal_raises_on_duplicate_rows():
    df = pd.concat([_valid_long_df(), _valid_long_df().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_longitudinal(df)


def test_validate_longitudinal_sorts_and_returns():
    df = _valid_long_df().iloc[::-1].reset_index(drop=True)
    out = validate_longitudinal(df)

    assert list(out.columns) == list(df.columns)
    assert len(out) == len(df)
    # sorted by participant_id, then visit_month within each participant
    assert list(out["participant_id"]) == ["P0001", "P0001", "P0002"]
    assert list(out[out["participant_id"] == "P0001"]["visit_month"]) == [0, 6]


def test_sessions_wide_to_long_computes_visit_month_from_first_session():
    sessions = pd.DataFrame({
        "participant_id": ["P0001", "P0001", "P0001"],
        "session_timestamp": ["2026-01-01", "2026-02-01", "2026-04-02"],
    })
    out = sessions_wide_to_long(sessions)

    assert out.loc[out["session_timestamp"] == pd.Timestamp("2026-01-01"), "visit_month"].iloc[0] == pytest.approx(0.0)
    # ~31 days later ~= 1.02 months
    assert out.loc[out["session_timestamp"] == pd.Timestamp("2026-02-01"), "visit_month"].iloc[0] == pytest.approx(1.02, abs=0.05)
    # ~91 days later ~= 3.0 months
    assert out.loc[out["session_timestamp"] == pd.Timestamp("2026-04-02"), "visit_month"].iloc[0] == pytest.approx(3.0, abs=0.05)


def test_sessions_wide_to_long_is_sorted_by_participant_then_time():
    sessions = pd.DataFrame({
        "participant_id": ["P0002", "P0001", "P0001"],
        "session_timestamp": ["2026-01-01", "2026-02-01", "2026-01-01"],
    })
    out = sessions_wide_to_long(sessions)
    assert list(out["participant_id"]) == ["P0001", "P0001", "P0002"]


def _canonical_frame(source_rows):
    return pd.DataFrame(source_rows, columns=[c for c in CANONICAL_COLUMNS if c != "source"])


def test_merge_simulated_and_real_preserves_row_count_and_schema():
    sim = _canonical_frame([
        {"participant_id": "P0001", "visit_month": 0, "age_baseline": 70, "education_years": 12,
         "sex": "M", "memory_z": 0.1, "attention_z": 0.2, "language_z": 0.3},
    ])
    real = _canonical_frame([
        {"participant_id": "R0001", "visit_month": 0, "age_baseline": 68, "education_years": 14,
         "sex": "F", "memory_z": -0.1, "attention_z": -0.2, "language_z": float("nan")},
    ])

    merged = merge_simulated_and_real(sim, real)

    assert len(merged) == len(sim) + len(real)
    assert set(merged.columns) == set(CANONICAL_COLUMNS)
    assert set(merged["source"]) == {"simulated", "real"}


def test_merge_simulated_and_real_raises_on_missing_canonical_column():
    sim = _canonical_frame([
        {"participant_id": "P0001", "visit_month": 0, "age_baseline": 70, "education_years": 12,
         "sex": "M", "memory_z": 0.1, "attention_z": 0.2, "language_z": 0.3},
    ]).drop(columns=["memory_z"])
    real = _canonical_frame([
        {"participant_id": "R0001", "visit_month": 0, "age_baseline": 68, "education_years": 14,
         "sex": "F", "memory_z": -0.1, "attention_z": -0.2, "language_z": float("nan")},
    ])

    with pytest.raises(ValueError, match="missing canonical columns"):
        merge_simulated_and_real(sim, real)

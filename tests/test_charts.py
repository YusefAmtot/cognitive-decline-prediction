import pandas as pd

from webapp.charts import build_domain_history_chart, build_score_history_charts


def _sessions_df(**columns):
    n = len(next(iter(columns.values())))
    data = {"session_timestamp": pd.to_datetime([f"2026-01-{i + 1:02d}" for i in range(n)])}
    data.update(columns)
    return pd.DataFrame(data)


def test_build_score_history_charts_needs_at_least_two_points():
    sessions = _sessions_df(memory_immediate_score=[6])
    assert build_score_history_charts(sessions) == []


def test_build_score_history_charts_produces_one_chart_per_metric_with_data():
    sessions = _sessions_df(
        memory_immediate_score=[6, 7],
        memory_delayed_score=[4, 5],
        reaction_speed_score=[2.0, 2.5],
        multidomain_percent=[0.8, 0.9],
    )
    charts = build_score_history_charts(sessions)
    assert {c["title"] for c in charts} == {
        "Immediate recall", "Delayed recall", "Reaction speed", "Attention / executive screen",
    }


def test_build_score_history_charts_skips_a_metric_stuck_below_two_valid_points():
    sessions = _sessions_df(memory_immediate_score=[6, None, None])
    assert build_score_history_charts(sessions) == []


def test_build_score_history_charts_axis_floor_never_shows_negative_zero():
    sessions = _sessions_df(memory_immediate_score=[0, 1])
    chart = build_score_history_charts(sessions)[0]
    assert not chart["y_axis_bottom_label"].startswith("-")


def test_build_domain_history_chart_returns_none_without_two_points_per_series():
    domain_long = pd.DataFrame({
        "session_timestamp": pd.to_datetime(["2026-01-01"]),
        "memory_z": [0.5],
        "attention_z": [0.2],
    })
    assert build_domain_history_chart(domain_long) is None


def test_build_domain_history_chart_builds_both_series_on_one_shared_axis():
    domain_long = pd.DataFrame({
        "session_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "memory_z": [-1.0, -0.5, 0.0],
        "attention_z": [2.0, 2.0, 2.0],
    })
    chart = build_domain_history_chart(domain_long)

    assert [s["key"] for s in chart["series"]] == ["memory_z", "attention_z"]
    assert chart["table_rows"][0]["memory_z"] == "-1.00"
    assert chart["table_rows"][0]["attention_z"] == "+2.00"


def test_build_domain_history_chart_keeps_zero_reference_line_in_view():
    # Every value here is positive - a naive min/max axis would clip the
    # z=0 "population average" line off the bottom of the plot entirely.
    domain_long = pd.DataFrame({
        "session_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "memory_z": [1.0, 1.5],
        "attention_z": [2.0, 2.5],
    })
    chart = build_domain_history_chart(domain_long)
    assert chart["y_top"] <= chart["zero_y"] <= chart["y_bottom"]


def test_build_domain_history_chart_ignores_columns_with_no_data():
    domain_long = pd.DataFrame({
        "session_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "memory_z": [-1.0, -0.5],
        "attention_z": [float("nan"), float("nan")],
    })
    chart = build_domain_history_chart(domain_long)
    assert [s["key"] for s in chart["series"]] == ["memory_z"]

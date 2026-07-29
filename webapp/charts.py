# webapp/charts.py
"""Builds accessible inline SVG line-chart data for the results page.

Plain SVG with no JS or CDN dependency, so it keeps working in this app's
offline, single-user setup (see webapp/app.py). Each chart pairs the visual
line with a text trend summary (used as the SVG's accessible name) and an
on-page data table, so someone using a screen reader - or who just prefers
exact numbers over reading a line's shape - gets the same information as
someone looking at the plot.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

CHART_WIDTH = 480
CHART_HEIGHT = 180
PAD_LEFT = 48
PAD_RIGHT = 16
PAD_TOP = 20
PAD_BOTTOM = 28

# (session-log column, display title, unit label, value format)
METRICS: List[Tuple[str, str, str, str]] = [
    ("memory_immediate_score", "Immediate recall", "words recalled", "{:.0f}"),
    ("memory_delayed_score", "Delayed recall", "words recalled", "{:.0f}"),
    ("reaction_speed_score", "Reaction speed", "speed score (higher = faster)", "{:.2f}"),
    ("multidomain_percent", "Attention / executive screen", "% correct", "{:.0%}"),
]


def _y_range(values: List[float]) -> Tuple[float, float]:
    """Padded axis bounds. Clamped at 0 since every tracked metric (word
    counts, speed score, percent correct) is non-negative - otherwise the
    padding below a low/flat score renders as a confusing "-0" axis label.
    """
    lo, hi = min(values), max(values)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        return max(0.0, lo - pad), hi + pad
    pad = (hi - lo) * 0.1
    return max(0.0, lo - pad), hi + pad


def build_score_history_charts(sessions: pd.DataFrame) -> List[Dict[str, Any]]:
    """One line-chart config per tracked metric with >=2 recorded values.

    Sessions with a missing value for a given metric are skipped for that
    metric's line but don't shift the x-position of the sessions around
    them, so a gap in "reaction speed" doesn't misalign "delayed recall".
    """
    n = len(sessions)
    if n == 0:
        return []

    dates = sessions["session_timestamp"].dt.strftime("%b %d").tolist()
    x0, x1 = PAD_LEFT, CHART_WIDTH - PAD_RIGHT
    y0, y1 = CHART_HEIGHT - PAD_BOTTOM, PAD_TOP

    def sx(i: int) -> float:
        return x0 if n == 1 else x0 + (x1 - x0) * i / (n - 1)

    charts = []
    for key, title, unit, fmt in METRICS:
        if key not in sessions.columns:
            continue
        values = sessions[key].tolist()
        plot_idx = [i for i, v in enumerate(values) if v is not None and v == v]
        if len(plot_idx) < 2:
            continue

        plot_values = [values[i] for i in plot_idx]
        y_lo, y_hi = _y_range(plot_values)

        def sy(v: float, y_lo=y_lo, y_hi=y_hi) -> float:
            return y0 if y_hi == y_lo else y0 + (y1 - y0) * (v - y_lo) / (y_hi - y_lo)

        points = []
        for i in plot_idx:
            v = values[i]
            points.append({
                "x": round(sx(i), 1),
                "y": round(sy(v), 1),
                "value_label": fmt.format(v),
                "session_label": f"Session {i + 1} ({dates[i]}): {fmt.format(v)}",
            })

        first_v, last_v = plot_values[0], plot_values[-1]
        if last_v > first_v:
            direction = "increased"
        elif last_v < first_v:
            direction = "decreased"
        else:
            direction = "stayed about the same"

        charts.append({
            "title": title,
            "unit": unit,
            "width": CHART_WIDTH,
            "height": CHART_HEIGHT,
            "x_left": x0,
            "x_right": x1,
            "y_top": round(y1, 1),
            "y_bottom": round(y0, 1),
            "y_axis_top_label": fmt.format(y_hi),
            "y_axis_bottom_label": fmt.format(y_lo),
            "points": points,
            "polyline": " ".join(f"{p['x']},{p['y']}" for p in points),
            "summary": (
                f"{title} across {len(plot_values)} sessions: {direction}, "
                f"from {fmt.format(first_v)} to {fmt.format(last_v)}."
            ),
            "table_rows": [
                {"session": f"Session {i + 1}", "date": dates[i], "value": fmt.format(values[i])}
                for i in plot_idx
            ],
        })

    return charts

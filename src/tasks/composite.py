# src/tasks/composite.py
from __future__ import annotations
from typing import Dict, Any

from src.tasks.norms import NormModel, zscore_task_outputs


def _extract_raw_task_scores(outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "memory_immediate_score": outputs.get("memory_immediate_recall", {}).get("score"),
        "memory_delayed_score": outputs.get("memory_delayed_recall", {}).get("score"),
        "reaction_speed_score": outputs.get("reaction_simple", outputs.get("reaction_go_nogo", {}))
            .get("metrics", {}).get("speed_score"),
        "multidomain_percent": outputs.get("multidomain_screen", {}).get("metrics", {}).get("percent"),
    }


def composite_from_normed_scores(
    outputs: Dict[str, Dict[str, Any]],
    age_baseline: float,
    education_years: float,
    norms: Dict[str, NormModel],
) -> Dict[str, Any]:
    """Composite score built from demographically-normed (z-scored) subtests.

    Same 0.25/0.30/0.25/0.20 weighting scheme as composite_from_task_outputs,
    but applied to each subtest's z-score (age/education-adjusted) instead of
    its raw value - this is what makes the composite personalized rather than
    blind to who took the test. Weights are renormalized over components
    actually present (a component is "present" when its raw score exists AND
    its z-score is not NaN).
    """
    raw_scores = _extract_raw_task_scores(outputs)
    z_scores = zscore_task_outputs(raw_scores, age_baseline, education_years, norms)

    weights = {
        "memory_immediate_score_z": 0.25,
        "memory_delayed_score_z": 0.30,
        "reaction_speed_score_z": 0.25,
        "multidomain_percent_z": 0.20,
    }

    parts, used_weights = [], []
    for key, weight in weights.items():
        value = z_scores.get(key)
        if value is not None and value == value:  # not NaN
            parts.append(value)
            used_weights.append(weight)

    if not parts:
        return {"task": "composite_normed", "score": None, "metrics": {}, "raw": {"z_scores": z_scores}}

    wsum = sum(used_weights)
    score = sum(p * w for p, w in zip(parts, used_weights)) / wsum

    return {
        "task": "composite_normed",
        "score": score,
        "metrics": {"components_present": len(parts), "weights_sum": wsum},
        "raw": {"raw_scores": raw_scores, "z_scores": z_scores},
    }


def composite_from_task_outputs(outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    outputs: dict of {task_name: task_result_dict}
    Produces a single composite score with higher=better.
    """
    # Pull what we need
    mem = outputs.get("memory_immediate_recall", {}).get("score")
    delayed = outputs.get("memory_delayed_recall", {}).get("score")
    rt = outputs.get("reaction_simple", {}).get("metrics", {}).get("avg_rt_ms")
    multi = outputs.get("multidomain_screen", {}).get("metrics", {}).get("percent")

    # Convert RT to a higher=better component
    speed = (1000.0 / rt) if rt and rt > 0 else None

    # Weighted blend; tweak as desired
    parts = []
    weights = []

    if mem is not None:
        parts.append(mem); weights.append(0.25)
    if delayed is not None:
        parts.append(delayed); weights.append(0.30)
    if speed is not None:
        parts.append(speed); weights.append(0.25)
    if multi is not None:
        # multi is 0-1, bring it to 0-10 scale
        parts.append(multi * 10.0); weights.append(0.20)

    if not parts:
        return {"task": "composite", "score": None, "max_score": None, "metrics": {}, "raw": {}}

    # Normalize by sum weights actually present
    wsum = sum(weights)
    score = sum(p * w for p, w in zip(parts, weights)) / wsum

    return {
        "task": "composite",
        "score": score,
        "max_score": None,
        "metrics": {"components_present": len(parts), "weights_sum": wsum},
        "raw": {"parts": parts, "weights": weights}
    }

# webapp/app.py
"""Local, single-user web app for the standardized cognitive test battery.

Run from the repo root with `python -m webapp.app`, then open
http://127.0.0.1:5000/. Not a diagnosis - see docs/limitations.md and
docs/ethics.md.

State design: this is intentionally a single global in-memory STATE dict
rather than per-visitor session storage - the app is scoped to one person
testing themselves locally (see the "Single user, local" decision this was
built against), not a multi-user server. Reusing it concurrently from
multiple browser tabs/participants would clobber the shared state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, redirect, render_template, request, url_for

from src.longitudinal.prediction import estimate_personalized_decline
from src.sessions.persistence import (
    MIN_SESSIONS_FOR_PERSONALIZATION,
    append_session,
    count_sessions,
    find_participant_profile,
    save_participant_profile,
)
from src.tasks.composite import composite_from_normed_scores
from src.tasks.delayed_recall import score_delayed_recall
from src.tasks.memory import DEFAULT_WORD_BANK, generate_targets, score_free_recall
from src.tasks.multidomain import (
    combine_multidomain_result,
    generate_alternating_sequence,
    generate_digits,
    score_alternating_sequence,
    score_digit_span,
    score_serial_subtraction,
)
from src.tasks.norms import fit_task_norms
from src.tasks.reaction import score_reaction_trials

app = Flask(__name__)

MEMORY_CONFIG = {"n_words": 10, "study_seconds": 12}
REACTION_CONFIG = {"n_trials": 10, "min_foreperiod_ms": 800, "max_foreperiod_ms": 2200}
SERIAL_CONFIG = {"serial_start": 100, "serial_step": 7, "serial_n": 5}
DIGIT_SPAN_LEN = 6
TRAIL_N = 6
DIGIT_STUDY_MS = 3000

STATE: Dict[str, Any] = {}


def _active() -> bool:
    return "participant_id" in STATE and "profile" in STATE


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    participant_id = request.form.get("participant_id", "").strip()
    if not participant_id:
        return redirect(url_for("index"))

    profile = find_participant_profile(participant_id)

    STATE.clear()
    STATE["participant_id"] = participant_id
    STATE["results"] = {}

    if profile is None:
        return redirect(url_for("profile"))

    STATE["profile"] = profile
    return redirect(url_for("task_memory"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "participant_id" not in STATE:
        return redirect(url_for("index"))

    if request.method == "POST":
        new_profile = save_participant_profile(
            STATE["participant_id"],
            age_baseline=float(request.form["age"]),
            education_years=int(request.form["education"]),
            sex=request.form["sex"].strip().upper(),
        )
        STATE["profile"] = new_profile
        return redirect(url_for("task_memory"))

    return render_template("profile.html")


@app.route("/task/memory", methods=["GET", "POST"])
def task_memory():
    if not _active():
        return redirect(url_for("index"))

    if request.method == "POST":
        result = score_free_recall(STATE["memory_targets"], request.form.get("recall", ""))
        STATE["results"]["memory_immediate_recall"] = result
        return redirect(url_for("task_reaction"))

    targets = generate_targets(DEFAULT_WORD_BANK, MEMORY_CONFIG["n_words"], seed=None)
    STATE["memory_targets"] = targets
    return render_template("memory.html", targets=targets, study_seconds=MEMORY_CONFIG["study_seconds"])


@app.route("/task/reaction", methods=["GET", "POST"])
def task_reaction():
    if not _active():
        return redirect(url_for("index"))

    if request.method == "POST":
        trials = json.loads(request.form.get("trials_json", "[]"))
        result = score_reaction_trials(trials, go_no_go=False)
        STATE["results"]["reaction_simple"] = result
        return redirect(url_for("task_serial"))

    return render_template("reaction.html", **REACTION_CONFIG)


@app.route("/task/serial", methods=["GET", "POST"])
def task_serial():
    if not _active():
        return redirect(url_for("index"))

    cfg = SERIAL_CONFIG

    if request.method == "POST":
        answers = [request.form.get(f"answer_{i}", "") for i in range(cfg["serial_n"])]
        STATE["serial_result"] = score_serial_subtraction(
            cfg["serial_start"], cfg["serial_step"], cfg["serial_n"], answers
        )
        return redirect(url_for("task_digitspan"))

    return render_template("serial.html", **cfg)


@app.route("/task/digitspan", methods=["GET", "POST"])
def task_digitspan():
    if not _active():
        return redirect(url_for("index"))

    if request.method == "POST":
        digits = STATE["multidomain_digits"]
        STATE["digit_result"] = score_digit_span(digits, request.form.get("digits", ""))
        return redirect(url_for("task_sequence"))

    digits = generate_digits(DIGIT_SPAN_LEN, seed=None)
    STATE["multidomain_digits"] = digits
    return render_template("digitspan.html", digits=digits, study_ms=DIGIT_STUDY_MS)


@app.route("/task/sequence", methods=["GET", "POST"])
def task_sequence():
    if not _active():
        return redirect(url_for("index"))

    if request.method == "POST":
        expected_seq = STATE["multidomain_sequence"]
        trail_result = score_alternating_sequence(expected_seq, request.form.get("sequence", ""))

        result = combine_multidomain_result(STATE["serial_result"], STATE["digit_result"], trail_result)
        STATE["results"]["multidomain_screen"] = result
        return redirect(url_for("task_delayed_recall"))

    expected_seq = generate_alternating_sequence(TRAIL_N)
    STATE["multidomain_sequence"] = expected_seq
    return render_template("sequence.html", trail_n=TRAIL_N)


@app.route("/task/delayed_recall", methods=["GET", "POST"])
def task_delayed_recall():
    if not _active():
        return redirect(url_for("index"))

    if request.method == "POST":
        result = score_delayed_recall(STATE["memory_targets"], request.form.get("recall", ""))
        STATE["results"]["memory_delayed_recall"] = result
        _finalize_session()
        return redirect(url_for("results"))

    return render_template("delayed_recall.html")


def _finalize_session() -> None:
    """Compute the normed composite, persist the session, and (once enough
    sessions exist) compute a personalized decline estimate. Runs once, right
    after the last task is submitted - not on every /results page load, so
    refreshing the results page doesn't re-append duplicate sessions.
    """
    participant_id = STATE["participant_id"]
    profile = STATE["profile"]
    outputs = STATE["results"]

    task_norms = fit_task_norms()
    normed = composite_from_normed_scores(outputs, profile["age_baseline"], profile["education_years"], task_norms)

    append_session(participant_id, outputs)
    n_sessions = count_sessions(participant_id)

    personalized = None
    personalization_error = None
    if n_sessions >= MIN_SESSIONS_FOR_PERSONALIZATION:
        try:
            estimate = estimate_personalized_decline(participant_id, profile)
            personalized = estimate.to_dict("records") if estimate is not None and not estimate.empty else None
        except Exception as exc:
            personalization_error = str(exc)

    STATE["final"] = {
        "outputs": outputs,
        "normed": normed,
        "n_sessions": n_sessions,
        "min_sessions": MIN_SESSIONS_FOR_PERSONALIZATION,
        "personalized": personalized,
        "personalization_error": personalization_error,
    }


@app.route("/results")
def results():
    if "final" not in STATE:
        return redirect(url_for("index"))

    return render_template("results.html", **STATE["final"])


if __name__ == "__main__":
    app.run(debug=True)

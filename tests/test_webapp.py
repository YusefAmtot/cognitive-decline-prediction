import functools
import json

import pytest

from src.sessions import persistence
from webapp import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Flask test client whose participant/session storage is redirected
    to a temp directory, so tests never read or write the real
    data/sessions/ (real participant data - see .gitignore).

    Patches the names webapp.app imported (rather than the persistence
    module's defaults), since Python binds default-argument values like
    `sessions_path: str = DEFAULT_SESSIONS_LOG_PATH` once at function
    definition time - patching the module-level constant afterward
    wouldn't change what an already-defined function falls back to.
    """
    participants_path = str(tmp_path / "participants.csv")
    sessions_path = str(tmp_path / "sessions_log.csv")

    monkeypatch.setattr(
        app_module, "find_participant_profile",
        functools.partial(persistence.find_participant_profile, participants_path=participants_path),
    )
    monkeypatch.setattr(
        app_module, "save_participant_profile",
        functools.partial(persistence.save_participant_profile, participants_path=participants_path),
    )
    monkeypatch.setattr(
        app_module, "append_session",
        functools.partial(persistence.append_session, sessions_path=sessions_path),
    )
    monkeypatch.setattr(
        app_module, "load_participant_sessions",
        functools.partial(persistence.load_participant_sessions, sessions_path=sessions_path),
    )
    monkeypatch.setattr(
        app_module, "count_sessions",
        functools.partial(persistence.count_sessions, sessions_path=sessions_path),
    )

    app_module.STATE.clear()
    with app_module.app.test_client() as test_client:
        yield test_client
    app_module.STATE.clear()


def _run_session(client, participant_id, first=False):
    client.post("/start", data={"participant_id": participant_id}, follow_redirects=True)
    if first:
        client.post("/profile", data={"age": "70", "education": "16", "sex": "F"}, follow_redirects=True)

    client.post("/task/memory", data={"recall": "apple dog river"}, follow_redirects=True)

    trials = [{"trial": i + 1, "stimulus": "GO", "rt_ms": 300.0 + i * 5, "correct": True} for i in range(10)]
    client.post("/task/reaction", data={"trials_json": json.dumps(trials)}, follow_redirects=True)

    client.post(
        "/task/serial",
        data={"answer_0": "93", "answer_1": "86", "answer_2": "79", "answer_3": "72", "answer_4": "65"},
        follow_redirects=True,
    )
    client.post("/task/digitspan", data={"digits": "123456"}, follow_redirects=True)
    client.post("/task/sequence", data={"sequence": "1A2B3C4D5E6F"}, follow_redirects=True)

    return client.post("/task/delayed_recall", data={"recall": "apple dog"}, follow_redirects=True)


def test_full_session_flow_reaches_results_page(client):
    resp = _run_session(client, "webtest-flow", first=True)
    assert resp.status_code == 200
    assert b"Session Complete" in resp.data


def test_composite_raw_score_is_populated_after_a_session(client):
    """Regression test: sessions_log.csv's composite_raw_score column used to
    always be empty because nothing ever set outputs["composite"] before the
    session was persisted (see webapp/app.py:_finalize_session).
    """
    _run_session(client, "webtest-composite", first=True)

    outputs = app_module.STATE["final"]["outputs"]
    assert outputs["composite"]["score"] is not None


def test_two_sessions_produce_both_line_graph_sections(client):
    _run_session(client, "webtest-history", first=True)
    resp = _run_session(client, "webtest-history", first=False)
    html = resp.get_data(as_text=True)

    assert "Domain trends" in html
    assert "chart-legend" in html
    assert "Score history across sessions" in html
    assert html.count('class="line-chart"') >= 2  # at least the domain chart + one per-task chart


def test_single_session_shows_fallback_messages_instead_of_charts(client):
    resp = _run_session(client, "webtest-single", first=True)
    html = resp.get_data(as_text=True)

    assert 'class="line-chart"' not in html
    assert html.count("Complete at least two sessions") == 2  # domain chart + per-task charts


def test_profile_rejects_non_numeric_age(client):
    client.post("/start", data={"participant_id": "webtest-badage"}, follow_redirects=True)
    resp = client.post("/profile", data={"age": "not-a-number", "education": "12", "sex": "F"})

    assert resp.status_code == 400
    assert b"must be numbers" in resp.data


def test_profile_rejects_out_of_range_age(client):
    client.post("/start", data={"participant_id": "webtest-outofrange"}, follow_redirects=True)
    resp = client.post("/profile", data={"age": "500", "education": "12", "sex": "F"})

    assert resp.status_code == 400
    assert b"between 1 and 120" in resp.data


def test_profile_rejects_invalid_sex(client):
    client.post("/start", data={"participant_id": "webtest-badsex"}, follow_redirects=True)
    resp = client.post("/profile", data={"age": "70", "education": "12", "sex": "X"})

    assert resp.status_code == 400
    assert b"Sex must be M or F" in resp.data

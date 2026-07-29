import pytest

from src.tasks.memory import generate_targets, score_free_recall
from src.tasks.delayed_recall import score_delayed_recall
from src.tasks.reaction import score_reaction_trials
from src.tasks.multidomain import (
    combine_multidomain_result,
    generate_alternating_sequence,
    generate_expected_serial,
    score_alternating_sequence,
    score_digit_span,
    score_serial_subtraction,
)


def test_generate_targets_is_deterministic_with_seed():
    first = generate_targets(n_words=10, seed=0)
    second = generate_targets(n_words=10, seed=0)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_score_free_recall_counts_correct_and_intrusions():
    targets = ["apple", "river", "chair"]
    result = score_free_recall(targets, "apple, chair, banana")

    assert result["score"] == 2
    assert result["max_score"] == 3
    assert result["metrics"]["n_correct"] == 2
    assert result["metrics"]["n_intrusions"] == 1
    assert result["raw"]["intrusions"] == ["banana"]


def test_score_free_recall_is_case_and_punctuation_insensitive():
    targets = ["apple", "river"]
    result = score_free_recall(targets, "Apple! RIVER.")
    assert result["score"] == 2


def test_score_delayed_recall_matches_free_recall_semantics():
    targets = ["apple", "river", "chair"]
    result = score_delayed_recall(targets, "apple river")
    assert result["task"] == "memory_delayed_recall"
    assert result["score"] == 2
    assert result["max_score"] == 3


def test_score_reaction_trials_computes_avg_rt_and_speed_score():
    trials = [
        {"trial": 1, "stimulus": "GO", "rt_ms": 400.0, "correct": True},
        {"trial": 2, "stimulus": "GO", "rt_ms": 600.0, "correct": True},
    ]
    result = score_reaction_trials(trials)

    assert result["metrics"]["avg_rt_ms"] == pytest.approx(500.0)
    assert result["metrics"]["speed_score"] == pytest.approx(1000.0 / 500.0)
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["n_go_trials"] == 2


def test_score_reaction_trials_handles_no_trials():
    result = score_reaction_trials([])
    assert result["metrics"]["avg_rt_ms"] is None
    assert result["metrics"]["speed_score"] is None
    assert result["metrics"]["accuracy"] is None


def test_generate_expected_serial():
    assert generate_expected_serial(100, 7, 5) == [93, 86, 79, 72, 65]


def test_score_serial_subtraction_counts_correct_answers():
    result = score_serial_subtraction(100, 7, 5, ["93", "86", "oops", "72", "65"])
    assert result["correct"] == 4
    assert result["max"] == 5


def test_score_digit_span_exact_match_required():
    digits = ["1", "2", "3", "4"]
    assert score_digit_span(digits, "1234")["correct"] == 1
    assert score_digit_span(digits, "1 2 3 4")["correct"] == 1  # spaces ignored
    assert score_digit_span(digits, "1235")["correct"] == 0


def test_generate_alternating_sequence():
    assert generate_alternating_sequence(3) == ["1", "a", "2", "b", "3", "c"]


def test_score_alternating_sequence_exact_and_no_spaces():
    expected_seq = ["1", "a", "2", "b", "3", "c"]
    assert score_alternating_sequence(expected_seq, "1 a 2 b 3 c")["correct"] == 6
    assert score_alternating_sequence(expected_seq, "1a2b3c")["correct"] == 6
    assert score_alternating_sequence(expected_seq, "1 a 2 x 3 c")["correct"] == 5


def test_combine_multidomain_result_sums_across_subtasks():
    serial_result = {"correct": 4, "max": 5, "raw": {"a": 1}}
    digit_result = {"correct": 1, "max": 1, "raw": {"b": 2}}
    trail_result = {"correct": 5, "max": 6, "raw": {"c": 3}}

    result = combine_multidomain_result(serial_result, digit_result, trail_result)

    assert result["task"] == "multidomain_screen"
    assert result["score"] == 10
    assert result["max_score"] == 12
    assert result["metrics"]["percent"] == pytest.approx(10 / 12)

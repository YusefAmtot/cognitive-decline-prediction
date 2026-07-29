# src/tasks/multidomain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import random
import time


@dataclass
class MultiDomainConfig:
    serial_start: int = 100
    serial_step: int = 7
    serial_n: int = 5

    digit_span_len: int = 6

    trail_n: int = 6  # will generate 1..N and A..(A+N-1)
    seed: int | None = None


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# Part 1: serial subtraction (MMSE "serial 7s" analog)
# ---------------------------------------------------------------------------

def generate_expected_serial(serial_start: int, serial_step: int, serial_n: int) -> List[int]:
    expected = []
    cur = serial_start
    for _ in range(serial_n):
        cur -= serial_step
        expected.append(cur)
    return expected


def score_serial_subtraction(serial_start: int, serial_step: int, serial_n: int, answers: List[str]) -> Dict[str, Any]:
    """Pure scoring for the serial-subtraction subtask - no I/O, so it can be
    driven by either the CLI loop below or a web form submission.
    """
    expected = generate_expected_serial(serial_start, serial_step, serial_n)

    correct = 0
    for given, exp in zip(answers, expected):
        try:
            if int(given) == exp:
                correct += 1
        except ValueError:
            pass

    return {
        "correct": correct,
        "max": serial_n,
        "raw": {"answers": answers, "expected": expected, "correct": correct},
    }


# ---------------------------------------------------------------------------
# Part 2: digit span (forward)
# ---------------------------------------------------------------------------

def generate_digits(digit_span_len: int, seed: int | None = None) -> List[str]:
    rng = random.Random(seed)
    return [str(rng.randint(0, 9)) for _ in range(digit_span_len)]


def score_digit_span(digits: List[str], typed: str) -> Dict[str, Any]:
    """Pure scoring for the digit-span subtask."""
    typed_norm = _normalize(typed)
    expected_norm = "".join(digits)
    correct = 1 if typed_norm == expected_norm else 0

    return {
        "correct": correct,
        "max": 1,
        "raw": {"digits": digits, "typed": typed, "correct": correct},
    }


# ---------------------------------------------------------------------------
# Part 3: alternating sequence (trail-making-lite / executive switching)
# ---------------------------------------------------------------------------

def generate_alternating_sequence(trail_n: int) -> List[str]:
    nums = list(range(1, trail_n + 1))
    letters = [chr(ord("a") + i) for i in range(trail_n)]
    expected_seq = []
    for n, L in zip(nums, letters):
        expected_seq.append(str(n))
        expected_seq.append(L)
    return expected_seq


def score_alternating_sequence(expected_seq: List[str], typed: str) -> Dict[str, Any]:
    """Pure scoring for the alternating-sequence subtask (proportion correct by position)."""
    typed_list = [t for t in typed.strip().lower().replace(",", " ").split() if t]
    # allow without spaces: if they typed "1a2b3c"
    if len(typed_list) <= 2:
        typed_list = list(_normalize(typed))

    pos_correct = 0
    for i in range(min(len(typed_list), len(expected_seq))):
        if typed_list[i] == expected_seq[i]:
            pos_correct += 1

    return {
        "correct": pos_correct,
        "max": len(expected_seq),
        "raw": {"expected_seq": expected_seq, "typed_tokens": typed_list, "pos_correct": pos_correct},
    }


def combine_multidomain_result(
    serial_result: Dict[str, Any],
    digit_result: Dict[str, Any],
    trail_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the three subtask scores into the overall multidomain_screen result."""
    total_points = serial_result["correct"] + digit_result["correct"] + trail_result["correct"]
    max_points = serial_result["max"] + digit_result["max"] + trail_result["max"]

    return {
        "task": "multidomain_screen",
        "score": total_points,
        "max_score": max_points,
        "metrics": {"percent": (total_points / max_points) if max_points else None},
        "raw": {
            "serial": serial_result["raw"],
            "digit_span": digit_result["raw"],
            "trail": trail_result["raw"],
        },
    }


def run_multidomain_task(
    participant_id: str,
    config: MultiDomainConfig = MultiDomainConfig(),
    *,
    input_fn=input,
    print_fn=print
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    print_fn("\n=== Multi-domain Screening ===")

    # 1) Serial subtraction
    print_fn("\nPart 1: Serial subtraction")
    print_fn(f"Start at {config.serial_start} and subtract {config.serial_step} each time.")
    current = config.serial_start
    serial_answers = []
    for i in range(config.serial_n):
        ans = input_fn(f"Step {i+1}: {current} - {config.serial_step} = ").strip()
        serial_answers.append(ans)
        current -= config.serial_step
    serial_result = score_serial_subtraction(config.serial_start, config.serial_step, config.serial_n, serial_answers)

    # 2) Digit span (forward)
    print_fn("\nPart 2: Digit span (forward)")
    digits = generate_digits(config.digit_span_len, config.seed)
    print_fn("Memorize these digits:")
    print_fn(" ".join(digits))
    time.sleep(3.0)
    print_fn("\n" * 20)
    typed_digits = input_fn("Type the digits in the same order (no spaces needed): ")
    digit_result = score_digit_span(digits, typed_digits)

    # 3) Trail-making lite (alternating sequence)
    # True Trails is visuomotor; here we approximate executive switching as a sequence rule.
    print_fn("\nPart 3: Alternating sequence (executive switching)")
    expected_seq = generate_alternating_sequence(config.trail_n)
    print_fn("Type the alternating sequence like: 1 a 2 b 3 c ...")
    typed_seq = input_fn("Your sequence: ")
    trail_result = score_alternating_sequence(expected_seq, typed_seq)

    result = combine_multidomain_result(serial_result, digit_result, trail_result)
    elapsed = time.perf_counter() - start_time

    result["participant_id"] = participant_id
    result["metrics"]["elapsed_seconds"] = elapsed
    return result

# src/tasks/reaction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import random
import time


@dataclass
class ReactionConfig:
    n_trials: int = 12
    min_foreperiod: float = 0.8   # seconds
    max_foreperiod: float = 2.2
    go_no_go: bool = False
    no_go_rate: float = 0.25      # fraction of NO trials if go_no_go=True
    seed: int | None = None


def _now() -> float:
    return time.perf_counter()


def score_reaction_trials(trials: List[Dict[str, Any]], go_no_go: bool = False) -> Dict[str, Any]:
    """Score a list of completed reaction trials.

    Each trial dict needs `stimulus` ("GO"/"NO"), `rt_ms` (float or None), and
    `correct` (bool). Pure function (no I/O, no timing) so both the CLI task
    below and a web request handler can reuse the same scoring - a browser
    can time trials precisely with JS and submit the resulting trial list
    here, avoiding the CLI's keyboard/OS input() latency.
    """
    go_rts = [t["rt_ms"] for t in trials if t["stimulus"] == "GO" and t["rt_ms"] is not None]
    avg_rt = sum(go_rts) / len(go_rts) if go_rts else None

    accuracy = sum(1 for t in trials if t["correct"]) / len(trials) if trials else None

    # Score direction: lower RT is better, so define a "speed score" higher=better for easy combining:
    speed_score = (1000.0 / avg_rt) if avg_rt and avg_rt > 0 else None

    return {
        "task": "reaction_go_nogo" if go_no_go else "reaction_simple",
        "score": avg_rt,  # keep raw RT as primary score
        "max_score": None,
        "metrics": {
            "avg_rt_ms": avg_rt,
            "speed_score": speed_score,
            "accuracy": accuracy,
            "n_go_trials": len(go_rts),
            "n_trials": len(trials),
        },
        "raw": {"trials": trials},
    }


def run_reaction_task(
    participant_id: str,
    config: ReactionConfig = ReactionConfig(),
    *,
    input_fn=input,
    print_fn=print
) -> Dict[str, Any]:
    """
    Measures reaction time via CLI.
    Caveat: CLI timing includes keyboard/OS latency. Still fine for learning + relative tracking.
    """
    rng = random.Random(config.seed)
    trials: List[Dict[str, Any]] = []

    print_fn("\n=== Reaction Speed Task ===")
    if config.go_no_go:
        print_fn("Rule: Press Enter ONLY when you see 'GO'. Do nothing on 'NO'.")
    else:
        print_fn("Rule: Press Enter as fast as possible when you see 'GO'.")

    input_fn("Press Enter to start...")

    for i in range(config.n_trials):
        fore = rng.uniform(config.min_foreperiod, config.max_foreperiod)
        time.sleep(max(0.0, fore))

        is_go = True
        if config.go_no_go:
            is_go = rng.random() > config.no_go_rate

        stimulus = "GO" if is_go else "NO"
        print_fn(f"\nTrial {i+1}/{config.n_trials}: {stimulus}")

        stim_time = _now()

        # For GO: require a keypress (Enter) and measure RT
        # For NO: participant should NOT press; we wait a short window and record false alarm if they press
        if is_go:
            input_fn("Press Enter NOW!")
            rt = (_now() - stim_time) * 1000.0
            trials.append({
                "trial": i + 1,
                "stimulus": stimulus,
                "foreperiod_s": fore,
                "rt_ms": rt,
                "correct": True
            })
        else:
            # No-go: give a response window
            response_window = 1.2
            print_fn(f"(No-go) Wait {response_window:.1f}s... Do NOT press Enter.")
            # crude detection: ask after the window; if user pressed enter early, we can't detect in pure input() easily
            # so we implement as a self-report prompt (still useful for learning)
            time.sleep(response_window)
            ans = input_fn("Did you press Enter during the NO trial? (y/n): ").strip().lower()
            false_alarm = ans.startswith("y")
            trials.append({
                "trial": i + 1,
                "stimulus": stimulus,
                "foreperiod_s": fore,
                "rt_ms": None,
                "correct": not false_alarm
            })

    result = score_reaction_trials(trials, go_no_go=config.go_no_go)
    result["participant_id"] = participant_id
    result["raw"]["config"] = config.__dict__
    return result

# src/tasks/delayed_recall.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import time


@dataclass
class DelayedRecallConfig:
    # In real setups you would do a true delay (10–30 mins) filled with other tasks.
    # Here, the delay is handled by your session orchestration (run other tasks before calling this).
    free_recall: bool = True


def _normalize_tokens(text: str) -> List[str]:
    tokens = []
    for part in text.replace(",", " ").split():
        t = part.strip().lower().strip(".!?;:'\"()[]{}")
        if t:
            tokens.append(t)
    return tokens


def score_delayed_recall(targets: List[str], recalled_text: str) -> Dict[str, Any]:
    """Score a delayed free-recall response against the original target words.

    Pure function (no I/O) so both the CLI task below and a web request
    handler can reuse the same scoring logic.
    """
    recalled = _normalize_tokens(recalled_text)
    target_set = set(t.lower() for t in targets)
    recalled_set = set(recalled)

    correct = sorted(target_set & recalled_set)
    intrusions = sorted(recalled_set - target_set)

    return {
        "task": "memory_delayed_recall",
        "score": len(correct),
        "max_score": len(targets),
        "metrics": {
            "n_recalled": len(recalled_set),
            "n_correct": len(correct),
            "n_intrusions": len(intrusions),
        },
        "raw": {
            "targets": list(targets),
            "recalled": recalled,
            "correct": correct,
            "intrusions": intrusions,
        },
    }


def run_delayed_recall_task(
    participant_id: str,
    targets: List[str],
    config: DelayedRecallConfig = DelayedRecallConfig(),
    *,
    input_fn=input,
    print_fn=print
) -> Dict[str, Any]:
    """
    Delayed recall. Requires the original 'targets' from the earlier memory task.
    """
    start_time = time.perf_counter()

    print_fn("\n=== Delayed Recall Task ===")
    print_fn("Earlier, you saw a list of words. Now recall as many as you can.")

    typed = input_fn("Your recall: ")

    result = score_delayed_recall(targets, typed)
    elapsed = time.perf_counter() - start_time

    result["participant_id"] = participant_id
    result["metrics"]["elapsed_seconds"] = elapsed
    result["raw"]["config"] = config.__dict__
    return result

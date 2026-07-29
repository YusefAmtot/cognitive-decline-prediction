# src/tasks/memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import random
import time


@dataclass
class MemoryConfig:
    n_words: int = 10
    study_seconds: int = 12
    # if True, participant types recalled words separated by spaces/commas
    free_recall: bool = True
    # if False, will do recognition (multiple choice) instead of recall
    recognition: bool = False
    recognition_lures: int = 10
    seed: int | None = None


DEFAULT_WORD_BANK = [
    # keep these simple, common nouns; expand as you like
    "apple", "river", "chair", "pencil", "window", "garden", "orange", "planet",
    "mirror", "coffee", "school", "button", "mountain", "pillow", "guitar",
    "station", "ticket", "camera", "bottle", "cookie", "forest", "doctor",
    "street", "yellow", "pocket", "circle", "castle", "shadow", "rocket",
    "blanket", "violin", "battery", "bridge", "candle", "silver", "engine",
    "market", "thunder", "flower", "wallet", "island", "sweater", "kitchen"
]


def _normalize_tokens(text: str) -> List[str]:
    # split on commas or whitespace, lowercase, strip punctuation-ish edges
    tokens = []
    for part in text.replace(",", " ").split():
        t = part.strip().lower().strip(".!?;:'\"()[]{}")
        if t:
            tokens.append(t)
    return tokens


def _choose_words(word_bank: Sequence[str], n: int, rng: random.Random) -> List[str]:
    if n > len(word_bank):
        raise ValueError("n_words is larger than the word bank size.")
    return rng.sample(list(word_bank), n)


def generate_targets(
    word_bank: Sequence[str] = DEFAULT_WORD_BANK,
    n_words: int = 10,
    seed: int | None = None,
) -> List[str]:
    """Pick the word list a participant will be asked to study.

    Split out from run_immediate_memory_task so a UI that can't block on
    input() (e.g. a web app, which shows the words, waits client-side, then
    submits recall in a separate request) can still reuse the same stimulus
    generation as the CLI.
    """
    rng = random.Random(seed)
    return _choose_words(word_bank, n_words, rng)


def score_free_recall(targets: Sequence[str], recalled_text: str) -> Dict[str, Any]:
    """Score a free-recall response against the target word list.

    Pure function (no I/O) so it can be called from both the CLI task below
    and a web request handler.
    """
    recalled = _normalize_tokens(recalled_text)
    target_set = set(t.lower() for t in targets)
    recalled_set = set(recalled)

    correct = sorted(target_set & recalled_set)
    intrusions = sorted(recalled_set - target_set)

    return {
        "task": "memory_immediate_recall",
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


def run_immediate_memory_task(
    participant_id: str,
    word_bank: Sequence[str] = DEFAULT_WORD_BANK,
    config: MemoryConfig = MemoryConfig(),
    *,
    input_fn=input,
    print_fn=print
) -> Dict[str, Any]:
    """
    CLI-style immediate recall task.
    - Shows words for config.study_seconds
    - Collects recall (free recall) OR recognition choices
    Returns a dict with score + metadata.
    """
    rng = random.Random(config.seed)

    targets = _choose_words(word_bank, config.n_words, rng)
    start_time = time.perf_counter()

    print_fn("\n=== Immediate Memory Task ===")
    print_fn(f"You will see {config.n_words} words. Try to remember them.")
    print_fn("Words:")
    print_fn("  " + "  ".join(targets))

    print_fn(f"\nStudy time: {config.study_seconds} seconds...")
    time.sleep(max(0, config.study_seconds))

    # "clear screen" lite
    print_fn("\n" * 30)

    if config.recognition:
        # Recognition: mix targets + lures, ask participant to mark which were shown
        lures = _choose_words([w for w in word_bank if w not in targets], config.recognition_lures, rng)
        options = targets + lures
        rng.shuffle(options)

        print_fn("Recognition test: For each word, type Y if it was shown, else N.")
        responses = {}
        for w in options:
            ans = input_fn(f"Was '{w}' shown? (Y/N): ").strip().lower()
            responses[w] = ans.startswith("y")

        hits = sum(1 for w in targets if responses.get(w, False))
        false_alarms = sum(1 for w in lures if responses.get(w, False))

        # Simple corrected recognition score (hits - false alarms), floor at 0
        corrected = max(0, hits - false_alarms)

        elapsed = time.perf_counter() - start_time
        return {
            "task": "memory_immediate_recognition",
            "participant_id": participant_id,
            "score": corrected,
            "max_score": config.n_words,
            "metrics": {
                "hits": hits,
                "false_alarms": false_alarms,
                "elapsed_seconds": elapsed
            },
            "raw": {
                "targets": targets,
                "lures": lures,
                "responses": responses,
                "config": config.__dict__
            }
        }

    # Free recall
    print_fn("Recall as many words as you can. Type them separated by spaces or commas.")
    typed = input_fn("Your recall: ")

    result = score_free_recall(targets, typed)
    elapsed = time.perf_counter() - start_time

    result["participant_id"] = participant_id
    result["metrics"]["elapsed_seconds"] = elapsed
    result["raw"]["config"] = config.__dict__
    return result

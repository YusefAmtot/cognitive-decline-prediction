
# src/sessions/run_session.py
from __future__ import annotations
from typing import Dict, Any

from src.tasks.memory import run_immediate_memory_task, MemoryConfig
from src.tasks.reaction import run_reaction_task, ReactionConfig
from src.tasks.delayed_recall import run_delayed_recall_task
from src.tasks.multidomain import run_multidomain_task
from src.tasks.composite import composite_from_task_outputs


def run_session(participant_id: str) -> Dict[str, Any]:
    outputs: Dict[str, Dict[str, Any]] = {}

    mem_out = run_immediate_memory_task(
        participant_id,
        config=MemoryConfig(n_words=10, study_seconds=10, seed=0)
    )
    outputs[mem_out["task"]] = mem_out

    react_out = run_reaction_task(
        participant_id,
        config=ReactionConfig(n_trials=10, go_no_go=False, seed=0)
    )
    outputs[react_out["task"]] = react_out

    multi_out = run_multidomain_task(participant_id)
    outputs[multi_out["task"]] = multi_out

    # Delayed recall uses the same targets from the immediate task
    targets = mem_out["raw"]["targets"]
    delayed_out = run_delayed_recall_task(participant_id, targets=targets)
    outputs[delayed_out["task"]] = delayed_out

    comp = composite_from_task_outputs(outputs)
    outputs[comp["task"]] = comp

    return outputs

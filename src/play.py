from src.sessions.run_session import run_session
from src.sessions.persistence import (
    MIN_SESSIONS_FOR_PERSONALIZATION,
    append_session,
    count_sessions,
    get_or_create_participant_profile,
)
from src.tasks.composite import composite_from_normed_scores
from src.tasks.norms import fit_task_norms
from src.longitudinal.prediction import estimate_personalized_decline


def main():
    print("\nWelcome to the cognitive test demo.")
    print("This is NOT a diagnosis. Just a learning tool.\n")

    participant_id = input("Enter your participant ID (e.g. 'me'): ")
    profile = get_or_create_participant_profile(participant_id)

    print("\nStarting session...\n")
    results = run_session(participant_id)

    print("\n=== SESSION COMPLETE ===\n")

    for task_name, result in results.items():
        print(f"{task_name.upper()}")
        print(f"  Score: {result['score']}")
        if "metrics" in result:
            for k, v in result["metrics"].items():
                print(f"  {k}: {v}")
        print()

    task_norms = fit_task_norms()
    normed = composite_from_normed_scores(
        results, profile["age_baseline"], profile["education_years"], task_norms
    )
    print("COMPOSITE_NORMED (age/education-adjusted, higher=better)")
    print(f"  Score: {normed['score']}")
    print()

    append_session(participant_id, results)

    n_sessions = count_sessions(participant_id)
    if n_sessions >= MIN_SESSIONS_FOR_PERSONALIZATION:
        try:
            estimate = estimate_personalized_decline(participant_id, profile)
        except Exception as exc:
            estimate = None
            print(f"(Could not compute a personalized estimate this time: {exc})\n")

        if estimate is not None and not estimate.empty:
            print("=== PERSONALIZED DECLINE ESTIMATE ===")
            print(
                f"Based on your {n_sessions} sessions, pooled with the simulated cohort "
                "for statistical robustness (not a diagnosis):"
            )
            print(estimate.to_string(index=False))
            print()
    else:
        remaining = MIN_SESSIONS_FOR_PERSONALIZATION - n_sessions
        print(
            f"Not enough sessions yet for a personalized estimate "
            f"({remaining} more needed) - showing population-level info only.\n"
        )


if __name__ == "__main__":
    main()

# Methodology

## Overview

This project predicts rates of cognitive decline from longitudinal test
scores and personalizes those predictions using a series of standardized
tests. Two data sources feed the same pipeline:

1. A simulated longitudinal cohort (`notebooks/00_generate_simulated_data.ipynb`)
   of 250 people with `memory`/`attention`/`language` scores across up to 8
   visits, generated with per-person random decline slopes, dropout, and
   noise.
2. Real sessions from the interactive CLI test battery (`src/tasks/*.py`,
   `src/play.py`) - immediate recall, delayed recall, reaction speed, and a
   multidomain screen (serial-7s, digit span, alternating-sequence) - which
   accumulate per participant in `data/sessions/` every time someone runs the
   demo.

## Normative (personalized) scoring

The simulated domains and the CLI subtests are on different, non-comparable
raw scales. Both are converted to age/education-adjusted z-scores
(`src/tasks/norms.py`) using regression-based norming: a normative reference
sample is fit with `score ~ age + education`, and each raw score becomes
`(raw - predicted) / residual_std`. This is what makes scoring personalized
rather than blind to who took the test. The CLI battery's reference sample is
a fabricated illustrative distribution, not real clinical norms (see
`docs/limitations.md`).

CLI subtests are mapped onto the simulated cohort's domain vocabulary:
`memory_z` = weighted z of immediate + delayed recall; `attention_z` =
weighted z of reaction speed + multidomain screen; `language_z` has no CLI
equivalent and is always missing for real participants.

## Decline-rate (slope) extraction

Three estimators are used per participant per domain
(`src/longitudinal/slope_extraction.py`):

- **OLS**: an independent linear regression per participant.
- **Theil-Sen**: a robust median-based slope, less sensitive to a single
  noisy visit.
- **Mixed-effects (MixedLM)**: a population model with a random intercept and
  random slope per participant. Its per-subject BLUP (best linear unbiased
  predictor) slope shrinks a person's estimate toward the population trend in
  proportion to how little data they have - exactly the statistical property
  that makes it more robust than an isolated per-subject regression for
  someone with only a few visits or CLI sessions.

Additional trend features (`src/longitudinal/trends.py`) capture visit-to-visit
variability, curvature (accelerating/decelerating decline), and missingness
patterns.

## Prediction

`src/models/models.py` trains classical ML models (Ridge, Random Forest,
Gradient Boosting) to predict each simulated participant's true generative
decline rate (`data/simulated/true_slopes.csv`) from demographics plus
cold-start features (computed from only their first 2 visits), evaluated
against a naive population-mean baseline with GroupKFold cross-validation
(`src/evaluation/evaluation.py`) so no participant leaks between train/test.
A separate temporal-holdout evaluation scores the slope estimators themselves
by projecting each participant's held-out final visit from their own prior
history.

## Personalization loop for real users

When someone runs `src/play.py`, their demographic profile is collected once
(`src/sessions/persistence.py`) and every session's normed scores are
appended to `data/sessions/sessions_log.csv`. Once a participant has at least
3 sessions, `src/longitudinal/prediction.py` merges their real history into
the simulated cohort (a mixed model needs many groups to estimate
between-person variance, so a single person's data can't be fit alone) and
refits the slope-extraction pipeline, producing a personalized decline-rate
estimate that shifts from population-level toward individually-anchored as
more sessions accumulate.

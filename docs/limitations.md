# Limitations

- **Synthetic data only.** The longitudinal cohort is entirely simulated
  (`notebooks/00_generate_simulated_data.ipynb`); no model here has been
  validated against real clinical or research data. Nothing in this
  repository has been clinically validated.
- **Not diagnostic.** This is a learning/demo tool, not a medical device.
  `src/play.py` says so explicitly, and every "personalized decline estimate"
  it prints is illustrative, not a clinical assessment.
- **Fabricated CLI norms.** The normative reference sample used to z-score
  the CLI test battery (`src/tasks/norms.py:generate_task_norm_reference`) is
  a synthetic, illustrative distribution invented for this project - it is
  not a validated neuropsychological norm.
- **Missing language domain for real users.** The CLI battery has no
  language subtest, so `language_z` is always missing for real participants;
  only `memory_z`/`attention_z` can be personalized from real sessions.
- **Domain mapping is a design choice.** Mapping CLI subtests onto the
  simulated cohort's `memory`/`attention` vocabulary (e.g. attention =
  reaction speed + multidomain screen) is a reasonable but unvalidated
  approximation, not a proven clinical equivalence.
- **CLI reaction-time noise.** `src/tasks/reaction.py` measures reaction time
  via `input()`, so timings include keyboard/OS/terminal latency on top of
  true response time.
- **Practice effects.** The memory task draws from a fixed 40-word bank
  (`src/tasks/memory.py:DEFAULT_WORD_BANK`). Repeated real sessions may show
  apparent improvement from familiarity with the word list rather than true
  cognitive change - a real confound the current pipeline does not correct
  for.
- **Small cohort.** n=250 simulated participants limits how much signal any
  ML model can extract; the naive population-mean baseline is expected to be
  fairly competitive (see `notebooks/06_model_comparison.ipynb`).
- **MNAR dropout.** Visit dropout probability grows with time in the
  simulator by design. Using missingness as a predictive feature is
  reasonable here but would not necessarily transfer to a real clinical
  dropout mechanism.
- **MixedLM convergence.** Sparse per-person data (heavy dropout, or a real
  user with only a few sessions) can cause the random-slope mixed model to
  fail to converge; `src/longitudinal/slope_extraction.py:fit_mixedlm` falls
  back to a random-intercept-only model in that case and surfaces
  `converged` so this is never silent.
- **Real-visit cadence mismatch.** The simulated cohort visits every 6
  months; a real person running the CLI demo repeatedly over days/weeks
  operates on a very different timescale, conflating short-term practice
  effects with the multi-year decline the simulation models.

# Ethics

- **Not a medical device.** This project is a learning tool for exploring
  longitudinal modeling and personalization techniques. It must never be
  used, marketed, or relied upon for clinical screening, diagnosis, or
  care decisions about cognitive decline or dementia. `src/play.py` states
  this on every run; any extension of this tool should preserve that
  disclaimer.
- **Fabricated norms are not clinical norms.** The CLI battery's normative
  reference sample (`src/tasks/norms.py:generate_task_norm_reference`) is
  synthetic and illustrative. Any "z-score" or "personalized estimate" it
  produces must be labeled as such and never presented as a validated
  neuropsychological score.
- **Local, unencrypted personal data.** Running `src/play.py` writes
  individually identifiable test performance, tied to a self-chosen
  participant ID, to `data/sessions/participants.csv` and
  `data/sessions/sessions_log.csv` in plaintext on the local filesystem.
  This directory is gitignored, but there is no encryption, access control,
  retention limit, or consent flow. Anyone extending this into a real
  multi-user deployment would need to add explicit informed consent, data
  retention/deletion policies, and appropriate storage security before
  collecting real people's cognitive test data.
- **Self-reported demographics, no verification.** Age/education/sex are
  collected via an unvalidated CLI prompt with no identity or accuracy
  checks - fine for a demo, not sufficient for anything higher-stakes.
- **Scope of "personalization."** The personalized decline-rate estimate
  produced after 3+ sessions reflects a small amount of self-reported data
  pooled with a synthetic cohort. It should never be communicated to a real
  user in a way that implies clinical confidence.

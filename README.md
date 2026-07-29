# Cognitive Decline Progression Prediction

Predicting rates of cognitive decline using longitudinal cognitive test data and classical machine learning models.

Not a diagnosis - see `docs/limitations.md` and `docs/ethics.md`.

## Running the model pipeline

```powershell
pip install -r requirements.txt
```

Run `notebooks/00` through `notebooks/07` in order (each writes data the next one reads). See `docs/methodology.md` for what each stage does.

## Running the test battery

Two ways to take the standardized cognitive tests yourself:

**Web app** (recommended - accurate reaction timing via the browser):
```powershell
python -m webapp.app
```
Then open http://127.0.0.1:5000/. Local, single-user only - not built for multiple concurrent users.

**CLI**:
```powershell
python -m src.play
```

Use the same participant ID across sessions - after 3 sessions you'll get a personalized decline-rate estimate instead of just population-level info.
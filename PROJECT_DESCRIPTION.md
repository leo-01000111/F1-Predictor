# F1 Race Prediction Pipeline — Project Description

> Written for comparison against similar projects. Describes what the system is, what it does, and how it works end to end.

---

## What it is

An end-to-end machine learning pipeline that predicts Formula 1 race podium probabilities (P1/P2/P3) for every driver before a race weekend. Given qualifying results and contextual data, the system outputs per-driver win and podium probabilities, displayed in a Streamlit dashboard.

**Scope:** 2014–2025 hybrid-era training data. Current target: 2026 season inference.

---

## What it predicts

Three binary classification targets per driver per race:

| Target | Meaning | Base rate |
|--------|---------|-----------|
| `p1`   | Driver finishes 1st (wins) | ~4.6% |
| `p2`   | Driver finishes 2nd | ~5.0% |
| `p3`   | Driver finishes 3rd | ~5.0% |

Each target is modelled independently. The result is a probability triplet per driver: `(p_win, p_p2, p_p3)`, plus a derived `p_podium = 1 - (1-p1)(1-p2)(1-p3)` approximation.

---

## Data sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| **FastF1** (Python library) | Historical race results, qualifying times, lap data, driver/team metadata | Free, cached locally |
| **OpenF1 API** | Live qualifying and free practice session data for the current race weekend | Free REST API |
| **Open-Meteo** | Race day weather per circuit (temperature, precipitation, wind, humidity) | Free, no key required |
| *(Legacy)* Reddit API | Sentiment signals — collected but not used in the active pipeline | Optional, unused |

---

## Feature matrix — 27 features

One row per driver per race. Features fall into seven groups:

### Qualifying
- `grid_position` — absolute grid slot (1 = pole)
- `quali_gap_to_pole_s` — raw seconds behind pole (clipped at 5s)
- `quali_gap_relative` — gap as fraction of max gap in that race (0–1, circuit-invariant)
- `grid_position_rel` — grid slot normalised by field size (0 = pole, 1 = last)
- `teammate_quali_delta` — driver's quali gap minus teammate's quali gap

### Driver form
- `driver_form_5` — rolling mean finishing position, last 5 races
- `driver_form_10` — rolling mean finishing position, last 10 races
- `driver_form_ewm` — exponential weighted mean finishing position (span=5, recent races weighted more)

### Circuit history (per driver, per circuit)
- `circuit_avg_finish` — historical average finish at this circuit
- `circuit_podiums` — historical podium count at this circuit
- `circuit_appearances` — number of prior starts at this circuit

### Reliability
- `dnf_rate` — driver's historical DNF rate

### Championship standing
- `championship_position` — driver's current WDC position
- `cum_points_before` — driver's cumulative points before this race

### Constructor
- `constructor_standing` — team's current WCC position
- `team_cum_points` — team's cumulative points
- `team_reliability_rate` — team-level DNF/finish rate

### Weather
- `temp_c_mean` — race day mean temperature (°C)
- `precip_mm_total` — total precipitation (mm)
- `windspeed_ms_mean` — mean wind speed (m/s)
- `humidity_pct_mean` — mean humidity (%)
- `is_wet_race` — binary flag (1 if precipitation above threshold)

### Free practice
- `fp_best_gap_s` — FP1 best lap gap to session-fastest
- `fp2_best_gap_s` — FP2 best lap gap to session-fastest
- `fp_long_run_delta` — FP long-run pace delta vs field
- `fp_total_laps` — total FP laps completed (proxy for setup work / reliability)

### Season progression
- `round_fraction` — (round − 1) / (total rounds − 1); 0 = season opener, 1 = finale

---

## Model architecture

### Primary model: XGBoost (default for inference)
- Three independent binary classifiers: one per target (p1, p2, p3)
- Dynamic `scale_pos_weight` to handle class imbalance (~20x for P1)
- Winner accuracy on holdout: **~52%**

### Secondary model: PyTorch MLP with embeddings
- Categorical embeddings for driver (dim 16), team (dim 8), circuit (dim 12)
- Dynamic `pos_weight` in `BCEWithLogitsLoss` for class imbalance
- Trained alongside XGBoost but underperforms it on this dataset size

### Stacking ensemble (meta-learner)
- Logistic regression meta-learner (C=0.01, heavy L2) combining XGB and NN out-of-fold predictions
- NN weight clamped to ≤2× XGB weight to prevent NN over-weighting
- Winner accuracy on holdout: **~25%** (worse than XGB alone due to noisy NN OOF predictions)
- **Not used for inference by default** — XGB-only is active until more data justifies stacking

### Isotonic calibration
- Isotonic regression calibrator trained on the final year of training data
- Converts raw model probabilities to calibrated probabilities
- Degeneracy guard: if calibrated sum < 0.05 (distribution shift, e.g. new season), falls back to raw probs normalised by sum

---

## Training strategy

**Two-pass approach** — both `quick_train.py` (XGB only) and `train.py` (full ensemble) run two sequential passes:

| Pass | Purpose | Training window | Output |
|------|---------|----------------|--------|
| Pass 1 — Evaluation | Honest holdout metrics | All years except last 1–2 | `holdout_metrics.json`, SHAP values, calibration curve |
| Pass 2 — Production | Best model for inference | All years except final year (used for calibration) | `xgb_podium.pkl`, `ensemble.pkl`, `calibrator.pkl` |

Year splits are computed dynamically from available data, not hardcoded.

---

## Inference pipeline

Triggered post-qualifying on race Saturday:

1. **Load historical data** — pulls completed races from the current season (in-season ingestion via FastF1)
2. **Fetch live quali results** — OpenF1 API for grid positions and gap times
3. **Fetch live FP data** — OpenF1 API for practice session laps (falls back to zeros if unavailable)
4. **Feature assembly** — mirrors the training feature matrix: clips quali gap, computes relative features, merges weather, computes rolling form
5. **Model inference** — XGBoost (default) or stacking ensemble via `--ensemble` flag
6. **Calibration** — isotonic calibrator with degeneracy fallback
7. **Output** — JSON saved to `data/predictions/YYYY_RNN.json`

---

## Post-race tracking

After each race completes:

```
python scripts/update_race_result.py --year YYYY --round N
```

- Fetches actual finishing order via FastF1
- Computes: winner hit?, podium overlap count, Brier score per target, ECE (Expected Calibration Error)
- Saves to `data/results/YYYY_RNN_actual.json`
- Used to accumulate season-level calibration data for future recalibration

---

## Streamlit dashboard — 4 pages

### Race Desk
- Per-driver win probability bar chart (team-coloured)
- Podium probability heatmap (drivers × P1/P2/P3)
- Top-3 podium cards with delta vs prior round
- Full driver table with probability columns + delta column
- Season win probability trend chart (round-by-round, multi-driver)
- Post-race result tracker: predicted vs actual, Brier score, podium overlap

### Historical Accuracy
- Winner accuracy and Brier score summary cards
- Calibration curve (predicted vs actual probability)
- Per-race accuracy leaderboard across seasons

### Feature Explorer
- SHAP bar chart, beeswarm, and waterfall plots
- XGBoost gain-based feature importance
- Feature correlation heatmap

### Control Panel
- Queue inference, quick retrain, full retrain — all in-process (no subprocess calls)
- Background task monitor with progress and status
- Active model metadata display (family, training window, artifact paths)

---

## Model registry and artifact management

- `models/active_model.json` — records which model family is active, training/calibration year ranges, artifact file paths
- `models/holdout_metrics.json` — family-agnostic holdout results
- `models/holdout_metrics_xgb.json` / `models/holdout_metrics_ensemble.json` — family-specific metrics
- `models/shap_values_p1.parquet` — pre-computed at train time; dashboard reads directly (no live SHAP computation)

---

## Known limitations / open issues

- **Stacking underperforms XGB alone** — NN out-of-fold predictions are noisier than XGB on this dataset size; stacking is disabled for inference by default
- **Calibrator distribution shift** — calibrator trained on 2025 data; degeneracy fallback active for 2026 until enough 2026 races are completed for recalibration
- **FP features zero-filled for 2025** — historical data was collected without `--fp` flag; FP columns are all 0 for 2025 rows
- **`round_fraction` not yet a trained feature** — added to feature matrix but current XGB pkl predates it; forward-compat shim silently drops it until next retrain
- **Sentiment pipeline unused** — `sentiment.py` and `sentiment_features.py` are preserved but not wired into any active path

---

## Repository structure (key paths)

```
src/
  data_collection/
    f1_historical.py     # FastF1 batch historical pull
    f1_live.py           # OpenF1 live quali/FP + in-season race results
    weather.py           # Open-Meteo fetch
  features/
    build_features.py    # Main feature pipeline (FEATURE_COLS = 27)
    driver_features.py   # Form, circuit history, DNF rate, EWM
    team_features.py     # Constructor standing, reliability, pit stops
    weather_features.py  # Weather merge
    practice_features.py # FP gap and long-run delta
  models/
    xgb_model.py         # XGBoost 3-classifier
    nn_model.py          # PyTorch MLP with embeddings
    ensemble.py          # Stacking meta-learner
    calibration.py       # Isotonic calibration + ECE utilities
    model_registry.py    # Active model metadata read/write
  training/
    train.py             # Full ensemble training (two-pass)
    evaluate.py          # Brier score, log loss, winner accuracy, ECE
    cross_validate.py    # Leave-one-season-out CV
  inference/
    predict.py           # Inference engine
  pipeline/
    actions.py           # Shared in-process action runner
  ui/
    app.py               # Streamlit shell (layout, navigation, context bar)
    services.py          # UI data layer
    views/               # Page renderers (race_prediction, historical_accuracy, ...)

scripts/
  quick_train.py               # XGB-only train (~5 min)
  race_weekend_inference.py    # Saturday post-quali inference script
  update_race_result.py        # Post-race actual result fetch + comparison

data/
  raw/                         # FastF1 cache, weather CSVs
  processed/
    feature_matrix.parquet     # 27-feature training matrix (one row per driver per race)
  predictions/
    YYYY_RNN.json              # Pre-race probability outputs
  results/
    YYYY_RNN_actual.json       # Post-race actuals + accuracy metrics

models/
  xgb_podium.pkl
  ensemble.pkl
  calibrator.pkl
  active_model.json
  holdout_metrics.json
  shap_values_p1.parquet
```

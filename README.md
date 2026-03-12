# F1 Race Outcome Prediction Pipeline

Predicts P1/P2/P3 podium probabilities for each F1 race using an ensemble of XGBoost + PyTorch, trained on 2014–2024 historical data.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
```

### 3. Collect historical data (one-time, takes ~1-2 hours)
```bash
# F1 session data (2014–2024)
python -m src.data_collection.f1_historical --start 2014 --end 2024

# Historical weather
python -m src.data_collection.weather

# Optional: sentiment history (only if you still want the legacy feature)
# python -m src.data_collection.sentiment --start-year 2021 --end-year 2024
```

### 4. Build feature matrix
```bash
python -m src.features.build_features
```

### 5. Train the model
```bash
# Full ensemble (XGBoost + NN, ~2-4 hours on CPU)
python -m src.training.train

# Quick XGBoost-only (~5 minutes)
python -m src.training.train --xgb-only
```

### 6. Race Weekend Inference (after qualifying concludes)
```bash
python scripts/race_weekend_inference.py --year 2025 --round 5

# Options:
#   --no-sentiment   Skip Reddit sentiment (faster)
#   --xgb-only       Use XGBoost only (faster, slightly less accurate)
#   --dry-run        Print data without running model (for debugging)
```

### 7. Launch Dashboard
```bash
streamlit run src/ui/app.py
```

Web UI is the only supported interface in this repository (desktop UI retired).

---

## Project Structure

```
2_F1/
├── data/
│   ├── raw/            # Fetched data (parquet)
│   ├── processed/      # Feature matrix
│   ├── cache/          # FastF1 disk cache
│   └── predictions/    # Per-race prediction JSONs
├── models/             # Trained model artifacts
├── src/
│   ├── data_collection/
│   │   ├── f1_historical.py   # FastF1 historical sessions
│   │   ├── f1_live.py         # OpenF1 live quali data
│   │   ├── weather.py         # Open-Meteo
│   │   └── sentiment.py       # Reddit PRAW + RoBERTa NLP
│   ├── features/
│   │   ├── build_features.py  # Main feature pipeline (entry point)
│   │   ├── driver_features.py
│   │   ├── team_features.py
│   │   ├── weather_features.py
│   │   └── sentiment_features.py
│   ├── models/
│   │   ├── xgb_model.py       # XGBoost classifier
│   │   ├── nn_model.py        # PyTorch MLP with embeddings
│   │   ├── ensemble.py        # Stacking meta-learner
│   │   └── calibration.py     # Isotonic probability calibration
│   ├── training/
│   │   ├── train.py           # Training entry point
│   │   ├── evaluate.py        # Metrics: Brier, log loss, accuracy
│   │   └── cross_validate.py  # Leave-one-season-out CV
│   ├── inference/
│   │   └── predict.py         # Feature assembly + inference
│   └── ui/
│       └── app.py             # Streamlit dashboard
├── scripts/
│   └── race_weekend_inference.py  # Saturday post-quali entry point
├── notebooks/
│   └── eda.ipynb
├── config/
│   ├── config.yaml
│   └── settings.py
└── requirements.txt
```

## Features Used

| Category | Features |
|---|---|
| Qualifying | Grid position, gap to pole (seconds), teammate delta |
| Driver form | Rolling avg finish (last 5, 10 races) |
| Circuit history | Avg finish, podium count, appearances at this track |
| Reliability | Rolling DNF rate |
| Championship | WDC position, cumulative points |
| Constructor | WCC position, team reliability rate |
| Weather | Temperature, precipitation, wind speed, wet/dry flag |
| Free practice | FP1/FP2/FP3 best gap, long-run pace delta, total laps |
| Sentiment (optional legacy) | 7-day Reddit sentiment score per driver/team |

## Model Architecture

```
XGBoost (P1 classifier) ─┐
XGBoost (P2 classifier) ─┤
XGBoost (P3 classifier) ─┤
                          ├─→ Stacking Meta-Learner (LogReg) ─→ Isotonic Calibration ─→ Output
NN (P1 logit) ───────────┤
NN (P2 logit) ───────────┤
NN (P3 logit) ───────────┘
```

- XGBoost: gradient-boosted trees on all tabular features
- NN: PyTorch MLP with driver/team/circuit embeddings
- Stacking: logistic meta-learner trained on OOF (out-of-fold) predictions
- Calibration: isotonic regression ensures well-calibrated probabilities

## Evaluation

- **Leave-one-season-out CV**: trains on all seasons before the test season
- **Holdout**: 2023–2024 seasons never seen during training
- **Metrics**: Brier score, log loss, ROC-AUC, winner accuracy, podium overlap

## API Keys Required

| Service | Key Required | Where to Get |
|---|---|---|
| FastF1 | None | Built into library |
| OpenF1 | None | Public API |
| Open-Meteo | None | Public API |
| Reddit | Optional (legacy sentiment collector) | reddit.com/prefs/apps |



### UI Cockpit Configuration (Optional)
```bash
# Defaults shown
UI_DENSITY_DEFAULT=dense
UI_ENABLE_LIVE_WIDGETS=false
UI_TASK_POLL_SECONDS=5
UI_MAX_VISIBLE_BLOCKS=6
```


### Race Ops Cockpit (Web UI)
- Three-region shell: context rail, main workspace, persistent right rail.
- Density modes: `dense` (default) and `comfort`.
- Right rail panels: Driver Focus, Task Monitor (polling), Quick Actions.
- Optional live widgets with artifact fallback (OpenF1 when available).
- Command-deck Control Panel with preflight checks and task severity states.



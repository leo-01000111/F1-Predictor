# F1 race outcome prediction pipeline

Predicts finishing-position probabilities (win, P2, P3, podium, top 6, top 10) for every driver before an F1 race. A single XGBoost regressor predicts each driver's finish position, and a Plackett-Luce simulation turns those predictions into a position probability matrix. Training data starts in 2014. A Streamlit dashboard displays the predictions.

The data and trained models are not committed to the repository (`data/` and `models/` are git-ignored). You have to collect data and train a model before the dashboard shows anything.

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` still lists packages used only by archived experiments (torch, transformers, praw). The active pipeline does not import them.

### 2. Set environment variables (optional)
There is no `.env.example`. Every variable has a default, so nothing is required to train a model or run inference. See [Configuration](#configuration) for the list.

### 3. Collect historical data
```bash
# Race and qualifying results, plus FP1/FP2/FP3 (FastF1)
python -m src.data_collection.f1_historical --start 2014 --end 2024

# Historical weather (Open-Meteo) for the races in data/raw/race_results.parquet
python -m src.data_collection.weather
```
<!-- TODO: add measured run times for data collection (depends on FastF1 cache state and rate limits) -->

`f1_historical` also accepts `--no-fp` (skip practice sessions), `--merge` (only collect years not already saved) and `--sleep` (seconds between requests).

Three feature groups come from separate collectors. Without them the matching features are filled with neutral values (see `src/features/build_features.py`):
```bash
# Lap-1 position and start compound (FastF1 lap data exists from 2018)
python scripts/backfill_lap_extras.py --start 2018 --end 2025 --merge

# Pit stop times and sprint results
python scripts/backfill_pit_sprint.py --pit --start 2014 --end 2025 --merge
python scripts/backfill_pit_sprint.py --sprint --start 2021 --end 2025 --merge
```

### 4. Build features and train
```bash
python scripts/quick_train.py
```

`quick_train.py` loads the raw data, builds the feature matrix (`data/processed/feature_matrix.parquet`), trains the model, and writes the artifacts the dashboard reads. It runs in two passes, described under [Evaluation](#evaluation). To rebuild only the feature matrix, run `python -m src.features.build_features`.

<!-- TODO: add measured training time for quick_train.py -->

Optional hyperparameter search with Optuna, then training with the result:
```bash
python scripts/tune_hyperparams.py --n-trials 60 --timeout 900
python scripts/quick_train.py --use-tuned-params
```
`tune_hyperparams.py` also accepts `--holdout-years 2024 2025`. It saves the best parameters to `models/optuna_best_params.json`.

### 5. Race weekend inference (after qualifying)
```bash
python scripts/race_weekend_inference.py --year 2026 --round 5

# Pass start tyres read from the grid walk (overrides OpenF1; default is Medium)
python scripts/race_weekend_inference.py --year 2026 --round 5 --compounds VER:SOFT NOR:MEDIUM RUS:HARD

# Print the fetched qualifying, practice and weather data without running the model
python scripts/race_weekend_inference.py --year 2024 --round 10 --dry-run
```
The script pulls qualifying and practice data from OpenF1, completed in-season results from FastF1, and a race-day forecast from Open-Meteo. It writes `data/predictions/<year>_R<round>.json`. Valid compounds are SOFT, MEDIUM, HARD, INTERMEDIATE and WET.

### 6. After the race
```bash
# Record the actual result and compare it with the saved prediction (writes under data/results/)
python scripts/update_race_result.py --year 2026 --round 5

# Add the new results to the raw data, rebuild features, retrain
python scripts/post_race_retrain.py --year 2026 --round 5

# Per-round accuracy and calibration summary for a season, from data/results/
python scripts/season_accuracy.py --year 2026
```

### 7. Launch the dashboard
```bash
streamlit run src/ui/app.py
```

The dashboard is a Streamlit app with four tabs: Race Desk, SHAP Analysis, Historical, and Control Panel. The Control Panel runs inference, result recording and retraining as background tasks. The Race Desk includes an expected-value table that takes a bookmaker odds CSV with the columns `driver, bm_win, bm_1-3, bm_1-6, bm_1-10`.

## Project structure

```
.
├── config/
│   ├── config.yaml
│   └── settings.py              # pydantic settings, paths, UI options
├── data/                        # not committed
│   ├── raw/                     # parquet files from the collectors
│   ├── processed/               # feature_matrix.parquet
│   ├── cache/                   # FastF1 cache
│   ├── predictions/             # one JSON per race
│   └── results/                 # recorded actual results
├── models/                      # not committed (model pickle, metrics, SHAP values)
├── experiments/archived/
│   ├── nn_ensemble/             # XGBoost + MLP + logistic stacking experiment (not used)
│   └── sentiment/               # Reddit sentiment experiment (not used)
├── scripts/
│   ├── quick_train.py           # feature build + training + evaluation artifacts
│   ├── tune_hyperparams.py      # Optuna search
│   ├── race_weekend_inference.py
│   ├── update_race_result.py
│   ├── post_race_retrain.py
│   ├── season_accuracy.py
│   ├── backfill_lap_extras.py
│   ├── backfill_pit_sprint.py
│   └── pipeline_after_ratelimit.py
├── src/
│   ├── data_collection/
│   │   ├── f1_historical.py     # FastF1 race, quali, practice, lap extras, pit stops, sprints
│   │   ├── f1_live.py           # OpenF1 live quali and practice data
│   │   ├── weather.py           # Open-Meteo archive and forecast
│   │   └── odds.py              # The Odds API race winner odds
│   ├── features/
│   │   ├── build_features.py    # FEATURE_COLS and the feature pipeline
│   │   ├── driver_features.py
│   │   ├── team_features.py
│   │   ├── circuit_features.py
│   │   ├── practice_features.py
│   │   └── weather_features.py
│   ├── models/
│   │   ├── xgb_model.py         # XGBRacePredictor (regressor)
│   │   ├── plackett_luce.py     # position probability matrix
│   │   ├── calibration.py       # expected calibration error
│   │   └── model_registry.py    # active model manifest, metrics files
│   ├── training/
│   │   ├── cross_validate.py    # leave-one-season-out CV
│   │   ├── evaluate.py          # holdout metrics
│   │   └── weights.py           # recency sample weights
│   ├── inference/
│   │   └── predict.py           # feature assembly and prediction JSON
│   ├── pipeline/
│   │   └── actions.py           # actions used by the dashboard's Control Panel
│   └── ui/
│       ├── app.py               # Streamlit entry point
│       ├── services.py
│       └── views/
├── tests/
│   └── test_ui_services.py
├── PROJECT_DESCRIPTION.md       # older description, see note below
└── requirements.txt
```

`scripts/pipeline_after_ratelimit.py` is a one-off script with a hard-coded start time and still passes `--xgb-only` to the inference script, which no longer accepts that flag. Treat it as a leftover.

## Model

```
features (35 columns) -> XGBoost regressor -> predicted finish position per driver
                      -> Plackett-Luce simulation -> P[driver, position] matrix
                      -> p_win, p_p2, p_p3, p_podium, p_top6, p_top10
```

- The regressor (`XGBRacePredictor` in `src/models/xgb_model.py`) predicts `target_position`, the finishing position with DNFs set to the number of starters plus one. It uses the `reg:absoluteerror` objective, and the number of trees is chosen by early stopping on an internal 10% split before the final fit.
- Training uses recency sample weights (`src/training/weights.py`): older seasons get lower weight, linearly from 0.65 to 1.35 before normalising.
- `PlackettLuceSampler` (`src/models/plackett_luce.py`) draws 10,000 simulated race orderings from the predicted positions using the Gumbel-max trick, with temperature 1.0. Each driver's row of the resulting matrix sums to 1, and each position's column sums to 1, so podium probabilities across the field add up to 3.
- There is no separate calibration step. `src/models/calibration.py` only computes expected calibration error for evaluation, and the `calibration` entry in `config/config.yaml` is not read by the active pipeline.
- The three-classifier XGBoost plus PyTorch MLP plus logistic-regression stacking ensemble is archived in `experiments/archived/nn_ensemble/`, and the Reddit sentiment features are archived in `experiments/archived/sentiment/`. Neither is imported by the active code.

## Features

The 35 model inputs are listed in `FEATURE_COLS` in `src/features/build_features.py`. All rolling and history features use only races before the one being predicted. Missing values are filled with the column median.

| Group | Features |
|---|---|
| Qualifying | `grid_position`, `quali_gap_relative`, `grid_position_rel`, `teammate_quali_delta`, `grid_penalty_places` |
| Car pace | `car_pace_delta_pct` |
| Driver form | `driver_form_5`, `driver_form_10`, `driver_form_ewm`, `driver_elo`, `driver_wet_avg_finish`, `driver_experience_norm` |
| Start | `driver_start_delta_avg`, `start_compound_code`, `compound_aggression_delta`, `circuit_grid_conversion` |
| Circuit | `circuit_avg_finish`, `circuit_podiums`, `circuit_appearances`, `is_street_circuit`, `circuit_safety_car_rate` |
| Reliability | `dnf_rate`, `team_reliability_rate` |
| Championship | `championship_position`, `cum_points_before`, `championship_pressure`, `round_fraction` |
| Constructor | `constructor_standing`, `team_cum_points`, `team_form_5`, `team_pit_delta_s` |
| Weather | `temp_c_mean`, `precip_mm_total`, `windspeed_ms_mean`, `humidity_pct_mean`, `is_wet_race` |
| Free practice | `fp_best_gap_s`, `fp2_best_gap_s`, `fp_long_run_delta` |
| Sprint | `sprint_position_rel` |

## Evaluation

`scripts/quick_train.py` runs two passes.

1. Evaluation pass. The last two seasons present in the feature matrix are the holdout (2023 and 2024 for data collected through 2024, or 2024 and 2025 if you also collected 2025). The third-from-last season is also kept out of training, and the rest are used for training. Probabilities come from the Plackett-Luce sampler. Metrics (winner accuracy, podium accuracy, P1 Brier score, Spearman correlation, mean position error, expected calibration error) are written to `models/holdout_metrics.json`, alongside per-race results, a P1 calibration curve and SHAP values for the dashboard. With fewer than 8 seasons the split shrinks (see the script).
2. Production pass. The active model (`models/xgb_race.pkl`) is retrained on every season except the most recent one in the data. This includes seasons that were holdout in pass 1 (with data through 2024, 2023 is in the production training set and 2024 is not). The holdout metrics therefore describe the pass 1 model, not the production model.

`python -m src.training.cross_validate` runs leave-one-season-out cross-validation: each season from `--min-year` (default 2019) to `--max-year` is held out in turn and the model is trained on all other seasons, including later ones. Results go to `models/loso_cv_results.parquet` and `models/loso_cv_summary.json`.

`scripts/update_race_result.py` and `scripts/season_accuracy.py` track accuracy on real races once a season is under way.

<!-- TODO: add measured results (holdout metrics, CV summary) once the author has run the final training -->

## Configuration

Settings are defined in `config/settings.py` and read from environment variables or a `.env` file in the repository root (git-ignored). `config/config.yaml` holds data ranges, API URLs, circuit coordinates and the XGBoost defaults used before tuning.

| Variable | Default | Used for |
|---|---|---|
| `ODDS_API_KEY` | empty | The Odds API key for `src/data_collection/odds.py`. Read with `os.getenv`, so export it in the shell (a `.env` file is not loaded into the environment by this code). Nothing in the pipeline or dashboard calls the odds fetcher yet. |
| `UI_DENSITY_DEFAULT` | `dense` | Dashboard density, `dense` or `comfort` |
| `UI_ENABLE_LIVE_WIDGETS` | `false` | Optional live widgets in the dashboard |
| `UI_TASK_POLL_SECONDS` | `5` | Background task polling interval |
| `UI_MAX_VISIBLE_BLOCKS` | `6` | Maximum visible dashboard blocks |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | empty, empty, `f1_prediction_bot/1.0` | Legacy, only for the archived sentiment experiment |

FastF1, OpenF1 and Open-Meteo need no keys.

## Tests

```bash
pip install pytest
pytest tests
```
The tests cover the dashboard service helpers (`tests/test_ui_services.py`). `pytest` is not listed in `requirements.txt`.

## Notes

`PROJECT_DESCRIPTION.md` was written for the earlier three-classifier version and does not match the current model.

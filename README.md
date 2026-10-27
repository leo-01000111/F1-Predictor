# F1 race outcome prediction pipeline

Predicts finishing-position probabilities (win, P2, P3, podium, top 6, top 10) for every driver before an F1 race. A single XGBoost regressor predicts each driver's finish position, and a Plackett-Luce simulation turns those predictions into a position probability matrix. Training data starts in 2014. GitHub Actions runs it every race weekend and publishes the predictions on [leongorecki.eu](https://leongorecki.eu/projects/f1predictor.html).

The raw and processed data (about 1 MB of parquet), the predictions and results, and the active model (`models/xgb_race.pkl`) are committed, because the GitHub Actions automation below uses them as state. The FastF1 cache and other model artifacts are git-ignored.

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
# Race and qualifying results (FastF1, via its Ergast/Jolpica fallback)
python -m src.data_collection.f1_historical --start 2014 --end 2024

# Historical weather (Open-Meteo) for the races in data/raw/race_results.parquet
python -m src.data_collection.weather
```
<!-- TODO: add measured run times for data collection (depends on FastF1 cache state and rate limits) -->

`f1_historical` also accepts `--merge` (only collect years not already saved) and `--sleep` (seconds between requests).

Lap-level data (practice pace, lap-1 positions, start tyres) is not used: F1's livetiming archive, which FastF1 reads it from, now requires a paid F1 TV subscription.

Two feature groups come from separate collectors. Without them the matching features are filled with neutral values (see `src/features/build_features.py`):
```bash
# Pit stop times and sprint results
python scripts/backfill_pit_sprint.py --pit --start 2014 --end 2025 --merge
python scripts/backfill_pit_sprint.py --sprint --start 2021 --end 2025 --merge
```

### 4. Build features and train
```bash
python scripts/quick_train.py
```

`quick_train.py` loads the raw data, builds the feature matrix (`data/processed/feature_matrix.parquet`), trains the model, and writes the model and evaluation artifacts. It runs in two passes, described under [Evaluation](#evaluation). To rebuild only the feature matrix, run `python -m src.features.build_features`.

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

# Print the fetched qualifying and weather data without running the model
python scripts/race_weekend_inference.py --year 2024 --round 10 --dry-run
```
The script pulls qualifying from OpenF1, completed in-season results from FastF1, and a race-day forecast from Open-Meteo. It writes `data/predictions/<year>_R<round>.json`.

For a round that was missed live, `python scripts/predict_past_race.py --year 2026 --round 18` predicts it from data available before the race (history, qualifying, sprint), with the measured race-day weather in place of the forecast.

### 6. After the race
```bash
# Record the actual result and compare it with the saved prediction (writes under data/results/)
python scripts/update_race_result.py --year 2026 --round 5

# Add the new results to the raw data, rebuild features, retrain
python scripts/post_race_retrain.py --year 2026 --round 5

# Per-round accuracy and calibration summary for a season, from data/results/
python scripts/season_accuracy.py --year 2026
```

## Automation (GitHub Actions)

`.github/workflows/race-weekend.yml` runs `scripts/ci_tick.py` every hour from Saturday to Monday (UTC). Each tick reads the FastF1 calendar and the files in `data/`, then does whatever is due:

- 45 minutes after qualifying ends, it predicts the race (`race_weekend_inference.py`).
- 4 hours after the race, once the result is published, it collects the season's race, qualifying, weather, pit stop and sprint data, records the result against the prediction, and retrains (`quick_train.py`).
- Every tick, it writes `site/f1-predictions.json` (`scripts/export_site_json.py`). The website copies that file for the live predictions on [leongorecki.eu/projects/f1predictor.html](https://www.leongorecki.eu/projects/f1predictor.html).

The workflow commits the updated data, model and JSON. CI installs `requirements-ci.txt`, whose versions are pinned to the ones that trained the committed pickle. Scheduled workflows are disabled after 60 days without commits. Over the off-season, `scripts/keepalive.py` (run weekly by a Windows scheduled task named "F1 keepalive") commits a timestamp to `.github/keepalive.txt` in this repo and the website repo whenever the last commit is more than 45 days old.

Check what a tick would do without running anything:
```bash
python scripts/ci_tick.py --dry-run --now 2026-10-31T23:50:00Z
```

`scripts/check_feature_skew.py 2026 18 austin` builds the features of a past race through both the training and the inference path and lists every feature that differs. Run it after changing feature code.

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
│   ├── predict_past_race.py
│   ├── keepalive.py
│   ├── backfill_pit_sprint.py
│   └── pipeline_after_ratelimit.py
├── src/
│   ├── data_collection/
│   │   ├── f1_historical.py     # FastF1 race, quali, sprints; Jolpica pit stops
│   │   ├── f1_live.py           # OpenF1 live quali and practice data
│   │   ├── weather.py           # Open-Meteo archive and forecast
│   │   └── odds.py              # The Odds API race winner odds
│   ├── features/
│   │   ├── build_features.py    # FEATURE_COLS and the feature pipeline
│   │   ├── driver_features.py
│   │   ├── team_features.py
│   │   ├── circuit_features.py
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
│   │   └── actions.py           # inference and result-recording actions
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

The 34 model inputs are listed in `FEATURE_COLS` in `src/features/build_features.py`. All rolling and history features use only races before the one being predicted. Missing values are filled with the column median.

| Group | Features |
|---|---|
| Qualifying | `grid_position`, `quali_gap_relative`, `grid_position_rel`, `teammate_quali_delta`, `grid_penalty_places` |
| Car pace | `car_pace_delta_pct` |
| Driver form | `driver_form_5`, `driver_form_10`, `driver_form_ewm`, `driver_elo`, `driver_wet_avg_finish`, `driver_experience_norm` |
| Start | `circuit_grid_conversion` |
| Circuit | `circuit_avg_finish`, `circuit_podiums`, `circuit_appearances`, `is_street_circuit`, `circuit_safety_car_rate` |
| Reliability | `dnf_rate`, `team_reliability_rate` |
| Championship | `championship_position`, `cum_points_before`, `championship_pressure`, `round_fraction` |
| Constructor | `constructor_standing`, `team_cum_points`, `team_form_5`, `team_pit_delta_s` |
| Weather | `temp_c_mean`, `precip_mm_total`, `windspeed_ms_mean`, `humidity_pct_mean`, `is_wet_race` |
| Sprint | `sprint_position_rel` |

## Evaluation

`scripts/quick_train.py` runs two passes.

1. Evaluation pass. The last two seasons present in the feature matrix are the holdout (2023 and 2024 for data collected through 2024, or 2024 and 2025 if you also collected 2025). The third-from-last season is also kept out of training, and the rest are used for training. Probabilities come from the Plackett-Luce sampler. Metrics (winner accuracy, podium accuracy, P1 Brier score, Spearman correlation, mean position error, expected calibration error) are written to `models/holdout_metrics.json`, alongside per-race results, a P1 calibration curve and SHAP values. With fewer than 8 seasons the split shrinks (see the script).
2. Production pass. The active model (`models/xgb_race.pkl`) is retrained on every season except the most recent one in the data. This includes seasons that were holdout in pass 1 (with data through 2024, 2023 is in the production training set and 2024 is not). The holdout metrics therefore describe the pass 1 model, not the production model.

`python -m src.training.cross_validate` runs leave-one-season-out cross-validation: each season from `--min-year` (default 2019) to `--max-year` is held out in turn and the model is trained on all other seasons, including later ones. Results go to `models/loso_cv_results.parquet` and `models/loso_cv_summary.json`.

`scripts/update_race_result.py` and `scripts/season_accuracy.py` track accuracy on real races once a season is under way.

<!-- TODO: add measured results (holdout metrics, CV summary) once the author has run the final training -->

## Configuration

Settings are defined in `config/settings.py` and read from environment variables or a `.env` file in the repository root (git-ignored). `config/config.yaml` holds data ranges, API URLs, circuit coordinates and the XGBoost defaults used before tuning.

| Variable | Default | Used for |
|---|---|---|
| `ODDS_API_KEY` | empty | The Odds API key for `src/data_collection/odds.py`. Read with `os.getenv`, so export it in the shell (a `.env` file is not loaded into the environment by this code). Nothing in the pipeline calls the odds fetcher yet. |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | empty, empty, `f1_prediction_bot/1.0` | Legacy, only for the archived sentiment experiment |

FastF1, OpenF1 and Open-Meteo need no keys.

## Notes

`PROJECT_DESCRIPTION.md` was written for the earlier three-classifier version and does not match the current model.

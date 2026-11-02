# F1 race outcome prediction pipeline

Predicts finishing-position probabilities (win, P2, P3, podium, top 6, top 10) for every driver before an F1 race. An XGBoost ranking model orders the drivers, a logistic model gives each car's chance of finishing, and a race simulation (retirements plus a Plackett-Luce order) turns both into a position probability matrix. A Kalman-filter season simulation gives championship odds. Training data starts in 2014. GitHub Actions runs it every race weekend and publishes the predictions on [leongorecki.eu](https://leongorecki.eu/projects/f1predictor.html).

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
python -m src.data_collection.f1_historical --start 2014 --end 2026

# Historical weather (Open-Meteo) for the races in data/raw/race_results.parquet
python -m src.data_collection.weather
```
<!-- TODO: add measured run times for data collection (depends on FastF1 cache state and rate limits) -->

`f1_historical` also accepts `--merge` (only collect years not already saved) and `--sleep` (seconds between requests).

FastF1's lap data (lap-1 positions, start tyres) is not used: F1's livetiming archive now requires a paid F1 TV subscription. Practice laps come from OpenF1 instead (2023 on):
```bash
python -m src.data_collection.practice_openf1
```

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

The production ranker's settings come from `scripts/tune_ranker.py` (walk-forward, picked on 2018-2022). The Optuna search below tunes the archived regressor (`XGBRacePredictor`) and does not affect the production model:
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

`.github/workflows/race-weekend.yml` runs `scripts/ci_tick.py` every hour from Saturday to Monday (UTC), plus an extra Sunday run at :48. Each tick reads the FastF1 calendar and the files in `data/`, then does whatever is due:

- 45 minutes after qualifying ends, it predicts the race (`race_weekend_inference.py`). Until lights out it re-predicts whenever OpenF1's starting grid changes (penalties), and snapshots the live Kalshi win and podium books (`data/raw/kalshi_snapshots.parquet`).
- 4 hours after the race, it collects the season's race, qualifying, weather, pit stop, sprint, practice and Kalshi data, records the result against the prediction, retrains (`quick_train.py`, behind the promotion gate) and updates the championship odds (`title_odds.py`).
- Every tick, it writes `site/f1-predictions.json` (`scripts/export_site_json.py`). The website copies that file hourly for the live section, the model-vs-market calculator and the title odds on [leongorecki.eu/projects/f1predictor.html](https://www.leongorecki.eu/projects/f1predictor.html).

Every tick appends to `data/logs/ci_<season>.jsonl` (decisions, each step's exit code and output tail, raw-file row counts, prediction inputs, data-source availability probes). `python scripts/ci_log_report.py --year 2026 --round 19` prints a round's timeline.

The workflow commits the updated data, model and JSON: `data/` and the production model files are committed as CI state. CI installs `requirements-ci.txt`, pinned to the versions that trained the committed pickle. Scheduled workflows are disabled after 60 days without commits; over the off-season `scripts/keepalive.py` (a weekly Windows scheduled task, "F1 keepalive") commits a timestamp to both repos when the last commit is more than 45 days old.

Check what a tick would do without running anything:
```bash
python scripts/ci_tick.py --dry-run --now 2026-10-31T23:50:00Z
```

`scripts/check_feature_skew.py 2026 18 austin` builds a past race's features through both the training and the inference path and lists every feature that differs. Run it after changing feature code.

## Project structure

```
.
├── config/                      # settings.py (paths, env), config.yaml (ranges, URLs, coordinates)
├── data/
│   ├── raw/                     # parquet files from the collectors (committed, CI state)
│   ├── processed/               # feature_matrix.parquet, feature_fill_values.json
│   ├── predictions/             # one JSON per race, YYYY_title.json
│   ├── results/                 # recorded actual results
│   └── logs/                    # CI event log per season
├── models/                      # xgb_race.pkl, active_model.json, holdout_metrics.json
│   └── experiments/             # backtest, calibration, gym and Kalman results
├── experiments/archived/        # earlier models (regressor, stacking + MLP, sentiment), not imported
├── site/f1-predictions.json     # what the website reads
├── scripts/
│   ├── ci_tick.py               # the scheduled race-weekend driver
│   ├── quick_train.py           # features + two-pass training + promotion gate
│   ├── race_weekend_inference.py, predict_past_race.py, replay_season.py
│   ├── update_race_result.py, post_race_retrain.py, season_accuracy.py
│   ├── title_odds.py            # championship simulation for the site
│   ├── export_site_json.py
│   ├── backtest.py              # walk-forward model choice (top-3 log loss vs the grid)
│   ├── gym.py                   # stepwise toggle search with a lockbox
│   ├── reliability.py, market_backtest.py, championship_backtest.py, kalman_backtest.py, ...
│   └── check_feature_skew.py, ci_log_report.py, keepalive.py
└── src/
    ├── data_collection/         # f1_historical (FastF1/Jolpica), f1_live (OpenF1, Jolpica live),
    │                            # practice_openf1, weather (Open-Meteo), kalshi_odds, source_probe
    ├── features/                # build_features (FEATURE_COLS), driver/team/circuit/weather
    │                            # features, practice_pace, strength_ratings, circuits, teams
    ├── models/
    │   ├── xgb_model.py         # TwoStageRaceModel (production), XGBRaceRanker
    │   ├── two_stage.py         # FinishModel, top-k PL likelihood, race simulation
    │   ├── race_calibration.py  # PL temperature + parked post-hoc layers (default off)
    │   ├── season_sim.py        # championship Monte Carlo
    │   ├── kalman_strength.py   # driver + car Kalman filter (title odds)
    │   └── model_registry.py, calibration.py (ECE), plackett_luce.py
    ├── training/                # backtest, promotion, gym, evaluate, weights, cross_validate
    ├── inference/predict.py     # feature assembly at race time, prediction JSON
    ├── pipeline/actions.py
    └── ci_log.py
```

## Model

```
35 features -> 5 averaged XGBoost NDCG rankers (finishers only) -> score per driver, / T
            -> logistic FinishModel                             -> P(finish) per car
            -> race simulation: retirements from P(finish), finishers in Plackett-Luce order
            -> p_win, p_p2, p_p3, p_podium, p_top6, p_top10
```

- **Ranker** (`TwoStageRaceModel` in `src/models/xgb_model.py`, family `two_stage_v5`): XGBoost `rank:ndcg`, one query group per race, trained on classified finishers only. Depth 3, 300 trees, learning rate 0.03 (`scripts/tune_ranker.py`, picked on 2018-2022), 5 seeds averaged. Recency sample weights from 0.65 (oldest season) to 1.35 (newest).
- **Finish model** (`src/models/two_stage.py`): logistic regression on reliability, circuit, grid, weather and experience inputs. Retired cars go to the back of the simulated order.
- **Probabilities**: `two_stage.simulate` (10,000 Gumbel-max draws). The Plackett-Luce temperature T is the only calibration step, fitted on a season the model did not train on. Each driver's row and each position's column of the result sum to 1.
- **Championship odds** (`scripts/title_odds.py`): a Kalman filter over every qualifying session and race (`kalman_strength.py`) gives car and driver pace with uncertainty; each of 20,000 simulated seasons draws from it and lets pace drift race by race. Retirements come from a pre-qualifying finish model.
- Earlier models (a finishing-position regressor, three classifiers + MLP stacking with an isotonic calibrator) are in `experiments/archived/`.

## Features

The 35 inputs are `FEATURE_COLS` in `src/features/build_features.py`. Every rolling or history feature uses only races before the one being predicted; missing values get the training fill values saved in `data/processed/feature_fill_values.json`, except the practice features (`NAN_OK_COLS`), which stay missing before 2023.

| Group | Features |
|---|---|
| Qualifying | `grid_position`, `quali_gap_relative`, `grid_position_rel`, `teammate_quali_delta`, `grid_penalty_places` |
| Car pace | `car_pace_delta_pct` |
| Driver form | `driver_form_5`, `driver_form_10`, `driver_form_ewm`, `driver_elo`, `driver_wet_avg_finish`, `driver_experience_norm` |
| Start | `circuit_grid_conversion` |
| Circuit | `circuit_avg_finish`, `circuit_podiums`, `circuit_appearances`, `is_street_circuit`, `circuit_safety_car_rate` |
| Reliability | `dnf_rate`, `team_reliability_rate` |
| Championship | `championship_position`, `cum_points_before`, `championship_pressure`, `round_fraction` |
| Constructor | `constructor_standing`, `team_cum_points`, `team_form_5` |
| Weather | `temp_c_mean`, `precip_mm_total`, `windspeed_ms_mean`, `humidity_pct_mean`, `is_wet_race` |
| Sprint | `sprint_position_rel` |
| Practice (OpenF1, 2023 on) | `fp_longrun_gap_pct`, `team_longrun_gap_pct` |

The grid at inference is OpenF1's starting grid (after penalties) when published, else the qualifying order. FastF1 lap data needs a paid F1 TV subscription, so practice laps come from OpenF1.

## Evaluation

- **Model choice: walk-forward backtest** (`scripts/backtest.py`). Each season from 2018 is predicted by a model trained only on earlier seasons, with T fitted on the previous season's out-of-sample predictions. Primary metric: per-race Plackett-Luce log loss of the ordered top 3, compared with the starting grid as a probability model, with a paired bootstrap. v5 over 2018-2026 (192 races): 4.50 vs 5.30 for the grid.
- **Calibration** (`scripts/reliability.py`): P(win) is calibrated (slope 1.09 [0.95, 1.27]); P(podium) overstates the favourite (0.85 predicted, 0.78 observed), from the order among finishers rather than the finish model.
- **Against the market** (`scripts/market_backtest.py`): at lights out the model beats Kalshi's de-vigged prices on winner log loss (2025-2026).
- **`quick_train.py` runs two passes.** Pass 1 holds out the last two seasons and writes `models/holdout_metrics.json` (2025-2026, 43 races: winner 62.8%, podium drivers 71.3% vs 67.4% / 69.0% for the grid, Brier P1 0.025). Pass 2 trains the production candidate on every season except the latest. A 43-race holdout swings by 10+ points with the training window, so it is a sanity check, not a model-selection tool.
- **Promotion gate** (`src/training/promotion.py`): the candidate replaces `models/xgb_race.pkl` unless it is clearly worse than the current model or the grid on the current season's races neither has trained on.
- **Experiments that did not ship** are recorded in `ROADMAP.md` with their numbers (Benter correction, compound-adjusted practice, the gym search, Kalman inputs to the race model, market blend).

## Configuration

Settings are defined in `config/settings.py` and read from environment variables or a `.env` file in the repository root (git-ignored). FastF1, OpenF1, Jolpica, Open-Meteo and Kalshi's public API need no keys. `ODDS_API_KEY` (The Odds API, `src/data_collection/odds.py`) and the Reddit variables are only used by unused or archived code.

## Notes

`ROADMAP.md` holds the longer plan and every experiment's result; `TODO_FP_MIGRATION.md` (not committed) is the working log. `scripts/pipeline_after_ratelimit.py` is a one-off leftover.

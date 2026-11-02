# F1 Race Prediction Pipeline — Project Description

> Written for comparison against similar projects. Describes what the system is, what it does, and how it works end to end. Current as of November 2026 (model family `two_stage_v5`). README.md has the commands; ROADMAP.md has every experiment and its numbers.

---

## What it is

An end-to-end pipeline that predicts the result of a Formula 1 race once qualifying is over: for every driver, the probability of winning, finishing 2nd or 3rd, making the podium, the top 6 and the top 10. It also simulates the rest of the season for championship odds. GitHub Actions runs it every race weekend and publishes the results on leongorecki.eu, next to a model-vs-betting-market comparison.

**Scope:** training data from 2014 (the hybrid era) on; predictions for the 2026 season, live since round 19.

---

## What it predicts

One ranking of the field per race, turned into probabilities by simulation:

- a **position probability matrix** P[driver, position]: each row and each column sums to 1, so the podium probabilities across the field add up to 3;
- from it `p_win`, `p_p2`, `p_p3`, `p_podium`, `p_top6`, `p_top10`, plus `p_finish` (chance of being classified);
- **championship odds** after each race: probability of the drivers' and constructors' titles, top-3 chances and expected points.

---

## Data sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| **FastF1** (with its Jolpica/Ergast fallback) | Race and qualifying results, calendar | Free. Lap data needs an F1 TV subscription and is not used |
| **Jolpica** | Pit stops, sprint results, live sprint classification, backfills | Free REST API |
| **OpenF1** | Live qualifying, the official starting grid (after penalties), practice laps with compounds (2023 on) | Free REST API |
| **Open-Meteo** | Race-day forecast at inference; measured / day-before forecast weather for history | Free, no key |
| **Kalshi** | Race winner and podium market prices (2025 on), snapshotted live from qualifying to lights out | Free public API |

All raw data are stored as parquet files and committed: the CI uses them as state.

---

## Features — 35 inputs

All history features use only races before the one being predicted. Missing values get fill values saved at training time, so training and inference impute identically. Team and circuit names go through one canonical mapping (`teams.py`, `circuits.py`), so rebrands and data sources don't split the history.

| Group | Inputs |
|---|---|
| Qualifying | grid position (starting grid after penalties), relative grid, gap to pole (relative), gap to teammate, grid penalty places |
| Car pace | team's qualifying pace vs the field |
| Driver form | finishes over the last 5 / 10 races and exponentially weighted, Elo, average finish in wet races, experience |
| Start | how drivers convert this grid slot at this circuit |
| Circuit | driver's average finish and podiums at the circuit, appearances, street circuit, safety-car rate |
| Reliability | driver and team DNF rates |
| Championship | position, points, title pressure, share of the season done |
| Constructor | standing, points, team form |
| Weather | temperature, rain, wind, humidity, wet race |
| Sprint | sprint result on sprint weekends (fetched live from Jolpica) |
| Practice | driver's and team's long-run pace gap (OpenF1 laps, 2023 on; left missing before) |

Computed but not used (tested, no gain): team pit-stop speed, Elo-style driver/team strength ratings, Kalman strength estimates, a circuit overtaking index.

---

## Model architecture

**Production: two-stage model** (`TwoStageRaceModel`).

1. **Ranker.** Five XGBoost rankers (NDCG objective, one query group per race), differing only in their random seed, averaged. Trained on classified finishers only, with recency weights from 0.65 (oldest season) to 1.35 (newest). Depth 3, 300 trees, learning rate 0.03, chosen on 2018-2022.
2. **Finish model.** A logistic regression giving each car's chance of being classified.
3. **Race simulation.** 10,000 simulated races: each car retires with its own probability and goes to the back; the others finish in a Plackett-Luce order (Gumbel-max sampling) of the ranker's scores divided by a temperature T. T is the only calibration step, fitted by maximum likelihood of the top-3 order on a season the ranker did not train on.

**Championship odds.** A Kalman filter tracks a pace value per driver and per car, with a full covariance (teammates share the car, which separates driver from car). It is updated by every qualifying session (gap to the session, scaled to the season's typical spread; mixed wet/dry sessions skipped) and every race finishing order (a Laplace step on the Plackett-Luce likelihood). Car uncertainty widens at each new season and much more at regulation resets. Each of 20,000 simulated seasons draws pace from the filter and lets it drift race by race; retirements come from a pre-qualifying finish model.

**Archived:** a finishing-position regressor; three per-position XGBoost classifiers plus a PyTorch MLP in a stacking ensemble with an isotonic calibrator; Reddit sentiment features.

---

## Training and model choice

- **Walk-forward backtest** chooses models: each season from 2018 is predicted by a model trained only on earlier seasons. Primary metric: per-race log loss of the ordered top 3 under the model's probabilities, against the starting grid used as a probability model, with paired bootstrap intervals. A 43-race holdout swings 10+ points with the training window and cannot tell models apart.
- **Two passes** in training: an evaluation pass on the last two seasons (sanity metrics), then a production candidate trained on every season except the latest (T fitted on the latest).
- **Promotion gate:** after each race the candidate replaces the production model unless it is clearly worse than the current model or the grid on the current season's races neither has trained on. Decisions are logged.
- **A "gym"** runs stepwise searches over feature groups and calibration layers with a lockbox season range, to test combinations of ideas that did not help on their own.

---

## Results

| Measure | Value |
|---|---|
| Walk-forward 2018-2026 (192 races), top-3 log loss | 4.50, grid 5.30 |
| Holdout 2025-2026 (43 races) | winner 62.8% (pole 67.4%), podium drivers 71.3% (grid top 3 69.0%), Brier P1 0.025 |
| P(win) calibration | slope 1.09 [0.95, 1.27] |
| vs Kalshi at lights out (2025-2026) | better winner log loss than the de-vigged market |
| Championship odds, 32 checkpoints 2018-2025 | champion log loss 0.31 (drivers), 0.33 (constructors); points-only baseline 0.38 / 0.38 |

Ideas that did not ship, each with its backtest numbers in ROADMAP.md: a Benter correction for places 2-3, compound-adjusted practice pace, strength ratings and Kalman estimates as inputs, safety-car and rating-blend calibration layers, a market blend, in-season retraining.

---

## Inference and publishing

- **After qualifying** (CI, 45 minutes after the session): qualifying from OpenF1, starting grid from OpenF1 (falls back to the qualifying order), sprint result from Jolpica, practice laps from OpenF1, forecast from Open-Meteo. Features are built through the same code as training; `check_feature_skew.py` diffs both paths for a past race.
- **Until lights out:** re-prediction when the official grid changes (penalties); Kalshi snapshots every tick.
- **After the race:** data ingest, result recorded against the prediction, gated retrain, championship odds, site export.
- **Site:** `site/f1-predictions.json` (next race, last race, season record, feature importance, title odds, model vs market per race, current Kalshi prices) feeds the project page: live podium cards, the season record, a betting-strategy replay against Kalshi with a gambling disclaimer, and a current-race odds calculator behind an 18+ self-declaration.
- **Logging:** every CI tick appends decisions, step outputs, row counts and data-source availability to a per-season event log.

---

## Known limitations

- The grid penalty is only known once OpenF1 publishes the starting grid; before that the qualifying order is used.
- P(podium) overstates the favourite (0.85 predicted, 0.78 observed): the Plackett-Luce order among finishers is too sharp for places 2-3. P(finish) is slightly under-confident.
- Practice features exist only from 2023, so most training seasons have them missing.
- Kalshi prices exist from 2025: too few races to fit a model-market blend.
- One season of live predictions is a small sample for any claim about real-world accuracy.

---

## Repository structure (key paths)

| Purpose | Path |
|---|---|
| Feature pipeline / list | `src/features/build_features.py` |
| Production model | `src/models/xgb_model.py`, `src/models/two_stage.py` |
| Championship simulation | `src/models/season_sim.py`, `src/models/kalman_strength.py`, `scripts/title_odds.py` |
| Training + gate | `scripts/quick_train.py`, `src/training/promotion.py` |
| Walk-forward backtest | `src/training/backtest.py`, `scripts/backtest.py` |
| Inference | `src/inference/predict.py`, `scripts/race_weekend_inference.py` |
| CI driver | `scripts/ci_tick.py`, `.github/workflows/race-weekend.yml` |
| Site export | `scripts/export_site_json.py` → `site/f1-predictions.json` |
| Experiments | `models/experiments/`, `scripts/*_backtest.py`, `scripts/gym.py` |

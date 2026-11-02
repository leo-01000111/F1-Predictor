# F1 Predictor roadmap

Where the model stands against how serious sports forecasters work, and what to build next.
Written 2026-10-27 from a research review (professional sports-prediction frameworks + F1 literature).
Day-to-day status stays in `TODO_FP_MIGRATION.md`; this file is the longer plan.

## Where we are

- (2026-10-27) One XGBoost regressor (MAE loss on finish position, 34 features) -> Plackett-Luce
  sampler with strength `exp(-pred_pos / T)`, `T = 1.0` fixed, never fitted.
- (2026-10-28) Now: two-stage model. P(finish) per car + XGBoost NDCG ranker on finishers
  (33 features), PL temperature fitted out of sample. Beats the grid on log loss over 191
  walk-forward races by 0.65; winner picks tie with pole.
- (2026-10-28, later) v5: same structure, 35 features (practice long-run pace), tuned ranker
  settings, 5 seeds averaged, official starting grid live. Top-3 log loss 4.501 vs 5.296 for the
  grid (-0.796 [-1.050, -0.570]).
- Holdout 2025-26 (42 races), before this roadmap: winner 45.2%, podium 63.5%.
- **The starting grid beats us on that holdout: pole sitter wins 69.0%, grid top 3 overlaps the podium 69.8%.**
- About 26 races of 2022-23 were missing or empty in the training data (fixed in Phase 0).

The gap to professional systems is mostly method, not data. They benchmark against the grid
and the betting market, score full probability distributions, model driver and car strength
explicitly, and simulate the race.

---

## Phase 0: data integrity and an honest benchmark

Nothing later means anything until the data is complete and the model is measured against the grid.

- [x] `scripts/backfill_results_jolpica.py`: refill empty/missing race + quali rounds from Jolpica
      (2022 R8-16, 2023 R2-22 race; 2022-23 quali). Verified against stored 2022 R7: identical.
- [ ] **Run it**, then weather for 2023, then `quick_train.py` (commands in TODO).
- [x] DNF labels: since late 2023 Jolpica writes lapped finishers as `"Lapped"`, which counted as
      DNF (364 rows, inflating `dnf_rate`). Shared `is_dnf()` in `f1_historical.py`, also used by
      `f1_live.py`; the backfill script relabels all rows.
- [x] Grid baselines in `holdout_metrics.json` (`baselines.winner_accuracy_pole`,
      `baselines.podium_accuracy_grid_top3`, `beats_baselines`) + winner log loss.
- [x] Team names were split (2026 R1-2 live-timing names vs R3-18 Jolpica names; OpenF1 at
      inference). `src/features/teams.py` `canonical_team()` gives one name per lineage
      (2026 branding), so rebrands share history. Applied in collectors, build_features,
      predict.py and the site export; stored parquets rewritten.
- [ ] If the retrained model still loses to the grid: test monotone constraint on grid position,
      check whether form/history features override the grid, try grid-only + quali gap model.
- [x] Known skews: sprint result now fetched live (Jolpica), official starting grid from OpenF1
      with a re-prediction when penalties change it, pit-lane starts (grid 0) moved behind the
      last slot. 2026-10-28: top-3 log loss 4.648 -> 4.595.
- [ ] `driver_wet_avg_finish` one race behind at inference.
- [ ] Update CLAUDE.md: it still describes per-target binary classifiers + isotonic calibrator.

**Done when:** data complete for 2014-2026, team names canonical, and the model beats both grid
baselines on the holdout (or we know why not).

## Phase 1: evaluation done right

A 42-race hit rate has a 95% interval of about +-15 points. It cannot tell two models apart.

- [x] Walk-forward backtest, retrain per season (`src/training/backtest.py`, `scripts/backtest.py`):
      test 2018-2026 (~190 races), each season trained only on earlier seasons. Per-round
      retraining is a later refinement.
- [x] Primary metric: per-race PL log loss of the ordered top 3 (`ll_top3`) plus winner log loss;
      hit rates secondary. Grid scored as a probability model too (PL on grid, T fitted on past races).
- [x] Paired bootstrap on per-race differences vs the grid.
- [x] PL temperature fitted walk-forward (previous season's out-of-sample predictions). Phase 2.
- [x] Run it and pick the production variant on `ll_top3` vs grid (two_stage_v5).
- [x] Reliability diagram (`scripts/reliability.py`, 2026-11-02): P(win) calibrated (slope 1.09
      [0.95, 1.27]); P(podium) favourite 0.85 predicted vs 0.78 observed, 4th/5th favourites low.
- [x] CI promotion gate (`src/training/promotion.py`, 2026-11-02): quick_train writes a candidate,
      scored on unseen races of the current season (ll_top3, T refit leave-one-race-out) vs the
      current model and the grid; refused only when clearly worse.

## Phase 2: the probability layer

Our probabilities come from a hand-set transform, so they are uncalibrated by construction.

- [x] Fit PL temperature `T` by maximum likelihood of the ordered top 3, on a season the model
      did not train on (backtest: previous season; quick_train: calibration season).
- [x] Benter correction (2026-11-02, not shipped): per-place exponents in `two_stage` (`lambdas`,
      default off), `scripts/benter_backtest.py`. ll_top3 vs v5 +0.003 [-0.026, +0.028]; lambdas
      noisy per season (pooled: l2 1.0, l3 0.85-0.95). The favourite's podium overstatement stays.
- [ ] Favourite podium overstatement (0.85 vs 0.78): test whether P(finish) is too high for
      favourites (finish-model reliability by grid/favourite rank).
- [x] Learn strengths directly: `XGBRaceRanker` (NDCG, one group per race) is production since
      2026-10-28. Walk-forward 191 races, top-3 log loss: ranker 4.861, regressor 5.160, grid 5.296;
      ranker vs grid -0.435 [-0.718, -0.163]. Winner hit 52.4% vs pole 53.9% (tie).
      Pairwise ranking objective was no better than the regressor (5.267).
- [x] Removed isotonic-calibrator leftovers (2026-11-02): registry key, config block,
      season_accuracy recalibration text. PL temperature is the only calibration step.

## Phase 3: structural model (closed 2026-10-28)

What the F1 literature agrees on: the car explains ~85-88% of results, constructor strength drifts
quickly, driver skill is stable, and about a quarter of starts end in a DNF.

Every item was judged on the walk-forward backtest (2018-2026, 191 races, top-3 log loss, 95%
paired bootstrap) and ships only if it helps. One shipped; the rest are built, tested and kept.

- [x] **Two-stage model, SHIPPED** (`TwoStageRaceModel`, production): logistic P(finish) on
      reliability, circuit, grid and weather features; NDCG ranker on finishers only; combined by
      simulation (exact likelihood for scoring). Top-3 log loss 4.648 vs 4.861 for the v3 ranker
      (-0.225 [-0.351, -0.114]) and 5.296 for the grid; 4.595 after the pit-lane grid fix. The finishers-only ranker alone gave
      nothing (4.888): the gain is from P(finish). Holdout 2025-26: winner 64.3%, podium 70.6%.
- [x] Driver + team strength ratings (`src/features/strength_ratings.py`): PL-gradient updates,
      team reset at regulation changes (2014, 2017, 2022, 2026), driver skill carried over, rookies
      start below average. Real signal (-1.525 one-race-ahead log-lik per finisher vs -1.950).
      Not shipped: as ranker inputs no gain (-0.361 vs -0.435 vs grid); as a blend after the model
      (`race_calibration.py`, weight refitted per season) top-3 -0.036 [-0.091, +0.017], winner
      log loss -0.064 [-0.090, -0.036] but the weight bounces 0-1 and fits to 0 for 2025-26;
      with a fixed weight 0.5: top-3 -0.021 [-0.075, +0.036], winner hit drops to 51.8%.
- [x] Safety-car shock (`race_calibration.py`: mixture with a flatter order, probability scaled by
      the circuit's SC rate). Not shipped: -0.031 [-0.092, +0.014]; the fit picks no shock for 2026.
- [x] Circuit overtaking index (`add_overtaking_index`, leak-free per-circuit grid-vs-finish
      Spearman, computed in training and inference) used to scale each race's randomness.
      Not shipped: +0.003 [-0.026, +0.035].
- [x] Exponential recency half-life (`compute_recency_sample_weight(half_life=)`), 2/4/8 years.
      Not shipped: best -0.027 [-0.064, +0.011] (8 years). Linear 0.65-1.35 stays.
- [x] In-season retraining (`walk_forward_by_round`, `scripts/backtest.py round_two_stage[_bK]`):
      retrain before every race incl. the current season, T from the previous 22 races. Not
      shipped: 1x -0.010 [-0.092, +0.072], 2x current-season weight -0.030 [-0.112, +0.058]
      (peak), 4x +0.006. Helps in runaway / new-rules seasons (2023, 2026), costs in close ones.
      Production keeps training on completed seasons; current-season races enter via features
      (form, standings, reliability, this weekend's quali) and the temperature.
- [x] Ensembling: covered by the rating blend (the constrained 2-weight log-pool). All three
      calibration pieces together overfit: +0.015 [-0.091, +0.140].
- [x] Kalman phase A (2026-11-02): `src/models/kalman_strength.py` + `scripts/kalman_backtest.py`.
      Qualifying gaps only (sessions with time order vs position Spearman < 0.9 skipped: 21/271;
      gaps rescaled per session to the season's spread; Huber; shared team-session error).
      Tuned on 2014-2020 for the next qualifying order. Eval 2021-2026, PL log-lik per event vs
      current ratings: quali order +0.52 [+0.04, +1.04]; race order -0.42 [-1.00, +0.17];
      Kalman+ratings vs ratings: quali +0.85 [+0.50, +1.23], race +0.35 [+0.05, +0.66].
      Caveats: car/driver split unreliable (2021 Racing Bulls car > Red Bull; rookie prior -0.99),
      new teams anchored at 0 not the field level (Cadillac), tuned values partly degenerate
      (initial_driver_sd ~0, q1/q2 bias ~0). Tuning on gap likelihood ranked worse (0.61 vs 0.69).
- [x] Kalman phase B (2026-11-02): new teams / rookies anchored to the active field mean; race
      finishing order as a Laplace (PL gradient + Hessian) update; tuned jointly on next-quali
      order + race order 2014-2020; sd floor 0.01. Eval 2021-2026 vs ratings: quali order
      +0.79 [+0.37, +1.24]; race order (pre-weekend) +0.05 [-0.41, +0.55] (tie, target missed);
      Kalman+ratings race +0.37 [+0.07, +0.70]; post-quali Kalman vs grid on race order
      +0.69 [+0.02, +1.37]. Sanity: 2021 Mercedes > Red Bull > AlphaTauri, VER/HAM top drivers;
      2026 reset car sd 1.40 -> 0.55 by R3. Still: new-team prior sd tuned to 0.06 (only Haas
      2016 in the window), q1/q2 bias at the floor, practice not used (2023+ only).
- [x] Kalman phase D (2026-11-02): gym toggles add_kalman (post-quali pace) / add_kalman_all
      (+ pre-weekend sd, car, driver), `scripts/gym.py --with-kalman`, scored on 2021-2026 only
      (filter tuned on 2014-2020). vs v5 ll_top3: -0.008 [-0.070, +0.051] / +0.023 [-0.045,
      +0.092]. No gain as race-model inputs (same as the Elo ratings): not shipped.
- [ ] Kalman phase C: season simulation (posterior draws + random walk instead of the fixed
      team shock); benchmark championship_backtest.py.
- [ ] Not attempted before 2026-11-02: a full Kalman / state-space strength model (Lindner et al. 2026) with
      uncertainty per rating. The cheap version above did not help as an input, so this is low
      priority until Phase 4 gives a stronger benchmark.

Tools: `scripts/backtest.py` (model variants), `scripts/calibration_backtest.py` (calibration
variants on cached two-stage predictions).

### Phase 3b: more information and less variance (2026-10-28)

- [x] **Ranker tuning + seed ensemble, SHIPPED** (`scripts/tune_ranker.py`). Grid over depth x
      trees x learning rate chosen on 2018-2022 only: depth 3, 300 trees, lr 0.03 (was 4/300/0.05).
      On 2023-2026 (88 races): -0.077 [-0.157, -0.003] with 1 seed, -0.095 [-0.175, -0.021] with
      5 seeds averaged. Seeds alone on the old settings: +0.018 (nothing).
- [x] **Practice long-run pace, SHIPPED** (`src/data_collection/practice_openf1.py`,
      `src/features/practice_pace.py`, `scripts/practice_backtest.py`). OpenF1 laps 2023-2026
      (99k laps). Per session: best long run (>= 5 clean laps) vs the session median, %, driver and
      team (faster car). Missing before 2023, never imputed. Walk-forward 2024-2026 (66 races):
      both features -0.029 [-0.056, -0.004], better in each season; either alone not significant
      (best of 3 variants on the same races: weaker evidence than the tuning result). Per-race
      Spearman with the finish 0.30 (grid 0.66): most of it is already in qualifying.
- [x] Full walk-forward 2018-2026 (v5): top-3 log loss 4.501 (4.595 before 3b, 4.648 before the
      live-grid fixes), winner log loss 1.191, winner hits 52.9% (pole 53.9%).
- [x] 2026 site record replayed with v5 (`scripts/replay_season.py`): the old record was made in
      one batch on 2026-10-27 by the archived v2 regressor. 13/18 winners, 2.0 podium drivers per
      race. Old files in `data/predictions/archive/xgb_v2/`.
- [x] Compound-adjusted long runs + FP-session weighting (2026-11-02, not shipped): options in
      `practice_pace.py` (default off), `scripts/practice_compound_backtest.py`. Primary (cleaning +
      tyre age + compound + FP2-weighted) vs v5 on 2024-2026: +0.025 [-0.001, +0.051] (worse);
      degradation feature +0.029 [+0.002, +0.053]. Better Spearman, fewer covered rows.
- [ ] Re-test the practice features when 2027 adds a season.

### Gym: combined toggle search (2026-11-02)
- [x] `src/training/gym.py` + `scripts/gym.py`: registry of 17 model-level flips (add pit/ratings/
      overtaking, drop each feature group / reputation, monotone, half-life, no finish model) + 35
      single-feature drops (ablation map only) + post-hoc layers (pooled T, rating blend, overtaking
      T, safety car, Benter). Prediction cache; stepwise on dev, lockbox scored once.
- [x] Result: both splits picked add_pit + drop_car + rating_blend 0.5 + safety_car on dev
      (-0.17 summed); lockbox rejected it: 2024-26 +0.189 [+0.051, +0.334], 2025-26 +0.271
      [+0.102, +0.449]. Model-level part alone +0.016 / +0.057 (ns). v5 stays.
- Ablation map (dev 2018-23): qualifying +0.75, finish model +0.26, reputation +0.24, driver
  +0.13, driver_elo alone +0.06, monotone +0.05 (all hurt when removed/added); practice +0.015
  when dropped (dev 2018-24). Nothing removable for a clear gain.
- [ ] If re-run: accept a flip only when its CI upper bound < 0; register compound practice
      columns (`add_columns_toggle(..., first_test_season=2024)`) once 2027 adds a season.

## Phase 4: the betting market

The market is the real benchmark. Benter's model alone barely beat the public odds; the blend won.

Source: Kalshi (free public API, US-regulated exchange; Polymarket is blocked in France, Betfair
needs an account). F1 race markets exist from 2025: winner (`KXF1RACE`) and podium
(`KXF1RACEPODIUM`), settled ones under `/historical`. Prices about 10% over 1 (the margin).

- [x] `src/data_collection/kalshi_odds.py` -> `data/raw/kalshi_odds.parquet`: bid/ask/last and a
      price at our prediction time (quali + 2h45) and at race start, 2025 R1 - 2026 R18
      (42 winner, 39 podium markets). Empty books are "no price", never their 0.5 midpoint.
      Collected automatically in the post-race CI step (ci_tick ingest, cannot fail the tick).
- [x] De-vig (normalise) and compare (`scripts/market_backtest.py`, out-of-sample model):
      at prediction time, winner log loss model 1.108 / market 1.160 / grid 1.369; podium
      0.1941 / 0.2023 / 0.2148. Model vs market -0.052 [-0.209, +0.104] and -0.008: the model
      matches a real-money market at the time it predicts. At race start the market has caught
      up (podium 0.1902 vs model 0.1941).
- [x] Benter-style blend (2 weights, fitted on one season, scored on the other): not shipped.
      vs model at prediction time: winner -0.058 [-0.228, +0.134], podium +0.0003; weights flip
      between seasons (0.3/1.3 vs 1.5/0.4). With race-start prices: winner -0.145 [-0.274, +0.013].
- [x] 2026-10-29: price fix. `price_from_quote` refused any book with a zero bid and fell back to
      the last trade, so long shots (bid 0 / ask 0.01) were priced at stale trades (~10%). The
      spread limit alone now rejects empty books; 1,004 of 1,748 stored quotes recomputed from
      the saved bid/ask/last.
- [x] Re-run with v5 and fixed prices (42 races): winner log loss model 0.944 / market 1.119 at
      prediction time, 1.102 at race start / grid 1.369. Model vs market at race start
      -0.158 [-0.283, -0.043]: v5 beats the market alone. Blend vs model -0.067 [-0.183, +0.082]:
      still nothing to ship. Podium: model 0.1883 vs market 0.1871 at the start (tie).
- [x] Live pre-start prices: ci_tick snapshots the open Kalshi win/podium books every tick from
      qualifying to lights out (`kalshi_odds.snapshot` -> `data/raw/kalshi_snapshots.parquet`,
      `market_snapshot` events in the CI log), plus a Sunday :48 cron so the last one lands
      ~12 min before an on-the-hour start. Exact pre-start prices for the re-test.
- [ ] Re-test the blend when ~2 more seasons of prices exist (CI keeps collecting). If the
      race-start blend holds up, add a final prediction update just before the start.
- [x] Track "model vs market" per race on the site: betting calculator with bookmaker margin
      (2026-11-02, `market` block in export_site_json.py, assets/f1-market.js on the website).

## Phase 5: product

- [x] Championship simulation (`src/models/season_sim.py`, `scripts/title_odds.py`, run by CI
      after each race, `title` block in the site JSON): pre-qualifying two-stage model (20 inputs,
      monotone constraints: unconstrained it learned reversed form effects among front-runners),
      future-race features from the normal pipeline, one form shock per team per simulated season
      (sd 0.8), sprints included, standings checked against Jolpica. Backtest 32 checkpoints
      2018-2025 (`scripts/championship_backtest.py`): champion log loss 0.396 drivers / 0.366
      constructors vs 0.375 / 0.380 for points-only "equal cars" (within noise); more probability on
      the eventual champion on average (0.76 vs 0.75, 0.84 vs 0.82); in line with Kalshi's 2025
      title market. Biggest miss: 2021 after R11 (Verstappen 10%).
- [ ] Methodology/backtest page: calibration plot, model vs grid vs market, walk-forward log loss.
- [ ] Own site (same pattern as Evans: `site/` + `build.py` in this repo, Pages workflow,
      served at `/f1/`) once two of these exist:
  - per-race pages with their own URLs (archive, full grid, predicted vs actual)
  - the methodology/backtest page
  - championship simulation
  - driver/team rating pages
  - more data than one JSON file holds (several seasons, models or markets)
  Until then the section on the f1predictor project page is enough.

---

## Sources

- van Kesteren & Bergkamp 2023, Bayesian rank-ordered logit, driver vs constructor: https://arxiv.org/abs/2203.08489
- Lindner et al. 2026, state-space driver/constructor abilities: https://arxiv.org/abs/2608.04629
- Henderson & Kirrane 2018, time-weighted Plackett-Luce for F1: https://projecteuclid.org/euclid.ba/1488250819
- Powell 2023, Generalizing Elo (PL strengths online): https://www-users.york.ac.uk/~bp787/Generalizing_Elo_arxiv.pdf
- Heilmeier et al. 2020, Monte Carlo race simulation: https://www.mdpi.com/2076-3417/10/12/4229
- Benter 1994, computer handicapping (fundamental model + public odds): https://www.gwern.net/doc/statistics/decision/1994-benter.pdf
- Walsh & Joshi, model selection by calibration vs accuracy: https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/
- Opta supercomputer (market + ratings, season simulation): https://theanalyst.com/2024/10/premier-league-predictions-2024-25-opta-supercomputer-october
- Jolpica API: https://github.com/jolpica/jolpica-f1

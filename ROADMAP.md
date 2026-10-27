# F1 Predictor roadmap

Where the model stands against how serious sports forecasters work, and what to build next.
Written 2026-10-27 from a research review (professional sports-prediction frameworks + F1 literature).
Day-to-day status stays in `TODO_FP_MIGRATION.md`; this file is the longer plan.

## Where we are

- One XGBoost regressor (MAE loss on finish position, 34 features) -> Plackett-Luce sampler with
  strength `exp(-pred_pos / T)`, `T = 1.0` fixed, never fitted.
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
- [ ] Known skews (from TODO): sprint result missing at live inference, grid penalties unknown,
      `driver_wet_avg_finish` one race behind.
- [ ] Update CLAUDE.md: it still describes per-target binary classifiers + isotonic calibrator.

**Done when:** data complete for 2014-2026, team names canonical, and the model beats both grid
baselines on the holdout (or we know why not).

## Phase 1: evaluation done right

A 42-race hit rate has a 95% interval of about +-15 points. It cannot tell two models apart.

- [ ] Walk-forward backtest: for each round from 2018 on, train on everything before it, predict
      it. Hundreds of out-of-sample races instead of 42. (Retrain per season is a cheap first cut.)
- [ ] Primary metric: per-race log loss of the full finishing order under the PL model
      (plus winner log loss and RPS). Keep hit rates as secondary, human-readable numbers.
- [ ] Paired bootstrap (or Diebold-Mariano) on per-race log loss when comparing two models.
- [ ] Reliability diagram for P(win) and P(podium) over the walk-forward set.
- [ ] CI promotion gate: `post_race_retrain` / `pipeline/actions.py` currently compare winner
      accuracy. Switch to walk-forward log loss, and refuse a model that loses to the grid.

## Phase 2: the probability layer

Our probabilities come from a hand-set transform, so they are uncalibrated by construction.

- [ ] Fit PL temperature `T` by maximum likelihood of actual finishing orders (walk-forward).
- [ ] Benter correction: Harville/PL overstates favourites for 2nd and 3rd. Fit separate exponents
      for positions 2 and 3.
- [ ] Learn strengths directly: XGBoost `rank:pairwise` / `rank:ndcg`, or a custom PL loss,
      instead of MAE regression on position (MAE with a DNF spike at n+1 predicts medians).
- [ ] Remove isotonic-calibrator leftovers once PL temperature replaces them.

## Phase 3: structural model

What the F1 literature agrees on: the car explains ~85-88% of results, constructor strength drifts
quickly, driver skill is stable, and about a quarter of starts end in a DNF.

- [ ] Latent strengths: driver + team-season ratings updated race by race (Powell's generalised Elo
      for PL is the cheap version; a Kalman / state-space model like Lindner et al. 2026 the full one).
      Quali gaps update team pace fastest. Start as features for XGBoost, then as a standalone model.
- [ ] Regulation resets (2014, 2017, 2022, 2026): inflate team-rating uncertainty, keep driver skill.
- [ ] Shrinkage: rookies and new teams start at a prior (team mean / reset distribution) with wide
      uncertainty instead of "missing form".
- [ ] Two-stage DNF: P(finish) from team-season reliability, circuit and lap-1 incident rate;
      pace ranking among finishers; combine by simulation.
- [ ] Result-level Monte Carlo: sample DNFs, a safety-car shock that reshuffles order, and race pace
      as noise around grid scaled by a circuit overtaking index (per-circuit Spearman grid vs finish).
      No lap data needed.
- [ ] Replace linear recency weights (0.65-1.35) with an exponential half-life tuned on log loss.
- [ ] Ensembling: XGB-only until a blend wins on walk-forward log loss. If blending, use a
      constrained log-pool with few weights shrunk toward equal (the old stack put 1.8x on the NN).

## Phase 4: the betting market

The market is the real benchmark. Benter's model alone barely beat the public odds; the blend won.

- [ ] Start logging pre-race odds now (Polymarket F1 markets are free; Betfair historical is paid):
      winner + podium prices after quali, to `data/raw/odds.parquet`. One season is needed before
      it is useful.
- [ ] De-vig, then compare log loss: model vs market vs grid.
- [ ] Benter-style blend: logistic combination of model and market log-probabilities (2 weights).
- [ ] Track "beat the closing price" per race on the site record.

## Phase 5: product

- [ ] Championship simulation: drivers' and constructors' title odds after each race
      (reuses the Phase 3 Monte Carlo).
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

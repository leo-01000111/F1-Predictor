# -*- coding: utf-8 -*-
"""
Why does the model lose to the starting grid?
=============================================
Trains several XGBRacePredictor variants in memory on the saved feature matrix
and scores each against the grid baselines. No model files are written.

Variants:
  all              current FEATURE_COLS (33 features)
  with_pit         + team_pit_delta_s (dropped 2026-10-27 after the first run)
  all_mono         + monotone constraint: worse grid / bigger quali gap -> worse finish
  grid_quali       grid + quali-gap features only (monotone)
  no_reputation    drops driver form/Elo/experience, circuit history, championship
                   standing

First run (34 features incl. pit, 88 races): all 44.3% winner vs pole 61.4%,
no_pit 53.4% with the best winner log loss (1.292), grid_quali 62.5%.

Two splits, like quick_train: train on years before the calibration year,
fit the Plackett-Luce temperature on the calibration year, test on the next two.
  A: train <=2023, cal 2024, test 2025-26
  B: train <=2021, cal 2022, test 2023-24

Reported per variant (both splits pooled): winner / podium hit rate, how often the
pick is the pole sitter, winner log loss at T=1 and at the fitted T, and a paired
bootstrap 95% interval for (variant - pole baseline) winner hit rate.

Usage:
    python scripts/grid_diagnostics.py
Output: printed table + models/experiments/grid_diagnostics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from src.features.build_features import FEATURE_COLS, load_feature_matrix
from src.models.plackett_luce import PlackettLuceSampler
from src.models.xgb_model import XGBRacePredictor
from src.training.evaluate import evaluate_race_predictions
from src.training.weights import compute_recency_sample_weight

OUT_PATH = ROOT / "models" / "experiments" / "grid_diagnostics.json"

# Higher value -> worse (higher) predicted finish position
MONOTONE = {"grid_position": 1, "grid_position_rel": 1, "quali_gap_relative": 1}

GRID_QUALI = ["grid_position", "grid_position_rel", "quali_gap_relative",
              "teammate_quali_delta", "car_pace_delta_pct"]
REPUTATION = ["driver_form_5", "driver_form_10", "driver_form_ewm", "driver_elo",
              "driver_experience_norm", "driver_wet_avg_finish", "circuit_avg_finish",
              "circuit_podiums", "circuit_appearances", "championship_position",
              "cum_points_before", "championship_pressure"]

VARIANTS: dict[str, tuple[list[str], bool]] = {
    "all": (FEATURE_COLS, False),
    "with_pit": (FEATURE_COLS + ["team_pit_delta_s"], False),
    "all_mono": (FEATURE_COLS, True),
    "grid_quali": (GRID_QUALI, True),
    "no_reputation": ([c for c in FEATURE_COLS if c not in REPUTATION], False),
}

SPLITS = {
    "A": {"train_max": 2023, "cal": 2024, "test": [2025, 2026]},
    "B": {"train_max": 2021, "cal": 2022, "test": [2023, 2024]},
}

T_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def _winner_log_loss(df: pd.DataFrame, pred: np.ndarray, temperature: float) -> float:
    """Mean -log P(actual winner). Under Plackett-Luce, P(first) = softmax(-pred / T)."""
    losses = []
    d = df.assign(_pred=pred)
    for _, g in d.groupby(["year", "round"]):
        if not (g["p1"] == 1).any():
            continue
        s = -g["_pred"].values / temperature
        p = np.exp(s - s.max())
        p /= p.sum()
        losses.append(-np.log(max(p[(g["p1"] == 1).values][0], 1e-6)))
    return float(np.mean(losses))


def _fit(train: pd.DataFrame, cols: list[str], mono: bool) -> XGBRacePredictor:
    params = dict(XGBRacePredictor.DEFAULT_PARAMS)
    if mono:
        params["monotone_constraints"] = {c: v for c, v in MONOTONE.items() if c in cols}
    model = XGBRacePredictor(params=params)
    model.fit(train[cols], train["target_position"].values,
              sample_weight=compute_recency_sample_weight(train["year"].values))
    return model


def _pole_pick(df: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    """Per race: is the model's winner pick (lowest predicted position) the pole sitter?"""
    d = df.assign(_pred=pred)
    out = {}
    for key, g in d.groupby(["year", "round"]):
        grid = pd.to_numeric(g["grid_position"], errors="coerce")
        grid = grid.where(grid > 0, np.inf)
        out[key] = g.loc[g["_pred"].idxmin(), "driver"] == g.loc[grid.idxmin(), "driver"]
    return pd.Series(out)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    fm = load_feature_matrix()
    sampler = PlackettLuceSampler(temperature=1.0, n_simulations=10_000, seed=42)
    results: dict[str, dict] = {}
    per_race: dict[str, list[pd.DataFrame]] = {}

    for split_name, s in SPLITS.items():
        train = fm[fm["year"] <= s["train_max"]].reset_index(drop=True)
        cal = fm[fm["year"] == s["cal"]].reset_index(drop=True)
        test = fm[fm["year"].isin(s["test"])].reset_index(drop=True)
        print(f"Split {split_name}: train {len(train)} rows, cal {s['cal']}, test {s['test']} "
              f"({test.groupby(['year', 'round']).ngroups} races)")

        for name, (cols, mono) in VARIANTS.items():
            model = _fit(train, cols, mono)
            pred_cal = model.predict(cal[cols])
            best_t = min(T_GRID, key=lambda t: _winner_log_loss(cal, pred_cal, t))

            pred = model.predict(test[cols])
            metrics, races = evaluate_race_predictions(test, pred, sampler)
            races = races.set_index(["year", "round"])
            races["pick_is_pole"] = _pole_pick(test, pred)
            races["split"] = split_name
            per_race.setdefault(name, []).append(races)

            r = results.setdefault(name, {"splits": {}})
            r["splits"][split_name] = {
                "winner_accuracy": metrics["winner_accuracy"],
                "podium_accuracy": metrics["podium_accuracy"],
                "pole_baseline": metrics["baselines"]["winner_accuracy_pole"],
                "grid_top3_baseline": metrics["baselines"]["podium_accuracy_grid_top3"],
                "log_loss_T1": _winner_log_loss(test, pred, 1.0),
                "fitted_T": best_t,
                "log_loss_fitted_T": _winner_log_loss(test, pred, best_t),
            }
            print(f"  {name:14s} winner {metrics['winner_accuracy']:.1%}  "
                  f"podium {metrics['podium_accuracy']:.1%}  T={best_t}")

    rng = np.random.default_rng(0)
    print()
    print(f"{'variant':14s} {'races':>5s} {'winner':>7s} {'pole':>6s} {'diff 95% CI':>18s} "
          f"{'podium':>7s} {'grid3':>6s} {'pick=pole':>9s} {'LL T=1':>7s} {'LL fitT':>7s}")
    for name in VARIANTS:
        races = pd.concat(per_race[name])
        diff = races["winner_correct"].astype(float) - races["grid_pole_correct"].astype(float)
        boots = [diff.sample(len(diff), replace=True, random_state=int(rng.integers(1e9))).mean()
                 for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sp = results[name]["splits"]
        n = len(races)
        pooled = {
            "n_races": n,
            "winner_accuracy": float(races["winner_correct"].mean()),
            "pole_baseline": float(races["grid_pole_correct"].astype(float).mean()),
            "winner_minus_pole": float(diff.mean()),
            "winner_minus_pole_ci95": [float(lo), float(hi)],
            "podium_accuracy": float(races["podium_overlap"].mean() / 3),
            "grid_top3_baseline": float(races["grid_podium_overlap"].astype(float).mean() / 3),
            "pick_is_pole": float(races["pick_is_pole"].mean()),
            "log_loss_T1": float(np.average([sp[k]["log_loss_T1"] for k in sp],
                                            weights=[(races["split"] == k).sum() for k in sp])),
            "log_loss_fitted_T": float(np.average([sp[k]["log_loss_fitted_T"] for k in sp],
                                                  weights=[(races["split"] == k).sum() for k in sp])),
        }
        results[name]["pooled"] = pooled
        print(f"{name:14s} {n:5d} {pooled['winner_accuracy']:7.1%} {pooled['pole_baseline']:6.1%} "
              f"{pooled['winner_minus_pole']:+6.1%} [{lo:+.0%},{hi:+.0%}] "
              f"{pooled['podium_accuracy']:7.1%} {pooled['grid_top3_baseline']:6.1%} "
              f"{pooled['pick_is_pole']:9.1%} {pooled['log_loss_T1']:7.3f} {pooled['log_loss_fitted_T']:7.3f}")

    # Reference: log loss of a grid-only probability model is not defined without
    # a temperature, so the pole hit rate is the baseline for hit rates only.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()

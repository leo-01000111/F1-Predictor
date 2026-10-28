# -*- coding: utf-8 -*-
"""
Walk-forward test of the race calibration layer (ROADMAP Phase 3)
=================================================================
Trains the two-stage model once per test season (on earlier seasons only), then
scores calibration variants on the same predictions. Every calibration parameter
is fitted on the previous test season's out-of-sample predictions.

Variants (src/models/race_calibration.py):
  base         temperature only (= production two-stage)
  overtaking   + per-race temperature from circuit_overtaking
  safety_car   + safety-car shock (mixture with a flatter order)
  rating       + blend of the driver/team strength rating into the score
  all          all of the above

Usage:
    python scripts/calibration_backtest.py
Writes models/experiments/calibration_backtest.json and _races.parquet.
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
from src.models.race_calibration import RaceCalibration, race_contexts
from src.training.backtest import _race_orders, make_two_stage_predict, paired_bootstrap

OUT_DIR = ROOT / "models" / "experiments"

VARIANTS = {
    "base": [],
    "overtaking": ["overtaking_slope"],
    "safety_car": ["sc_scale", "sc_flatten"],
    "rating": ["rating_weight"],
    "all": ["overtaking_slope", "sc_scale", "sc_flatten", "rating_weight"],
    # rating weight fixed at 0.5 (median of the per-season fits in run 1), only T refitted
    "rating_fixed": [],
}
START = {"rating_fixed": {"rating_weight": 0.5}}


def _scored_races(df: pd.DataFrame, raw: np.ndarray, q: np.ndarray):
    """[(key, context, actual top-3 order)] for races with a full podium."""
    tops = {key: top for key, _, top in _race_orders(df)}
    return [(key, ctx, tops[key]) for key, _, ctx in race_contexts(df, raw, q) if key in tops]


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] == "__main__")

    fm = load_feature_matrix().reset_index(drop=True)
    years = list(range(2018, int(fm["year"].max()) + 1))
    predict = make_two_stage_predict(FEATURE_COLS)

    seasons = {}
    for year in years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        pred, q = predict(train, test)
        seasons[year] = _scored_races(test, -pred, q)   # pred at T=1, so raw = -pred
        logger.info(f"{year}: trained, {len(seasons[year])} races")

    rows, fitted = [], {}
    for name, params in VARIANTS.items():
        prev = None
        for year in years:
            races = seasons[year]
            cal = RaceCalibration(**START.get(name, {}))
            if prev is not None:
                cal = cal.fit([(ctx, top) for _, ctx, top in prev], params)
            fitted.setdefault(name, {})[year] = {k: v for k, v in cal.__dict__.items() if k != "fitted"}
            for key, ctx, top in races:
                m = cal.simulate(ctx, n_sims=5000, seed=0)
                rows.append({
                    "variant": name, "year": int(key[0]), "round": int(key[1]),
                    "ll_top3": cal.nll(ctx, top),
                    "ll_winner": cal.nll(ctx, top[:1]),
                    "winner_hit": bool(np.argmax(m[:, 0]) == top[0]),
                })
            prev = races
        r = pd.DataFrame([x for x in rows if x["variant"] == name])
        logger.info(f"{name:11s} ll_top3 {r['ll_top3'].mean():.3f}  ll_winner {r['ll_winner'].mean():.3f}  "
                    f"winner {r['winner_hit'].mean():.1%}")

    res = pd.DataFrame(rows)
    wide = res.pivot_table(index=["year", "round"], columns="variant", values=["ll_top3", "ll_winner"])
    print(f"\n{len(wide)} races. Differences vs base (two-stage, temperature only), 95% bootstrap.\n")
    summary = {}
    for name in VARIANTS:
        r = res[res["variant"] == name]
        s = {"ll_top3": float(r["ll_top3"].mean()), "ll_winner": float(r["ll_winner"].mean()),
             "winner_accuracy": float(r["winner_hit"].mean()), "params_by_season": fitted[name]}
        line = f"{name:12s} ll_top3 {s['ll_top3']:.3f}  ll_winner {s['ll_winner']:.3f}  winner {s['winner_accuracy']:.1%}"
        if name != "base":
            for metric in ("ll_top3", "ll_winner"):
                d = paired_bootstrap(wide[(metric, name)], wide[(metric, "base")])
                s[f"{metric}_minus_base"] = list(d)
                line += f"   d {metric} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]"
        summary[name] = s
        print(line)
    print("\nFitted parameters for the latest season:")
    for name in VARIANTS:
        print(f"  {name:11s} {fitted[name][years[-1]]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_DIR / "calibration_backtest_races.parquet", index=False)
    with open(OUT_DIR / "calibration_backtest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved {OUT_DIR / 'calibration_backtest.json'}")


if __name__ == "__main__":
    main()

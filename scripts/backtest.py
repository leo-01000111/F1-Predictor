# -*- coding: utf-8 -*-
"""
Walk-forward backtest of model variants against the grid (ROADMAP Phase 1)
==========================================================================
Each test season is predicted by a model trained only on earlier seasons.
Scored per race by Plackett-Luce log loss (winner and ordered top 3) and hit rates,
against a grid-only probability model. See src/training/backtest.py.

Variants:
  grid            starting grid as the prediction (baseline)
  all             current FEATURE_COLS
  with_pit        + team_pit_delta_s (dropped 2026-10-27)
  no_reputation   drops driver form/Elo/experience, circuit history, championship standing
  grid_quali      grid + quali-gap features only
  all_ratings     all + driver/team strength ratings
  ratings_swap    reputation features replaced by the ratings
  grid_ratings    grid + quali gap + ratings
  rank_<v>        XGBRanker (pairwise) on variant <v>'s columns, e.g. rank_all
  ndcg_<v>        XGBRanker (NDCG, top-weighted) on variant <v>'s columns
  finishers_ranker  Phase 3 stage 2 alone: production ranker trained on finishers only
  two_stage       Phase 3: finishers-only ranker + finish-probability model, combined
  two_stage_h<N>  two_stage with exponential recency weights, N-year half-life
  round_two_stage[_b<K>]  two_stage retrained before EVERY race (current season included),
                  T from the previous 22 races; _b<K>: current-season rows weighted K x

Usage:
    python scripts/backtest.py                       # all variants, test 2018-2026
    python scripts/backtest.py --variants grid all
    python scripts/backtest.py --start 2021

Uses the saved data/processed/feature_matrix.parquet (run quick_train or
build_features first). Writes models/experiments/backtest.json and
backtest_races.parquet only; no model files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from src.features.build_features import FEATURE_COLS, load_feature_matrix
from src.models.xgb_model import XGBRacePredictor
from src.training.backtest import (
    grid_predict,
    make_ranker_predict,
    make_two_stage_predict,
    make_xgb_predict,
    paired_bootstrap,
    walk_forward,
    walk_forward_by_round,
)

OUT_DIR = ROOT / "models" / "experiments"

GRID_QUALI = ["grid_position", "grid_position_rel", "quali_gap_relative",
              "teammate_quali_delta", "car_pace_delta_pct"]
REPUTATION = ["driver_form_5", "driver_form_10", "driver_form_ewm", "driver_elo",
              "driver_experience_norm", "driver_wet_avg_finish", "circuit_avg_finish",
              "circuit_podiums", "circuit_appearances", "championship_position",
              "cum_points_before", "championship_pressure"]

RATINGS = ["driver_rating", "team_rating", "car_rating"]

VARIANT_COLS: dict[str, list[str]] = {
    "all": FEATURE_COLS,
    "with_pit": FEATURE_COLS + ["team_pit_delta_s"],
    "no_reputation": [c for c in FEATURE_COLS if c not in REPUTATION],
    "grid_quali": GRID_QUALI,
    # Phase 3: driver + team strength ratings (src/features/strength_ratings.py)
    "all_ratings": FEATURE_COLS + RATINGS,
    "ratings_swap": [c for c in FEATURE_COLS if c not in REPUTATION] + RATINGS,
    "grid_ratings": GRID_QUALI + RATINGS,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest against the grid")
    parser.add_argument("--variants", nargs="+", default=["grid", *VARIANT_COLS])
    parser.add_argument("--start", type=int, default=2018, help="First test season")
    parser.add_argument("--end", type=int, default=None, help="Last test season (default: latest)")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="Override tree count (small values only for smoke tests)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] == "src.training.backtest")

    fm = load_feature_matrix()
    end = args.end or int(fm["year"].max())
    years = list(range(args.start, end + 1))
    params = dict(XGBRacePredictor.DEFAULT_PARAMS)
    if args.n_estimators:
        params["n_estimators"] = args.n_estimators

    variants = ["grid"] + [v for v in args.variants if v != "grid"]
    races = []
    for name in variants:
        if name == "grid":
            races.append(walk_forward(fm, grid_predict, years, name, fit_t_on_train=True))
        elif name == "finishers_ranker":
            races.append(walk_forward(fm, make_two_stage_predict(FEATURE_COLS, use_finish_model=False), years, name))
        elif name == "two_stage":
            races.append(walk_forward(fm, make_two_stage_predict(FEATURE_COLS), years, name))
        elif name.startswith("round_two_stage"):
            # round_two_stage[_b<K>]: retrain before every race incl. the current season,
            # current-season rows weighted K times extra
            boost = float(name.split("_b")[1]) if "_b" in name else 1.0
            races.append(walk_forward_by_round(
                fm, make_two_stage_predict(FEATURE_COLS, current_boost=boost), years, name,
                warmup_year=years[0] - 1))
        elif name.startswith("two_stage_h"):
            # two_stage_h<N>: exponential recency weights with an N-year half-life
            races.append(walk_forward(
                fm, make_two_stage_predict(FEATURE_COLS, half_life=float(name[11:])), years, name))
        elif name.startswith("rank_"):
            # rank_<variant> / ndcg_... : XGBRanker on that variant's columns
            races.append(walk_forward(fm, make_ranker_predict(VARIANT_COLS[name[5:]]), years, name))
        elif name.startswith("ndcg_"):
            races.append(walk_forward(fm, make_ranker_predict(VARIANT_COLS[name[5:]], "rank:ndcg"), years, name))
        else:
            races.append(walk_forward(fm, make_xgb_predict(VARIANT_COLS[name], params), years, name))
    races = pd.concat(races, ignore_index=True)

    wide = races.pivot_table(index=["year", "round"], columns="variant",
                             values=["ll_winner", "ll_top3", "winner_hit", "podium_overlap"])
    n = len(wide)
    summary: dict[str, dict] = {}
    print(f"\nWalk-forward {years[0]}-{years[-1]}: {n} races. Log loss: lower is better. "
          f"Differences are vs grid, with 95% bootstrap intervals.\n")
    print(f"{'variant':14s} {'winner':>7s} {'podium':>7s} {'ll_win':>7s} {'ll_top3':>8s} "
          f"{'d ll_top3 vs grid':>24s} {'d winner vs grid':>22s}")
    for name in variants:
        r = races[races["variant"] == name]
        s = {
            "winner_accuracy": float(r["winner_hit"].mean()),
            "podium_accuracy": float(r["podium_overlap"].mean() / 3),
            "ll_winner": float(r["ll_winner"].mean()),
            "ll_winner_t1": float(r["ll_winner_t1"].mean()),
            "ll_top3": float(r["ll_top3"].mean()),
            "temperatures": {int(y): float(t) for y, t in r.groupby("year")["temperature"].first().items()},
        }
        line = (f"{name:14s} {s['winner_accuracy']:7.1%} {s['podium_accuracy']:7.1%} "
                f"{s['ll_winner']:7.3f} {s['ll_top3']:8.3f}")
        if name != "grid":
            d, lo, hi = paired_bootstrap(wide[("ll_top3", name)], wide[("ll_top3", "grid")])
            w, wlo, whi = paired_bootstrap(wide[("winner_hit", name)].astype(float),
                                           wide[("winner_hit", "grid")].astype(float))
            s["ll_top3_minus_grid"] = [d, lo, hi]
            s["winner_minus_grid"] = [w, wlo, whi]
            line += f" {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {w:+6.1%} [{wlo:+.0%},{whi:+.0%}]"
        summary[name] = s
        print(line)

    print("\nll_top3 by season:")
    print(races.pivot_table(index="year", columns="variant", values="ll_top3").round(3)[variants].to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    races.to_parquet(OUT_DIR / "backtest_races.parquet", index=False)
    with open(OUT_DIR / "backtest.json", "w", encoding="utf-8") as f:
        json.dump({"test_years": years, "n_races": n, "variants": summary}, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'backtest.json'}")


if __name__ == "__main__":
    main()

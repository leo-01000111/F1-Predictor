"""
Tune the two-stage ranker's settings and seed-ensemble size (ROADMAP idea 4).

Selection and test use different seasons, so the reported gain is not the
best of many tries on the same races:
  1. tune     grid over max_depth x n_estimators x learning_rate, walk-forward
              on TUNE_YEARS (2018-2022), one seed; pick the lowest top-3 log loss
  2. confirm  on TEST_YEARS (2023-2026): current settings vs the pick, each with
              1 and N seeds; paired bootstrap vs the current production setup

Usage:
    python scripts/tune_ranker.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import FEATURE_COLS, load_feature_matrix  # noqa: E402
from src.models.xgb_model import XGBRaceRanker  # noqa: E402
from src.training.backtest import make_two_stage_predict, paired_bootstrap, walk_forward  # noqa: E402

TUNE_YEARS = list(range(2018, 2023))
GRID = {"max_depth": [3, 4, 6], "n_estimators": [150, 300, 600], "learning_rate": [0.03, 0.05, 0.1]}
OUT = ROOT / "models" / "experiments" / "tune_ranker.json"


def run(fm: pd.DataFrame, years: list[int], name: str, params: dict, n_seeds: int, cols=FEATURE_COLS) -> pd.DataFrame:
    return walk_forward(fm, make_two_stage_predict(cols, params=params, n_seeds=n_seeds), years, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the ranker on 2018-2022, confirm on 2023+")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    fm = load_feature_matrix()
    test_years = list(range(2023, int(fm["year"].max()) + 1))
    base = dict(XGBRaceRanker.DEFAULT_PARAMS)

    # 1. Tune
    tune = []
    for depth, trees, lr in itertools.product(*GRID.values()):
        p = dict(base, max_depth=depth, n_estimators=trees, learning_rate=lr)
        r = run(fm, TUNE_YEARS, f"d{depth}_n{trees}_lr{lr}", p, 1)
        tune.append({"max_depth": depth, "n_estimators": trees, "learning_rate": lr,
                     "ll_top3": float(r["ll_top3"].mean()), "ll_winner": float(r["ll_winner"].mean())})
        print(f"tune d={depth} n={trees:3d} lr={lr:<5} ll_top3 {tune[-1]['ll_top3']:.4f}", flush=True)
    tune_df = pd.DataFrame(tune).sort_values("ll_top3")
    best = tune_df.iloc[0]
    pick = dict(base, max_depth=int(best["max_depth"]), n_estimators=int(best["n_estimators"]),
                learning_rate=float(best["learning_rate"]))
    cur = tune_df[(tune_df.max_depth == base["max_depth"]) & (tune_df.n_estimators == base["n_estimators"])
                  & (tune_df.learning_rate == base["learning_rate"])].iloc[0]
    print(f"\nTune {TUNE_YEARS[0]}-{TUNE_YEARS[-1]}: current {cur['ll_top3']:.4f}, "
          f"pick d={pick['max_depth']} n={pick['n_estimators']} lr={pick['learning_rate']} {best['ll_top3']:.4f}")

    # 2. Confirm on seasons the pick was not chosen on
    variants = {
        "current": (base, 1),
        f"current_s{args.seeds}": (base, args.seeds),
        "pick": (pick, 1),
        f"pick_s{args.seeds}": (pick, args.seeds),
    }
    res = pd.concat([run(fm, test_years, n, p, s) for n, (p, s) in variants.items()], ignore_index=True)
    wide = res.pivot_table(index=["year", "round"], columns="variant", values="ll_top3")
    summary = {}
    print(f"\nConfirm {test_years[0]}-{test_years[-1]} ({len(wide)} races), top-3 log loss, diff vs current:")
    for n in variants:
        r = res[res["variant"] == n]
        s = {"ll_top3": float(r["ll_top3"].mean()), "ll_winner": float(r["ll_winner"].mean()),
             "winner_accuracy": float(r["winner_hit"].mean())}
        line = f"  {n:14s} {s['ll_top3']:.4f}  winner {s['winner_accuracy']:.1%}"
        if n != "current":
            d, lo, hi = paired_bootstrap(wide[n], wide["current"])
            s["minus_current"] = [d, lo, hi]
            line += f"  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        summary[n] = s
        print(line)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"tune_years": TUNE_YEARS, "test_years": test_years, "grid": tune_df.to_dict("records"),
                   "pick": {k: pick[k] for k in GRID}, "seeds": args.seeds, "confirm": summary}, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()

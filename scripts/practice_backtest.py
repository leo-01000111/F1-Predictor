"""
Does practice long-run pace help? (ROADMAP idea 2)

Adds the OpenF1 long-run features (src/features/practice_pace.py, 2023 on) to
the feature matrix, left missing before 2023, and runs the walk-forward
backtest of the production two-stage model with and without them on
2024-2026 (the first seasons whose training data contains the feature).

Usage:
    python scripts/practice_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_collection.practice_openf1 import load_practice_laps  # noqa: E402
from src.features.build_features import FEATURE_COLS, load_feature_matrix  # noqa: E402
from src.features.practice_pace import add_practice_pace  # noqa: E402
from src.training.backtest import make_two_stage_predict, paired_bootstrap, walk_forward  # noqa: E402

TEST_YEARS = [2024, 2025, 2026]
VARIANTS = {
    "base": FEATURE_COLS,
    "team_longrun": FEATURE_COLS + ["team_longrun_gap_pct"],
    "driver_longrun": FEATURE_COLS + ["fp_longrun_gap_pct"],
    "both_longrun": FEATURE_COLS + ["fp_longrun_gap_pct", "team_longrun_gap_pct"],
}
OUT = ROOT / "models" / "experiments" / "practice_backtest.json"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    fm = add_practice_pace(load_feature_matrix(), load_practice_laps())
    cov = fm[fm["year"] >= 2023].groupby("year")["team_longrun_gap_pct"].apply(lambda s: s.notna().mean())
    print("Rows with a team long run, by season:", cov.round(2).to_dict())
    fin = fm[(fm["year"] >= 2023) & fm["finish_position"].notna()]
    for c in ("fp_longrun_gap_pct", "team_longrun_gap_pct", "grid_position"):
        rho = fin.groupby(["year", "round"]).apply(lambda g: g[c].corr(g["finish_position"], method="spearman")).mean()
        print(f"  mean per-race Spearman({c}, finish) = {rho:.3f}")

    res = pd.concat([walk_forward(fm, make_two_stage_predict(cols), TEST_YEARS, name)
                     for name, cols in VARIANTS.items()], ignore_index=True)
    wide = res.pivot_table(index=["year", "round"], columns="variant", values="ll_top3")
    summary = {}
    print(f"\nWalk-forward {TEST_YEARS[0]}-{TEST_YEARS[-1]} ({len(wide)} races), top-3 log loss, diff vs base:")
    for name in VARIANTS:
        r = res[res["variant"] == name]
        s = {"ll_top3": float(r["ll_top3"].mean()), "ll_winner": float(r["ll_winner"].mean()),
             "winner_accuracy": float(r["winner_hit"].mean()),
             "by_year": r.groupby("year")["ll_top3"].mean().round(4).to_dict()}
        line = f"  {name:15s} {s['ll_top3']:.4f}  winner {s['winner_accuracy']:.1%}  by year {s['by_year']}"
        if name != "base":
            d, lo, hi = paired_bootstrap(wide[name], wide["base"])
            s["minus_base"] = [d, lo, hi]
            line += f"\n                  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        summary[name] = s
        print(line)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"test_years": TEST_YEARS, "coverage": {int(k): float(v) for k, v in cov.items()},
                   "variants": summary}, f, indent=2, default=str)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()

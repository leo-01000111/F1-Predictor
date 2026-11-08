"""
Do quali-reliability ("quali distrust") features help?

Walk-forward backtest of the production two-stage model with and without the
features from src/features/quali_features.py (rain-hit / shuffled qualifying),
scored by top-3 log loss with paired bootstrap intervals vs the base model.
Also reports the subset of races whose qualifying looked unreliable.

Usage:
    python scripts/quali_reliability_backtest.py [--start 2018]

Uses data/processed/feature_matrix.parquet (run build_features first).
Writes models/experiments/quali_reliability_backtest.json only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import FEATURE_COLS, load_feature_matrix  # noqa: E402
from src.training.backtest import make_two_stage_predict, paired_bootstrap, walk_forward  # noqa: E402

NEW = ["quali_spread_pct", "quali_segment_order", "quali_shuffle", "quali_vs_prior"]
BASE = [c for c in FEATURE_COLS if c not in NEW]
VARIANTS = {
    "base": BASE,
    "race_level": BASE + ["quali_spread_pct", "quali_segment_order", "quali_shuffle"],
    "vs_prior": BASE + ["quali_vs_prior"],
    "shuffle_prior": BASE + ["quali_shuffle", "quali_vs_prior"],
    "all_quali_rel": BASE + NEW,
}
OUT = ROOT / "models" / "experiments" / "quali_reliability_backtest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    fm = load_feature_matrix()
    years = list(range(args.start, int(fm["year"].max()) + 1))
    names = ["base"] + [v for v in args.variants if v != "base"]
    res = pd.concat([walk_forward(fm, make_two_stage_predict(VARIANTS[n]), years, n) for n in names],
                    ignore_index=True)

    race = fm.groupby(["year", "round"])[["quali_shuffle", "quali_spread_pct", "quali_segment_order"]].first()
    unreliable = race[(race["quali_shuffle"] >= race["quali_shuffle"].quantile(0.8))
                      | (race["quali_spread_pct"] >= 2.0) | (race["quali_segment_order"] < 0.9)].index

    summary = {}
    for metric in ("ll_top3", "ll_winner"):
        wide = res.pivot_table(index=["year", "round"], columns="variant", values=metric)
        sub = wide[wide.index.isin(unreliable)]
        print(f"\nWalk-forward {years[0]}-{years[-1]} ({len(wide)} races, {len(sub)} unreliable-quali), {metric}:")
        for n in names:
            s = summary.setdefault(n, {})
            s[metric] = float(wide[n].mean())
            s[f"{metric}_unreliable"] = float(sub[n].mean())
            line = f"  {n:14s} all {wide[n].mean():.4f}  unreliable {sub[n].mean():.4f}"
            if n != "base":
                d, lo, hi = paired_bootstrap(wide[n], wide["base"])
                du, ulo, uhi = paired_bootstrap(sub[n], sub["base"])
                s[f"{metric}_minus_base"] = [d, lo, hi]
                s[f"{metric}_minus_base_unreliable"] = [du, ulo, uhi]
                line += f"   d {d:+.4f} [{lo:+.4f},{hi:+.4f}]  d_unrel {du:+.4f} [{ulo:+.4f},{uhi:+.4f}]"
            print(line)
    for n in names:
        r = res[res["variant"] == n]
        summary[n]["winner_accuracy"] = float(r["winner_hit"].mean())
        summary[n]["by_year_ll_top3"] = {int(k): round(float(v), 4) for k, v in r.groupby("year")["ll_top3"].mean().items()}
    print("\nwinner hit:", {n: round(summary[n]["winner_accuracy"], 3) for n in names})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"test_years": years, "n_unreliable": len(unreliable), "variants": summary}, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

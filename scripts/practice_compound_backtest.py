"""
Compound-adjusted long runs and FP-session weighting (ROADMAP Phase 3b).

Rebuilds the two practice features (src/features/practice_pace.py) with the
refinements switched on and runs the walk-forward backtest of the production
two-stage model (DEFAULT_PARAMS, 5 seeds) for each variant against v5.

Pre-registered primary variant (2026-11-02, before any backtest was run):
  v6 = clean + adjust_age + adjust_compound + session weights FP1 0.5 / FP2 1 / FP3 0.5,
  replacing both v5 features. Ship rule: ll_top3 vs v5 better on 2024-2026 with a 95%
  paired-bootstrap CI excluding zero, and not worse on 2018-2026.
The other variants are diagnostics (which part helps); they cannot ship on their own.

Practice data starts in 2023, so every variant's model is identical to v5 for test
seasons up to 2023 (no practice rows in training, the feature never splits).
Variants are run on 2023-2026 (2023 only seeds T for 2024); v5 runs on 2018-2026
and its 2018-2022 races are shared. 2023 is checked to be identical.

Usage:
    python scripts/practice_compound_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_collection.practice_openf1 import load_practice_laps  # noqa: E402
from src.features.build_features import FEATURE_COLS, load_feature_matrix  # noqa: E402
from src.features.practice_pace import V6, add_practice_pace  # noqa: E402
from src.training.backtest import make_two_stage_predict, paired_bootstrap, walk_forward  # noqa: E402

PRIMARY = "v6"
SW = V6["session_weights"]
VARIANTS = {
    # name: (practice_pace options, extra feature columns)
    "v6": (V6, []),
    "v6_deg": (dict(V6, deg=True), ["fp_longrun_deg_pct"]),
    "clean": ({"clean": True}, []),
    "session_weights": ({"session_weights": SW}, []),
    "fp2_only": ({"session_weights": {"FP1": 0.0, "FP2": 1.0, "FP3": 0.0}}, []),
    "compound_age": ({"clean": True, "adjust_age": True, "adjust_compound": True}, []),
    "clean_sw": ({"clean": True, "session_weights": SW}, []),
}
ALL_YEARS = list(range(2018, 2027))
VAR_YEARS = [2023, 2024, 2025, 2026]
OUT = ROOT / "models" / "experiments" / "practice_compound_backtest.json"
OUT_RACES = ROOT / "models" / "experiments" / "practice_compound_backtest_races.parquet"


def _spearman(fm: pd.DataFrame, col: str) -> float:
    fin = fm[(fm["year"] >= 2023) & fm["finish_position"].notna()]
    return float(fin.groupby(["year", "round"])[[col, "finish_position"]]
                 .apply(lambda g: g[col].corr(g["finish_position"], method="spearman")).mean())


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    base_fm = load_feature_matrix()
    laps = load_practice_laps()
    v5_fm = add_practice_pace(base_fm, laps)
    cols = list(FEATURE_COLS)

    res = [walk_forward(v5_fm, make_two_stage_predict(cols), ALL_YEARS, "v5")]
    print(f"v5 done: {res[0].groupby('year')['ll_top3'].mean().round(3).to_dict()}", flush=True)
    diag = {"v5": {"rows_fp": int(v5_fm["fp_longrun_gap_pct"].notna().sum()),
                   "rows_team": int(v5_fm["team_longrun_gap_pct"].notna().sum()),
                   "spearman_fp": _spearman(v5_fm, "fp_longrun_gap_pct"),
                   "spearman_team": _spearman(v5_fm, "team_longrun_gap_pct")}}
    for name, (opts, extra) in VARIANTS.items():
        fm = add_practice_pace(base_fm, laps, opts)
        diag[name] = {"rows_fp": int(fm["fp_longrun_gap_pct"].notna().sum()),
                      "rows_team": int(fm["team_longrun_gap_pct"].notna().sum()),
                      "spearman_fp": _spearman(fm, "fp_longrun_gap_pct"),
                      "spearman_team": _spearman(fm, "team_longrun_gap_pct")}
        r = walk_forward(fm, make_two_stage_predict(cols + extra), VAR_YEARS, name)
        res.append(r)
        print(f"{name} done: {r.groupby('year')['ll_top3'].mean().round(3).to_dict()}", flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_parquet(OUT_RACES, index=False)
    summarise(res, diag)


def summarise(raw: pd.DataFrame, diag: dict) -> None:
    """Splice v5's races up to 2023 into every variant and write the comparison.

    A variant's 2023 run has T = 1 (no earlier season in its run), so only its
    T-free score ll_winner_t1 can be compared with v5's; when that matches, its
    2023 predictions are v5's and v5's 2023 rows (fitted T) are used.
    """
    v5 = raw[raw["variant"] == "v5"]
    same_2023, res = {}, [v5]
    for name in VARIANTS:
        r = raw[raw["variant"] == name]
        a = r[r["year"] == 2023].set_index("round")["ll_winner_t1"]
        b = v5[v5["year"] == 2023].set_index("round")["ll_winner_t1"].reindex(a.index)
        same_2023[name] = bool(np.allclose(a, b, atol=1e-6))
        res += [v5[v5["year"] <= 2023].assign(variant=name), r[r["year"] > 2023]]
    res = pd.concat(res, ignore_index=True)
    wide = res.pivot_table(index=["year", "round"], columns="variant", values="ll_top3")
    print("2023 predictions identical to v5 (ll_winner_t1):", same_2023)
    windows = {"2024-2026": [2024, 2025, 2026], "2018-2026": ALL_YEARS}
    summary = {}
    for name in ["v5"] + list(VARIANTS):
        r = res[res["variant"] == name]
        s = {"features": diag[name],
             "by_year": r.groupby("year")["ll_top3"].mean().round(4).to_dict()}
        for wname, yrs in windows.items():
            rw = r[r["year"].isin(yrs)]
            w = {"n_races": int(len(rw)), "ll_top3": float(rw["ll_top3"].mean()),
                 "ll_winner": float(rw["ll_winner"].mean()), "winner_accuracy": float(rw["winner_hit"].mean())}
            if name != "v5":
                ww = wide[wide.index.get_level_values("year").isin(yrs)]
                w["minus_v5"] = list(paired_bootstrap(ww[name], ww["v5"]))
            s[wname] = w
        summary[name] = s
        line = f"  {name:16s}"
        for wname in windows:
            w = s[wname]
            line += f"  {wname}: {w['ll_top3']:.4f}"
            if "minus_v5" in w:
                d, lo, hi = w["minus_v5"]
                line += f" ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])"
        print(line + f"  rho fp {diag[name]['spearman_fp']:.3f} team {diag[name]['spearman_team']:.3f}")

    p = summary[PRIMARY]
    d24, lo24, hi24 = p["2024-2026"]["minus_v5"]
    d_all = p["2018-2026"]["minus_v5"][0]
    ship = bool(hi24 < 0 and d_all <= 0)
    print(f"\nPrimary {PRIMARY}: ship = {ship}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"primary": PRIMARY, "primary_opts": V6, "ship_rule":
                   "2024-2026 ll_top3 vs v5 95% CI below 0 and 2018-2026 mean diff <= 0",
                   "ship": ship, "same_2023_as_v5": same_2023,
                   "variants": {n: {"opts": VARIANTS[n][0], "extra_cols": VARIANTS[n][1]} for n in VARIANTS},
                   "results": summary}, f, indent=2, default=str)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    if "--summarise" in sys.argv:   # re-score the saved races without rerunning the backtest
        saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        summarise(pd.read_parquet(OUT_RACES),
                  {n: s["features"] for n, s in saved.get("results", {}).items()})
    else:
        main()

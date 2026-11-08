# -*- coding: utf-8 -*-
"""
What is special about the races the model gets wrong?
=====================================================
Out-of-sample v5 predictions (gym walk-forward, each season predicted by a model
trained on earlier seasons) for 2018-2026, scored per race. Every race gets
a set of conditions, split into
  PRE  known before lights out -> could become a feature or a post-hoc switch
  POST only known afterwards   -> explains failures, cannot be used to predict
and each condition is compared between the worst quarter of races (top-3 log
loss) and the rest, with Spearman vs the per-race log loss.

Usage:  python scripts/failure_analysis.py
Writes models/experiments/failure_analysis.json and failure_races.parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import load_feature_matrix  # noqa: E402
from src.training import gym as G  # noqa: E402

YEARS = list(range(2018, 2027))
OUT = ROOT / "models" / "experiments"


def race_conditions(fm: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    df = fm.merge(preds, on=["year", "round", "driver"])
    rows = []
    for (y, r), g in df.groupby(["year", "round"]):
        g = g.sort_values("pred")                       # lower pred = stronger
        fav, second = g.iloc[0], g.iloc[1]
        grid = pd.to_numeric(g["grid_position"], errors="coerce")
        fin = pd.to_numeric(g["finish_position"], errors="coerce")
        elo = pd.to_numeric(g["driver_elo"], errors="coerce")
        top_elo = g.loc[elo.nlargest(5).index]
        team_pace = g.groupby("team")["car_pace_delta_pct"].first().sort_values()
        winner = g[fin == 1]
        rows.append({
            "year": y, "round": r, "circuit_key": g["circuit_key"].iloc[0],
            # ---------------- PRE ----------------
            "PRE_wet": int(g["is_wet_race"].iloc[0] == 1),
            "PRE_precip_mm": float(g["precip_mm_total"].iloc[0]),
            "PRE_street": int(g["is_street_circuit"].iloc[0] == 1),
            "PRE_sc_rate": float(g["circuit_safety_car_rate"].iloc[0]),
            "PRE_sprint_weekend": int((g["sprint_position_rel"] != 0.5).any()),
            "PRE_reg_reset_year": int(y in (2014, 2017, 2022, 2026)),
            "PRE_season_opener_3": int(r <= 3),
            "PRE_new_circuit": int(pd.to_numeric(g["circuit_appearances"], errors="coerce").max() <= 1),
            "PRE_quali_shuffle": float(g["quali_shuffle"].iloc[0]),
            "PRE_quali_spread": float(g["quali_spread_pct"].iloc[0]),
            "PRE_fav_not_on_pole": int(fav["grid_position"] != 1),
            "PRE_fav_grid": float(fav["grid_position"]),
            "PRE_fav_margin": float(second["pred"] - fav["pred"]),   # model's confidence
            "PRE_fav_p_finish": float(fav["q"]),
            "PRE_top5_elo_out_of_top6": int((pd.to_numeric(top_elo["grid_position"], errors="coerce") > 6).sum()),
            "PRE_front_row_teammates": int(g.loc[grid <= 2, "team"].nunique() == 1),
            "PRE_top2_team_gap_pct": float(team_pace.iloc[1] - team_pace.iloc[0]) if len(team_pace) > 1 else np.nan,
            "PRE_teams_within_0.3pct": int((team_pace - team_pace.iloc[0] <= 0.3).sum()),
            "PRE_temp_c": float(g["temp_c_mean"].iloc[0]),
            "PRE_wind_ms": float(g["windspeed_ms_mean"].iloc[0]),
            "PRE_title_pressure": float(pd.to_numeric(g["championship_pressure"], errors="coerce").min()),
            # ---------------- POST ----------------
            "POST_fav_dnf": int(fav["dnf"] == 1),
            "POST_n_dnf": int((g["dnf"] == 1).sum()),
            "POST_top5_elo_dnf": int((top_elo["dnf"] == 1).sum()),
            "POST_winner_grid": float(winner["grid_position"].iloc[0]) if len(winner) else np.nan,
            "POST_winner_model_rank": float(g.reset_index(drop=True).index[g["driver"].values == winner["driver"].iloc[0]][0] + 1)
            if len(winner) else np.nan,
            "POST_pole_dnf": int((g.loc[grid == 1, "dnf"] == 1).any()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    fm = load_feature_matrix()
    gym = G.Gym(fm)
    races = gym.evaluate({}, YEARS)
    sp = G.season_predictions(gym.fm, gym.spec({}), YEARS)
    preds = pd.concat([pd.DataFrame({"year": t["year"], "round": t["round"], "driver": t["driver"],
                                     "pred": p, "q": q}) for t, p, q in sp.values()], ignore_index=True)
    cond = race_conditions(fm, preds)
    df = races[["year", "round", "ll_top3", "ll_winner", "winner_hit"]].merge(cond, on=["year", "round"])
    df["fail"] = (df["ll_top3"] >= df["ll_top3"].quantile(0.75)).astype(int)
    df.to_parquet(OUT / "failure_races.parquet", index=False)

    cols = [c for c in df.columns if c.startswith(("PRE_", "POST_"))]
    res = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna()
        rho, p_rho = spearmanr(x[ok], df.loc[ok, "ll_top3"])
        a, b = x[ok & (df["fail"] == 1)], x[ok & (df["fail"] == 0)]
        try:
            p_mw = mannwhitneyu(a, b).pvalue
        except ValueError:
            p_mw = np.nan
        res.append({"condition": c, "worst_q_mean": a.mean(), "rest_mean": b.mean(),
                    "rho_ll_top3": rho, "p_rho": p_rho, "p_mw": p_mw})
    res = pd.DataFrame(res).sort_values("p_rho")
    pd.set_option("display.width", 200)
    print(f"{len(df)} races 2018-2026, worst quarter = ll_top3 >= {df['ll_top3'].quantile(0.75):.2f} "
          f"(mean ll_top3 {df['ll_top3'].mean():.3f}, winner hit {df['winner_hit'].mean():.1%})\n")
    print(res.round(3).to_string(index=False))

    # How much of the bad-race excess is post-race chaos?
    chaos = (df["POST_fav_dnf"] == 1) | (df["POST_top5_elo_dnf"] >= 2) | (df["POST_winner_grid"] > 5)
    print(f"\nRaces with chaos (fav DNF, 2+ top-Elo DNFs, or winner from P6+): {chaos.mean():.1%} of all, "
          f"{chaos[df['fail'] == 1].mean():.1%} of the worst quarter")
    print(f"ll_top3 chaos {df.loc[chaos, 'll_top3'].mean():.3f} vs clean {df.loc[~chaos, 'll_top3'].mean():.3f}")
    print("\nWorst 15 races:")
    print(df.sort_values("ll_top3", ascending=False).head(15)[
        ["year", "round", "circuit_key", "ll_top3", "PRE_wet", "PRE_fav_grid", "POST_fav_dnf",
         "POST_n_dnf", "POST_winner_grid", "POST_winner_model_rank"]].to_string(index=False))
    with open(OUT / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"n_races": len(df), "conditions": res.to_dict("records"),
                   "chaos_share_all": float(chaos.mean()), "chaos_share_worst_q": float(chaos[df["fail"] == 1].mean())},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()

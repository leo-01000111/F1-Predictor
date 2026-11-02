# -*- coding: utf-8 -*-
"""
Championship odds for the current season (ROADMAP Phase 5)
==========================================================
Simulates the remaining races of `--year` from the standings after the last
completed round (src/models/season_sim.py). Pace comes from the Kalman strength
filter (src/models/kalman_strength.py) run through every qualifying and race so
far: each simulated season draws car + driver pace from its posterior and steps
the random walk once per remaining race. Retirements come from the pre-qualifying
two-stage model's finish probabilities (seasons before the latest).

Backtest (scripts/championship_backtest.py, 32 checkpoints 2018-2025), log loss
of the actual champion: Kalman 0.307 drivers / 0.334 constructors; the previous
simulation (model score + team shock 0.8) 0.389 / 0.353; points-only "equal
cars" 0.375 / 0.380. Kalman - equal cars (drivers) -0.068 [-0.133, -0.008];
Kalman - previous within noise. Switched 2026-11-02.

Usage:
    python scripts/title_odds.py --year 2026
Writes data/predictions/YYYY_title.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

N_SIMS = 20_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    year = args.year

    import fastf1

    from src.data_collection.f1_historical import (
        load_pit_stops, load_quali_results, load_race_results, load_sprint_results,
    )
    from src.features.build_features import load_feature_matrix
    from src.models.kalman_strength import filter_through, load_params
    from src.models.season_sim import (
        PREQUALI_COLS, future_features, prequali_params, simulate_season, standings,
    )
    from src.models.xgb_model import TwoStageRaceModel
    from src.training.weights import compute_recency_sample_weight

    race, quali, pit, sprint = load_race_results(), load_quali_results(), load_pit_stops(), load_sprint_results()
    done = race[race["year"] == year]
    if done.empty:
        logger.info(f"No {year} races yet; nothing to simulate")
        return
    last = int(done["round"].max())

    sched = fastf1.get_event_schedule(year, include_testing=False)
    sched = sched[sched["RoundNumber"] >= 1]
    total = int(sched["RoundNumber"].max())
    sprint_rounds = set(sched.loc[sched["EventFormat"].str.contains("sprint"), "RoundNumber"].astype(int))
    rem = sched[sched["RoundNumber"] > last]
    remaining = pd.DataFrame({
        "round": rem["RoundNumber"].astype(int),
        "circuit": rem["Location"],
        "circuit_key": rem["EventName"].str.lower().str.replace(" ", "_").str.replace("grand_prix", "gp"),
        "race_date": rem["EventDate"].dt.date.astype(str),
    })

    dp, tp = standings(race, sprint, year, last)
    if remaining.empty:
        drivers = pd.DataFrame({"driver": dp.index, "points_now": dp.values, "expected_points": dp.values})
        drivers["p_champion"] = (drivers["points_now"] == drivers["points_now"].max()).astype(float)
        teams = pd.DataFrame({"team": tp.index, "points_now": tp.values, "expected_points": tp.values})
        teams["p_champion"] = (teams["points_now"] == teams["points_now"].max()).astype(float)
        team_of = done[done["round"] == last].set_index("driver")["team"]
        drivers["team"] = drivers["driver"].map(team_of)
        drivers["p_top3"] = drivers["points_now"].rank(ascending=False, method="first").le(3).astype(float)
    else:
        fm = load_feature_matrix()
        train = fm[fm["year"] < year]
        model = TwoStageRaceModel(params=prequali_params())
        model.fit(train[PREQUALI_COLS], train["target_position"].values,
                  sample_weight=compute_recency_sample_weight(train["year"].values),
                  qid=(train["year"] * 100 + train["round"]).to_numpy(), dnf=train["dnf"].to_numpy())
        fut = future_features(race, quali, pit, sprint, year, last, remaining)
        kf = filter_through(quali, race, year, last, load_params())
        res = simulate_season(model, fut, dp, tp, sprint_rounds, year, n_sims=N_SIMS, kalman={"filter": kf})
        drivers, teams = res.drivers, res.teams

    drivers = drivers.sort_values(["expected_points", "points_now"], ascending=False)
    teams = teams.sort_values(["expected_points", "points_now"], ascending=False)
    out = {
        "year": year,
        "after_round": last,
        "rounds_total": total,
        "races_left": total - last,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": f"kalman_strength + two_stage_prequali finish model, n_sims={N_SIMS}",
        "drivers": [
            {"driver": r.driver, "team": r.team, "points": round(float(r.points_now), 1),
             "expected_points": round(float(r.expected_points), 1),
             "p_champion": round(float(r.p_champion), 4), "p_top3": round(float(r.p_top3), 4)}
            for r in drivers.itertuples()
        ],
        "teams": [
            {"team": r.team, "points": round(float(r.points_now), 1),
             "expected_points": round(float(r.expected_points), 1),
             "p_champion": round(float(r.p_champion), 4)}
            for r in teams.itertuples()
        ],
    }
    path = ROOT / "data" / "predictions" / f"{year}_title.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    top = out["drivers"][0]
    print(f"Wrote {path.relative_to(ROOT)} (after R{last}, {out['races_left']} left; "
          f"{top['driver']} {top['p_champion']:.0%})")


if __name__ == "__main__":
    main()

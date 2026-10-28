# -*- coding: utf-8 -*-
"""
Backtest of the championship simulation (ROADMAP Phase 5)
=========================================================
For each finished season 2018-2025, simulate the title from four checkpoints
(after 25%, 50%, 75% of the races, and with two races left), using only what
was known then: the pre-qualifying model is trained on earlier seasons, its
temperature fitted on the previous season, features built from completed races.

Scored by log loss of the actual champion (drivers and constructors), for
several team-form spreads (team_sigma), against an "equal cars" baseline
(every car the same strength and finishing chance: points-only uncertainty).

Usage:
    python scripts/championship_backtest.py
Writes models/experiments/championship_backtest.json and _rows.parquet.
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

from src.data_collection.f1_historical import (
    load_pit_stops, load_quali_results, load_race_results, load_sprint_results,
)
from src.features.build_features import load_feature_matrix
from src.models.season_sim import PREQUALI_COLS, future_features, prequali_params, simulate_season, standings
from src.models.xgb_model import TwoStageRaceModel
from src.training.backtest import paired_bootstrap
from src.training.weights import compute_recency_sample_weight

OUT_DIR = ROOT / "models" / "experiments"
SIGMAS = [0.0, 0.15, 0.3, 0.5, 0.8]
FRACTIONS = [0.25, 0.5, 0.75]


class EqualCars:
    """Baseline: every car equally fast, 85% finish chance."""
    def predict(self, X):
        return np.zeros(len(X))

    def predict_finish(self, X):
        return np.full(len(X), 0.85)


def _fit(fm: pd.DataFrame, train_max: int) -> TwoStageRaceModel:
    tr = fm[fm["year"] <= train_max]
    m = TwoStageRaceModel(params=prequali_params())
    m.fit(tr[PREQUALI_COLS], tr["target_position"].values,
          sample_weight=compute_recency_sample_weight(tr["year"].values),
          qid=(tr["year"] * 100 + tr["round"]).to_numpy(), dnf=tr["dnf"].to_numpy())
    return m


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] == "__main__")
    race, quali, pit, sprint = load_race_results(), load_quali_results(), load_pit_stops(), load_sprint_results()
    fm = load_feature_matrix().reset_index(drop=True)

    rows = []
    for year in range(2018, 2026):
        # Temperature from the previous season, scored by a model that never saw it
        m_prev = _fit(fm, year - 2)
        prev = fm[fm["year"] == year - 1].reset_index(drop=True)
        m_prev.fit_temperature(prev[PREQUALI_COLS], prev)
        model = _fit(fm, year - 1)
        model.temperature = m_prev.temperature

        season = race[race["year"] == year]
        n_rounds = int(season["round"].max())
        final_d, final_t = standings(race, sprint, year, n_rounds)
        champ_d, champ_t = final_d.idxmax(), final_t.idxmax()
        sprint_rounds = set(sprint.loc[sprint["year"] == year, "round"].astype(int))
        checkpoints = sorted({max(1, round(n_rounds * f)) for f in FRACTIONS} | {n_rounds - 2})
        for k in checkpoints:
            remaining = (season[season["round"] > k]
                         .drop_duplicates("round")[["round", "circuit", "circuit_key", "race_date"]])
            fut = future_features(race, quali, pit, sprint, year, k, remaining)
            dp, tp = standings(race, sprint, year, k)
            for name, mdl, sig in [("equal_cars", EqualCars(), 0.0)] + [(f"model_s{s}", model, s) for s in SIGMAS]:
                res = simulate_season(mdl, fut, dp, tp, sprint_rounds, year, team_sigma=sig, n_sims=20_000)
                pd_ = res.drivers.set_index("driver")["p_champion"]
                pt_ = res.teams.set_index("team")["p_champion"]
                rows.append({
                    "year": year, "after_round": k, "races_left": n_rounds - k, "variant": name,
                    "champion": champ_d, "p_champion_driver": float(pd_.get(champ_d, 0.0)),
                    "constructor": champ_t, "p_champion_team": float(pt_.get(champ_t, 0.0)),
                    "leader_now": dp.idxmax(),
                })
            logger.info(f"{year} after R{k}: champion {champ_d} ({champ_t}), leader {dp.idxmax()}, "
                        f"T={model.temperature}")

    df = pd.DataFrame(rows)
    eps = 1e-3
    df["ll_driver"] = -np.log(df["p_champion_driver"].clip(lower=eps))
    df["ll_team"] = -np.log(df["p_champion_team"].clip(lower=eps))
    n = df.groupby("variant").size().iloc[0]
    print(f"\n{n} checkpoints (8 seasons). Log loss of the actual champion, lower is better.\n")
    print(f"{'variant':12s} {'ll_driver':>10s} {'ll_team':>9s} {'P(champ) drv':>13s} {'P(champ) team':>14s}")
    summary = {}
    for v, g in df.groupby("variant", sort=False):
        summary[v] = {c: float(g[c].mean()) for c in ("ll_driver", "ll_team", "p_champion_driver", "p_champion_team")}
        print(f"{v:12s} {g['ll_driver'].mean():10.3f} {g['ll_team'].mean():9.3f} "
              f"{g['p_champion_driver'].mean():13.3f} {g['p_champion_team'].mean():14.3f}")
    wide = df.pivot_table(index=["year", "after_round"], columns="variant", values=["ll_driver", "ll_team"])
    best = min((v for v in summary if v.startswith("model")), key=lambda v: summary[v]["ll_driver"] + summary[v]["ll_team"])
    for metric in ("ll_driver", "ll_team"):
        d = paired_bootstrap(wide[(metric, best)], wide[(metric, "equal_cars")])
        summary[f"{best}_minus_equal_{metric}"] = list(d)
        print(f"\n{best} - equal_cars {metric}: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
    print("\nBy races left (best variant), mean P(actual champion):")
    print(df[df["variant"] == best].groupby("races_left")[["p_champion_driver", "p_champion_team"]].mean().round(3).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "championship_backtest_rows.parquet", index=False)
    with open(OUT_DIR / "championship_backtest.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, **summary}, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'championship_backtest.json'}")


if __name__ == "__main__":
    main()

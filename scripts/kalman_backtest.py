# -*- coding: utf-8 -*-
"""
Kalman strength filter, phase A (ROADMAP Phase 3: state-space strengths)
========================================================================
Qualifying-gap Kalman filter (src/models/kalman_strength.py) vs the current
Elo-style strength ratings (src/features/strength_ratings.py) and a naive
baseline (each driver's gap in the previous usable session).

Each method gives every driver a pre-weekend score from earlier data only. Scored
by Plackett-Luce log-likelihood of
  - the qualifying order of the session (all classified qualifiers), and
  - the race finishing order (classified finishers),
with one scale per method and target fitted on FIT_YEARS and evaluated on
EVAL_YEARS. Kalman hyperparameters are tuned on 2014-2020 for the next qualifying
order (--tune, starts from the saved values) or read from
models/experiments/kalman_params.json.

Usage:
    python scripts/kalman_backtest.py [--tune]
Writes models/experiments/kalman_backtest.json and kalman_params.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize_scalar

from src.features.strength_ratings import add_strength_ratings
from src.models.kalman_strength import KalmanParams, pl_loglik, run, session_measurements, tune
from src.training.backtest import paired_bootstrap

FIT_YEARS = (2015, 2020)
EVAL_YEARS = (2021, 2026)
OUT = ROOT / "models" / "experiments"
PARAMS_PATH = OUT / "kalman_params.json"


def per_event(df: pd.DataFrame, score_col: str, pos_col: str, scale: float) -> pd.Series:
    out = {}
    for key, g in df.groupby(["year", "round"]):
        g = g.dropna(subset=[pos_col])
        if len(g) < 2:
            continue
        sc = g[score_col].fillna(g[score_col].median()).to_numpy(float) * scale
        order = np.argsort(g[pos_col].to_numpy(float), kind="stable")
        out[key] = pl_loglik(sc, order)
    return pd.Series(out)


def fit_scale(df: pd.DataFrame, score_col: str, pos_col: str) -> float:
    fit = df[df["year"].between(*FIT_YEARS)]
    res = minimize_scalar(lambda b: -per_event(fit, score_col, pos_col, b).sum(),
                          bounds=(0.01, 20.0), method="bounded")
    return float(res.x)


def fit_combo(df: pd.DataFrame, cols: list[str], pos_col: str) -> np.ndarray:
    """Weights of a linear score over `cols`, fitted on FIT_YEARS (PL log-lik)."""
    from scipy.optimize import minimize

    fit = df[df["year"].between(*FIT_YEARS)]
    res = minimize(lambda w: -per_event(fit.assign(_s=fit[cols].fillna(0.0) @ w), "_s", pos_col, 1.0).sum(),
                   np.ones(len(cols)), method="Nelder-Mead")
    return res.x


def naive_scores(meas: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Minus each driver's gap (%) in the previous usable session (NaN if absent)."""
    last: dict[str, float] = {}
    out = pd.Series(np.nan, index=rows.index)
    used = {k: g for k, g in meas.groupby(["year", "round"])}
    for key, g in rows.sort_values(["year", "round"]).groupby(["year", "round"], sort=True):
        out.loc[g.index] = [-last[d] if d in last else np.nan for d in g["driver"]]
        s = used.get(key)
        if s is not None:
            last.update(dict(zip(s["driver"], s["y"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="re-tune Kalman hyperparameters on 2014-2020")
    args = ap.parse_args()

    quali = pd.read_parquet(ROOT / "data" / "raw" / "quali_results.parquet")
    race = pd.read_parquet(ROOT / "data" / "raw" / "race_results.parquet")
    meas = session_measurements(quali)
    logger.info(f"{meas.groupby(['year', 'round']).ngroups} usable qualifying sessions, {len(meas)} rows")

    if args.tune or not PARAMS_PATH.exists():
        start = KalmanParams(**json.loads(PARAMS_PATH.read_text())) if PARAMS_PATH.exists() else KalmanParams()
        params, ll = tune(meas, start=start, restarts=2)
        PARAMS_PATH.write_text(json.dumps(asdict(params), indent=2))
        logger.info(f"Tuned on 2014-2020: log-lik {ll:.1f} -> {PARAMS_PATH.name}")
    params = KalmanParams(**json.loads(PARAMS_PATH.read_text()))

    # Pre-weekend Kalman state for every qualifier and every race entrant
    q = quali.reset_index(drop=True)
    r = race.reset_index(drop=True)
    entries = pd.concat([q[["year", "round", "driver", "team"]].assign(src="q"),
                         r[["year", "round", "driver", "team"]].assign(src="r")], ignore_index=True)
    state, _ = run(meas, params, all_sessions=entries)
    entries = entries.join(state)
    q["kalman"] = entries.loc[entries["src"] == "q", "pace"].to_numpy()
    r["kalman"] = entries.loc[entries["src"] == "r", "pace"].to_numpy()

    # Current ratings (pre-race, from earlier races); qualifying uses the same weekend's pre-race value
    rated = add_strength_ratings(race)
    r["ratings"] = rated["car_rating"].to_numpy()
    q = q.merge(r[["year", "round", "driver", "ratings"]].drop_duplicates(["year", "round", "driver"]),
                on=["year", "round", "driver"], how="left")
    q["naive"] = naive_scores(meas, q).to_numpy()
    r["naive"] = naive_scores(meas, r).to_numpy()
    r["finish_ok"] = np.where(pd.to_numeric(r["dnf"], errors="coerce").fillna(0) == 0,
                              pd.to_numeric(r["finish_position"], errors="coerce"), np.nan)

    results: dict = {"params": asdict(params), "fit_years": FIT_YEARS, "eval_years": EVAL_YEARS}
    for target, df, pos in (("quali_order", q, "quali_position"), ("race_order", r, "finish_ok")):
        ev = df[df["year"].between(*EVAL_YEARS)]
        per = {}
        block = {}
        df = df.assign(ratings=df["ratings"].fillna(df["ratings"].median()))
        for method in ("kalman", "ratings", "naive"):
            b = fit_scale(df, method, pos)
            per[method] = per_event(ev, method, pos, b)
            block[method] = {"scale": round(b, 4), "mean_ll": round(float(per[method].mean()), 4),
                             "n_events": int(len(per[method]))}
        # Does the filter add information to the ratings? (weights fitted on FIT_YEARS)
        w = fit_combo(df, ["kalman", "ratings"], pos)
        ev_c = ev.assign(combo=ev[["kalman", "ratings"]].fillna(0.0) @ w)
        per["combo"] = per_event(ev_c, "combo", pos, 1.0)
        block["combo"] = {"weights": [round(float(x), 4) for x in w],
                          "mean_ll": round(float(per["combo"].mean()), 4)}
        for a, b_ in (("kalman", "ratings"), ("kalman", "naive"), ("combo", "ratings")):
            d, lo, hi = paired_bootstrap(per[a], per[b_])
            block[f"{a}_minus_{b_}"] = [round(d, 4), round(lo, 4), round(hi, 4)]
        by_year = pd.DataFrame(per).assign(year=[k[0] for k in per["kalman"].index]).groupby("year").mean()
        block["by_year"] = by_year.round(3).to_dict(orient="index")
        results[target] = block
        logger.info(f"{target}: " + ", ".join(f"{m} {block[m]['mean_ll']:.3f}" for m in per if m != "combo")
                    + f", combo {block['combo']['mean_ll']:.3f} | kalman-ratings {block['kalman_minus_ratings']}"
                    + f", combo-ratings {block['combo_minus_ratings']} (higher = better)")

    (OUT / "kalman_backtest.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote models/experiments/kalman_backtest.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Model vs betting market vs grid, and a Benter-style blend (ROADMAP Phase 4)
==========================================================================
Uses Kalshi prices (src/data_collection/kalshi_odds.py) taken when our model
predicts (qualifying start + 2h45). Market probabilities are de-vigged by
normalising: win prices to sum 1 per race, podium prices to sum 3.

Model probabilities are out of sample: the two-stage model for season S is
trained on seasons before S, with its temperature fitted on season S-1's
out-of-sample predictions (as scripts/backtest.py).

Blend (Benter 1994): log p = a * log p_model + b * log p_market, renormalised
per race (win); logit(p) = a * logit p_model + b * logit p_market (podium,
rescaled to sum 3). a, b are fitted on one season and scored on the other
(two-fold cross-fit by season), so every blended number is out of sample.

Usage:
    python scripts/market_backtest.py [--price price_pred|price_start]
Writes models/experiments/market_backtest.json and _drivers.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from src.data_collection.kalshi_odds import load_kalshi_odds
from src.features.build_features import FEATURE_COLS, load_feature_matrix
from src.models.two_stage import simulate
from src.training.backtest import (
    _grid_score,
    fit_temperature,
    grid_predict,
    make_two_stage_predict,
    paired_bootstrap,
)

OUT_DIR = ROOT / "models" / "experiments"
FLOOR = 0.002          # probability for a driver the market did not list
BLEND_GRID = np.round(np.arange(0.0, 1.61, 0.1), 2)


def model_probs(fm: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Out-of-sample P(win), P(podium) per driver from the two-stage model."""
    predict = make_two_stage_predict(FEATURE_COLS)
    rows, prev = [], None
    for year in [years[0] - 1] + years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        pred, q = predict(train, test)
        if year in years:
            t = fit_temperature(*prev) if prev is not None else 1.0
            for _, idx in test.groupby(["year", "round"]).indices.items():
                m = simulate(pred[idx], q[idx], t, n_sims=40_000, seed=0)
                g = test.iloc[idx]
                rows.append(pd.DataFrame({
                    "year": g["year"].values, "round": g["round"].values, "driver": g["driver"].values,
                    "p_win_model": m[:, 0], "p_podium_model": m[:, :3].sum(axis=1),
                }))
            logger.info(f"model {year}: T={t}")
        prev = (pred, test, q)
    return pd.concat(rows, ignore_index=True)


def grid_probs(fm: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Grid as a probability model: PL on grid position, T fitted on earlier races."""
    rows = []
    for year in years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        t = fit_temperature(grid_predict(train, train), train)
        for _, idx in test.groupby(["year", "round"]).indices.items():
            g = test.iloc[idx]
            m = simulate(_grid_score(g["grid_position"]), np.ones(len(g)), t, n_sims=40_000, seed=0)
            rows.append(pd.DataFrame({
                "year": g["year"].values, "round": g["round"].values, "driver": g["driver"].values,
                "p_win_grid": m[:, 0], "p_podium_grid": m[:, :3].sum(axis=1),
            }))
    return pd.concat(rows, ignore_index=True)


def market_probs(odds: pd.DataFrame, price: str) -> pd.DataFrame:
    """De-vigged market probabilities. A race's market counts only if its listed
    prices sum to at least half the total (otherwise the book was too thin)."""
    out = []
    for market, total in (("win", 1.0), ("podium", 3.0)):
        o = odds[(odds["market"] == market) & odds["driver"].notna()].copy()
        o["p"] = pd.to_numeric(o[price], errors="coerce").clip(lower=0.001)
        o = o.dropna(subset=["p"])
        raw_sum = o.groupby(["year", "round"])["p"].transform("sum")
        o = o[raw_sum >= 0.5 * total]
        o["p"] = o["p"] * total / o.groupby(["year", "round"])["p"].transform("sum")
        out.append(o[["year", "round", "driver", "p"]].rename(columns={"p": f"p_{market}_market"}))
    return out[0].merge(out[1], on=["year", "round", "driver"], how="outer")


def _renorm(p: pd.Series, keys: list[pd.Series], total: float) -> pd.Series:
    return p * total / p.groupby(keys).transform("sum")


def blend_win(df: pd.DataFrame, a: float, b: float) -> pd.Series:
    lp = a * np.log(df["p_win_model"].clip(lower=1e-4)) + b * np.log(df["p_win_market"].clip(lower=1e-4))
    w = np.exp(lp - lp.groupby([df["year"], df["round"]]).transform("max"))
    return _renorm(w, [df["year"], df["round"]], 1.0)


def blend_podium(df: pd.DataFrame, a: float, b: float) -> pd.Series:
    def logit(p):
        p = p.clip(1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    z = a * logit(df["p_podium_model"]) + b * logit(df["p_podium_market"])
    p = 1 / (1 + np.exp(-z))
    return _renorm(p, [df["year"], df["round"]], 3.0).clip(1e-4, 1 - 1e-4)


def per_race(df: pd.DataFrame, col_win: str, col_pod: str) -> pd.DataFrame:
    """Per race: winner log loss, podium log loss (mean binary per driver), win Brier."""
    rows = []
    for (y, r), g in df.groupby(["year", "round"]):
        pw = g[col_win].clip(1e-6, 1)
        pp = g[col_pod].clip(1e-4, 1 - 1e-4)
        won, pod = g["won"].astype(bool), g["podium"].astype(bool)
        rows.append({
            "year": y, "round": r,
            "ll_winner": float(-np.log(pw[won]).sum()) if won.any() else np.nan,
            "ll_podium": float(-(np.log(pp[pod]).sum() + np.log(1 - pp[~pod]).sum()) / len(g)),
            "brier_win": float(((pw - won) ** 2).mean()),
            "winner_hit": bool(won[pw.idxmax()]),
        })
    return pd.DataFrame(rows)


def fit_blend(df: pd.DataFrame, kind: str) -> tuple[float, float]:
    best, best_loss = (1.0, 0.0), np.inf
    for a, b in product(BLEND_GRID, BLEND_GRID):
        if a == 0 and b == 0:
            continue
        if kind == "win":
            d = df.assign(p=blend_win(df, a, b))
            loss = per_race(d.assign(pp=d["p_podium_model"]), "p", "pp")["ll_winner"].mean()
        else:
            d = df.assign(p=blend_podium(df, a, b))
            loss = per_race(d.assign(pw=d["p_win_model"]), "pw", "p")["ll_podium"].mean()
        if loss < best_loss:
            best, best_loss = (float(a), float(b)), loss
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", default="price_pred", choices=["price_pred", "price_start"])
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] == "__main__")

    odds = load_kalshi_odds()
    fm = load_feature_matrix().reset_index(drop=True)
    years = sorted(int(y) for y in odds["year"].unique())
    mk = market_probs(odds, args.price)
    races_with_market = set(zip(mk.loc[mk["p_win_market"].notna(), "year"],
                                mk.loc[mk["p_win_market"].notna(), "round"]))
    races_with_podium = set(zip(mk.loc[mk["p_podium_market"].notna(), "year"],
                                mk.loc[mk["p_podium_market"].notna(), "round"]))

    base = fm[fm["year"].isin(years)][["year", "round", "driver", "p1", "podium"]].rename(columns={"p1": "won"})
    df = (base.merge(model_probs(fm, years), on=["year", "round", "driver"])
              .merge(grid_probs(fm, years), on=["year", "round", "driver"])
              .merge(mk, on=["year", "round", "driver"], how="left"))
    df = df[[k in races_with_market for k in zip(df["year"], df["round"])]].copy()
    df["p_win_market"] = df["p_win_market"].fillna(FLOOR)
    df["p_podium_market"] = df["p_podium_market"].fillna(FLOOR * 3)
    df["p_win_market"] = _renorm(df["p_win_market"], [df["year"], df["round"]], 1.0)
    has_podium_mkt = pd.Series([k in races_with_podium for k in zip(df["year"], df["round"])], index=df.index)

    # Out-of-sample blend: weights fitted on the other season
    df["p_win_blend"], df["p_podium_blend"], weights = np.nan, np.nan, {}
    for y in years:
        fit_on = df[df["year"] != y]
        aw, bw = fit_blend(fit_on, "win")
        ap, bp = fit_blend(fit_on[has_podium_mkt[fit_on.index]], "podium")
        weights[y] = {"win": (aw, bw), "podium": (ap, bp)}
        m = df["year"] == y
        df.loc[m, "p_win_blend"] = blend_win(df[m], aw, bw)
        df.loc[m, "p_podium_blend"] = blend_podium(df[m], ap, bp)
        logger.info(f"blend weights fitted without {y}: win a={aw} b={bw}, podium a={ap} b={bp}")

    scores = {name: per_race(df, f"p_win_{name}", f"p_podium_{name}") for name in ("grid", "model", "market", "blend")}
    for s in scores.values():
        # podium scores only where a usable podium market existed
        s.loc[[k not in races_with_podium for k in zip(s["year"], s["round"])], "ll_podium"] = np.nan
    n = len(scores["model"])
    n_pod = int(scores["model"]["ll_podium"].notna().sum())
    print(f"\n{n} races with a winner market, {n_pod} with a podium market ({args.price}). "
          f"Lower log loss / Brier is better.\n")
    print(f"{'':8s} {'ll_winner':>10s} {'ll_podium':>10s} {'brier_win':>10s} {'winner':>8s}")
    summary = {"n_races": n, "price": args.price, "blend_weights": {str(k): v for k, v in weights.items()}}
    for name, s in scores.items():
        summary[name] = {c: float(s[c].mean()) for c in ("ll_winner", "ll_podium", "brier_win", "winner_hit")}
        print(f"{name:8s} {s['ll_winner'].mean():10.3f} {s['ll_podium'].mean():10.4f} "
              f"{s['brier_win'].mean():10.4f} {s['winner_hit'].mean():8.1%}")
    print("\nPaired differences per race, 95% bootstrap (negative = first is better):")
    for a, b in (("model", "market"), ("blend", "market"), ("blend", "model"), ("market", "grid")):
        for metric in ("ll_winner", "ll_podium"):
            d = paired_bootstrap(scores[a][metric], scores[b][metric])
            summary[f"{a}_minus_{b}_{metric}"] = list(d)
            print(f"  {a:6s} - {b:6s} {metric:10s} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "market_backtest_drivers.parquet", index=False)
    with open(OUT_DIR / "market_backtest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'market_backtest.json'}")


if __name__ == "__main__":
    main()

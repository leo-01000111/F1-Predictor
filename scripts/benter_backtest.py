# -*- coding: utf-8 -*-
"""
Walk-forward test of the Benter correction (ROADMAP Phase 2)
=============================================================
Harville / Plackett-Luce tends to overstate favourites for 2nd and 3rd place.
Benter's fix: place k is drawn with log-strengths lambda_k * s (lambda_1 = 1,
absorbed in T). src.models.two_stage implements it in the likelihood
(top_k_nll, lambdas=) and the simulation (simulate, lambdas=).

The two-stage model (production settings) is trained once per test season on
earlier seasons only. Calibration is fitted on the previous test season's
out-of-sample predictions (2018 has none: T = 1, no correction, as in
scripts/backtest.py).

Variants:
  base          T only (= production v5, scripts/backtest.py two_stage)
  benter        T as base, then (lambda_2, lambda_3) by coordinate descent  [primary]
  benter_joint  (T, lambda_2, lambda_3) jointly by coordinate descent
  benter_pooled T as base, (lambda_2, lambda_3) fitted on ALL earlier test seasons'
                OOS predictions (each at its own walk-forward T): less noisy

Metrics per race: top-3 log loss (primary), winner log loss, winner hit,
P(podium) Brier (sum over drivers). Paired bootstrap vs base, overall and per season.
Also P(podium) calibration by favourite rank (rank of the variant's own P(win)).

Usage:
    python scripts/benter_backtest.py [--check]
--check only runs the brute-force test of the likelihood and the simulation.
Writes models/experiments/benter_backtest.json and _races.parquet.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from src.models.two_stage import simulate, simulate_logw, top_k_nll, top_k_nll_logw
from src.training.backtest import T_GRID, _race_orders, fit_temperature, paired_bootstrap

OUT_DIR = ROOT / "models" / "experiments"
LAMBDA_GRID = np.round(np.arange(0.30, 1.501, 0.05), 2)
FIT_SAMPLES = 500
FAV_RANKS = 6


def brute_force_check() -> None:
    """Exact enumeration on a 6-car race vs top_k_nll_logw and simulate_logw."""
    rng = np.random.default_rng(1)
    n, lam = 6, [1.0, 0.7, 0.5]
    s, q = rng.normal(0, 1.2, n), rng.uniform(0.6, 0.95, n)
    exact, marg = {}, np.zeros((n, 3))     # exact: podiums of finishers only
    for fin in itertools.product([0, 1], repeat=n):
        p_set = np.prod([q[i] if f else 1 - q[i] for i, f in enumerate(fin)])
        ret = [i for i in range(n) if not fin[i]]

        def rec(avail, ret, prefix, pr):
            if len(prefix) == 3:
                if all(fin[i] for i in prefix):
                    exact[tuple(prefix)] = exact.get(tuple(prefix), 0.0) + p_set * pr
                for j, i in enumerate(prefix):
                    marg[i, j] += p_set * pr
                return
            if avail:
                w = np.exp(lam[min(len(prefix), 2)] * s[avail])
                for a, wa in zip(avail, w / w.sum()):
                    rec([x for x in avail if x != a], ret, prefix + [a], pr * wa)
            else:   # fewer than 3 finishers: retired cars fill the places at random
                for a in ret:
                    rec(avail, [x for x in ret if x != a], prefix + [a], pr / len(ret))
        rec([i for i in range(n) if fin[i]], ret, [], 1.0)
    m = simulate_logw(s, q, n_sims=400_000, seed=0, lambdas=lam)
    print(f"simulate: max |P(place k) - exact| over places 1-3 = {np.abs(m[:, :3] - marg).max():.4f}")
    # The likelihood assumes the podium cars finished (true for every scored race)
    for order in ([0, 1, 2], [int(i) for i in np.argsort(-s)[:3]], [5, 3, 1]):
        print(f"order {order}: exact -logP {-np.log(exact[tuple(order)]):.4f}  "
              f"top_k_nll_logw {top_k_nll_logw(s, q, order, n_samples=200_000, lambdas=lam):.4f}")


def _train_seasons(years: list[int]) -> dict:
    from src.features.build_features import FEATURE_COLS, load_feature_matrix
    from src.training.backtest import make_two_stage_predict

    fm = load_feature_matrix().reset_index(drop=True)
    predict = make_two_stage_predict(FEATURE_COLS)
    seasons = {}
    for year in years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        pred, q = predict(train, test)
        seasons[year] = (pred, q, test)
        logger.info(f"{year}: trained, {len(test)} rows")
    return seasons


def _season_loss(races, t: float, lam: tuple) -> float:
    return float(np.mean([top_k_nll(p, q, top, t, n_samples=FIT_SAMPLES, lambdas=lam) for p, q, top in races]))


def fit_lambdas(races, t: float, fit_t: bool = False, rounds: int = 3) -> tuple[float, tuple]:
    """Coordinate descent on mean top-3 log loss: lambda_2, lambda_3 (and T if fit_t)."""
    lam = [1.0, 1.0, 1.0]
    best = _season_loss(races, t, tuple(lam))
    for _ in range(rounds):
        moved = False
        if fit_t:
            for v in T_GRID:
                loss = _season_loss(races, float(v), tuple(lam))
                if loss < best - 1e-9:
                    t, best, moved = float(v), loss, True
        for k in (1, 2):
            for v in LAMBDA_GRID:
                cand = list(lam)
                cand[k] = float(v)
                loss = _season_loss(races, t, tuple(cand))
                if loss < best - 1e-9:
                    lam, best, moved = cand, loss, True
        if not moved:
            break
    return t, tuple(lam)


def _races(pred, q, df):
    return [(pred[idx], q[idx], top) for _, idx, top in _race_orders(df)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    brute_force_check()
    if args.check:
        return

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] == "__main__")
    from src.features.build_features import load_feature_matrix
    years = list(range(2018, int(load_feature_matrix()["year"].max()) + 1))
    seasons = _train_seasons(years)

    rows, drivers, params = [], [], {}
    prev = None
    history = []   # earlier seasons' races at their walk-forward T: (pred / T, q, top)
    for year in years:
        pred, q, test = seasons[year]
        if prev is None:
            fits = {"base": (1.0, (1.0, 1.0, 1.0))}
            fits["benter"] = fits["benter_joint"] = fits["benter_pooled"] = fits["base"]
        else:
            t = fit_temperature(*prev)
            prev_races = _races(prev[0], prev[2], prev[1])
            fits = {"base": (t, (1.0, 1.0, 1.0)),
                    "benter": fit_lambdas(prev_races, t),
                    "benter_joint": fit_lambdas(prev_races, t, fit_t=True)}
            fits["benter_pooled"] = (t, fit_lambdas(history, 1.0)[1]) if history else fits["benter"]
        for name, (t, lam) in fits.items():
            params.setdefault(name, {})[year] = {"temperature": t, "lambda_2": lam[1], "lambda_3": lam[2]}
            for (y, rnd), idx, top in _race_orders(test):
                p, qq = pred[idx], q[idx]
                m = simulate(p, qq, t, n_sims=5000, seed=0, lambdas=lam)
                on_podium = np.zeros(len(idx))
                on_podium[top] = 1.0
                fav = np.argsort(-m[:, 0], kind="stable")
                for rank, i in enumerate(fav[:FAV_RANKS], start=1):
                    drivers.append({"variant": name, "year": int(y), "round": int(rnd), "fav_rank": rank,
                                    "p_win": float(m[i, 0]), "p_podium": float(m[i, :3].sum()),
                                    "on_podium": float(on_podium[i])})
                rows.append({
                    "variant": name, "year": int(y), "round": int(rnd),
                    "ll_top3": top_k_nll(p, qq, top, t, lambdas=lam),
                    "ll_winner": top_k_nll(p, qq, top[:1], t, lambdas=lam),
                    "winner_hit": bool(np.argmax(m[:, 0]) == top[0]),
                    "brier_podium": float(((m[:, :3].sum(axis=1) - on_podium) ** 2).sum()),
                })
        logger.info(f"{year}: " + "  ".join(f"{k} T={v[0]} l2={v[1][1]} l3={v[1][2]}" for k, v in fits.items()))
        if prev is not None:
            history += _races(pred / fits["base"][0], q, test)
        prev = (pred, test, q)

    res = pd.DataFrame(rows)
    metrics = ["ll_top3", "ll_winner", "brier_podium"]
    wide = res.pivot_table(index=["year", "round"], columns="variant", values=metrics)
    print(f"\n{len(wide)} races. Differences vs base (two-stage v5, T only), 95% paired bootstrap.\n")
    summary = {"n_races": int(len(wide)), "test_years": years, "variants": {}}
    for name in fits:
        r = res[res["variant"] == name]
        s = {m: float(r[m].mean()) for m in metrics}
        s["winner_accuracy"] = float(r["winner_hit"].mean())
        s["params_by_season"] = params[name]
        line = (f"{name:13s} ll_top3 {s['ll_top3']:.3f}  ll_winner {s['ll_winner']:.3f}  "
                f"brier {s['brier_podium']:.4f}  winner {s['winner_accuracy']:.1%}")
        if name != "base":
            for m in metrics:
                d = paired_bootstrap(wide[(m, name)], wide[(m, "base")])
                s[f"{m}_minus_base"] = list(d)
                line += f"\n{'':15s}d {m:12s} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"
            s["by_season"] = {}
            for y in years:
                w = wide.loc[y]
                d = paired_bootstrap(w[("ll_top3", name)], w[("ll_top3", "base")])
                s["by_season"][y] = {"n": int(len(w)), "ll_top3": float(w[("ll_top3", name)].mean()),
                                     "ll_top3_base": float(w[("ll_top3", "base")].mean()),
                                     "ll_top3_minus_base": list(d),
                                     "brier_minus_base": float((w[("brier_podium", name)]
                                                                - w[("brier_podium", "base")]).mean())}
        summary["variants"][name] = s
        print(line)

    for name in ("benter", "benter_joint", "benter_pooled"):
        print(f"\n{name}: per season (ll_top3 vs base, 95% CI; fitted on earlier seasons only)")
        for y in years:
            b, p = summary["variants"][name]["by_season"][y], params[name][y]
            d = b["ll_top3_minus_base"]
            print(f"  {y} n={b['n']:2d} T={p['temperature']:.3f} l2={p['lambda_2']:.2f} l3={p['lambda_3']:.2f}  "
                  f"ll_top3 {b['ll_top3']:.3f} vs {b['ll_top3_base']:.3f}  d {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  "
                  f"d brier {b['brier_minus_base']:+.4f}")

    drv = pd.DataFrame(drivers)
    print("\nP(podium) by favourite rank: mean predicted vs observed (obs 95% bootstrap over races)")
    rng = np.random.default_rng(0)
    summary["podium_by_favourite_rank"] = {}
    for name in fits:
        g = drv[drv["variant"] == name].groupby("fav_rank")
        out = {}
        for k, d in g:
            x = d["on_podium"].to_numpy()
            lo, hi = np.percentile(rng.choice(x, size=(2000, len(x))).mean(axis=1), [2.5, 97.5])
            out[int(k)] = {"pred": float(d["p_podium"].mean()), "obs": float(x.mean()),
                           "obs_ci": [float(lo), float(hi)]}
        summary["podium_by_favourite_rank"][name] = out
        print(f"  {name:13s} " + "  ".join(f"#{k} {v['pred']:.3f}/{v['obs']:.3f}" for k, v in out.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drv.to_parquet(OUT_DIR / "benter_backtest_favourites.parquet", index=False)
    res.to_parquet(OUT_DIR / "benter_backtest_races.parquet", index=False)
    with open(OUT_DIR / "benter_backtest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved {OUT_DIR / 'benter_backtest.json'}")


if __name__ == "__main__":
    main()

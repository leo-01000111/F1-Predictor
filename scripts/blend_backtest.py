# -*- coding: utf-8 -*-
"""
Model blend ("mixture of experts") walk-forward test
====================================================
Components, all as Plackett-Luce log-strengths (higher = faster), out of sample:
  v5    production two-stage ranker (gym cache: trained on earlier seasons)
  grid  -log(grid slot)
  glm   linear Plackett-Luce on a handful of standardised features, fitted by
        exact top-3 likelihood on earlier seasons (a different model family)
Blend: s = sum_k w_k * c_k. For each test season the weights are fitted on the
components' OOS outputs of all earlier test seasons (expanding window, exact
top-3 PL likelihood). Baseline "v5_only" is the same protocol with one weight
(= a fitted temperature), so the comparison isolates the extra experts.
Scored with the production two-stage likelihood (v5's P(finish)).

Usage:  python scripts/blend_backtest.py [--two-stage-fit]
--two-stage-fit: refine the weights on the two-stage likelihood (with P(finish)), as scored.
Writes models/experiments/blend_backtest[_ts].json only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import load_feature_matrix  # noqa: E402
from src.models.two_stage import top_k_nll_logw  # noqa: E402
from src.training import gym as G  # noqa: E402
from src.training.backtest import _grid_score, _race_orders, paired_bootstrap  # noqa: E402

GLM_FEATURES_RATINGS = ["log_grid", "quali_gap_relative", "driver_elo", "driver_form_ewm", "team_form_5",
                        "car_pace_delta_pct", "driver_rating", "car_rating", "teammate_quali_delta"]
# Production (src/models/blend.py): FEATURE_COLS inputs only, no strength ratings
GLM_FEATURES = ([c for c in GLM_FEATURES_RATINGS if c not in ("driver_rating", "car_rating")]
                if "--no-ratings" in sys.argv else GLM_FEATURES_RATINGS)
OOS_YEARS = list(range(2018, 2027))   # component predictions
TEST_YEARS = list(range(2019, 2027))  # blend scored (2018 only trains the first weights)
FIT_TWO_STAGE = "--two-stage-fit" in sys.argv   # weights by the scoring likelihood (slow)
OUT = ROOT / "models" / "experiments" / (("blend_backtest_ts" if FIT_TWO_STAGE else "blend_backtest")
                                         + ("_noratings" if "--no-ratings" in sys.argv else "") + ".json")


# ----------------------------------------------------------------------------- #
# Exact top-3 Plackett-Luce likelihood for linear scores s = X @ w
# ----------------------------------------------------------------------------- #
def _races(df: pd.DataFrame, X: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    return [(X[idx], top) for _, idx, top in _race_orders(df)]


def pl_nll_grad(w: np.ndarray, races: list, l2: float = 0.0) -> tuple[float, np.ndarray]:
    nll, grad = 0.0, np.zeros_like(w)
    for X, top in races:
        s = X @ w
        alive = np.ones(len(s), bool)
        for i in top:
            sa = s[alive]
            m = sa.max()
            e = np.exp(sa - m)
            z = e.sum()
            p = e / z
            nll -= s[i] - m - np.log(z)
            grad -= X[i] - p @ X[alive]
            alive[i] = False
    n = max(len(races), 1)
    return nll / n + l2 * (w @ w), grad / n + 2 * l2 * w


def fit_pl(races: list, k: int, l2: float = 0.0, w0=None) -> np.ndarray:
    w0 = np.zeros(k) if w0 is None else w0
    r = minimize(pl_nll_grad, w0, args=(races, l2), jac=True, method="L-BFGS-B")
    return r.x


# ----------------------------------------------------------------------------- #
def glm_matrix(df: pd.DataFrame, mu=None, sd=None):
    X = df.reindex(columns=GLM_FEATURES).copy()
    X["log_grid"] = np.log(_grid_score(df["grid_position"]))
    X = X.apply(pd.to_numeric, errors="coerce")
    if mu is None:
        mu, sd = X.mean(), X.std().replace(0, 1)
    X = ((X - mu) / sd).fillna(0.0)
    return X.to_numpy(float), mu, sd


def glm_oos(fm: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for y in OOS_YEARS:
        train = fm[fm["year"] < y].reset_index(drop=True)
        test = fm[fm["year"] == y].reset_index(drop=True)
        Xtr, mu, sd = glm_matrix(train)
        w = fit_pl(_races(train, Xtr), Xtr.shape[1], l2=1e-3)
        out[y] = glm_matrix(test, mu, sd)[0] @ w
        logger.info(f"glm {y}: " + ", ".join(f"{f}={v:+.2f}" for f, v in zip(GLM_FEATURES, w)))
    return out


def fit_two_stage(prev: list, cols: list[int], w0: np.ndarray) -> np.ndarray:
    """Weights minimising the production two-stage top-3 likelihood (uses P(finish))."""
    data = [(Cp[idx][:, cols], qp[idx], top) for t, Cp, qp in prev for _, idx, top in _race_orders(t)]

    def f(w):
        return float(np.mean([top_k_nll_logw(X @ w, qq, top, n_samples=300) for X, qq, top in data]))
    return minimize(f, w0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 200}).x


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    fm = load_feature_matrix()
    gym = G.Gym(fm)
    sp = G.season_predictions(gym.fm, gym.spec({}), OOS_YEARS)
    glm = glm_oos(gym.fm)

    comp: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for y in OOS_YEARS:
        test, pred, q = sp[y]
        grid = -np.log(_grid_score(test["grid_position"]))
        C = np.column_stack([-np.asarray(pred, float), grid, glm[y]])
        comp[y] = (test.reset_index(drop=True), C, np.asarray(q, float))

    variants = {"v5_only": [0], "v5+grid": [0, 1], "v5+glm": [0, 2], "v5+grid+glm": [0, 1, 2],
                "glm_only": [2], "grid_only": [1]}
    if FIT_TWO_STAGE:
        variants = {k: v for k, v in variants.items() if k in ("v5_only", "v5+grid", "v5+glm")}
    rows = []
    weights: dict[str, dict] = {v: {} for v in variants}
    for y in TEST_YEARS:
        prev = [comp[p] for p in OOS_YEARS if p < y]
        test, C, q = comp[y]
        for name, cols in variants.items():
            races = [r for t, Cp, _ in prev for r in _races(t, Cp[:, cols])]
            w = fit_pl(races, len(cols), w0=np.ones(len(cols)) * (1.0 / len(cols)))
            if FIT_TWO_STAGE:
                w = fit_two_stage(prev, cols, w)
            weights[name][y] = [round(float(v), 3) for v in w]
            s_all = C[:, cols] @ w
            for (yy, rr), idx, top in _race_orders(test):
                s = s_all[idx]
                rows.append({"year": yy, "round": rr, "variant": name,
                             "ll_top3": top_k_nll_logw(s, q[idx], top, n_samples=2000),
                             "ll_winner": top_k_nll_logw(s, q[idx], top[:1], n_samples=2000),
                             "winner_hit": int(np.argmax(s) == top[0])})
        logger.info(f"{y}: weights " + ", ".join(f"{k}={weights[k][y]}" for k in variants))

    res = pd.DataFrame(rows)
    res.to_parquet(OUT.with_suffix(".parquet"), index=False)
    summary = {}
    print(f"\nWalk-forward {TEST_YEARS[0]}-{TEST_YEARS[-1]}, {res.query('variant==\"v5_only\"').shape[0]} races")
    for metric in ("ll_top3", "ll_winner", "winner_hit"):
        wide = res.pivot_table(index=["year", "round"], columns="variant", values=metric)
        print(f"\n{metric}:")
        for name in variants:
            s = summary.setdefault(name, {})
            s[metric] = float(wide[name].mean())
            line = f"  {name:12s} {wide[name].mean():.4f}"
            if name != "v5_only":
                d, lo, hi = paired_bootstrap(wide[name].astype(float), wide["v5_only"].astype(float))
                s[f"{metric}_minus_v5"] = [d, lo, hi]
                line += f"   d {d:+.4f} [{lo:+.4f},{hi:+.4f}]"
            print(line)
    print("\nll_top3 by season:")
    print(res.pivot_table(index="year", columns="variant", values="ll_top3").round(3)[list(variants)].to_string())
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"test_years": TEST_YEARS, "glm_features": GLM_FEATURES, "variants": summary,
                   "weights": weights}, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()

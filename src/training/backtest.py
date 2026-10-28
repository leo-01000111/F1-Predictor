"""
Walk-forward backtest (ROADMAP Phase 1).

For each test season S, the model is trained only on seasons before S and
predicts every race of S. Unlike cross_validate.py (leave-one-season-out, which
also trains on later seasons), nothing from the future is seen, so the scores
are what a live forecaster would have got.

Scoring is per race, on Plackett-Luce probabilities with log-strength -pred / T:
  - ll_winner : -log P(actual winner)
  - ll_top3   : -log P(actual P1, P2, P3 in that order)
  - winner / podium hit (picks = lowest predicted positions)

T is fitted on the previous test season's out-of-sample predictions (T=1 for
the first season), so it is never fitted on the season it scores.

The grid is scored the same way: log-strength -grid / T_grid, with T_grid fitted
on all races before the test season. That gives a log-loss baseline, not only
"pole wins".
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.training.weights import compute_recency_sample_weight

# Geometric grid: regressor outputs need T ~ 1-3, ranker scores are smaller (T < 0.5)
T_GRID = np.round(np.geomspace(0.05, 8.0, 60), 3)

PredictFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def pl_neg_loglik(pred: np.ndarray, order: list[int], temperature: float) -> float:
    """-log P(order) under Plackett-Luce for the drivers in `order` (top-k, in finishing order)."""
    s = -np.asarray(pred, dtype=float) / temperature
    remaining = np.ones(len(s), dtype=bool)
    nll = 0.0
    for i in order:
        r = s[remaining]
        m = r.max()
        nll -= s[i] - (m + np.log(np.exp(r - m).sum()))
        remaining[i] = False
    return float(nll)


def _grid_score(grid: pd.Series) -> np.ndarray:
    """Grid as a predicted position: pit-lane starts (0) and unknown go last."""
    g = pd.to_numeric(grid, errors="coerce").to_numpy(dtype=float)
    last = np.nanmax(np.where(g > 0, g, np.nan)) + 1 if np.any(g > 0) else len(g)
    return np.where(np.isfinite(g) & (g > 0), g, last)


def _race_orders(df: pd.DataFrame) -> list[tuple[tuple, np.ndarray, list[int]]]:
    """Per race: key, row positions in df, and indices of actual P1-P3 in finishing order."""
    out = []
    fin = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy()
    for key, idx in df.groupby(["year", "round"]).indices.items():
        f = fin[idx]
        top = [int(np.where(f == p)[0][0]) for p in (1, 2, 3) if (f == p).any()]
        if len(top) == 3:
            out.append((key, idx, top))
    return out


def fit_temperature(pred: np.ndarray, df: pd.DataFrame) -> float:
    """T minimising mean top-3 PL log loss over the races in df."""
    races = _race_orders(df)
    if not races:
        return 1.0
    losses = [np.mean([pl_neg_loglik(pred[idx], top, t) for _, idx, top in races]) for t in T_GRID]
    return float(T_GRID[int(np.argmin(losses))])


def score_races(df: pd.DataFrame, pred: np.ndarray, temperature: float) -> pd.DataFrame:
    """Per-race scores for predicted positions `pred` aligned with df rows."""
    rows = []
    for (year, rnd), idx, top in _race_orders(df):
        p = pred[idx]
        pick = np.argsort(p, kind="stable")
        rows.append({
            "year": int(year),
            "round": int(rnd),
            "ll_winner": pl_neg_loglik(p, top[:1], temperature),
            "ll_top3": pl_neg_loglik(p, top, temperature),
            "winner_hit": bool(pick[0] == top[0]),
            "podium_overlap": len(set(pick[:3]) & set(top)),
        })
    return pd.DataFrame(rows)


def make_xgb_predict(cols: list[str], params: Optional[dict] = None) -> PredictFn:
    """Train XGBRacePredictor on `cols` and predict the test rows."""
    from src.models.xgb_model import XGBRacePredictor

    def _predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = XGBRacePredictor(params=params)
        model.fit(train[cols], train["target_position"].values,
                  sample_weight=compute_recency_sample_weight(train["year"].values))
        return model.predict(test[cols])

    return _predict


def make_ranker_predict(cols: list[str], objective: str = "rank:pairwise") -> PredictFn:
    """XGBRaceRanker (the production model class) on `cols`. Returns -score (T=1)."""
    from src.models.xgb_model import XGBRaceRanker

    params = dict(XGBRaceRanker.DEFAULT_PARAMS, objective=objective)
    if objective != "rank:ndcg":
        params["lambdarank_pair_method"] = "mean"

    def _predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = XGBRaceRanker(params=params)
        model.fit(train[cols], train["target_position"].values,
                  sample_weight=compute_recency_sample_weight(train["year"].values),
                  qid=(train["year"] * 100 + train["round"]).to_numpy())
        return model.predict(test[cols])

    return _predict


def grid_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Grid-only 'model': predicted position = starting grid."""
    out = np.empty(len(test))
    for _, idx in test.groupby(["year", "round"]).indices.items():
        out[idx] = _grid_score(test["grid_position"].iloc[idx])
    return out


def walk_forward(
    fm: pd.DataFrame,
    predict: PredictFn,
    test_years: list[int],
    name: str,
    fit_t_on_train: bool = False,
) -> pd.DataFrame:
    """Run one variant over all test seasons. Returns per-race scores.

    fit_t_on_train: fit T on the training rows' own predictions (only sensible for
    models that cannot overfit, i.e. the grid). Otherwise T comes from the previous
    test season's out-of-sample predictions.
    """
    fm = fm.reset_index(drop=True)
    results = []
    prev: Optional[tuple[np.ndarray, pd.DataFrame]] = None
    for year in test_years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        if train.empty or test.empty:
            continue
        pred = predict(train, test)
        if fit_t_on_train:
            t = fit_temperature(predict(train, train), train)
        else:
            t = fit_temperature(*prev) if prev is not None else 1.0
        prev = (pred, test)

        scores = score_races(test, pred, t)
        scores["ll_winner_t1"] = [
            pl_neg_loglik(pred[idx], top[:1], 1.0) for _, idx, top in _race_orders(test)
        ]
        scores["temperature"] = t
        scores["variant"] = name
        results.append(scores)
        logger.info(
            f"{name:14s} {year}: {len(scores)} races, winner {scores['winner_hit'].mean():.0%}, "
            f"ll_winner {scores['ll_winner'].mean():.3f}, ll_top3 {scores['ll_top3'].mean():.3f}, T={t}"
        )
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over races and its 95% bootstrap interval."""
    d = (a - b).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

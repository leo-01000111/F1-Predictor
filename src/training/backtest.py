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


def _nll(pred: np.ndarray, q: Optional[np.ndarray], order: list[int], temperature: float,
         n_samples: int = 2000) -> float:
    """PL log loss of `order`; with finish probabilities q, the two-stage version."""
    if q is None:
        return pl_neg_loglik(pred, order, temperature)
    from src.models.two_stage import top_k_nll
    return top_k_nll(pred, q, order, temperature, n_samples=n_samples)


def fit_temperature(pred: np.ndarray, df: pd.DataFrame, q: Optional[np.ndarray] = None) -> float:
    """T minimising mean top-3 log loss over the races in df."""
    races = _race_orders(df)
    if not races:
        return 1.0
    losses = [
        np.mean([_nll(pred[idx], None if q is None else q[idx], top, t, n_samples=500)
                 for _, idx, top in races])
        for t in T_GRID
    ]
    return float(T_GRID[int(np.argmin(losses))])


def score_races(df: pd.DataFrame, pred: np.ndarray, temperature: float,
                q: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Per-race scores for predicted positions `pred` (and finish probabilities q) aligned with df rows."""
    rows = []
    for (year, rnd), idx, top in _race_orders(df):
        p = pred[idx]
        qq = None if q is None else q[idx]
        if qq is None:
            pick = np.argsort(p, kind="stable")
            podium = pick[:3]
        else:
            from src.models.two_stage import simulate
            m = simulate(p, qq, temperature, n_sims=5000, seed=0)
            pick = np.argsort(-m[:, 0], kind="stable")
            podium = np.argsort(-m[:, :3].sum(axis=1), kind="stable")[:3]
        rows.append({
            "year": int(year),
            "round": int(rnd),
            "ll_winner": _nll(p, qq, top[:1], temperature),
            "ll_top3": _nll(p, qq, top, temperature),
            "winner_hit": bool(pick[0] == top[0]),
            "podium_overlap": len(set(podium) & set(top)),
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


def make_two_stage_predict(cols: list[str], use_finish_model: bool = True,
                           half_life: Optional[float] = None, current_boost: float = 1.0,
                           params: Optional[dict] = None, n_seeds: Optional[int] = None) -> PredictFn:
    """Phase 3: pace ranker trained on finishers only (+ finish model for P(finish)).

    use_finish_model=False isolates stage 2: the finishers-only ranker scored as if
    every car finishes. half_life: exponential recency weights (years) instead of
    the default linear 0.65-1.35. current_boost: extra weight multiplier for training
    rows from the test race's own season (only matters with in-season retraining).
    params / n_seeds: ranker settings and seed-ensemble size (scripts/tune_ranker.py).
    """
    from src.models.xgb_model import TwoStageRaceModel

    def _predict(train: pd.DataFrame, test: pd.DataFrame):
        w = compute_recency_sample_weight(train["year"].values, half_life=half_life)
        if current_boost != 1.0:
            w = np.where(train["year"].to_numpy() == test["year"].iloc[0], w * current_boost, w)
            w = w / w.mean()
        model = TwoStageRaceModel(params=params, n_seeds=n_seeds)
        model.fit(train[cols], train["target_position"].values,
                  sample_weight=w,
                  qid=(train["year"] * 100 + train["round"]).to_numpy(),
                  dnf=pd.to_numeric(train["dnf"], errors="coerce").fillna(0).to_numpy())
        pred = model.predict(test[cols])
        if not use_finish_model:
            return pred
        return pred, model.predict_finish(test[cols])

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
    prev: Optional[tuple] = None
    for year in test_years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        if train.empty or test.empty:
            continue
        out = predict(train, test)
        # Two-stage models return (pred, finish probabilities)
        pred, q = out if isinstance(out, tuple) else (out, None)
        if fit_t_on_train:
            t = fit_temperature(predict(train, train), train)
        else:
            t = fit_temperature(*prev) if prev is not None else 1.0
        prev = (pred, test, q)

        scores = score_races(test, pred, t, q)
        scores["ll_winner_t1"] = [
            _nll(pred[idx], None if q is None else q[idx], top[:1], 1.0)
            for _, idx, top in _race_orders(test)
        ]
        scores["temperature"] = t
        scores["variant"] = name
        results.append(scores)
        logger.info(
            f"{name:14s} {year}: {len(scores)} races, winner {scores['winner_hit'].mean():.0%}, "
            f"ll_winner {scores['ll_winner'].mean():.3f}, ll_top3 {scores['ll_top3'].mean():.3f}, T={t}"
        )
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def walk_forward_by_round(
    fm: pd.DataFrame,
    predict: PredictFn,
    test_years: list[int],
    name: str,
    warmup_year: Optional[int] = None,
    t_window: int = 22,
) -> pd.DataFrame:
    """In-season retraining: each race is predicted by a model trained on every
    earlier race, including earlier rounds of its own season.

    T for each race is fitted on the out-of-sample predictions of the previous
    `t_window` races from this same process. Races of `warmup_year` are predicted
    (to seed T) but not scored.
    """
    fm = fm.reset_index(drop=True)
    first = warmup_year if warmup_year is not None else test_years[0]
    keys = sorted({(int(y), int(r)) for y, r in zip(fm["year"], fm["round"])
                   if first <= y <= test_years[-1]})
    history: list[tuple[np.ndarray, pd.DataFrame, Optional[np.ndarray]]] = []
    results = []
    for year, rnd in keys:
        before = (fm["year"] < year) | ((fm["year"] == year) & (fm["round"] < rnd))
        train = fm[before].reset_index(drop=True)
        test = fm[(fm["year"] == year) & (fm["round"] == rnd)].reset_index(drop=True)
        out = predict(train, test)
        pred, q = out if isinstance(out, tuple) else (out, None)
        if year in test_years:
            if history:
                recent = history[-t_window:]
                t = fit_temperature(
                    np.concatenate([h[0] for h in recent]),
                    pd.concat([h[1] for h in recent], ignore_index=True),
                    None if q is None else np.concatenate([h[2] for h in recent]),
                )
            else:
                t = 1.0
            scores = score_races(test, pred, t, q)
            scores["ll_winner_t1"] = [
                _nll(pred[idx], None if q is None else q[idx], top[:1], 1.0)
                for _, idx, top in _race_orders(test)
            ]
            scores["temperature"] = t
            scores["variant"] = name
            results.append(scores)
        history.append((pred, test, q))
    res = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    for y, g in res.groupby("year"):
        logger.info(f"{name:14s} {y}: {len(g)} races, winner {g['winner_hit'].mean():.0%}, "
                    f"ll_top3 {g['ll_top3'].mean():.3f}, T {g['temperature'].mean():.2f}")
    return res


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over races and its 95% bootstrap interval."""
    d = (a - b).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

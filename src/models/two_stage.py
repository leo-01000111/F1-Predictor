"""
Two-stage race model (ROADMAP Phase 3).

Stage 1, finish model: P(car is classified as a finisher), a logistic regression
on reliability, circuit and starting-position features. Retirements are partly
luck, so this stays a small, well-calibrated linear model.

Stage 2, pace model: XGBRaceRanker trained on finishers only, so a fast car that
retired is no longer taught as "slow".

Combination: in a race, each car finishes independently with probability q_i;
the finishers are ordered by Plackett-Luce with weights w_i = exp(-pred_i / T);
retired cars fill the places behind them. Probabilities come from Monte Carlo
(simulate). Scoring the actual top 3 uses the exact PL formula averaged over
sampled sets of other finishers (Rao-Blackwellised, low variance).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FINISH_COLS = [
    "team_reliability_rate",
    "dnf_rate",
    "is_street_circuit",
    "circuit_safety_car_rate",
    "grid_position_rel",
    "is_wet_race",
    "precip_mm_total",
    "driver_experience_norm",
]

N_SUBSET_SAMPLES = 2000


class FinishModel:
    """P(finish) per car. Logistic regression on FINISH_COLS."""

    def __init__(self, cols: Optional[list[str]] = None, C: float = 1.0) -> None:
        self.cols = cols or FINISH_COLS
        self._model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> "FinishModel":
        y = 1 - pd.to_numeric(df["dnf"], errors="coerce").fillna(0).astype(int)
        kw = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
        self._model.fit(df[self.cols].astype(float).fillna(0.0), y, **kw)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(df[self.cols].astype(float).fillna(0.0))[:, 1]


def top_k_nll(pred: np.ndarray, q: np.ndarray, order: list[int], temperature: float,
              n_samples: int = N_SUBSET_SAMPLES, seed: int = 0) -> float:
    """-log P(drivers `order` finish 1st, 2nd, ... in that order) under the two-stage model.

    P = prod_k q_k * E_S[ prod_j w_{o_j} / (W_S + sum_{m>=j} w_{o_m}) ], where S is
    the random set of other finishers (each other car finishes with prob q).
    """
    return top_k_nll_logw(-np.asarray(pred, float) / temperature, q, order, n_samples, seed)


def top_k_nll_logw(s: np.ndarray, q: np.ndarray, order: list[int],
                   n_samples: int = N_SUBSET_SAMPLES, seed: int = 0) -> float:
    """top_k_nll with Plackett-Luce log-strengths s given directly."""
    s = np.asarray(s, float)
    w = np.exp(s - s.max())
    q = np.clip(np.asarray(q, float), 1e-4, 1 - 1e-4)
    order = list(order)
    others = np.setdiff1d(np.arange(len(w)), order)
    rng = np.random.default_rng(seed)
    mask = rng.random((n_samples, len(others))) < q[others]
    w_other = mask @ w[others]                       # (n_samples,)
    tail = np.cumsum(w[order][::-1])[::-1]           # sum of w over order[j:], per j
    p = np.ones(n_samples)
    for j, i in enumerate(order):
        p *= w[i] / (w_other + tail[j])
    return float(-(np.log(q[order]).sum() + np.log(p.mean())))


def simulate(pred: np.ndarray, q: np.ndarray, temperature: float,
             n_sims: int = 10_000, seed: Optional[int] = None) -> np.ndarray:
    """Position probability matrix [driver, position] from the two-stage model."""
    return simulate_logw(-np.asarray(pred, float) / temperature, q, n_sims=n_sims, seed=seed)


def simulate_logw(s: np.ndarray, q: np.ndarray, n_sims: int = 10_000, seed: Optional[int] = None,
                  s_alt: Optional[np.ndarray] = None, p_alt: float = 0.0) -> np.ndarray:
    """simulate() with log-strengths s. With s_alt, a share p_alt of the races
    use s_alt instead (e.g. a safety-car race with a flatter order)."""
    s = np.asarray(s, float)
    n = len(s)
    rng = np.random.default_rng(seed)
    base = np.broadcast_to(s, (n_sims, n))
    if s_alt is not None and p_alt > 0:
        alt = rng.random(n_sims) < p_alt
        base = np.where(alt[:, None], np.asarray(s_alt, float), base)
    scores = base + rng.gumbel(size=(n_sims, n))
    finished = rng.random((n_sims, n)) < np.asarray(q, float)
    # Retired cars rank behind all finishers, in random order among themselves
    scores = np.where(finished, scores, -1e9 + rng.random((n_sims, n)))
    rankings = np.argsort(-scores, axis=1)
    counts = np.zeros((n, n))
    for pos in range(n):
        counts[:, pos] = np.bincount(rankings[:, pos], minlength=n)
    return counts / n_sims

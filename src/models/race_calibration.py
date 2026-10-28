"""
Race calibration layer (ROADMAP Phase 3): turns the pace model's raw scores
into per-race Plackett-Luce log-strengths, with race-level randomness.

    s_i = (raw_i + rating_weight * car_rating_i) / T_race
    T_race = temperature * exp(overtaking_slope * z),  z = (circuit_overtaking - 0.2) / 0.05

Safety-car shock: with probability p_sc = min(sc_scale * circuit_safety_car_rate, 0.9)
the race is a "safety-car race" whose log-strengths are multiplied by sc_flatten
(< 1: a flatter, more reshuffled order). The likelihood is the mixture of the
two cases; the simulation draws the case per simulated race.

All parameters are fitted by coordinate descent on top-3 log loss of races the
pace model did not train on. With the defaults (rating_weight 0, slope 0,
sc_scale 0) this is the plain two-stage model with temperature T.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from src.models.two_stage import simulate_logw, top_k_nll_logw

OVT_REF, OVT_SCALE = 0.2, 0.05

GRIDS = {
    "temperature": list(np.round(np.geomspace(0.3, 3.0, 25), 3)),
    "rating_weight": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "overtaking_slope": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
    "sc_scale": [0.0, 1.0, 2.0, 4.0, 8.0],
    "sc_flatten": [0.25, 0.5, 0.75],
}


@dataclass
class RaceContext:
    """What the calibration needs about one race besides the model outputs."""
    raw: np.ndarray            # pace model score, higher = faster (= -pred at T=1)
    q: np.ndarray              # P(finish)
    car_rating: np.ndarray
    overtaking: float          # circuit_overtaking
    sc_rate: float             # circuit_safety_car_rate


@dataclass
class RaceCalibration:
    temperature: float = 1.0
    rating_weight: float = 0.0
    overtaking_slope: float = 0.0
    sc_scale: float = 0.0
    sc_flatten: float = 0.5
    fitted: list = field(default_factory=list)   # which parameters were fitted

    def log_strength(self, r: RaceContext) -> np.ndarray:
        z = (r.overtaking - OVT_REF) / OVT_SCALE
        t_race = self.temperature * np.exp(self.overtaking_slope * z)
        return (r.raw + self.rating_weight * r.car_rating) / t_race

    def p_sc(self, r: RaceContext) -> float:
        return float(min(self.sc_scale * r.sc_rate, 0.9))

    def nll(self, r: RaceContext, order: list[int], n_samples: int = 2000) -> float:
        s = self.log_strength(r)
        p = self.p_sc(r)
        base = top_k_nll_logw(s, r.q, order, n_samples)
        if p <= 0:
            return base
        sc = top_k_nll_logw(s * self.sc_flatten, r.q, order, n_samples)
        return float(-np.log((1 - p) * np.exp(-base) + p * np.exp(-sc)))

    def simulate(self, r: RaceContext, n_sims: int = 10_000, seed: Optional[int] = None) -> np.ndarray:
        s = self.log_strength(r)
        return simulate_logw(s, r.q, n_sims=n_sims, seed=seed, s_alt=s * self.sc_flatten, p_alt=self.p_sc(r))

    def fit(self, races: list[tuple[RaceContext, list[int]]], params: list[str],
            rounds: int = 2) -> "RaceCalibration":
        """Coordinate descent over GRIDS for `params` (temperature always included)."""
        params = ["temperature"] + [p for p in params if p != "temperature"]
        best = self

        def loss(c: RaceCalibration) -> float:
            return float(np.mean([c.nll(r, order, n_samples=500) for r, order in races]))

        best_loss = loss(best)
        for _ in range(rounds):
            for name in params:
                for v in GRIDS[name]:
                    cand = replace(best, **{name: v})
                    cand_loss = loss(cand)
                    if cand_loss < best_loss - 1e-9:
                        best, best_loss = cand, cand_loss
        return replace(best, fitted=params)


def race_contexts(df: pd.DataFrame, raw: np.ndarray, q: np.ndarray) -> list[tuple[tuple, np.ndarray, RaceContext]]:
    """Split model outputs aligned with df rows into per-race contexts."""
    out = []
    df = df.reset_index(drop=True)
    rating = pd.to_numeric(df.get("car_rating", 0.0), errors="coerce").fillna(0.0).to_numpy()
    ovt = pd.to_numeric(df.get("circuit_overtaking", OVT_REF), errors="coerce").fillna(OVT_REF).to_numpy()
    sc = pd.to_numeric(df.get("circuit_safety_car_rate", 0.0), errors="coerce").fillna(0.0).to_numpy()
    for key, idx in df.groupby(["year", "round"]).indices.items():
        out.append((key, idx, RaceContext(raw=raw[idx], q=q[idx], car_rating=rating[idx],
                                          overtaking=float(ovt[idx[0]]), sc_rate=float(sc[idx[0]]))))
    return out

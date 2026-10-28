"""
Driver + team strength ratings (ROADMAP Phase 3).

Each car's strength is driver rating + team rating. After every race both are
moved along the gradient of the Plackett-Luce log-likelihood of the finishing
order of classified finishers (Powell, "Generalizing Elo"). DNFs are left out of
the update: a retirement says little about pace.

Between seasons ratings shrink toward 0 (the field average). Team ratings shrink
much harder at regulation resets, when the car order is rebuilt; driver skill
carries over. New drivers start at ROOKIE_PRIOR, new teams at 0.

Ratings are recorded BEFORE each race (no leakage). A race without results (the
upcoming race at inference) gets pre-race ratings and no update.

Hyperparameters were tuned by one-race-ahead log-likelihood of the finishing
order over 2014-2020 (tune_hyperparameters below).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

REGULATION_RESETS = {2014, 2017, 2022, 2026}


@dataclass(frozen=True)
class RatingParams:
    # Tuned 2026-10-28 on 2014-2020: -1.525 log-lik per finisher vs -1.950 for
    # equal strengths. The optimum is flat (+-0.002 across neighbouring values).
    lr_driver: float = 0.05
    lr_team: float = 0.10
    driver_carry: float = 0.8      # season-to-season shrink toward 0
    team_carry: float = 0.8
    reset_carry: float = 0.3       # team carry at a regulation reset
    rookie_prior: float = -0.75


DEFAULT_PARAMS = RatingParams()


def _pl_gradient(s: np.ndarray) -> tuple[np.ndarray, float]:
    """Gradient and log-likelihood of PL for strengths s already in finishing order."""
    g = np.zeros(len(s))
    ll = 0.0
    for j in range(len(s) - 1):
        r = s[j:]
        p = np.exp(r - r.max())
        p /= p.sum()
        g[j:] -= p
        g[j] += 1.0
        ll += np.log(p[0])
    return g, float(ll)


def _run(df: pd.DataFrame, params: RatingParams) -> tuple[pd.DataFrame, list[tuple[int, float, int]]]:
    """Walk races in order. Returns pre-race ratings per row and per-race (year, ll, n_finishers)."""
    driver: dict[str, float] = {}
    team: dict[str, float] = {}
    pre_d = np.zeros(len(df))
    pre_t = np.zeros(len(df))
    race_ll: list[tuple[int, float, int]] = []
    prev_year = None

    years = df["year"].to_numpy()
    drivers = df["driver"].astype(str).to_numpy()
    teams = df["team"].astype(str).to_numpy()
    fin = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy()
    dnf = pd.to_numeric(df.get("dnf", 0), errors="coerce").fillna(0).to_numpy()

    for _, idx in df.groupby(["year", "round"], sort=True).indices.items():
        year = int(years[idx[0]])
        if prev_year is not None and year > prev_year:
            tc = params.reset_carry if year in REGULATION_RESETS else params.team_carry
            for k in driver:
                driver[k] *= params.driver_carry
            for k in team:
                team[k] *= tc
        prev_year = year

        for i in idx:
            driver.setdefault(drivers[i], params.rookie_prior)
            team.setdefault(teams[i], 0.0)
            pre_d[i] = driver[drivers[i]]
            pre_t[i] = team[teams[i]]

        ok = [i for i in idx if np.isfinite(fin[i]) and dnf[i] == 0]
        if len(ok) < 2:
            continue
        ok.sort(key=lambda i: fin[i])
        s = np.array([pre_d[i] + pre_t[i] for i in ok])
        g, ll = _pl_gradient(s)
        race_ll.append((year, ll, len(ok)))
        for gi, i in zip(g, ok):
            driver[drivers[i]] += params.lr_driver * gi
            team[teams[i]] += params.lr_team * gi

    out = pd.DataFrame({"driver_rating": pre_d, "team_rating": pre_t}, index=df.index)
    return out, race_ll


def add_strength_ratings(race_df: pd.DataFrame, params: RatingParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Add pre-race driver_rating, team_rating and car_rating (their sum)."""
    df = race_df.sort_values(["year", "round"], kind="stable")
    ratings, _ = _run(df, params)
    out = race_df.copy()
    out["driver_rating"] = ratings["driver_rating"].reindex(out.index)
    out["team_rating"] = ratings["team_rating"].reindex(out.index)
    out["car_rating"] = out["driver_rating"] + out["team_rating"]
    return out


def tune_hyperparameters(race_df: pd.DataFrame, years: tuple[int, int] = (2014, 2020)) -> tuple[RatingParams, float]:
    """Grid search on mean one-race-ahead log-likelihood per finisher over `years`."""
    df = race_df[race_df["year"].between(*years)].sort_values(["year", "round"], kind="stable")
    best, best_score = DEFAULT_PARAMS, -np.inf
    grid = product([0.02, 0.05, 0.1], [0.05, 0.1, 0.2], [0.8, 0.95], [0.5, 0.8], [0.0, 0.3], [-0.5, 0.0])
    for lr_d, lr_t, dc, tc, rc, rp in grid:
        p = replace(DEFAULT_PARAMS, lr_driver=lr_d, lr_team=lr_t, driver_carry=dc,
                    team_carry=tc, reset_carry=rc, rookie_prior=rp)
        _, race_ll = _run(df, p)
        # skip the first season: ratings start empty
        scored = [(ll, n) for y, ll, n in race_ll if y > years[0]]
        score = sum(ll for ll, _ in scored) / sum(n for _, n in scored)
        if score > best_score:
            best, best_score = p, score
    return best, float(best_score)

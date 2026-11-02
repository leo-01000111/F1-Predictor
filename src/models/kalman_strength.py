"""
Kalman filter driver + car strengths from qualifying gaps (ROADMAP Phase 3, state-space idea).

State: one pace value per driver (d) and per team lineage (c), in % of lap time,
higher = faster, with a full covariance. A car's pace is c_team + d_driver.

Measurement (one per qualifying session): each driver's best time as a gap to the
session in %, y_i = 100 * log(t_i / t_min), modelled as

    y_i = o - (c_team(i) + d_i) + b_stage(i) + u_team(i) + e_i
    e_i ~ N(0, noise_sd^2) per driver, u ~ N(0, team_session_sd^2) per team and session

o is a per-session offset (diffuse prior, dropped after the update). b_stage is the
extra gap of drivers knocked out in Q2 / Q1, whose best lap was set on a slower,
earlier track. Teammates share c, so their gap measures driver vs driver and the
field measures the car: the covariance carries that split.

Gaps are rescaled per session to the season's typical spread (track length and
layout change what 1 % means). Robustness: sessions whose time order contradicts the qualifying order (Spearman
< MIN_ORDER_RHO: mixed wet/dry sessions, where a Q1 lap beats the Q3 laps) are
skipped. Within a session, a noise multiplier from the median squared innovation
(rough wet sessions) and Huber weights per driver (crashes, aborted laps).

Dynamics: per race, car variance + car_drift_sd^2 (development), driver variance +
driver_drift_sd^2. At a new season, car + car_season_sd^2 (car_reset_sd^2 at a
regulation reset), driver + driver_season_sd^2. New drivers start at rookie_mean
with driver_prior_sd, new teams at 0 with car_prior_sd. Drivers seen in the first
season of the data are not rookies: they start at 0 with initial_driver_sd.

The filter walks sessions in order and records the state BEFORE each session, so
the pre-session values use only earlier qualifying sessions (no leakage).
Hyperparameters: tune() maximises the one-session-ahead predictive likelihood.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from src.features.strength_ratings import REGULATION_RESETS

MIN_ORDER_RHO = 0.9
OFFSET_VAR = 1e4
MAX_GAP_PCT = 7.0             # 107 % rule: slower laps carry no pace information


@dataclass(frozen=True)
class KalmanParams:
    noise_sd: float = 0.25
    car_drift_sd: float = 0.08
    driver_drift_sd: float = 0.02
    car_season_sd: float = 0.4
    car_reset_sd: float = 1.0
    driver_season_sd: float = 0.1
    driver_prior_sd: float = 0.4
    rookie_mean: float = -0.2
    initial_driver_sd: float = 0.4  # drivers already racing when the data starts
    car_prior_sd: float = 1.0
    q2_bias: float = 0.2
    q1_bias: float = 0.4
    team_session_sd: float = 0.0   # car x track error shared by teammates in a session
    huber_k: float = 2.5


DEFAULT_PARAMS = KalmanParams()


def session_measurements(q: pd.DataFrame) -> pd.DataFrame:
    """Per-session gap y (%) and knockout stage; drops unusable sessions/rows."""
    q = q.dropna(subset=["quali_best_time_s", "quali_position"]).copy()
    q = q[q["quali_best_time_s"] > 0]
    out = []
    for _, g in q.groupby(["year", "round"], sort=True):
        if len(g) < 10:
            continue
        rho = g["quali_best_time_s"].corr(g["quali_position"], method="spearman")
        if not np.isfinite(rho) or rho < MIN_ORDER_RHO:
            continue
        g = g.copy()
        g["y"] = 100 * np.log(g["quali_best_time_s"] / g["quali_best_time_s"].min())
        n = len(g)
        q2_cut = 10 + (n - 10) // 2
        pos = g["quali_position"].to_numpy()
        g["stage"] = np.where(pos <= 10, 3, np.where(pos <= q2_cut, 2, 1))
        out.append(g[g["y"] <= MAX_GAP_PCT])
    if not out:
        return q.iloc[:0]
    m = pd.concat(out, ignore_index=True)
    # A 1 % gap means more at a short lap than a long one: scale each session's
    # gaps to its season's typical spread (median of session medians), so track
    # layout does not swing the car estimates. Raises per-session rank
    # correlation 0.63 -> 0.69 on 2021-2026.
    med = m.groupby(["year", "round"])["y"].transform("median")
    season_med = m.assign(med=med).drop_duplicates(["year", "round"]).groupby("year")["med"].median()
    m["y"] = m["y"] * m["year"].map(season_med) / med
    return m


class KalmanStrength:
    """Walks qualifying sessions in order; keeps mean + covariance over all entities."""

    def __init__(self, params: KalmanParams = DEFAULT_PARAMS) -> None:
        self.p = params
        self.index: dict[tuple[str, str], int] = {}     # ("d", "HAM") / ("c", "Mercedes")
        self.mean = np.zeros(0)
        self.cov = np.zeros((0, 0))
        self.year: Optional[int] = None
        self.first_year: Optional[int] = None

    def _add(self, key: tuple[str, str], mean: float, sd: float) -> int:
        i = len(self.mean)
        self.index[key] = i
        self.mean = np.append(self.mean, mean)
        cov = np.zeros((i + 1, i + 1))
        cov[:i, :i] = self.cov
        cov[i, i] = sd ** 2
        self.cov = cov
        return i

    def _idx(self, kind: str, name: str) -> int:
        key = (kind, name)
        if key not in self.index:
            if kind == "d" and self.year == self.first_year:
                return self._add(key, 0.0, self.p.initial_driver_sd)
            if kind == "d":
                return self._add(key, self.p.rookie_mean, self.p.driver_prior_sd)
            return self._add(key, 0.0, self.p.car_prior_sd)
        return self.index[key]

    def _inflate(self, car_sd: float, driver_sd: float) -> None:
        for (kind, _), i in self.index.items():
            self.cov[i, i] += (car_sd if kind == "c" else driver_sd) ** 2

    def advance(self, year: int) -> None:
        """Dynamics for one race weekend (and a season change)."""
        if self.year is not None and year > self.year:
            car = self.p.car_reset_sd if year in REGULATION_RESETS else self.p.car_season_sd
            self._inflate(car, self.p.driver_season_sd)
        self.year = year
        if self.first_year is None:
            self.first_year = year
        self._inflate(self.p.car_drift_sd, self.p.driver_drift_sd)

    def pace(self, driver: str, team: str) -> tuple[float, float]:
        """Mean and sd of car + driver pace (prior entities if unseen)."""
        di, ci = self._idx("d", driver), self._idx("c", team)
        h = np.zeros(len(self.mean))
        h[di] = h[ci] = 1.0
        return float(h @ self.mean), float(np.sqrt(h @ self.cov @ h))

    def get(self, kind: str, name: str) -> tuple[float, float]:
        i = self._idx(kind, name)
        return float(self.mean[i]), float(np.sqrt(self.cov[i, i]))

    def update(self, drivers: list[str], teams: list[str], y: np.ndarray,
               stage: np.ndarray) -> float:
        """Kalman update on one session. Returns the predictive log-likelihood."""
        p = self.p
        di = [self._idx("d", d) for d in drivers]
        ci = [self._idx("c", t) for t in teams]
        n, m = len(y), len(self.mean)
        # augmented state: entities + session offset
        x = np.append(self.mean, 0.0)
        P = np.zeros((m + 1, m + 1))
        P[:m, :m] = self.cov
        P[m, m] = OFFSET_VAR
        H = np.zeros((n, m + 1))
        H[np.arange(n), di] = -1.0
        H[np.arange(n), ci] = -1.0
        H[:, m] = 1.0
        bias = np.where(stage == 2, p.q2_bias, np.where(stage == 1, p.q1_bias, 0.0))
        v = y - bias - H @ x
        HP = H @ P
        S0 = HP @ H.T
        r = np.full(n, p.noise_sd ** 2)
        # session noise scale, then Huber weights on standardised innovations
        z2 = v ** 2 / (S0.diagonal() + r)
        scale = float(np.clip(np.median(z2) / 0.455, 1.0, 25.0))
        r = r * scale
        for _ in range(2):
            z = np.abs(v) / np.sqrt(S0.diagonal() + r)
            w = np.where(z > p.huber_k, (z / p.huber_k) ** 2, 1.0)
            r = p.noise_sd ** 2 * scale * w
        same_team = np.equal.outer(np.asarray(ci), np.asarray(ci)).astype(float)
        S = S0 + np.diag(r) + p.team_session_sd ** 2 * scale * same_team
        L = np.linalg.cholesky(S)
        a = np.linalg.solve(L, v)
        # offset prior is diffuse: drop its constant so sessions of equal size compare
        ll = float(-0.5 * (a @ a) - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)
                   + 0.5 * np.log(OFFSET_VAR))
        K = np.linalg.solve(S, HP).T
        x = x + K @ v
        P = P - K @ HP
        self.mean = x[:m]
        self.cov = 0.5 * (P[:m, :m] + P[:m, :m].T)
        return ll


def run(meas: pd.DataFrame, params: KalmanParams = DEFAULT_PARAMS,
        all_sessions: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, list[tuple[int, int, float]]]:
    """Filter through all sessions in order.

    meas: session_measurements() output. all_sessions (optional, year/round/driver/
    team rows, e.g. raw qualifying or race entries) gets pre-session state too,
    including skipped sessions. Returns per-row pre-session state and
    (year, round, predictive log-lik) per used session.
    """
    rows = all_sessions if all_sessions is not None else meas
    rows = rows.sort_values(["year", "round"], kind="stable")
    used = {k: g for k, g in meas.groupby(["year", "round"], sort=True)}
    kf = KalmanStrength(params)
    rec = {"driver_pace": [], "driver_pace_sd": [], "car_pace": [], "car_pace_sd": [], "pace": [], "pace_sd": []}
    idx_out, lls = [], []
    for (year, rnd), g in rows.groupby(["year", "round"], sort=True):
        kf.advance(int(year))
        for i, r in g.iterrows():
            dm, ds = kf.get("d", str(r["driver"]))
            cm, cs = kf.get("c", str(r["team"]))
            pm, ps = kf.pace(str(r["driver"]), str(r["team"]))
            for k, val in zip(rec, (dm, ds, cm, cs, pm, ps)):
                rec[k].append(val)
            idx_out.append(i)
        s = used.get((year, rnd))
        if s is not None:
            ll = kf.update(s["driver"].astype(str).tolist(), s["team"].astype(str).tolist(),
                           s["y"].to_numpy(float), s["stage"].to_numpy())
            lls.append((int(year), int(rnd), ll))
    return pd.DataFrame(rec, index=idx_out), lls


def pl_loglik(score: np.ndarray, order: np.ndarray) -> float:
    """log P(order) under Plackett-Luce with log-strengths `score` (order: indices, best first)."""
    sc = score[order]
    ll = 0.0
    for j in range(len(sc) - 1):
        r = sc[j:]
        ll += r[0] - (r.max() + np.log(np.exp(r - r.max()).sum()))
    return float(ll)


def order_loglik(meas: pd.DataFrame, state: pd.DataFrame, scale: float, after_year: int) -> float:
    """Summed PL log-lik of each session's qualifying order from the pre-session pace."""
    d = meas.join(state)
    total = 0.0
    for (year, _), g in d.groupby(["year", "round"]):
        if year > after_year:
            total += pl_loglik(scale * g["pace"].to_numpy(float), np.argsort(g["quali_position"].to_numpy(float)))
    return total


def tune(meas: pd.DataFrame, years: tuple[int, int] = (2014, 2020),
         start: KalmanParams = DEFAULT_PARAMS, maxiter: int = 400, restarts: int = 1,
         objective: str = "order") -> tuple[KalmanParams, float]:
    """Tune on `years` (Nelder-Mead, log scale), first season excluded as burn-in.

    objective "order": PL log-lik of the next qualifying order (what the filter is
    used for; a fitted scale is a nuisance parameter). "gap": predictive likelihood
    of the gaps themselves, which on 2021-2026 ranked drivers worse (Spearman 0.61
    vs 0.69).
    """
    from scipy.optimize import minimize

    m = meas[(meas["year"] >= years[0]) & (meas["year"] <= years[1])]
    start = replace(start, team_session_sd=max(start.team_session_sd, 0.05))
    names = [k for k in asdict(start) if k not in ("huber_k", "rookie_mean")]
    x0 = np.append(np.log([getattr(start, k) for k in names]), [start.rookie_mean, 0.0])

    def unpack(x: np.ndarray) -> KalmanParams:
        return replace(start, **{k: float(np.exp(v)) for k, v in zip(names, x[:-2])}, rookie_mean=float(x[-2]))

    def f(x: np.ndarray) -> float:
        state, lls = run(m, unpack(x))
        if objective == "gap":
            return -sum(ll for y, _, ll in lls if y > years[0])
        return -order_loglik(m, state, float(np.exp(x[-1])), years[0])

    for _ in range(restarts):      # Nelder-Mead in 14 dims stalls; restart from the optimum
        res = minimize(f, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 0.05})
        x0 = res.x
    return unpack(res.x), float(-res.fun)

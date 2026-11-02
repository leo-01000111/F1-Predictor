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
with driver_prior_sd, new teams with car_prior_sd. Both are anchored to the current
field (gaps fix only differences, so the overall level drifts): a new team starts at
the mean of the cars active this or last season, a rookie at the active drivers'
mean + rookie_mean. Drivers seen in the first season of the data are not rookies:
they start at the field mean with initial_driver_sd.

Race update (phase B, race_scale > 0): the classified finishing order is a
Plackett-Luce observation with log-strengths race_scale * (c + d). It is not
Gaussian, so the update is a Laplace step at the prior mean (gradient and Hessian
of the PL log-likelihood), as in TrueSkill / Glicko. Retirements are left out.

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
SD_FLOOR = 0.01               # % of lap time; tuning may not push an sd below this


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
    race_scale: float = 0.0        # PL log-strength per % of pace in the race update (0 = off)
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
        self.last_seen: dict[tuple[str, str], int] = {}

    def _add(self, key: tuple[str, str], mean: float, sd: float) -> int:
        i = len(self.mean)
        self.index[key] = i
        self.mean = np.append(self.mean, mean)
        cov = np.zeros((i + 1, i + 1))
        cov[:i, :i] = self.cov
        cov[i, i] = sd ** 2
        self.cov = cov
        return i

    def _field_mean(self, kind: str) -> float:
        """Mean of the active (seen this or last season) drivers or cars."""
        act = [i for key, i in self.index.items()
               if key[0] == kind and self.last_seen.get(key, -1) >= (self.year or 0) - 1]
        return float(self.mean[act].mean()) if act else 0.0

    def _idx(self, kind: str, name: str) -> int:
        key = (kind, name)
        if key not in self.index:
            base = self._field_mean(kind)
            if kind == "d" and self.year == self.first_year:
                i = self._add(key, base, self.p.initial_driver_sd)
            elif kind == "d":
                i = self._add(key, base + self.p.rookie_mean, self.p.driver_prior_sd)
            else:
                i = self._add(key, base, self.p.car_prior_sd)
        else:
            i = self.index[key]
        if self.year is not None:
            self.last_seen[key] = self.year
        return i

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

    def update_race(self, drivers: list[str], teams: list[str]) -> float:
        """Laplace update on a finishing order (drivers listed winner first).

        Returns the plug-in PL log-likelihood of the order at the prior mean.
        """
        n, b = len(drivers), self.p.race_scale
        if n < 2 or b <= 0:
            return 0.0
        di = [self._idx("d", d) for d in drivers]
        ci = [self._idx("c", t) for t in teams]
        m = len(self.mean)
        H = np.zeros((n, m))
        H[np.arange(n), di] += b
        H[np.arange(n), ci] += b
        s = H @ self.mean
        g = np.zeros(n)
        A = np.zeros((n, n))      # minus the Hessian of the PL log-likelihood in s
        ll = 0.0
        for j in range(n - 1):
            r = s[j:]
            pj = np.exp(r - r.max())
            pj /= pj.sum()
            ll += np.log(pj[0])
            g[j] += 1.0
            g[j:] -= pj
            A[j:, j:] += np.diag(pj) - np.outer(pj, pj)
        # (P^-1 + H'AH)^-1 = P - PH' (I + A H P H')^-1 A H P   (A is singular: shift invariance)
        PH = self.cov @ H.T
        M = np.linalg.solve(np.eye(n) + A @ (H @ PH), A @ PH.T)
        cov = self.cov - PH @ M
        self.cov = 0.5 * (cov + cov.T)
        self.mean = self.mean + self.cov @ (H.T @ g)
        return float(ll)


def run(meas: pd.DataFrame, params: KalmanParams = DEFAULT_PARAMS,
        all_sessions: Optional[pd.DataFrame] = None,
        races: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, list[tuple[int, int, float, float]]]:
    """Filter through all weekends in order: qualifying update, then race update.

    meas: session_measurements() output. all_sessions (optional, year/round/driver/
    team rows, e.g. raw qualifying or race entries) gets pre-weekend state too,
    including skipped sessions; rows with src == "r" also get `pace_post_quali`
    (after this weekend's qualifying, before the race). races (optional, raw race
    results) feed the race update when params.race_scale > 0. Returns per-row
    state and (year, round, quali log-lik, race log-lik) per weekend.
    """
    rows = all_sessions if all_sessions is not None else meas
    rows = rows.sort_values(["year", "round"], kind="stable")
    used = {k: g for k, g in meas.groupby(["year", "round"], sort=True)}
    finishers = {}
    if races is not None and params.race_scale > 0:
        ok = races[pd.to_numeric(races["dnf"], errors="coerce").fillna(0) == 0].copy()
        ok["_fin"] = pd.to_numeric(ok["finish_position"], errors="coerce")
        ok = ok.dropna(subset=["_fin"]).sort_values(["year", "round", "_fin"])
        finishers = {k: g for k, g in ok.groupby(["year", "round"], sort=True)}
    by_key = {k: g for k, g in rows.groupby(["year", "round"], sort=True)}
    weekends = sorted(set(by_key) | set(used) | set(finishers))
    kf = KalmanStrength(params)
    post = {}
    rec = {"driver_pace": [], "driver_pace_sd": [], "car_pace": [], "car_pace_sd": [], "pace": [], "pace_sd": []}
    idx_out, lls = [], []
    for year, rnd in weekends:
        g = by_key.get((year, rnd), rows.iloc[:0])
        kf.advance(int(year))
        for i, r in g.iterrows():
            dm, ds = kf.get("d", str(r["driver"]))
            cm, cs = kf.get("c", str(r["team"]))
            pm, ps = kf.pace(str(r["driver"]), str(r["team"]))
            for k, val in zip(rec, (dm, ds, cm, cs, pm, ps)):
                rec[k].append(val)
            idx_out.append(i)
        s = used.get((year, rnd))
        ll_q = ll_r = 0.0
        if s is not None:
            ll_q = kf.update(s["driver"].astype(str).tolist(), s["team"].astype(str).tolist(),
                             s["y"].to_numpy(float), s["stage"].to_numpy())
        if "src" in g.columns:
            for i, r in g[g["src"] == "r"].iterrows():
                post[i] = kf.pace(str(r["driver"]), str(r["team"]))[0]
        f = finishers.get((year, rnd))
        if f is not None:
            ll_r = kf.update_race(f["driver"].astype(str).tolist(), f["team"].astype(str).tolist())
        if s is not None or f is not None:
            lls.append((int(year), int(rnd), ll_q, ll_r))
    out = pd.DataFrame(rec, index=idx_out)
    if post:
        out["pace_post_quali"] = pd.Series(post)
    return out, lls


KALMAN_COLS = ["kalman_pace_postq", "kalman_pace_sd", "kalman_car", "kalman_driver"]


def kalman_features(quali: pd.DataFrame, race: pd.DataFrame,
                    params: KalmanParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Per race entry (year, round, driver): filter state known when the race is predicted.

    kalman_pace_postq is car + driver pace after this weekend's qualifying (before
    the race); the other columns are as of before the weekend.
    """
    meas = session_measurements(quali)
    rows = race[["year", "round", "driver", "team"]].reset_index(drop=True).assign(src="r")
    state, _ = run(meas, params, all_sessions=rows, races=race)
    out = rows.join(state)
    out = out.rename(columns={"pace_post_quali": "kalman_pace_postq", "pace_sd": "kalman_pace_sd",
                              "car_pace": "kalman_car", "driver_pace": "kalman_driver"})
    return out[["year", "round", "driver"] + KALMAN_COLS].drop_duplicates(["year", "round", "driver"])


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
         objective: str = "order", races: Optional[pd.DataFrame] = None) -> tuple[KalmanParams, float]:
    """Tune on `years` (Nelder-Mead, log scale), first season excluded as burn-in.

    objective "order": PL log-lik of the next qualifying order (what the filter is
    used for; a fitted scale is a nuisance parameter), plus, with `races`, the
    log-lik of each race's finishing order from the post-qualifying state (scale =
    race_scale, which is tuned too). "gap": predictive likelihood of the gaps
    themselves, which on 2021-2026 ranked drivers worse (Spearman 0.61 vs 0.69).
    Standard deviations are floored at SD_FLOOR (phase A drove some to ~0).
    """
    from scipy.optimize import minimize

    m = meas[(meas["year"] >= years[0]) & (meas["year"] <= years[1])]
    rc = None if races is None else races[(races["year"] >= years[0]) & (races["year"] <= years[1])]
    start = replace(start, team_session_sd=max(start.team_session_sd, 0.05),
                    race_scale=max(start.race_scale, 0.5) if rc is not None else 0.0)
    skip = ("huber_k", "rookie_mean") + (("race_scale",) if rc is None else ())
    names = [k for k in asdict(start) if k not in skip]
    x0 = np.append(np.log([max(getattr(start, k), SD_FLOOR) for k in names]), [start.rookie_mean, 0.0])

    def unpack(x: np.ndarray) -> KalmanParams:
        vals = {k: float(max(np.exp(v), SD_FLOOR)) for k, v in zip(names, x[:-2])}
        return replace(start, **vals, rookie_mean=float(x[-2]))

    def f(x: np.ndarray) -> float:
        state, lls = run(m, unpack(x), races=rc)
        if objective == "gap":
            return -sum(ll for y, _, ll, _r in lls if y > years[0])
        race_ll = sum(r for y, _, _q, r in lls if y > years[0])
        return -(order_loglik(m, state, float(np.exp(x[-1])), years[0]) + race_ll)

    for _ in range(restarts):      # Nelder-Mead in 14 dims stalls; restart from the optimum
        res = minimize(f, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 0.05})
        x0 = res.x
    return unpack(res.x), float(-res.fun)

"""
Championship simulation (ROADMAP Phase 5).

Plays out the rest of a season many times and counts how often each driver and
team ends up champion.

  1. Standings now: race points + sprint points from the raw data (matches the
     official table, checked against Jolpica after 2026 R18).
  2. Future races have no qualifying, so a pre-qualifying two-stage model
     (PREQUALI_COLS: no grid, quali gaps, weather or sprint inputs) predicts them.
     Their features are built by the normal pipeline from synthetic rows:
     "state" features (form, standings, reliability, Elo) as of the last completed
     race, circuit features (history at that track, street, safety-car rate) per race.
  3. Each simulated season draws one form shock per team (normal, sd team_sigma)
     that holds for all its remaining races: a car that is quick in one race is
     likely quick in the next. Each race: retirements from P(finish), finishers
     ordered by Plackett-Luce with the shocked strengths, points by position.
     Sprint weekends get a second, shorter draw with sprint points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLS, build_feature_matrix

QUALI_COLS = ["grid_position", "grid_position_rel", "quali_gap_relative", "teammate_quali_delta",
              "car_pace_delta_pct", "grid_penalty_places", "circuit_grid_conversion", "sprint_position_rel"]
WEATHER_COLS = ["temp_c_mean", "precip_mm_total", "windspeed_ms_mean", "humidity_pct_mean", "is_wet_race"]
PREQUALI_COLS = [c for c in FEATURE_COLS if c not in QUALI_COLS + WEATHER_COLS]
# Direction each pre-qualifying input may act in (+1: higher value -> faster).
# Without these the ranker learned reversed effects among front-runners (2019:
# Hamilton's best-in-field form lowered his score), which the grid masks in the
# full model but which dominate a season simulation.
PREQUALI_MONOTONE = {
    "driver_form_5": -1, "driver_form_10": -1, "driver_form_ewm": -1, "circuit_avg_finish": -1,
    "championship_position": -1, "constructor_standing": -1, "dnf_rate": -1,
    "driver_wet_avg_finish": -1, "team_form_5": -1,
    "driver_elo": 1, "cum_points_before": 1, "team_cum_points": 1, "team_reliability_rate": 1,
    "circuit_podiums": 1, "driver_experience_norm": 1,
}


def prequali_params() -> dict:
    from src.models.xgb_model import XGBRaceRanker
    return dict(XGBRaceRanker.DEFAULT_PARAMS,
                monotone_constraints={c: v for c, v in PREQUALI_MONOTONE.items() if c in PREQUALI_COLS})


# Features that depend on the circuit of each future race; everything else is
# the state after the last completed race
CIRCUIT_COLS = ["circuit_avg_finish", "circuit_podiums", "circuit_appearances",
                "is_street_circuit", "circuit_safety_car_rate", "round_fraction"]

RACE_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


def sprint_points(year: int) -> np.ndarray:
    return np.array([3, 2, 1], float) if year <= 2021 else np.arange(8, 0, -1, dtype=float)


def standings(race_df: pd.DataFrame, sprint_df: Optional[pd.DataFrame], year: int,
              through_round: int) -> tuple[pd.Series, pd.Series]:
    """Driver and team points after `through_round` (races + sprints)."""
    from src.features.teams import canonicalize_teams

    r = canonicalize_teams(race_df[(race_df["year"] == year) & (race_df["round"] <= through_round)])
    rows = r[["round", "driver", "team", "points"]].copy()
    if sprint_df is not None and not sprint_df.empty:
        s = sprint_df[(sprint_df["year"] == year) & (sprint_df["round"] <= through_round)].copy()
        pts = sprint_points(year)
        s["points"] = s["sprint_position"].map(lambda p: pts[int(p) - 1] if pd.notna(p) and int(p) <= len(pts) else 0.0)
        s = s.merge(r[["round", "driver", "team"]], on=["round", "driver"], how="left")
        rows = pd.concat([rows, s[["round", "driver", "team", "points"]]], ignore_index=True)
    rows["points"] = pd.to_numeric(rows["points"], errors="coerce").fillna(0.0)
    return rows.groupby("driver")["points"].sum(), rows.groupby("team")["points"].sum()


def future_features(race_df: pd.DataFrame, quali_df: pd.DataFrame, pit_df: Optional[pd.DataFrame],
                    sprint_df: Optional[pd.DataFrame], year: int, after_round: int,
                    remaining: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for the remaining races of `year`, as known after `after_round`.

    remaining: one row per future race with round, circuit, circuit_key, race_date.
    The lineup is the one of the last completed race.
    """
    done = (race_df["year"] < year) | ((race_df["year"] == year) & (race_df["round"] <= after_round))
    history = race_df[done]
    lineup = history[(history["year"] == year) & (history["round"] == after_round)][["driver", "team"]]
    synth = []
    for _, race in remaining.iterrows():
        rows = lineup.copy()
        rows["year"], rows["round"] = year, int(race["round"])
        rows["circuit"], rows["circuit_key"], rows["race_date"] = race["circuit"], race["circuit_key"], race["race_date"]
        rows["grid_position"] = np.nan
        rows["finish_position"] = np.nan
        rows["dnf"], rows["points"], rows["status"] = 0, 0.0, ""
        for c in ("p1", "p2", "p3", "podium"):
            rows[c] = 0
        synth.append(rows)
    combined = pd.concat([history, *synth], ignore_index=True)
    q_done = (quali_df["year"] < year) | ((quali_df["year"] == year) & (quali_df["round"] <= after_round))
    s_df = None
    if sprint_df is not None:
        s_df = sprint_df[(sprint_df["year"] < year) | ((sprint_df["year"] == year) & (sprint_df["round"] <= after_round))]
    fm = build_feature_matrix(combined, quali_df[q_done], None, pit_df, s_df)
    fut = fm[(fm["year"] == year) & (fm["round"] > after_round)].copy()

    # State features: as of the first future race (only completed races feed them)
    first = fut[fut["round"] == fut["round"].min()].set_index("driver")
    state_cols = [c for c in PREQUALI_COLS if c not in CIRCUIT_COLS]
    for c in state_cols:
        fut[c] = fut["driver"].map(first[c])
    return fut.sort_values(["round", "driver"]).reset_index(drop=True)


@dataclass
class SeasonResult:
    drivers: pd.DataFrame     # driver, team, points_now, p_champion, expected_points, p_position_k...
    teams: pd.DataFrame       # team, points_now, p_champion, expected_points


def simulate_season(model, future: pd.DataFrame, driver_pts: pd.Series, team_pts: pd.Series,
                    sprint_rounds: set[int], year: int, team_sigma: float = 0.3,
                    n_sims: int = 10_000, seed: Optional[int] = 0) -> SeasonResult:
    """Monte Carlo of the remaining races. `model` is a fitted TwoStageRaceModel on
    PREQUALI_COLS (its predict() already includes the temperature)."""
    rng = np.random.default_rng(seed)
    drivers = sorted(future["driver"].unique())
    d_idx = {d: i for i, d in enumerate(drivers)}
    team_of = future.drop_duplicates("driver").set_index("driver")["team"]
    teams = sorted(set(team_of) | set(team_pts.index))
    t_idx = {t: i for i, t in enumerate(teams)}
    drv_team = np.array([t_idx[team_of[d]] for d in drivers])
    n = len(drivers)

    pts = np.tile(np.array([driver_pts.get(d, 0.0) for d in drivers]), (n_sims, 1))
    team_now = np.array([team_pts.get(t, 0.0) for t in teams])
    gained = np.zeros((n_sims, n))
    shocks = rng.normal(0.0, team_sigma, size=(n_sims, len(teams)))[:, drv_team]
    sp = sprint_points(year)

    for rnd, race in future.groupby("round"):
        race = race.set_index("driver").reindex(drivers)
        X = race[PREQUALI_COLS].astype(float).fillna(0.0)
        s = -model.predict(X)
        q = np.clip(model.predict_finish(X), 1e-3, 1.0)
        for kind in (["race", "sprint"] if int(rnd) in sprint_rounds else ["race"]):
            table = RACE_POINTS if kind == "race" else sp
            scores = s[None, :] + shocks + rng.gumbel(size=(n_sims, n))
            finished = rng.random((n_sims, n)) < q
            scores = np.where(finished, scores, -1e9 + rng.random((n_sims, n)))
            order = np.argsort(-scores, axis=1)[:, :len(table)]
            for k, p in enumerate(table):
                np.add.at(gained, (np.arange(n_sims), order[:, k]), p)

    final = pts + gained
    final_tb = final + rng.random(final.shape) * 1e-6          # random tie-break
    champ = np.bincount(final_tb.argmax(axis=1), minlength=n) / n_sims
    ranks = (-final_tb).argsort(axis=1).argsort(axis=1)         # 0 = first
    team_final = np.tile(team_now, (n_sims, 1))
    for i in range(n):
        team_final[:, drv_team[i]] += gained[:, i]
    team_champ = np.bincount((team_final + rng.random(team_final.shape) * 1e-6).argmax(axis=1),
                             minlength=len(teams)) / n_sims

    d = pd.DataFrame({
        "driver": drivers, "team": [team_of[x] for x in drivers],
        "points_now": [driver_pts.get(x, 0.0) for x in drivers],
        "p_champion": champ, "expected_points": final.mean(axis=0),
        "p_top3": (ranks < 3).mean(axis=0),
        "p_position": [np.bincount(ranks[:, i], minlength=n) / n_sims for i in range(n)],
    }).sort_values("expected_points", ascending=False).reset_index(drop=True)
    t = pd.DataFrame({
        "team": teams, "points_now": team_now, "p_champion": team_champ,
        "expected_points": team_final.mean(axis=0),
    }).sort_values("expected_points", ascending=False).reset_index(drop=True)
    return SeasonResult(drivers=d, teams=t)

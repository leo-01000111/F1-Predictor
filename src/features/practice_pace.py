"""
Practice long-run pace (ROADMAP idea 2), from OpenF1 practice laps (2023 on).

Friday long runs on race fuel show race pace in a way a single qualifying lap
does not. Per practice session:

  long run   a stint with >= MIN_LAPS clean laps: timed, not an out-lap, not
             the stint's first lap, within 107% of the stint median
  pace       the median of those clean laps
  gap        driver's best long run vs the median long run of the session, in %
             (negative = faster than the field); clipped to +-CLIP_PCT

Features per (year, round, driver), averaged over the weekend's sessions:
  fp_longrun_gap_pct     the driver's own long-run gap
  team_longrun_gap_pct   the faster of the team's two cars (race pace is mostly car)
  fp_longrun_laps        clean long-run laps the driver did (how much evidence)

Seasons before 2023 have no data (NaN); XGBoost treats that as missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LAPS = 5
CLIP_PCT = 5.0
PACE_COLS = ["fp_longrun_gap_pct", "team_longrun_gap_pct"]


def stint_paces(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per long-run stint: year, round, session, driver, compound, pace_s, n_laps."""
    df = laps.dropna(subset=["lap_duration", "stint_number"]).copy()
    df = df[~df["is_pit_out_lap"]]
    first = df.groupby(["year", "round", "session", "driver", "stint_number"])["lap_number"].transform("min")
    df = df[df["lap_number"] > first]
    key = ["year", "round", "session", "driver", "stint_number"]
    med = df.groupby(key)["lap_duration"].transform("median")
    df = df[df["lap_duration"] <= med * 1.07]
    out = df.groupby(key).agg(pace_s=("lap_duration", "median"), n_laps=("lap_duration", "size"),
                              compound=("compound", "first")).reset_index()
    return out[out["n_laps"] >= MIN_LAPS]


def practice_pace(laps: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Weekend long-run features per (year, round, driver).

    teams: year, round, driver, team (the race entry list) for the team feature.
    """
    cols = ["year", "round", "driver", "fp_longrun_gap_pct", "team_longrun_gap_pct", "fp_longrun_laps"]
    if laps is None or laps.empty:
        return pd.DataFrame(columns=cols)
    st = stint_paces(laps)
    if st.empty:
        return pd.DataFrame(columns=cols)
    best = st.groupby(["year", "round", "session", "driver"]).agg(
        pace_s=("pace_s", "min"), n_laps=("n_laps", "sum")).reset_index()
    field = best.groupby(["year", "round", "session"])["pace_s"].transform("median")
    best["gap"] = ((best["pace_s"] / field - 1) * 100).clip(-CLIP_PCT, CLIP_PCT)
    # Sessions with only a handful of long runs say little about the field
    n_runs = best.groupby(["year", "round", "session"])["driver"].transform("size")
    best = best[n_runs >= 6]
    drv = best.groupby(["year", "round", "driver"]).agg(
        fp_longrun_gap_pct=("gap", "mean"), fp_longrun_laps=("n_laps", "sum")).reset_index()

    t = teams[["year", "round", "driver", "team"]].drop_duplicates(["year", "round", "driver"])
    best = best.merge(t, on=["year", "round", "driver"], how="left")
    team_sess = best.groupby(["year", "round", "session", "team"])["gap"].min().reset_index()
    team = team_sess.groupby(["year", "round", "team"])["gap"].mean().rename("team_longrun_gap_pct").reset_index()
    out = t.merge(drv, on=["year", "round", "driver"], how="left").merge(team, on=["year", "round", "team"], how="left")
    out = out[out[["fp_longrun_gap_pct", "team_longrun_gap_pct"]].notna().any(axis=1)]
    return out[cols]


def add_practice_pace(df: pd.DataFrame, laps: pd.DataFrame | None) -> pd.DataFrame:
    """Merge the practice features into a race-level frame (NaN where no data)."""
    df = df.drop(columns=[c for c in ["fp_longrun_gap_pct", "team_longrun_gap_pct", "fp_longrun_laps"]
                          if c in df.columns])
    if laps is None or laps.empty:
        for c in ["fp_longrun_gap_pct", "team_longrun_gap_pct", "fp_longrun_laps"]:
            df[c] = np.nan
        return df
    pace = practice_pace(laps, df)
    return df.merge(pace, on=["year", "round", "driver"], how="left")

"""Team/constructor-level features."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_constructor_standing(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative constructor points and position BEFORE each race.
    """
    df = race_df.copy().sort_values(["year", "round", "team"])

    # Sum points per team per race (both drivers)
    team_race_points = (
        df.groupby(["year", "round", "team"])["points"].sum().reset_index()
        .rename(columns={"points": "team_race_points"})
        .sort_values(["year", "round"])
    )

    team_race_points["team_cum_points"] = (
        team_race_points.groupby(["year", "team"])["team_race_points"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    team_race_points["constructor_standing"] = (
        team_race_points.groupby(["year", "round"])["team_cum_points"]
        .rank(ascending=False, method="min")
    )

    df = df.merge(
        team_race_points[["year", "round", "team", "constructor_standing", "team_cum_points"]],
        on=["year", "round", "team"],
        how="left",
    )

    return df


def add_team_reliability(race_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Team-level rolling DNF rate across both drivers.
    team_reliability_rate = 1 - rolling DNF rate
    """
    df = race_df.copy().sort_values(["team", "year", "round"])

    # Compute team DNF rate (by race)
    team_dnf = (
        df.groupby(["year", "round", "team"])["dnf"]
        .mean()
        .reset_index()
        .rename(columns={"dnf": "team_dnf_rate_race"})
        .sort_values(["team", "year", "round"])
    )

    team_dnf["team_reliability_rate"] = 1.0 - (
        team_dnf.groupby("team")["team_dnf_rate_race"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )

    df = df.merge(
        team_dnf[["year", "round", "team", "team_reliability_rate"]],
        on=["year", "round", "team"],
        how="left",
    )

    return df


def add_car_pace_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw constructor pace delta from qualifying.

    Computes per-constructor median qualifying gap to pole for the race weekend,
    then expresses each constructor's median as a percentage gap to the fastest
    constructor's median. Isolates car pace from driver skill.

    Formula: (team_median_quali - fastest_team_median) / fastest_team_median * 100
    Clipped at 3.0% to prevent outlier contamination.

    Requires 'team' and 'quali_gap_to_pole_s' columns (after quali merge).
    """
    if "quali_gap_to_pole_s" not in df.columns or "team" not in df.columns:
        out = df.copy()
        out["car_pace_delta_pct"] = 0.0
        return out

    out = df.copy()

    # Avoid groupby().apply() — its return type changed in pandas 2.x.
    # Use explicit aggregation + merge instead.
    team_med = (
        out.groupby(["year", "round", "team"])["quali_gap_to_pole_s"]
        .median()
        .reset_index()
        .rename(columns={"quali_gap_to_pole_s": "_team_med"})
    )
    race_fastest = (
        team_med.groupby(["year", "round"])["_team_med"]
        .min()
        .reset_index()
        .rename(columns={"_team_med": "_fastest"})
    )
    team_med = team_med.merge(race_fastest, on=["year", "round"])
    denom = team_med["_fastest"].clip(lower=0.01)
    team_med["car_pace_delta_pct"] = (
        ((team_med["_team_med"] - team_med["_fastest"]) / denom * 100)
        .clip(lower=0.0, upper=3.0)
        .fillna(0.0)
    )
    out = out.merge(
        team_med[["year", "round", "team", "car_pace_delta_pct"]],
        on=["year", "round", "team"],
        how="left",
    )
    out["car_pace_delta_pct"] = out["car_pace_delta_pct"].fillna(0.0)
    return out


def add_team_rolling_form(race_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Rolling mean of a team's best finish position over the last N races.

    team_form_5: rolling best-finish mean (lower = better car/team).
    Uses shift(1) so the current race is not included.
    """
    df = race_df.copy()

    team_best = (
        df.groupby(["year", "round", "team"])["finish_position"]
        .min()
        .reset_index()
        .rename(columns={"finish_position": "_team_best"})
        .sort_values(["team", "year", "round"])
    )

    col = f"team_form_{window}"
    team_best[col] = (
        team_best.groupby("team")["_team_best"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df = df.merge(team_best[["year", "round", "team", col]], on=["year", "round", "team"], how="left")
    return df


def add_pit_stop_performance(
    race_df: pd.DataFrame,
    pit_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Team pit stop performance delta vs the field.

    pit_df should have columns: year, round, driver, avg_pit_s
    (produced by src.data_collection.f1_historical.collect_pit_data).

    Computes team median pit time and field median pit time per race.
    team_pit_delta_s = team_median - field_median (negative = faster).

    If pit_df is not provided, returns a NaN placeholder to keep the schema.
    """
    df = race_df.copy()

    if pit_df is None or pit_df.empty:
        df["team_pit_delta_s"] = np.nan
        return df

    pit_sub = pit_df[["year", "round", "driver", "avg_pit_s"]].copy()
    pit_sub["avg_pit_s"] = pd.to_numeric(pit_sub["avg_pit_s"], errors="coerce")

    df = df.merge(pit_sub, on=["year", "round", "driver"], how="left")

    field_median = df.groupby(["year", "round"])["avg_pit_s"].transform("median")
    team_median = df.groupby(["year", "round", "team"])["avg_pit_s"].transform("median")
    df["team_pit_delta_s"] = team_median - field_median

    df = df.drop(columns=["avg_pit_s"], errors="ignore")
    return df

"""
Driver-level feature engineering.
All features are computed using only data available BEFORE the race
(no target leakage).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def add_driver_form(race_df: pd.DataFrame, windows: list[int] = [5, 10]) -> pd.DataFrame:
    """
    Rolling mean finish position over last N races per driver.
    Lower = better (P1 is best).
    Uses a shift(1) so the current race is never included.
    Also adds driver_form_ewm: exponential weighted mean (span=5) so the
    most recent race counts ~4x more than a race 5 rounds ago.
    """
    df = race_df.copy().sort_values(["driver", "year", "round"])

    for w in windows:
        df[f"driver_form_{w}"] = (
            df.groupby("driver")["finish_position"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    # Exponential weighted form: span=5 means the last race gets ~33% of the
    # total weight, vs 20% in a simple 5-race rolling mean. Captures momentum.
    df["driver_form_ewm"] = (
        df.groupby("driver")["finish_position"]
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    return df


def add_championship_standing(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative WDC points and position BEFORE each race (using shift).
    """
    df = race_df.copy().sort_values(["year", "round", "driver"])

    # Cumulative points before this race
    df["cum_points_before"] = (
        df.groupby(["year", "driver"])["points"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    # Championship position (rank within year, before this race)
    df["championship_position"] = (
        df.groupby(["year", "round"])["cum_points_before"]
        .rank(ascending=False, method="min")
    )

    return df


def add_dnf_rate(race_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Rolling DNF rate (fraction of last N races where driver retired).
    """
    df = race_df.copy().sort_values(["driver", "year", "round"])
    df["dnf_rate"] = (
        df.groupby("driver")["dnf"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    return df


def add_circuit_history(
    race_df: pd.DataFrame,
    circuit_history_seasons: int = 5,
) -> pd.DataFrame:
    """
    Per-driver, per-circuit historical performance features:
    - circuit_avg_finish: average finish at this circuit (past N seasons)
    - circuit_podiums: number of podiums at this circuit (past N seasons)
    - circuit_appearances: number of starts at this circuit

    Complexity is O(N) over rows (per-group streaming window), avoiding the
    previous O(N^2) full-frame filtering in each iteration.
    """
    df = race_df.copy().sort_values(["circuit_key", "driver", "year", "round"])

    avg_finishes = pd.Series(np.nan, index=df.index, dtype=float)
    podium_counts = pd.Series(0, index=df.index, dtype=int)
    appearances = pd.Series(0, index=df.index, dtype=int)

    season_window = int(circuit_history_seasons) if circuit_history_seasons else 0

    for _, grp in df.groupby(["circuit_key", "driver"], sort=False):
        # (year, finish_position_or_nan, podium_or_zero)
        hist: deque[tuple[int, float, float]] = deque()
        finish_sum = 0.0
        finish_count = 0
        podium_sum = 0.0

        for idx, row in grp.iterrows():
            current_year = int(row["year"])
            cutoff_year = current_year - season_window if season_window > 0 else None

            # Evict rows older than the configured season window.
            while hist and cutoff_year is not None and hist[0][0] < cutoff_year:
                old_year, old_finish, old_podium = hist.popleft()
                _ = old_year
                if pd.notna(old_finish):
                    finish_sum -= float(old_finish)
                    finish_count -= 1
                podium_sum -= float(old_podium)

            n_hist = len(hist)
            appearances.at[idx] = n_hist
            avg_finishes.at[idx] = (finish_sum / finish_count) if finish_count > 0 else np.nan
            podium_counts.at[idx] = int(round(podium_sum))

            finish_raw = row.get("finish_position")
            podium_raw = row.get("podium")
            finish_val = float(finish_raw) if pd.notna(finish_raw) else np.nan
            podium_val = float(podium_raw) if pd.notna(podium_raw) else 0.0

            hist.append((current_year, finish_val, podium_val))
            if pd.notna(finish_val):
                finish_sum += finish_val
                finish_count += 1
            podium_sum += podium_val

    df["circuit_avg_finish"] = avg_finishes
    df["circuit_podiums"] = podium_counts
    df["circuit_appearances"] = appearances

    return df


def add_driver_start_delta(race_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Rolling average of grid places gained or lost at race start (end of lap 1).

    Positive value = gained places (good starter).
    Negative value = lost places (poor starter).

    Formula: grid_position - lap1_position  (higher grid_pos = further back on grid)

    Requires 'lap1_position' column (position at end of lap 1) in the DataFrame.
    If that column is absent — which is the case until historical lap data is
    backfilled — the feature defaults to 0.0 for all drivers.

    Uses a rolling 10-race mean with min_periods=3 (same window as driver_form_10).
    Drivers with fewer than 3 races default to 0.0.
    """
    df = race_df.copy().sort_values(["driver", "year", "round"])

    if "lap1_position" not in df.columns:
        df["driver_start_delta_avg"] = 0.0
        return df

    df["_start_delta"] = (
        pd.to_numeric(df["grid_position"], errors="coerce")
        - pd.to_numeric(df["lap1_position"], errors="coerce")
    )

    df["driver_start_delta_avg"] = (
        df.groupby("driver")["_start_delta"]
        .transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean().fillna(0.0)
        )
    )
    df = df.drop(columns=["_start_delta"])
    return df


def add_elo_ratings(
    race_df: pd.DataFrame,
    k_base: float = 32.0,
    initial_elo: float = 1500.0,
    off_season_decay: float = 0.85,
) -> pd.DataFrame:
    """
    Pre-race driver Elo rating from pairwise head-to-head results.

    - K per pair = k_base / (n_finishers - 1): total max gain per race = k_base
    - DNF drivers excluded from pairings
    - Off-season: Elo decays 15% toward initial_elo between seasons
    - driver_elo is recorded BEFORE the race (no leakage)
    """
    df = race_df.copy().sort_values(["year", "round"])

    elo: dict[str, float] = {}
    # {(year, round, driver) -> pre-race elo}
    pre_race: dict[tuple[int, int, str], float] = {}
    prev_year: int | None = None

    for (year, rnd), grp in df.groupby(["year", "round"], sort=True):
        # Off-season decay when year changes
        if prev_year is not None and year > prev_year:
            for d in list(elo):
                elo[d] = elo[d] * off_season_decay + initial_elo * (1 - off_season_decay)

        # Initialise new drivers
        for d in grp["driver"].unique():
            if d not in elo:
                elo[d] = initial_elo

        # Record pre-race Elo for every driver in this race
        for d in grp["driver"].unique():
            pre_race[(year, rnd, d)] = elo[d]

        # Compute updates from non-DNF finishers only
        finishers = grp[grp["dnf"] == 0].dropna(subset=["finish_position"]).copy()
        finishers = finishers.sort_values("finish_position")
        driver_list = finishers["driver"].tolist()
        n = len(driver_list)

        if n >= 2:
            K = k_base / (n - 1)
            finish_lookup = dict(zip(finishers["driver"], finishers["finish_position"]))
            delta: dict[str, float] = {d: 0.0 for d in driver_list}

            for i in range(n):
                for j in range(i + 1, n):
                    d_i, d_j = driver_list[i], driver_list[j]
                    e_i = 1.0 / (1.0 + 10.0 ** ((elo[d_j] - elo[d_i]) / 400.0))
                    fi = finish_lookup[d_i]
                    fj = finish_lookup[d_j]
                    act_i = 1.0 if fi < fj else (0.5 if fi == fj else 0.0)
                    delta[d_i] += K * (act_i - e_i)
                    delta[d_j] += K * ((1.0 - act_i) - (1.0 - e_i))

            for d, dv in delta.items():
                elo[d] = elo[d] + dv

        prev_year = year

    df["driver_elo"] = df.apply(
        lambda r: pre_race.get((r["year"], r["round"], r["driver"]), initial_elo),
        axis=1,
    )
    return df


def add_wet_performance(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling average finish position in confirmed wet races (is_wet_race == 1).

    Uses all prior wet races via expanding window (min_periods=3) with shift(1).
    Requires 'is_wet_race' column; returns NaN placeholder column if absent.
    Call AFTER weather has been merged into race_df.
    """
    df = race_df.copy().sort_values(["driver", "year", "round"])

    if "is_wet_race" not in df.columns:
        df["driver_wet_avg_finish"] = np.nan
        return df

    # Only non-null wet race finishes contribute
    df["_wet_finish"] = df["finish_position"].where(df["is_wet_race"] == 1)

    df["driver_wet_avg_finish"] = (
        df.groupby("driver")["_wet_finish"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )
    df = df.drop(columns=["_wet_finish"])
    return df


def add_championship_pressure(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Championship pressure index: how much ground a driver must recover.

    championship_pressure = clip(points_gap_to_leader / max_remaining_pts, 0, 1)
    max_remaining_pts = 25 * remaining_races_in_season (floor 1 to avoid /0)

    Requires add_championship_standing() to be called first (needs cum_points_before).
    Leaders get 0; mathematically eliminated drivers get 1.
    """
    df = race_df.copy()

    if "cum_points_before" not in df.columns:
        df["championship_pressure"] = 0.0
        return df

    leader_pts = df.groupby(["year", "round"])["cum_points_before"].transform("max")
    season_max_round = df.groupby("year")["round"].transform("max")
    remaining_races = (season_max_round - df["round"]).clip(lower=0)
    max_remaining = (25 * remaining_races).clip(lower=1)
    gap = (leader_pts - df["cum_points_before"]).clip(lower=0)

    df["championship_pressure"] = (gap / max_remaining).clip(0.0, 1.0)
    return df


def add_teammate_quali_delta(qual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quali gap vs teammate: driver's quali_gap_to_pole - teammate's quali_gap_to_pole.
    Positive = slower than teammate. Negative = faster than teammate.

    If no teammate row exists in the same year/round/team group, value is NaN.
    """
    df = qual_df.copy()
    df["quali_gap_to_pole_s"] = pd.to_numeric(df.get("quali_gap_to_pole_s"), errors="coerce")

    grp = df.groupby(["year", "round", "team"])["quali_gap_to_pole_s"]
    team_sum = grp.transform("sum")
    team_count = grp.transform("count")

    teammate_count = (team_count - 1).clip(lower=0)
    teammate_avg = (team_sum - df["quali_gap_to_pole_s"]) / teammate_count.replace(0, np.nan)

    df["teammate_quali_delta"] = df["quali_gap_to_pole_s"] - teammate_avg
    return df

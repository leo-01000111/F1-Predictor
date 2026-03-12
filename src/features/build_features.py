"""
Main feature engineering pipeline.

Combines all raw data sources into a unified feature matrix ready for model training.

Usage:
    from src.features.build_features import build_feature_matrix, load_feature_matrix

Output:
    data/processed/feature_matrix.parquet
    - One row per driver per race
    - All features + targets (p1, p2, p3, podium, finish_position)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.features.driver_features import (
    add_driver_form,
    add_championship_standing,
    add_dnf_rate,
    add_circuit_history,
    add_teammate_quali_delta,
    add_driver_start_delta,
    add_elo_ratings,
    add_wet_performance,
    add_championship_pressure,
)
from src.features.circuit_features import add_grid_conversion
from src.features.team_features import (
    add_constructor_standing,
    add_team_reliability,
    add_pit_stop_performance,
    add_car_pace_delta,
    add_team_rolling_form,
)
from src.features.weather_features import merge_weather
from src.features.practice_features import merge_practice_features

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Circuit metadata lookups (hardcoded — changes very rarely)
# --------------------------------------------------------------------------- #

# Street / semi-street circuits.  Albert Park counts as semi-street.
STREET_CIRCUITS: frozenset[str] = frozenset({
    "monaco", "singapore", "baku", "miami", "las_vegas", "jeddah", "albert_park",
})

# Historical SC+VSC rate: fraction of laps run under safety car at each circuit
# (hybrid era, approximate).  Unlisted circuits default to _DEFAULT_SC_RATE.
CIRCUIT_SC_RATE: dict[str, float] = {
    "baku": 0.36,
    "monaco": 0.30,
    "singapore": 0.26,
    "jeddah": 0.22,
    "albert_park": 0.20,
    "interlagos": 0.19,
    "miami": 0.17,
    "suzuka": 0.16,
    "spa": 0.15,
    "silverstone": 0.13,
    "las_vegas": 0.12,
    "bahrain": 0.11,
    "monza": 0.10,
    "hungaroring": 0.10,
    "abu_dhabi": 0.09,
    "zandvoort": 0.09,
    "red_bull_ring": 0.09,
    "barcelona": 0.08,
    "imola": 0.14,
    "losail": 0.08,
    "cota": 0.11,
    "mexico_city": 0.09,
}
_DEFAULT_SC_RATE: float = 0.12

# Race start tire compound encoding: Soft=0, Medium=1, Hard=2, Intermediate=3, Wet=4
COMPOUND_MAP: dict[str, int] = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4,
}
_DEFAULT_COMPOUND: int = 1  # Medium fallback when tire data unavailable

# Historical modal start compound per circuit (hybrid era).
# Unlisted circuits default to Medium (1).
CIRCUIT_TYPICAL_COMPOUND: dict[str, int] = {
    "monaco": 0,       # Soft
    "singapore": 0,    # Soft
    "hungaroring": 0,  # Soft
    "baku": 1,         # Medium
    "miami": 1,        # Medium
    "albert_park": 1,  # Medium
    "jeddah": 1,       # Medium
    "bahrain": 1,      # Medium
    "spa": 1,          # Medium
    "silverstone": 1,  # Medium
    "monza": 2,        # Hard (long straights favour hard)
    "abu_dhabi": 1,    # Medium
}
_DEFAULT_TYPICAL_COMPOUND: int = 1

# Final feature column list (used by models)
FEATURE_COLS = [
    # Qualifying — circuit-invariant
    "grid_position",
    # quali_gap_to_pole_s removed (circuit-variant; quali_gap_relative is strictly better)
    "quali_gap_relative",           # quali_gap / max_gap_in_race (0–1): circuit-invariant
    "grid_position_rel",            # grid_position / n_drivers_in_race (0–1)
    "teammate_quali_delta",
    # Car pace (constructor level, isolates car from driver)
    "car_pace_delta_pct",           # (team_median_quali - fastest_team_median) / fastest_team_median * 100, clipped 3%
    # Driver form
    "driver_form_5",
    "driver_form_10",
    "driver_form_ewm",              # exponential weighted (span=5): recent races count more
    # Start performance
    "driver_start_delta_avg",       # rolling 10-race mean of grid_pos - lap1_pos (positive = gains places)
    # Circuit history
    "circuit_avg_finish",
    "circuit_podiums",
    "circuit_appearances",
    # Circuit type
    "is_street_circuit",            # 1 for Monaco/Singapore/Baku/Miami/Las Vegas/Jeddah/Albert Park
    "circuit_safety_car_rate",      # historical fraction of laps under SC/VSC at this circuit
    # Reliability
    "dnf_rate",
    # Championship
    "championship_position",
    "cum_points_before",
    # Constructor
    "constructor_standing",
    "team_cum_points",
    "team_reliability_rate",
    # Weather
    "temp_c_mean",
    "precip_mm_total",
    "windspeed_ms_mean",
    "humidity_pct_mean",
    "is_wet_race",
    # Free practice
    "fp_best_gap_s",
    "fp2_best_gap_s",
    "fp_long_run_delta",
    # fp_total_laps removed (near-zero SHAP; total laps is strategic, not a reliability signal)
    # Tire strategy
    "start_compound_code",          # Soft=0, Medium=1, Hard=2, Intermediate=3, Wet=4; fallback=1
    "compound_aggression_delta",    # start_compound_code - circuit_typical_start_compound
    # Season progression
    "round_fraction",   # (round-1)/(total_rounds-1): 0=opener, 1=finale; captures early vs late season dynamics
    # Driver Elo
    "driver_elo",               # pre-race Elo from pairwise head-to-head results (base=1500)
    # Wet-race performance
    "driver_wet_avg_finish",    # rolling avg finish in confirmed wet races (expanding, min_periods=3)
    # Circuit grid conversion
    "circuit_grid_conversion",  # historical avg finish from this grid bin at this circuit
    # Championship pressure
    "championship_pressure",    # points_gap_to_leader / max_remaining_pts, clipped [0,1]
    # Team rolling form
    "team_form_5",              # team's rolling best finish, last 5 races
    # Grid penalty
    "grid_penalty_places",      # max(0, grid_position - quali_position): places dropped due to penalties
    # Pit stop performance
    "team_pit_delta_s",         # team median pit time - field median (negative = faster)
    # Sprint result
    "sprint_position_rel",      # sprint_position / n_starters (0.5 = neutral for non-sprint weekends)
]

TARGET_COLS = ["p1", "p2", "p3", "podium", "finish_position"]

# Identifier columns (preserved but not features)
ID_COLS = ["year", "round", "circuit_key", "race_date", "driver", "team"]

# Categorical columns that need encoding
CATEGORICAL_COLS = ["driver", "team", "circuit_key"]


def build_feature_matrix(
    race_df: pd.DataFrame,
    quali_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    practice_df: Optional[pd.DataFrame] = None,
    lap_extras_df: Optional[pd.DataFrame] = None,
    pit_df: Optional[pd.DataFrame] = None,
    sprint_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw DataFrames.

    All lookback features use shift(1) to avoid target leakage.
    """
    logger.info("Building feature matrix...")

    # ------------------------------------------------------------------ #
    # 0. Merge lap extras (lap1_position + start_compound) into race_df
    # ------------------------------------------------------------------ #
    # These columns activate driver_start_delta_avg and start_compound_code.
    # If not provided, those features fall back to 0.0 / Medium respectively.
    df = race_df.copy()
    if lap_extras_df is not None and not lap_extras_df.empty:
        extras_cols = [c for c in ["year", "round", "driver", "lap1_position", "start_compound"]
                       if c in lap_extras_df.columns]
        extras_sub = lap_extras_df[extras_cols]
        df = df.merge(extras_sub, on=["year", "round", "driver"], how="left")
        n_lap1 = int(df["lap1_position"].notna().sum()) if "lap1_position" in df.columns else 0
        n_cmp = int(df["start_compound"].notna().sum()) if "start_compound" in df.columns else 0
        logger.info(
            f"Merged lap extras: {n_lap1}/{len(df)} rows with lap1_position, "
            f"{n_cmp}/{len(df)} rows with start_compound"
        )
    else:
        logger.debug("No lap_extras_df provided; driver_start_delta_avg and start_compound_code will be zero-filled")

    # ------------------------------------------------------------------ #
    # 1. Driver form and championship features (from race results)
    # ------------------------------------------------------------------ #
    df = add_driver_form(df, windows=[5, 10])
    df = add_championship_standing(df)
    df = add_dnf_rate(df)
    df = add_circuit_history(df, circuit_history_seasons=5)
    df = add_driver_start_delta(df)
    df = add_elo_ratings(df)
    df = add_championship_pressure(df)
    df = add_grid_conversion(df)

    # Team features
    df = add_constructor_standing(df)
    df = add_team_reliability(df)
    df = add_team_rolling_form(df)
    df = add_pit_stop_performance(df, pit_df=pit_df)

    # ------------------------------------------------------------------ #
    # 2. Merge qualifying features
    # ------------------------------------------------------------------ #
    quali_sub = quali_df[["year", "round", "driver", "quali_position", "quali_gap_to_pole_s"]].copy()
    # Rename quali_position -> grid_position for races where they match
    # (Sprint weekends may differ; for standard use they're identical)
    if "grid_position" not in df.columns:
        quali_sub = quali_sub.rename(columns={"quali_position": "grid_position"})
    else:
        quali_sub = quali_sub.rename(columns={"quali_position": "quali_position_from_q"})

    df = df.merge(quali_sub, on=["year", "round", "driver"], how="left")

    # Fill grid_position from qualifying if not already in race results
    if "grid_position" not in df.columns:
        df["grid_position"] = df.get("quali_position_from_q", np.nan)

    if "quali_gap_to_pole_s" not in df.columns:
        df["quali_gap_to_pole_s"] = np.nan

    # Teammate delta (requires quali data)
    quali_with_delta = add_teammate_quali_delta(
        df[["year", "round", "driver", "team", "quali_gap_to_pole_s"]].dropna()
    )
    df = df.merge(
        quali_with_delta[["year", "round", "driver", "teammate_quali_delta"]],
        on=["year", "round", "driver"],
        how="left",
    )

    # Car pace delta: constructor-level median quali gap vs fastest constructor
    df = add_car_pace_delta(df)

    # ------------------------------------------------------------------ #
    # 2b. Grid penalty (requires quali_position_from_q and grid_position)
    # ------------------------------------------------------------------ #
    if "quali_position_from_q" in df.columns and "grid_position" in df.columns:
        qp = pd.to_numeric(df["quali_position_from_q"], errors="coerce")
        gp = pd.to_numeric(df["grid_position"], errors="coerce")
        df["grid_penalty_places"] = (gp - qp).clip(lower=0).fillna(0.0)
    else:
        df["grid_penalty_places"] = 0.0

    # ------------------------------------------------------------------ #
    # 2c. Sprint results
    # ------------------------------------------------------------------ #
    if sprint_df is not None and not sprint_df.empty:
        sprint_sub = sprint_df[["year", "round", "driver", "sprint_position", "n_sprint_starters"]].copy()
        df = df.merge(sprint_sub, on=["year", "round", "driver"], how="left")
        sp = pd.to_numeric(df.get("sprint_position"), errors="coerce")
        ns = pd.to_numeric(df.get("n_sprint_starters"), errors="coerce").fillna(20)
        df["sprint_position_rel"] = (sp / ns).fillna(0.5)
        df = df.drop(columns=["sprint_position", "n_sprint_starters"], errors="ignore")
    else:
        df["sprint_position_rel"] = 0.5

    # ------------------------------------------------------------------ #
    # 3. Merge weather
    # ------------------------------------------------------------------ #
    if weather_df is not None and not weather_df.empty:
        df = merge_weather(df, weather_df)
    else:
        for col in ["temp_c_mean", "precip_mm_total", "windspeed_ms_mean", "humidity_pct_mean"]:
            df[col] = np.nan
        df["is_wet_race"] = 0

    # ------------------------------------------------------------------ #
    # 3b. Wet-race driver performance (requires is_wet_race from weather)
    # ------------------------------------------------------------------ #
    df = add_wet_performance(df)

    # ------------------------------------------------------------------ #
    # 4. Merge free practice features (FP1/FP2/FP3)
    # ------------------------------------------------------------------ #
    df = merge_practice_features(df, practice_df)

    # ------------------------------------------------------------------ #
    # 4b. Derived quali features (computed after quali merge)
    # ------------------------------------------------------------------ #

    # Clip raw gap at 5 s — values above this are almost always artefacts
    # (wet sessions where slow cars didn't set a flying lap, DNFs in Q1, etc.).
    # Austrian GP had a 62-second "gap" in the raw data.
    if "quali_gap_to_pole_s" in df.columns:
        df["quali_gap_to_pole_s"] = df["quali_gap_to_pole_s"].clip(upper=5.0)

    # quali_gap_relative: normalize gap within each race so the feature means
    # the same thing regardless of circuit (Monaco max ~2s vs Monza max ~1s).
    # Using max gap in the race as denominator; clip again to guard against /0.
    def _relative_gap(grp: pd.Series) -> pd.Series:
        max_gap = grp.clip(upper=5.0).max()
        if pd.isna(max_gap) or max_gap < 0.01:
            return pd.Series(np.nan, index=grp.index)
        return grp.clip(upper=5.0) / max_gap

    df["quali_gap_relative"] = (
        df.groupby(["year", "round"])["quali_gap_to_pole_s"]
        .transform(_relative_gap)
    )

    # grid_position_rel: relative grid position within each race (0 = pole, 1 = last).
    # n_drivers varies race-to-race (DNFs at start, retirements before grid, etc.).
    def _relative_grid(grp: pd.Series) -> pd.Series:
        n = grp.max()
        if pd.isna(n) or n < 1:
            return pd.Series(np.nan, index=grp.index)
        return (grp - 1) / (n - 1)  # pole → 0.0, last → 1.0

    df["grid_position_rel"] = (
        df.groupby(["year", "round"])["grid_position"]
        .transform(_relative_grid)
    )

    # round_fraction: (round-1)/(max_round_in_season-1)
    # 0.0 = season opener, 1.0 = season finale.
    # Captures early-vs-late-season dynamics (development race, tyre knowledge, etc.).
    if "round" in df.columns and "year" in df.columns:
        season_max = df.groupby("year")["round"].transform("max")
        df["round_fraction"] = (df["round"] - 1) / (season_max - 1).clip(lower=1)
    else:
        df["round_fraction"] = 0.0

    # ------------------------------------------------------------------ #
    # 4c. Circuit type and safety car rate features
    # ------------------------------------------------------------------ #
    if "circuit_key" in df.columns:
        df["is_street_circuit"] = df["circuit_key"].str.lower().isin(STREET_CIRCUITS).astype(float)
        df["circuit_safety_car_rate"] = (
            df["circuit_key"].str.lower().map(CIRCUIT_SC_RATE).fillna(_DEFAULT_SC_RATE)
        )
    else:
        df["is_street_circuit"] = 0.0
        df["circuit_safety_car_rate"] = _DEFAULT_SC_RATE

    # ------------------------------------------------------------------ #
    # 4d. Tire strategy features
    # ------------------------------------------------------------------ #
    # start_compound_code: encode starting tire compound
    # Source: 'start_compound' column (if available from data collection).
    # Fallback: Medium (1) when tire data is not present.
    if "start_compound" in df.columns:
        df["start_compound_code"] = (
            df["start_compound"].str.upper().map(COMPOUND_MAP).fillna(_DEFAULT_COMPOUND)
        )
    else:
        df["start_compound_code"] = float(_DEFAULT_COMPOUND)

    # compound_aggression_delta: start_compound_code - circuit_typical_start_compound
    # Negative = more aggressive (softer) than circuit norm; positive = conservative.
    if "circuit_key" in df.columns:
        typical = df["circuit_key"].str.lower().map(CIRCUIT_TYPICAL_COMPOUND).fillna(_DEFAULT_TYPICAL_COMPOUND)
    else:
        typical = _DEFAULT_TYPICAL_COMPOUND
    df["compound_aggression_delta"] = df["start_compound_code"] - typical

    # ------------------------------------------------------------------ #
    # 5. Ensure all feature columns exist; fill missing with medians
    # ------------------------------------------------------------------ #
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Impute missing numeric features with column median
    for col in FEATURE_COLS:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)

    # ------------------------------------------------------------------ #
    # 6. Encode categoricals as integer codes (for NN embeddings)
    # ------------------------------------------------------------------ #
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            # Reserve 0 for unknown categories at inference time.
            df[f"{col}_idx"] = df[col].astype("category").cat.codes + 1
    logger.info(f"Feature matrix: {len(df)} rows - {len(FEATURE_COLS)} features")
    return df


def save_feature_matrix(df: pd.DataFrame) -> None:
    path = PROCESSED_DIR / "feature_matrix.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved feature matrix -> {path}")


def _backfill_missing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Backfill feature columns for compatibility with older cached matrices."""
    patched = df.copy()
    patched_cols: list[str] = []

    # round_fraction can be reconstructed from season progression.
    if "round_fraction" not in patched.columns:
        if "round" in patched.columns and "year" in patched.columns:
            season_max = patched.groupby("year")["round"].transform("max")
            patched["round_fraction"] = (patched["round"] - 1) / (season_max - 1).clip(lower=1)
        else:
            patched["round_fraction"] = 0.0
        patched_cols.append("round_fraction")

    # Ensure all expected feature columns exist.
    for col in FEATURE_COLS:
        if col not in patched.columns:
            patched[col] = np.nan
            patched_cols.append(col)

    # Keep all feature columns numeric and imputed.
    for col in FEATURE_COLS:
        series = pd.to_numeric(patched[col], errors="coerce")
        if series.isna().any():
            median = series.median()
            fill_value = median if not np.isnan(median) else 0.0
            series = series.fillna(fill_value)
        patched[col] = series

    return patched, patched_cols


def load_feature_matrix() -> pd.DataFrame:
    path = PROCESSED_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run build_feature_matrix() first. Expected: {path}")

    df = pd.read_parquet(path)
    df, patched_cols = _backfill_missing_features(df)
    if patched_cols:
        logger.warning(
            "Loaded legacy feature matrix missing columns; backfilled: "
            + ", ".join(sorted(set(patched_cols)))
        )
        try:
            df.to_parquet(path, index=False)
            logger.info(f"Persisted repaired feature matrix -> {path}")
        except Exception as exc:
            logger.warning(f"Could not persist repaired feature matrix: {exc}")
    return df


def get_X_y(df: pd.DataFrame, target: str = "p1") -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target Series y."""
    X = df[FEATURE_COLS].copy()
    y = df[target].copy()
    return X, y


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from src.data_collection.f1_historical import (
        load_race_results, load_quali_results, load_practice_results,
        load_lap_extras, load_pit_stops, load_sprint_results,
    )
    from src.data_collection.weather import load_weather_history

    race_df = load_race_results()
    quali_df = load_quali_results()

    try:
        weather_df = load_weather_history()
    except FileNotFoundError:
        logger.warning("Weather data not found, proceeding without it")
        weather_df = None

    try:
        practice_df = load_practice_results()
    except FileNotFoundError:
        logger.warning("Practice data not found, proceeding without it")
        practice_df = None

    try:
        lap_extras_df = load_lap_extras()
        logger.info(f"Lap extras: {len(lap_extras_df)} rows")
    except FileNotFoundError:
        logger.info("No lap_extras.parquet found; start_delta/compound will be zero. "
                    "Run: python scripts/backfill_lap_extras.py --start 2018 --end 2025 --merge")
        lap_extras_df = None

    try:
        pit_df = load_pit_stops()
        logger.info(f"Pit stop data: {len(pit_df)} rows")
    except FileNotFoundError:
        logger.info("No pit_stops.parquet found; team_pit_delta_s will be NaN. "
                    "Run: python scripts/backfill_pit_sprint.py --pit --start 2014 --end 2025 --merge")
        pit_df = None

    try:
        sprint_df = load_sprint_results()
        logger.info(f"Sprint data: {len(sprint_df)} rows")
    except FileNotFoundError:
        logger.info("No sprint_results.parquet found; sprint_position_rel will be 0.5 (neutral). "
                    "Run: python scripts/backfill_pit_sprint.py --sprint --start 2021 --end 2025 --merge")
        sprint_df = None

    fm = build_feature_matrix(
        race_df=race_df,
        quali_df=quali_df,
        weather_df=weather_df,
        practice_df=practice_df,
        lap_extras_df=lap_extras_df,
        pit_df=pit_df,
        sprint_df=sprint_df,
    )
    save_feature_matrix(fm)
    logger.info("Done.")




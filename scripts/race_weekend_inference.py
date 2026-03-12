# -*- coding: utf-8 -*-
"""
Race Weekend Inference Script
==============================
Run this script after qualifying concludes on Saturday.
It will automatically:
  1. Pull qualifying results from OpenF1
  2. Pull FP1/FP2/FP3 lap metrics from OpenF1
  3. Attempt to fetch start tire compounds (available post-race for retroactive use)
  4. Fetch race-day weather forecast from Open-Meteo
  5. Build features and run the XGBoost model
  6. Save predictions to data/predictions/{year}_R{round}.json
  7. The Streamlit dashboard auto-refreshes from the latest prediction file

2026 workflow (22-driver grid):
  - Grid size is handled dynamically; no config changes needed
  - start_compound will default to Medium pre-race (correct behavior)
  - After each race completes, run update_race_result.py to record actuals

Usage:
    # After qualifying Saturday:
    python scripts/race_weekend_inference.py --year 2026 --round N

    # Dry run with a past race (for testing):
    python scripts/race_weekend_inference.py --year 2024 --round 10 --dry-run

    # After the race -- record actuals:
    python scripts/update_race_result.py --year 2026 --round N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def main(
    year: int,
    round_num: int,
    dry_run: bool = False,
) -> None:

    from src.data_collection.f1_live import (
        get_live_practice_dataframe,
        get_live_quali_dataframe,
        get_completed_season_race_results,
        get_start_compounds,
    )
    from src.data_collection.weather import get_race_day_forecast
    from src.data_collection.f1_historical import load_race_results, load_pit_stops, load_sprint_results
    from src.inference.predict import run_inference

    # ------------------------------------------------------------------ #
    # 1. Load historical race data (for form/circuit features)
    # ------------------------------------------------------------------ #
    logger.info("Loading historical race data...")
    try:
        race_df_history = load_race_results()
    except FileNotFoundError:
        logger.error(
            "Historical race data not found. "
            "Run: python -m src.data_collection.f1_historical first."
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1b. Append completed in-season races so rolling features are current
    # ------------------------------------------------------------------ #
    if round_num > 1:
        logger.info(f"Fetching completed {year} race results (rounds 1-{round_num - 1})...")
        inseason_df = get_completed_season_race_results(year, max_round=round_num)
        if not inseason_df.empty:
            # Drop any rows from the current year already in historical data
            # (historical may include some completed 2026 rounds if added manually)
            existing_keys = set(
                zip(race_df_history["year"], race_df_history["round"])
            )
            inseason_new = inseason_df[
                ~inseason_df.apply(lambda r: (r["year"], r["round"]) in existing_keys, axis=1)
            ]
            if not inseason_new.empty:
                race_df_history = pd.concat(
                    [race_df_history, inseason_new], ignore_index=True
                ).sort_values(["year", "round"])
                logger.info(
                    f"Appended {len(inseason_new)} in-season rows "
                    f"({inseason_new['round'].nunique()} rounds) to historical data"
                )
        else:
            logger.info("No completed in-season races found (first round of season?)")

    # ------------------------------------------------------------------ #
    # 2. Pull live qualifying results
    # ------------------------------------------------------------------ #
    logger.info(f"Fetching qualifying results for {year} R{round_num}...")
    quali_df = get_live_quali_dataframe(year, round_num)

    if quali_df.empty:
        logger.error("No qualifying data returned. Is qualifying finished?")
        sys.exit(1)

    logger.info(f"Got quali results for {len(quali_df)} drivers")
    print(quali_df[["driver_abbr", "team", "quali_position", "quali_gap_to_pole_s"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # 3. Pull live FP1/FP2/FP3 data
    # ------------------------------------------------------------------ #
    logger.info(f"Fetching FP1/FP2/FP3 results for {year} R{round_num}...")
    practice_df = get_live_practice_dataframe(year, round_num)
    if practice_df.empty:
        logger.warning("Could not fetch any FP sessions. Proceeding without FP features.")
    else:
        logger.info(f"Fetched practice data for {practice_df['driver'].nunique()} drivers")

    # ------------------------------------------------------------------ #
    # 3b. Fetch start tire compounds (available after race, not pre-race)
    # ------------------------------------------------------------------ #
    logger.info(f"Attempting to fetch start compounds for {year} R{round_num}...")
    compounds = get_start_compounds(year, round_num)
    if compounds:
        quali_df["start_compound"] = quali_df["driver_abbr"].map(compounds)
        logger.info(f"Start compounds applied: {compounds}")
    else:
        logger.info("Start compound data not yet available (pre-race is normal). Defaulting to Medium.")

    # ------------------------------------------------------------------ #
    # 4. Get race-day weather forecast
    # ------------------------------------------------------------------ #
    circuit_key = quali_df["circuit_key"].iloc[0] if "circuit_key" in quali_df.columns else ""
    race_date = quali_df["race_date"].iloc[0] if "race_date" in quali_df.columns else ""

    logger.info(f"Fetching weather forecast for {circuit_key} on {race_date}...")
    weather_dict = get_race_day_forecast(circuit_key, race_date)

    if not weather_dict:
        logger.warning("Could not fetch weather forecast, using defaults")
        weather_dict = {
            "temp_c_mean": 20.0,
            "precip_mm_total": 0.0,
            "windspeed_ms_mean": 5.0,
            "humidity_pct_mean": 50.0,
            "is_wet_race": 0,
        }

    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    temp_c = _safe_float(weather_dict.get("temp_c_mean"), 20.0)
    rain_mm = _safe_float(weather_dict.get("precip_mm_total"), 0.0)
    wind_ms = _safe_float(weather_dict.get("windspeed_ms_mean"), 0.0)

    logger.info(
        f"Weather: {temp_c:.1f}C | "
        f"Rain: {rain_mm:.1f}mm | "
        f"Wind: {wind_ms:.1f}m/s | "
        f"Wet: {'Yes' if weather_dict.get('is_wet_race') else 'No'}"
    )
    # ------------------------------------------------------------------ #
    # 5. Run inference
    # ------------------------------------------------------------------ #
    if dry_run:
        logger.info("DRY RUN: skipping model inference, printing data only")
        print("\nQuali results:")
        print(quali_df.to_string(index=False))
        if not practice_df.empty:
            print("\nPractice features:")
            print(practice_df.to_string(index=False))
        print(f"\nWeather: {weather_dict}")
        return

    # ------------------------------------------------------------------ #
    # 5b. Load supplemental data (pit stops + sprint results)
    # ------------------------------------------------------------------ #
    pit_df = None
    sprint_df = None
    try:
        pit_df = load_pit_stops()
        logger.info(f"Pit stop data loaded: {len(pit_df)} rows")
    except FileNotFoundError:
        logger.info("No pit_stops.parquet; team_pit_delta_s will be NaN. "
                    "Run: python scripts/backfill_pit_sprint.py --pit --merge")
    try:
        sprint_df = load_sprint_results()
        logger.info(f"Sprint data loaded: {len(sprint_df)} rows")
    except FileNotFoundError:
        logger.info("No sprint_results.parquet; sprint_position_rel will be 0.5. "
                    "Run: python scripts/backfill_pit_sprint.py --sprint --merge")

    logger.info("Running model inference...")
    result = run_inference(
        year=year,
        round_num=round_num,
        quali_df=quali_df,
        race_df_history=race_df_history,
        weather_dict=weather_dict,
        practice_df=practice_df,
        pit_df=pit_df,
        sprint_df=sprint_df,
    )

    # ------------------------------------------------------------------ #
    # 6. Print summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"F1 RACE PREDICTIONS -- {year} Round {round_num} ({circuit_key.upper()})")
    print(f"Race Date: {race_date}")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Driver':<12} {'Team':<20} {'P(Win)':>8} {'P(Pod)':>8}")
    print("-" * 55)
    for i, p in enumerate(result["predictions"], 1):
        print(
            f"{i:<5} {p['driver']:<12} {p['team']:<20} "
            f"{p['p_win']:>7.1%} {p['p_podium']:>7.1%}"
        )

    print(f"\nPredictions saved to: data/predictions/{year}_R{round_num:02d}.json")
    print("Launch the dashboard: streamlit run src/ui/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Weekend Inference")
    parser.add_argument("--year", type=int, required=True, help="Race year (e.g. 2025)")
    parser.add_argument("--round", type=int, required=True, dest="round_num", help="Round number")
    parser.add_argument("--dry-run", action="store_true", help="Print data without running model")
    args = parser.parse_args()

    main(
        year=args.year,
        round_num=args.round_num,
        dry_run=args.dry_run,
    )


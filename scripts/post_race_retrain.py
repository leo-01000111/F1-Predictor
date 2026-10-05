# -*- coding: utf-8 -*-
"""
Post-Race Retrain Script
========================
Run this after a race completes and you have recorded the actual result.

Workflow:
  1. Fetch all completed year races from FastF1 and merge into data/raw/race_results.parquet
  2. Rebuild the full feature matrix (data/processed/feature_matrix.parquet)
  3. Retrain XGBoost on the updated dataset (2014 -> current)
  4. Print accuracy before/after

Usage:
    python scripts/post_race_retrain.py --year 2026 --round 3

Then re-run inference for the next round once qualifying is done:
    python scripts/race_weekend_inference.py --year 2026 --round 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def main(year: int, round_num: int) -> None:
    from src.data_collection.f1_live import get_completed_season_race_results
    from src.data_collection.f1_historical import (
        load_quali_results,
        load_practice_results,
        load_lap_extras,
        load_pit_stops,
        load_sprint_results,
    )
    from src.data_collection.weather import load_weather_history
    from src.features.build_features import build_feature_matrix, save_feature_matrix
    from src.models.model_registry import load_active_model_metadata
    from scripts.quick_train import main as quick_train_main

    import pandas as pd

    RAW_DIR = ROOT / "data" / "raw"
    race_path = RAW_DIR / "race_results.parquet"

    # ------------------------------------------------------------------ #
    # 1. Fetch completed in-season races
    # ------------------------------------------------------------------ #
    logger.info(f"Fetching completed {year} race results via FastF1...")
    inseason_df = get_completed_season_race_results(year)
    if inseason_df.empty:
        logger.error(
            f"No completed race data found for {year}. "
            "Has the race finished and FastF1 published results?"
        )
        sys.exit(1)

    n_rounds = int(inseason_df["round"].nunique())
    logger.info(f"Fetched {len(inseason_df)} rows from {n_rounds} completed {year} rounds.")

    # ------------------------------------------------------------------ #
    # 2. Merge into raw parquet
    # ------------------------------------------------------------------ #
    logger.info("Merging into historical race_results.parquet...")
    if race_path.exists():
        existing = pd.read_parquet(race_path)
        old_count = int((existing["year"] == year).sum())
        base = existing[existing["year"] != year]
        merged = pd.concat([base, inseason_df], ignore_index=True).sort_values(
            ["year", "round", "driver"]
        )
        logger.info(
            f"Replaced {old_count} old {year} rows with {len(inseason_df)} fresh rows. "
            f"Total: {len(merged)} rows across {merged['year'].nunique()} seasons."
        )
    else:
        merged = inseason_df.sort_values(["year", "round", "driver"])
        logger.warning("No existing race_results.parquet found. Writing current year data only.")

    merged.to_parquet(race_path, index=False)

    # ------------------------------------------------------------------ #
    # 3. Rebuild feature matrix
    # ------------------------------------------------------------------ #
    logger.info("Rebuilding feature matrix...")

    try:
        quali_df = load_quali_results()
    except FileNotFoundError:
        logger.error("quali_results.parquet not found. Run f1_historical data collection first.")
        sys.exit(1)

    weather_df = None
    practice_df = None
    lap_extras_df = None
    pit_df = None
    sprint_df = None

    try:
        weather_df = load_weather_history()
    except FileNotFoundError:
        logger.warning("No weather data found. Proceeding without weather features.")
    try:
        practice_df = load_practice_results()
    except FileNotFoundError:
        logger.warning("No practice data found. FP features will be zero.")
    try:
        lap_extras_df = load_lap_extras()
        logger.info(f"Lap extras: {len(lap_extras_df)} rows")
    except FileNotFoundError:
        logger.warning("No lap_extras.parquet. Start delta/compound features will be zero.")
    try:
        pit_df = load_pit_stops()
        logger.info(f"Pit stop data: {len(pit_df)} rows")
    except FileNotFoundError:
        logger.warning("No pit_stops.parquet. team_pit_delta_s will be NaN.")
    try:
        sprint_df = load_sprint_results()
        logger.info(f"Sprint data: {len(sprint_df)} rows")
    except FileNotFoundError:
        logger.warning("No sprint_results.parquet. sprint_position_rel will be neutral.")

    fm = build_feature_matrix(
        race_df=merged,
        quali_df=quali_df,
        weather_df=weather_df,
        practice_df=practice_df,
        lap_extras_df=lap_extras_df,
        pit_df=pit_df,
        sprint_df=sprint_df,
    )
    save_feature_matrix(fm)
    logger.info(f"Feature matrix saved: {len(fm)} rows, {fm.shape[1]} columns.")

    # ------------------------------------------------------------------ #
    # 4. Retrain
    # ------------------------------------------------------------------ #
    old_acc = float(
        (load_active_model_metadata() or {}).get("winner_accuracy_holdout", 0.0) or 0.0
    )
    logger.info(f"Current holdout accuracy before retrain: {old_acc:.1%}")
    logger.info("Retraining XGBoost on updated dataset (2014 -> current year)...")

    try:
        quick_train_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            logger.error(f"quick_train exited with code {exc.code}")
            sys.exit(int(exc.code))

    new_acc = float(
        (load_active_model_metadata() or {}).get("winner_accuracy_holdout", 0.0) or 0.0
    )
    delta = new_acc - old_acc
    sign = "+" if delta >= 0 else ""

    logger.info("=" * 50)
    logger.info(f"Post-race retrain complete for {year} (through R{round_num})")
    logger.info(f"  Seasons in training: {sorted(merged['year'].unique().tolist())}")
    logger.info(f"  {year} rounds included: {n_rounds}")
    logger.info(f"  Holdout accuracy: {old_acc:.1%} -> {new_acc:.1%} ({sign}{delta:.1%})")
    logger.info("=" * 50)
    logger.info(f"Next step: run inference after R{round_num + 1} qualifying:")
    logger.info(f"  python scripts/race_weekend_inference.py --year {year} --round {round_num + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-race retrain: fetch results, rebuild features, retrain XGB")
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g. 2026)")
    parser.add_argument("--round", type=int, required=True, dest="round_num", help="Race round just completed")
    args = parser.parse_args()
    main(year=args.year, round_num=args.round_num)

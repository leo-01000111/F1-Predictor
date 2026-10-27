# -*- coding: utf-8 -*-
"""
Predict a race that has already run, using only data from before it.

For rounds that were missed live. Inputs match what race_weekend_inference.py
would have had after qualifying:
  - race history strictly before the round (form, standings, Elo, circuit history)
  - qualifying order and gaps from quali_results.parquet (no grid penalties,
    exactly as at live inference); starters missing from it go to the back
  - pit stops before the round; sprint results up to and including it (the
    sprint runs before the race)
  - the race-day weather forecast as it stood the day before (Open-Meteo
    previous-runs archive), not the measured weather
  - the active model, whose production pass never trains on the latest season

If the archived forecast is unavailable, measured weather is used with a warning.

Writes data/predictions/{year}_R{round}.json like a live run. Then record the
result: python scripts/update_race_result.py --year Y --round N

Usage:
    python scripts/predict_past_race.py --year 2026 --round 18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger


def main(year: int, round_num: int) -> None:
    from src.data_collection.f1_historical import (
        load_pit_stops,
        load_quali_results,
        load_race_results,
        load_sprint_results,
    )
    from src.data_collection.weather import get_past_race_forecast, load_weather_history
    from src.inference.predict import run_inference

    race = load_race_results()
    quali = load_quali_results()
    weather = load_weather_history()

    def before(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < round_num))]

    def this_round(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["year"] == year) & (df["round"] == round_num)]

    q = this_round(quali).sort_values("quali_position")
    r = this_round(race)
    if q.empty or r.empty:
        logger.error(f"No qualifying or race rows for {year} R{round_num}")
        sys.exit(1)

    # Drivers without a qualifying time still start: OpenF1 lists them at the back
    missing = r[~r["driver"].isin(q["driver"])].sort_values("grid_position")
    if not missing.empty:
        logger.info(f"Not in quali file, added at the back: {missing['driver'].tolist()}")
        back = pd.DataFrame({
            "driver": missing["driver"].values,
            "team": missing["team"].values,
            "quali_position": q["quali_position"].max() + 1 + pd.RangeIndex(len(missing)),
            "quali_gap_to_pole_s": float("nan"),
            "circuit": q["circuit"].iloc[0],
            "circuit_key": q["circuit_key"].iloc[0],
        })
        q = pd.concat([q, back], ignore_index=True)

    quali_df = pd.DataFrame({
        "driver_abbr": q["driver"].values,
        "team": q["team"].values,
        "quali_position": q["quali_position"].values,
        "quali_gap_to_pole_s": q["quali_gap_to_pole_s"].values,
        "circuit": q["circuit"].values,
        "circuit_key": q["circuit_key"].values,
        "race_date": r["race_date"].iloc[0],
    })

    weather_dict = get_past_race_forecast(
        str(r["circuit_key"].iloc[0]), str(r["race_date"].iloc[0]), location=str(r["circuit"].iloc[0])
    )
    source = "day-before forecast"
    if not weather_dict:
        w = this_round(weather)
        if w.empty:
            logger.error(f"No forecast or weather row for {year} R{round_num}")
            sys.exit(1)
        weather_dict = w.drop(columns=["year", "round", "circuit_key", "race_date"], errors="ignore").iloc[0].to_dict()
        source = "MEASURED (forecast unavailable)"
        logger.warning(f"{year} R{round_num}: using measured weather")
    logger.info(f"Weather ({source}): {weather_dict['precip_mm_total']:.1f} mm, wet={int(weather_dict['is_wet_race'])}")

    sprint = load_sprint_results()
    sprint = sprint[(sprint["year"] < year) | ((sprint["year"] == year) & (sprint["round"] <= round_num))]

    run_inference(
        year=year,
        round_num=round_num,
        quali_df=quali_df,
        race_df_history=before(race),
        weather_dict=weather_dict,
        pit_df=before(load_pit_stops()),
        sprint_df=sprint,
    )
    print(f"Wrote data/predictions/{year}_R{round_num:02d}.json")
    print(f"Next: python scripts/update_race_result.py --year {year} --round {round_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a past race from pre-race data only")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    args = parser.parse_args()
    main(args.year, args.round_num)

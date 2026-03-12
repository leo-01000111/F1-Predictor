"""Weather feature engineering — merging weather data into the race/feature matrix."""
from __future__ import annotations

import pandas as pd


def merge_weather(base_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join weather data onto base_df by (year, round).
    Missing weather → filled with global medians.
    """
    weather_cols = [
        "year", "round",
        "temp_c_mean", "temp_c_max",
        "precip_mm_total", "precip_mm_max_hourly",
        "windspeed_ms_mean", "windspeed_ms_max",
        "humidity_pct_mean", "cloudcover_pct_mean",
        "is_wet_race",
    ]
    weather_sub = weather_df[[c for c in weather_cols if c in weather_df.columns]]

    merged = base_df.merge(weather_sub, on=["year", "round"], how="left")

    # Fill missing weather with global median
    numeric_weather = [c for c in weather_cols if c not in ("year", "round", "is_wet_race")]
    for col in numeric_weather:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    if "is_wet_race" in merged.columns:
        merged["is_wet_race"] = merged["is_wet_race"].fillna(0).astype(int)

    return merged
